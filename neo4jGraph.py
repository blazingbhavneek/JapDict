import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import httpx
import neo4j
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import json_repair
except ImportError:
    print("Installing json_repair...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "json_repair"])
    import json_repair

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B-GPTQ-Int8"
MAX_MODEL_LEN = 8192
CHUNK_SIZE = 1000  # Increased chunk size as suggested
CHUNK_OVERLAP = 50  # tokens
BATCH_SIZE = 80  # Max concurrent requests overall (for Pass 1 LLM calls)
MAX_CONCURRENT_DEFS = 80  # Max concurrent requests for definition generation (Pass 2)

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# Paths
TEXT_FILE_PATH = "wiki.txt"
CACHE_DIR = "./neo4j_output"
OUTPUT_TERMS_FILE = CACHE_DIR + "/neo4j_all_terms.json"
OUTPUT_BASE_NAME = CACHE_DIR + "/neo4j_output"
os.makedirs(CACHE_DIR, exist_ok=True)

# Global semaphore for vLLM requests
VLLM_SEMAPHORE = None
TERM_DEF_SEMAPHORE = None


# ----------------------------
# CUSTOM LLM FOR NEO4J_GRAPHRAG (Modified for vLLM Server API)
# ----------------------------
class VLLMLLM(LLMInterface):
    """Wrapper to use vLLM Server API with neo4j-graphrag"""

    def __init__(self, api_base_url: str, api_key: str = "token-abc123"):
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.async_client = httpx.AsyncClient(timeout=120.0)
        self.sync_client = httpx.Client(timeout=120.0)

    def _prepare_request_payload(self, input_text: str) -> dict:
        return {
            "model": "",
            "messages": [{"role": "user", "content": input_text}],
            "temperature": 0.0,
            "top_p": 0.1,
            "max_tokens": 1500,
            "stop": ["</s>"],
        }

    def invoke(self, input: str, **kwargs) -> LLMResponse:
        """Single generation via sync HTTP request."""
        payload = self._prepare_request_payload(input)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self.sync_client.post(
                f"{self.api_base_url}/chat/completions", json=payload, headers=headers
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"].strip()
            return LLMResponse(content=generated_text)
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
            raise
        except KeyError as e:
            print(f"Key error - possibly malformed response: {e}, Response: {result}")
            raise

    async def ainvoke(self, input: str, **kwargs) -> LLMResponse:
        """Async generation via async HTTP request."""
        # Use general semaphore for Pass 1, specific for Pass 2
        semaphore_to_use = (
            TERM_DEF_SEMAPHORE
            if hasattr(asyncio.current_task(), "_is_def_task")
            else VLLM_SEMAPHORE
        )
        async with semaphore_to_use:
            payload = self._prepare_request_payload(input)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            try:
                response = await self.async_client.post(
                    f"{self.api_base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"].strip()
                return LLMResponse(content=generated_text)
            except httpx.HTTPStatusError as e:
                print(
                    f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
                )
                raise
            except httpx.RequestError as e:
                print(f"Request error occurred: {e}")
                raise
            except KeyError as e:
                print(
                    f"Key error - possibly malformed response: {e}, Response: {result}"
                )
                raise


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def chunk_text_by_tokens(
    text: str, tokenizer, chunk_size: int, overlap: int
) -> List[str]:
    """Chunk text by token count with overlap"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks


def extract_json_from_text(text: str) -> Optional[Dict]:
    """Attempt to extract a JSON object from potentially unstructured text."""
    start_idx = text.find("{")
    if start_idx == -1:
        return None
    brace_count = 0
    end_idx = -1
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    if end_idx == -1:
        return None  # Unmatched braces
    json_str = text[start_idx : end_idx + 1]
    try:
        parsed_json = json_repair.loads(json_str)
        return parsed_json
    except json.JSONDecodeError:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON string: {json_str[:100]}...")
            return None


# ----------------------------
# PROMPTS FOR TWO-PASS LOGIC
# ----------------------------

# 1st Pass: Term Extraction Prompt
extraction_prompt_pass1 = """
以下のテキスト断片から、技術的・専門的な用語を抽出してください。
対象とする用語の特性:
- 電力システム、制御、監視、保護、機器、送配電などに関連する固有名詞、複合語、専門用語。
- 例: 遠隔制御、周波数調整、地絡保護、変圧器、送電線、変電所、親局、子局、開閉器、遮断器、同調発電機。
- 複合語（2語以上の組み合わせ）は含める。
- 複合語の部分要素も含める（例：「個別制御」なら「個別」「制御」も検討）。
- 一般的な名詞（機器、装置、システム、データ、情報）は単体では除外（複合語の一部ならOK）。
- 人名、地名、企業名、UI用語（ボタン、クリック）は除外。
- 英字略語（例: HB, EM）は除外。ただし日本語複合語に含まれるなら対象。
抽出条件:
- テキスト中で説明・定義・使用例があるもの、または文脈的に専門用語であると判断されるもの。
- 重複する用語は1つにまとめます。
- 用語のリストをJSON形式で返してください。
テキスト断片:
{chunk_text}
出力形式 (JSONのみ):
{{"terms": ["用語1", "用語2", "用語3"]}}
"""

# 2nd Pass: Definition Generation Prompt
definition_prompt_pass2 = """
用語: {term}
以下の文脈から、この用語の定義を抽出・生成してください。
定義の条件:
- テキストに明示的な定義が含まれる場合（「○○とは..」「○○は..」）、それを抽出。
- テキストに説明や使用例がある場合、それから定義を推論（ただし、テキストに基づいてのみ）。
- テキストに該当する情報がない場合、空文字を返す（推測・外部知識は使用しない）。
定義の形式:
- 1～3文程度の簡潔な説明。
- 専門用語を含める（例：「66kV以上の電圧で電力を送る線路」）。
- 句読点を含める。
文脈:
{context}
出力形式:
成功例: {{"term": "遠隔制御", "definition": "遠隔制御とは、離れた場所から機器を制御する機能。"}}
失敗例: {{"term": "遠隔制御", "definition": ""}}
JSONオブジェクトのみを出力してください。
"""

# ----------------------------
# TWO-PASS PROCESSING LOGIC
# ----------------------------


async def pass1_extract_terms(
    text_path: str, vllm_model: VLLMLLM, cache_run_id: str
) -> Dict[int, List[str]]:
    """First pass: Extract terms from text chunks. Returns a map {chunk_idx: [term1, term2, ...]}."""
    print("\n[1/4] Starting Pass 1: Term Extraction...")
    if not Path(text_path).exists():
        print(f"✗ File not found: {text_path}")
        return {}

    # Load file
    with open(text_path, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"✓ Loaded file: {len(content)} characters")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Chunk text
    print(f"  Chunking text (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_text_by_tokens(content, tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        print(f"  ⊘ No chunks generated")
        return {}
    print(f"  Generated {len(chunks)} chunks")

    # Define cache path for extracted terms
    cache_terms_path = os.path.join(CACHE_DIR, f"{cache_run_id}_pass1_terms.json")

    # Check for cached results
    if os.path.exists(cache_terms_path):
        print(f"  ✓ Loading cached terms from {cache_terms_path}")
        with open(cache_terms_path, "r", encoding="utf-8") as f:
            cached_terms = json.load(f)
        # Convert keys back to int if needed by downstream logic
        return {int(k): v for k, v in cached_terms.items()}

    # Process chunks concurrently for term extraction
    chunk_to_terms = {}
    pending_chunks = [(i, c) for i, c in enumerate(chunks)]
    progress_bar = tqdm(total=len(pending_chunks), desc="Pass 1: Extracting terms")

    while pending_chunks:
        batch_size = min(BATCH_SIZE, len(pending_chunks))
        batch = pending_chunks[:batch_size]
        pending_chunks = pending_chunks[batch_size:]

        coros = [
            extract_terms_from_chunk(chunk_idx, chunk, vllm_model)
            for chunk_idx, chunk in batch
        ]
        tasks = [asyncio.create_task(coro) for coro in coros]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (chunk_idx, _), result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"  ✗ Error in term extraction for chunk {chunk_idx}: {result}")
                chunk_to_terms[chunk_idx] = []  # Add empty list on error
            else:
                chunk_to_terms[chunk_idx] = result
        progress_bar.update(batch_size)

    progress_bar.close()

    # Aggregate all unique terms found across all chunks
    all_extracted_terms = set()
    for terms_list in chunk_to_terms.values():
        all_extracted_terms.update(terms_list)

    print(f"  Extracted {len(all_extracted_terms)} unique terms from all chunks.")
    print(f"  Saving terms to cache: {cache_terms_path}")
    with open(cache_terms_path, "w", encoding="utf-8") as f:
        json.dump(
            chunk_to_terms, f, ensure_ascii=False, indent=2
        )  # Save map of chunk_idx -> terms
    print(f"✓ Pass 1 completed.")
    return chunk_to_terms


async def extract_terms_from_chunk(
    chunk_idx: int, chunk: str, vllm_model: VLLMLLM
) -> List[str]:
    """Extract terms from a single chunk."""
    try:
        prompt = extraction_prompt_pass1.format(chunk_text=chunk)
        llm_response = await vllm_model.ainvoke(prompt)
        response_text = (
            llm_response.content
            if hasattr(llm_response, "content")
            else str(llm_response)
        )

        if not response_text or not response_text.strip():
            return []

        # Attempt to parse JSON response
        try:
            result = json_repair.loads(response_text)
            if (
                isinstance(result, dict)
                and "terms" in result
                and isinstance(result["terms"], list)
            ):
                terms = [str(t).strip() for t in result["terms"] if str(t).strip()]
                return terms
            else:
                extracted_result = extract_json_from_text(response_text)
                if (
                    extracted_result
                    and isinstance(extracted_result, dict)
                    and "terms" in extracted_result
                ):
                    terms = [
                        str(t).strip()
                        for t in extracted_result["terms"]
                        if str(t).strip()
                    ]
                    return terms
        except json.JSONDecodeError:
            extracted_result = extract_json_from_text(response_text)
            if (
                extracted_result
                and isinstance(extracted_result, dict)
                and "terms" in extracted_result
            ):
                terms = [
                    str(t).strip() for t in extracted_result["terms"] if str(t).strip()
                ]
                return terms

        return []  # Return empty list if parsing fails
    except Exception as e:
        print(f"    ✗ Error processing chunk {chunk_idx} for term extraction: {e}")
        return []


async def pass2_generate_definitions(
    chunk_to_terms: Dict[int, List[str]],
    text_path: str,
    vllm_model: VLLMLLM,
    cache_run_id: str,
) -> Dict[str, str]:
    """Second pass: Generate definitions for extracted terms using their originating chunks."""
    print(f"\n[2/4] Starting Pass 2: Definition Generation...")
    if not Path(text_path).exists():
        print(f"✗ File not found: {text_path}")
        return {}

    # Load file content to get the actual chunks
    with open(text_path, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"✓ Loaded file content")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Re-chunk text to get the actual chunk text based on indices
    print(f"  Re-chunking text for context retrieval...")
    chunks = chunk_text_by_tokens(content, tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  Retrieved {len(chunks)} chunks for context")

    # Define cache path for definitions
    cache_defs_path = os.path.join(CACHE_DIR, f"{cache_run_id}_pass2_definitions.json")

    # Check for cached results
    if os.path.exists(cache_defs_path):
        print(f"  ✓ Loading cached definitions from {cache_defs_path}")
        with open(cache_defs_path, "r", encoding="utf-8") as f:
            cached_defs = json.load(f)
        return cached_defs

    # Prepare all tasks: (term, context) pairs
    all_definition_tasks = []
    term_to_chunk_info = (
        {}
    )  # Map term to its originating chunk index for potential debugging

    for chunk_idx, terms in chunk_to_terms.items():
        if chunk_idx >= len(chunks):
            print(
                f"  ! Chunk index {chunk_idx} out of range for retrieved chunks. Skipping."
            )
            continue
        chunk_text = chunks[chunk_idx]
        for term in terms:
            all_definition_tasks.append((term, chunk_text))
            term_to_chunk_info[term] = chunk_idx  # Store origin chunk index

    total_terms_to_process = len(all_definition_tasks)
    print(f"  Total terms to define: {total_terms_to_process}")
    if total_terms_to_process == 0:
        print("  No terms to define. Returning empty dictionary.")
        return {}

    # Create tqdm progress bar
    progress_bar = tqdm(
        total=total_terms_to_process, desc="Pass 2: Generating definitions"
    )

    # Mark the VLLM model instance for definition tasks so the semaphore works correctly
    # We can do this by setting an attribute on the model instance itself, used within ainvoke
    # Or, we can create tasks that explicitly acquire the semaphore.
    # The cleanest way within the existing structure is to mark the tasks.
    # However, the semaphore logic in ainvoke() relies on the task context.
    # A more robust way is to wrap the call inside the semaphore acquisition explicitly here.
    # This avoids needing to modify the ainvoke method itself for this specific use case.

    async def _define_single_term(term, context):
        # Acquire the definition-specific semaphore within the task
        async with TERM_DEF_SEMAPHORE:
            definition = await generate_definition_for_term(term, context, vllm_model)
            progress_bar.update(1)  # Update progress inside the task after completion
            return term, definition  # Return the term and its definition

    # Create task coroutines for all (term, context) pairs
    task_coroutines = [
        _define_single_term(term, ctx) for term, ctx in all_definition_tasks
    ]

    # Gather results concurrently, respecting the semaphore limit inside each task
    print(
        f"  Executing {total_terms_to_process} definition tasks concurrently (up to {MAX_CONCURRENT_DEFS} at a time)..."
    )
    task_results = await asyncio.gather(*task_coroutines, return_exceptions=True)

    progress_bar.close()

    # Process results
    all_definitions = {}
    successful_definitions = 0
    for i, result in enumerate(task_results):
        if isinstance(result, Exception):
            term_at_error = all_definition_tasks[i][
                0
            ]  # Get the term that caused the error
            print(
                f"    ✗ Error generating definition for term '{term_at_error}': {result}"
            )
            all_definitions[term_at_error] = ""  # Add empty definition on error
        elif isinstance(result, tuple) and len(result) == 2:
            term, definition = result
            if definition:  # If definition is not empty
                all_definitions[term] = definition
                successful_definitions += 1
            else:
                all_definitions[term] = ""
        else:
            # This case should ideally not happen if _define_single_term always returns (term, def)
            print(f"    ? Unexpected result format at index {i}: {result}")
            # We cannot reliably add this to all_definitions without the term name

    print(
        f"  Generated definitions for {successful_definitions} terms out of {total_terms_to_process}."
    )
    print(f"  Saving definitions to cache: {cache_defs_path}")
    with open(cache_defs_path, "w", encoding="utf-8") as f:
        json.dump(all_definitions, f, ensure_ascii=False, indent=2)
    print(f"✓ Pass 2 completed.")
    return all_definitions


async def generate_definition_for_term(
    term: str, context: str, vllm_model: VLLMLLM
) -> str:
    """Generate a definition for a single term using context."""
    try:
        # Semaphore is handled by the calling task marked with _is_def_task
        prompt = definition_prompt_pass2.format(term=term, context=context)
        llm_response = await vllm_model.ainvoke(prompt)
        response_text = (
            llm_response.content
            if hasattr(llm_response, "content")
            else str(llm_response)
        )

        if not response_text or not response_text.strip():
            return ""

        # Attempt to parse JSON response
        try:
            result = json_repair.loads(response_text)
            if isinstance(result, dict) and "definition" in result:
                return str(result["definition"]).strip()
            else:
                extracted_result = extract_json_from_text(response_text)
                if (
                    extracted_result
                    and isinstance(extracted_result, dict)
                    and "definition" in extracted_result
                ):
                    return str(extracted_result["definition"]).strip()
        except json.JSONDecodeError:
            extracted_result = extract_json_from_text(response_text)
            if (
                extracted_result
                and isinstance(extracted_result, dict)
                and "definition" in extracted_result
            ):
                return str(extracted_result["definition"]).strip()

        return ""  # Return empty string if parsing fails
    except Exception as e:
        print(f"    ✗ Error generating definition for term '{term}': {e}")
        return ""


# ----------------------------
# GRAPH MANAGER MATCHING CUSTOM STRUCTURE
# ----------------------------
class Neo4jGraphManager:
    """Manages Neo4j graph operations aligned with custom graph structure"""

    def __init__(self, driver):
        self.driver = driver
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Dict] = []

    def create_indexes(self):
        """Create necessary indexes for efficient querying"""
        with self.driver.session() as session:
            session.run(
                """
                CREATE INDEX term_name_idx IF NOT EXISTS
                FOR (t:Term) ON (t.name)
            """
            )

    def add_terms(self, terms_with_definitions: Dict[str, str]):
        """Add term nodes to Neo4j"""
        with self.driver.session() as session:
            for term, definition in terms_with_definitions.items():
                session.run(
                    """
                    MERGE (t:Term {name: $name})
                    ON CREATE SET t.definition = $definition
                    ON MATCH SET t.definition = $definition
                """,
                    name=term,
                    definition=definition,
                )

    def add_compound_relationships(self, all_terms: Set[str]):
        """Attempt to create COMPOUND_OF relationships based on substring matching."""
        print("Building COMPOUND_OF relationships...")
        with self.driver.session() as session:
            # Fetch all existing terms from the DB for comparison
            result = session.run("MATCH (t:Term) RETURN t.name as name")
            existing_terms = {record["name"] for record in result}

            for compound_term in all_terms:
                if len(compound_term) < 2:  # Skip very short terms
                    continue
                for existing_term in existing_terms:
                    if (
                        existing_term != compound_term
                        and existing_term in compound_term
                    ):
                        if len(existing_term) > 1:
                            session.run(
                                """
                                MATCH (compound:Term {name: $compound})
                                MATCH (component:Term {name: $component})
                                MERGE (compound)-[r:COMPOUND_OF {weight: 1.0}]->(component)
                            """,
                                compound=compound_term,
                                component=existing_term,
                            )
                            print(
                                f"  -> Created COMPOUND_OF: {compound_term} -> {existing_term}"
                            )

    def add_parent_term_relationships(self):
        """Build PARENT_TERM relationships by analyzing definitions"""
        print("Building PARENT_TERM relationships...")
        with self.driver.session() as session:
            result = session.run(
                "MATCH (t:Term) RETURN t.name as name, t.definition as definition"
            )
            all_terms = {
                record["name"]: record["definition"] or "" for record in result
            }

            for term_name, definition in all_terms.items():
                words_in_def = definition.split()
                check_limit = int(len(words_in_def) * 0.3)  # Check first 30%
                for word in words_in_def[:check_limit]:
                    if word in all_terms and word != term_name:
                        session.run(
                            """
                             MATCH (child:Term {name: $child_name})
                             MATCH (parent:Term {name: $parent_name})
                             MERGE (child)-[r:PARENT_TERM {weight: 1.0}]->(parent)
                         """,
                            child_name=term_name,
                            parent_name=word,
                        )
                        print(f"  -> Created PARENT_TERM: {term_name} -> {word}")

    def query_term_graph(self, term: str, max_depth: int = 2) -> Dict:
        """Query the graph using BFS to find related terms"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (start:Term {name: $term})
                CALL apoc.path.expandConfig(start, {
                    relationshipFilter: "PARENT_TERM|COMPOUND_OF",
                    maxLevel: $max_depth,
                    limit: 100
                })
                YIELD path
                RETURN nodes(path) as nodes, relationships(path) as rels
            """,
                term=term,
                max_depth=max_depth,
            )
            all_nodes = {}
            all_edges = []
            for record in result:
                nodes = record["nodes"]
                rels = record["rels"]
                for node in nodes:
                    all_nodes[node["name"]] = {"definition": node.get("definition", "")}
                for rel in rels:
                    all_edges.append(
                        {
                            "source": rel.start_node["name"],
                            "target": rel.end_node["name"],
                            "type": rel.type,
                            "weight": rel.get("weight", 1.0),
                        }
                    )
            return {
                "found": len(all_nodes) > 0,
                "query_term": term,
                "nodes": all_nodes,
                "edges": all_edges,
                "total_nodes": len(all_nodes),
                "total_edges": len(all_edges),
            }


# ----------------------------
# QUERY SYSTEM (matching custom code structure)
# ----------------------------
class QuerySystem:
    def __init__(self, driver):
        self.driver = driver
        self.graph = Neo4jGraphManager(driver)

    def dict_lookup(self, term: str) -> Dict:
        """Exact dictionary lookup"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (t:Term {name: $term}) RETURN t.definition as definition",
                term=term,
            )
            record = result.single()
            if record:
                return {
                    "found": True,
                    "exact_match": term,
                    "definition": record["definition"],
                }
            else:
                return {
                    "found": False,
                    "query": term,
                    "message": "No exact match found",
                }

    async def full_query(self, term: str, max_depth: int = 2) -> Dict:
        """Full query: lookup + graph traversal"""
        dict_result = self.dict_lookup(term)
        graph_result = None
        if dict_result["found"]:
            graph_result = self.graph.query_term_graph(term, max_depth=max_depth)
        return {"query": term, "dict_result": dict_result, "graph_result": graph_result}


# ----------------------------
# MAIN BUILD FUNCTION (TWO-PASS)
# ----------------------------
# New Pass 1 function that takes the list of chunks directly
async def pass1_extract_terms_from_chunks(
    chunks: List[str], vllm_model: VLLMLLM, cache_run_id: str
) -> Dict[int, List[str]]:
    """First pass: Extract terms from a list of pre-segmented chunks."""
    print("\n[3/4] Starting Pass 1: Term Extraction from pre-segmented chunks...")
    num_chunks = len(chunks)
    if num_chunks == 0:
        print(f"  ⊘ No chunks provided for term extraction")
        return {}

    # Define cache path for extracted terms
    cache_terms_path = os.path.join(CACHE_DIR, f"{cache_run_id}_pass1_terms.json")

    # Check for cached results
    if os.path.exists(cache_terms_path):
        print(f"  ✓ Loading cached terms from {cache_terms_path}")
        with open(cache_terms_path, "r", encoding="utf-8") as f:
            cached_terms = json.load(f)
        # Convert keys back to int if needed by downstream logic
        return {int(k): v for k, v in cached_terms.items()}

    # Process chunks concurrently for term extraction
    chunk_to_terms = {}
    pending_chunks = [(i, c) for i, c in enumerate(chunks)]
    progress_bar = tqdm(total=len(pending_chunks), desc="Pass 1: Extracting terms")

    while pending_chunks:
        batch_size = min(BATCH_SIZE, len(pending_chunks))
        batch = pending_chunks[:batch_size]
        pending_chunks = pending_chunks[batch_size:]

        coros = [
            extract_terms_from_chunk(chunk_idx, chunk, vllm_model)
            for chunk_idx, chunk in batch
        ]
        tasks = [asyncio.create_task(coro) for coro in coros]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (chunk_idx, _), result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"  ✗ Error in term extraction for chunk {chunk_idx}: {result}")
                chunk_to_terms[chunk_idx] = []  # Add empty list on error
            else:
                chunk_to_terms[chunk_idx] = result
        progress_bar.update(batch_size)

    progress_bar.close()

    # Aggregate all unique terms found across all chunks
    all_extracted_terms = set()
    for terms_list in chunk_to_terms.values():
        all_extracted_terms.update(terms_list)

    print(
        f"  Extracted {len(all_extracted_terms)} unique terms from {num_chunks} chunks."
    )
    print(f"  Saving terms to cache: {cache_terms_path}")
    with open(cache_terms_path, "w", encoding="utf-8") as f:
        json.dump(
            chunk_to_terms, f, ensure_ascii=False, indent=2
        )  # Save map of chunk_idx -> terms
    print(f"✓ Pass 1 completed.")
    return chunk_to_terms


# New Pass 2 function that takes the chunk_to_terms map and the list of chunks
async def pass2_generate_definitions_from_chunks(
    chunk_to_terms: Dict[int, List[str]],
    chunks: List[str],
    vllm_model: VLLMLLM,
    cache_run_id: str,
) -> Dict[str, str]:
    """Second pass: Generate definitions for extracted terms using their corresponding chunks."""
    print(
        f"\n[3.5/4] Starting Pass 2: Definition Generation from pre-segmented chunks..."
    )
    num_chunks = len(chunks)

    # Define cache path for definitions
    cache_defs_path = os.path.join(CACHE_DIR, f"{cache_run_id}_pass2_definitions.json")

    # Check for cached results
    if os.path.exists(cache_defs_path):
        print(f"  ✓ Loading cached definitions from {cache_defs_path}")
        with open(cache_defs_path, "r", encoding="utf-8") as f:
            cached_defs = json.load(f)
        return cached_defs

    # Prepare all tasks: (term, context_chunk_text) pairs
    all_definition_tasks = []

    for chunk_idx, terms in chunk_to_terms.items():
        if chunk_idx >= num_chunks:
            print(
                f"  ! Chunk index {chunk_idx} out of range for provided chunks list (length {num_chunks}). Skipping."
            )
            continue
        chunk_text = chunks[chunk_idx]  # Get the text from the pre-segmented list
        for term in terms:
            all_definition_tasks.append((term, chunk_text))

    total_terms_to_process = len(all_definition_tasks)
    print(f"  Total terms to define: {total_terms_to_process}")
    if total_terms_to_process == 0:
        print("  No terms to define. Returning empty dictionary.")
        return {}

    # Create tqdm progress bar
    progress_bar = tqdm(
        total=total_terms_to_process, desc="Pass 2: Generating definitions"
    )

    async def _define_single_term(term, context):
        # Acquire the definition-specific semaphore within the task
        async with TERM_DEF_SEMAPHORE:
            definition = await generate_definition_for_term(term, context, vllm_model)
            progress_bar.update(1)  # Update progress inside the task after completion
            return term, definition  # Return the term and its definition

    # Create task coroutines for all (term, context) pairs
    task_coroutines = [
        _define_single_term(term, ctx) for term, ctx in all_definition_tasks
    ]

    # Gather results concurrently, respecting the semaphore limit inside each task
    print(
        f"  Executing {total_terms_to_process} definition tasks concurrently (up to {MAX_CONCURRENT_DEFS} at a time)..."
    )
    task_results = await asyncio.gather(*task_coroutines, return_exceptions=True)

    progress_bar.close()

    # Process results
    all_definitions = {}
    successful_definitions = 0
    for i, result in enumerate(task_results):
        if isinstance(result, Exception):
            term_at_error = all_definition_tasks[i][
                0
            ]  # Get the term that caused the error
            print(
                f"    ✗ Error generating definition for term '{term_at_error}': {result}"
            )
            all_definitions[term_at_error] = ""  # Add empty definition on error
        elif isinstance(result, tuple) and len(result) == 2:
            term, definition = result
            if definition:  # If definition is not empty
                all_definitions[term] = definition
                successful_definitions += 1
            else:
                all_definitions[term] = ""
        else:
            # This case should ideally not happen if _define_single_term always returns (term, def)
            print(f"    ? Unexpected result format at index {i}: {result}")
            # We cannot reliably add this to all_definitions without the term name

    print(
        f"  Generated definitions for {successful_definitions} terms out of {total_terms_to_process}."
    )
    print(f"  Saving definitions to cache: {cache_defs_path}")
    with open(cache_defs_path, "w", encoding="utf-8") as f:
        json.dump(all_definitions, f, ensure_ascii=False, indent=2)
    print(f"✓ Pass 2 completed.")
    return all_definitions


async def build_neo4j(text_path: str):
    """Build graph from text file using two-pass logic."""
    global VLLM_SEMAPHORE, TERM_DEF_SEMAPHORE
    VLLM_SEMAPHORE = asyncio.Semaphore(BATCH_SIZE)
    TERM_DEF_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_DEFS)

    print("=" * 60)
    print("Neo4j GraphRAG Dictionary Builder (TWO-PASS LOGIC - CONCURRENT DEFS)")
    print("=" * 60)
    print(f"Text path: {text_path}")
    print(f"Neo4j URI: {NEO4J_URI}")
    print(f"Max concurrent Pass 1 requests: {BATCH_SIZE}")
    print(f"Max concurrent Pass 2 requests: {MAX_CONCURRENT_DEFS}")
    print(f"Increased Chunk Size: {CHUNK_SIZE}")
    print(f"Model Max Context Length: {MAX_MODEL_LEN}")
    print("=" * 60)

    # Generate a run ID based on the file path and current time to manage cache
    run_id_hash = hashlib.md5(f"{text_path}_{time.time()}".encode()).hexdigest()[:8]

    # 1. Initialize components
    print("\n[1/4] Initializing components...")
    VLLM_API_BASE = "http://localhost:8000/v1"
    vllm_model = VLLMLLM(api_base_url=VLLM_API_BASE)
    print("✓ Components initialized")

    # 2. Initialize Neo4j
    print("\n[2/4] Connecting to Neo4j...")
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    graph_manager = Neo4jGraphManager(driver)
    graph_manager.create_indexes()
    print("✓ Neo4j initialized")

    # --- NEW LOGIC: Read file in segments based on model context length ---
    print(
        f"\n[2.5/4] Reading and segmenting file based on model context length ({MAX_MODEL_LEN} tokens)..."
    )
    if not Path(text_path).exists():
        print(f"✗ File not found: {text_path}")
        return False

    # Initialize tokenizer for length checking
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Read the file in segments
    all_chunks_from_segments = []
    segment_counter = 0
    with open(text_path, "r", encoding="utf-8") as f:
        current_segment_lines = []
        current_segment_token_count = 0

        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n")  # Remove trailing newline
            if not line:  # Skip empty lines to avoid unnecessary processing
                continue

            # Estimate token count for the line
            line_tokens = tokenizer.encode(line, add_special_tokens=False)
            line_token_count = len(line_tokens)

            # Check if adding this line would exceed the max length significantly
            # We add a small buffer (e.g., 1000 tokens) to account for overlap and potential processing overhead during Pass 1/Pass 2
            estimated_next_count = (
                current_segment_token_count + line_token_count + 1000
            )  # +1 for potential newline separator if needed, +1000 buffer

            if estimated_next_count > MAX_MODEL_LEN:
                # If the current segment is empty and the single line is too long, warn and skip
                if current_segment_token_count == 0:
                    print(
                        f"  ! Warning: Line {line_num} is too long ({line_token_count} tokens) even for a single segment, skipping. Content: {line[:50]}..."
                    )
                    continue

                # Process the current segment
                print(
                    f"  Processing segment {segment_counter} (approx. {current_segment_token_count} tokens)..."
                )
                current_segment_text = "\n".join(current_segment_lines)
                segment_chunks = chunk_text_by_tokens(
                    current_segment_text, tokenizer, CHUNK_SIZE, CHUNK_OVERLAP
                )
                all_chunks_from_segments.extend(segment_chunks)
                print(
                    f"    -> Generated {len(segment_chunks)} chunks from segment {segment_counter}"
                )
                segment_counter += 1

                # Start a new segment with the current line
                current_segment_lines = [line]
                current_segment_token_count = line_token_count
            else:
                # Add the line to the current segment
                current_segment_lines.append(line)
                current_segment_token_count += line_token_count

        # Process the last segment if it has content
        if current_segment_lines:
            print(
                f"  Processing final segment {segment_counter} (approx. {current_segment_token_count} tokens)..."
            )
            current_segment_text = "\n".join(current_segment_lines)
            segment_chunks = chunk_text_by_tokens(
                current_segment_text, tokenizer, CHUNK_SIZE, CHUNK_OVERLAP
            )
            all_chunks_from_segments.extend(segment_chunks)
            print(
                f"    -> Generated {len(segment_chunks)} chunks from final segment {segment_counter}"
            )

    print(
        f"  Total chunks generated from all segments: {len(all_chunks_from_segments)}"
    )
    # --- END NEW LOGIC ---

    # 3. Pass 1: Extract Terms (per chunk) - Use the pre-segmented chunks
    chunk_to_terms = await pass1_extract_terms_from_chunks(
        all_chunks_from_segments, vllm_model, run_id_hash
    )

    # 4. Pass 2: Generate Definitions (per chunk, concurrently) - Use the pre-segmented chunks and their corresponding terms
    term_definitions = await pass2_generate_definitions_from_chunks(
        chunk_to_terms, all_chunks_from_segments, vllm_model, run_id_hash
    )

    # Filter out terms with empty definitions
    final_definitions = {k: v for k, v in term_definitions.items() if v}
    print(f"✓ Final definitions with content: {len(final_definitions)}")

    # 5. Save final definitions and edges to JSON file (as requested)
    graph_manager.add_terms(final_definitions)  # Add nodes
    graph_manager.add_compound_relationships(
        set(final_definitions.keys())
    )  # Add compound rels
    graph_manager.add_parent_term_relationships()  # Add parent rels

    # Fetch the final graph structure from Neo4j
    with driver.session() as session:
        node_query_result = session.run(
            "MATCH (t:Term) WHERE t.definition IS NOT NULL RETURN t.name as name, t.definition as definition"
        )
        final_nodes = {
            record["name"]: record["definition"] for record in node_query_result
        }
        edge_query_result = session.run(
            "MATCH (a:Term)-[r]->(b:Term) RETURN a.name as source, b.name as target, type(r) as rel_type, r.weight as weight"
        )
        final_edges = []
        for record in edge_query_result:
            final_edges.append(
                {
                    "source": record["source"],
                    "target": record["target"],
                    "type": record["rel_type"],
                    "weight": record["weight"],
                }
            )

    output_json_data = {
        "terms": list(final_nodes.keys()),
        "definitions": final_nodes,
        "edges": final_edges,
    }

    output_json_path = f"{OUTPUT_BASE_NAME}.json"
    print(
        f"\n[4/4] Saving final terms, definitions, and edges to {output_json_path}..."
    )
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_json_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved data to {output_json_path}")

    # 6. Save unique terms list separately
    with open(OUTPUT_TERMS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(final_nodes.keys()), f, ensure_ascii=False, indent=2)
    print(f"✓ Saved unique terms list to {OUTPUT_TERMS_FILE}")

    print("\n" + "=" * 60)
    print("TWO-PASS BUILD COMPLETED")
    print("=" * 60)

    driver.close()
    return True


async def query(term: str):
    """Query the dictionary for a term"""
    print("=" * 60)
    print("Neo4j GraphRAG Dictionary Query")
    print("=" * 60)
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    query_system = QuerySystem(driver)
    result = await query_system.full_query(term, max_depth=2)
    print("\n" + "-" * 60)
    print("RAW GRAPH RAG OUTPUT")
    print("-" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("-" * 60)
    # Formatted results
    dict_result = result["dict_result"]
    print("\n" + "=" * 60)
    print("FORMATTED RESULTS")
    print("=" * 60)
    if dict_result["found"]:
        print(f"\n✓ DEFINITION FOUND")
        print(f"Term: {dict_result['exact_match']}")
        print(f"Definition: {dict_result['definition']}")
    else:
        print(f"\n✗ No definition found for: {term}")
    graph_result = result["graph_result"]
    if graph_result and graph_result["found"]:
        print(f"\n{'-' * 60}")
        print(
            f"RELATED TERMS ({graph_result['total_nodes']} nodes, {graph_result['total_edges']} edges)"
        )
        print(f"{'-' * 60}")
        edge_types = {}
        for edge in graph_result["edges"]:
            edge_type = edge["type"]
            if edge_type not in edge_types:
                edge_types[edge_type] = []
            edge_types[edge_type].append(edge)
        for edge_type in sorted(edge_types.keys()):
            edges = edge_types[edge_type]
            print(f"\n{edge_type}:")
            for edge in edges[:10]:
                source_def = (
                    graph_result["nodes"].get(edge["source"], {}).get("definition", "")
                )
                target_def = (
                    graph_result["nodes"].get(edge["target"], {}).get("definition", "")
                )
                print(f"  {edge['source']} → {edge['target']}")
                if source_def:
                    print(f"    └─ {source_def[:60]}...")
    else:
        print(f"\nNo related terms found in graph")
    driver.close()
    print("\n" + "=" * 60)


# ----------------------------
# MAIN ENTRY POINT
# ----------------------------
async def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [options]")
        print("\nCommands:")
        print(
            "  build              Build the dictionary from text file using two-pass logic"
        )
        print("  query <term>       Query the dictionary for a term definition")
        print("\nExamples:")
        print("  python main.py build")
        print("  python main.py query 遠隔制御")
        return

    command = sys.argv[1]
    if command == "build":
        success = await build_neo4j(TEXT_FILE_PATH)
        if not success:
            sys.exit(1)
    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python main.py query <term>")
            return
        term = sys.argv[2]
        await query(term)
    else:
        print(f"✗ Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
