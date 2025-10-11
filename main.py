#!/usr/bin/env python3

import asyncio
import hashlib
import json
import os
import pickle
import re
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Third-party imports
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from pydantic import BaseModel
    import json_repair
    from openai import AsyncOpenAI
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install with: pip install numpy sentence-transformers pydantic json-repair openai")
    sys.exit(1)

# Japanese NLP libraries
try:
    from sudachipy import dictionary, tokenizer
    SUDACHIPY_AVAILABLE = True
except ImportError:
    SUDACHIPY_AVAILABLE = False
    print("Warning: SudachiPy not available. Install with: pip install sudachipy")

try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    print("Warning: MeCab not available. Install with: pip install mecab-python3")

try:
    from pyknp import KNP, Juman
    JUMAN_KNP_AVAILABLE = True
except ImportError:
    JUMAN_KNP_AVAILABLE = False

# Configuration constants
TERM_FOLDER = "./terms"
WORKING_DIR = "./ja_graph_rag_cache_3"
CACHE_DIR = WORKING_DIR + "/processing_cache"
VECTOR_DB_DIR = WORKING_DIR + "/vector_db"
MAX_TOKEN = 5120
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
SEMANTIC_SIMILARITY_THRESHOLD = 0.98
MAX_CONCURRENT_DEFINITIONS = 20

# VLLM configuration
VLLM_HOST = "http://localhost:8000/v1"
VLLM_MODEL = "kaitchup/Phi-4-mini-instruct-AutoRoundGPTQ-4bit"

# Initialize embedding model
try:
    EMBED_MODEL = SentenceTransformer("cl-nagoya/ruri-v3-310m", device="cpu")
    embedding_func = EMBED_MODEL.encode
    print("✓ Embedding model loaded")
except Exception as e:
    print(f"✗ Failed to load embedding model: {e}")
    sys.exit(1)

# Initialize VLLM client
vllm_client = AsyncOpenAI(api_key="EMPTY", base_url=VLLM_HOST)
GLOBAL_LLM_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_DEFINITIONS)

# Data models
class FilteredTerms(BaseModel):
    terms: List[str]

class TermDefinition(BaseModel):
    term: str
    definition: str

class Entity(BaseModel):
    name: str
    type: str
    description: str

class Relationship(BaseModel):
    source: str
    target: str
    description: str
    strength: float

# Tokenizer classes
class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[Dict[str, str]]:
        pass

class SudachiTokenizer(BaseTokenizer):
    def __init__(self):
        try:
            self.tokenizer_obj = dictionary.Dictionary(dict="full").create()
            self.mode = tokenizer.Tokenizer.SplitMode.C
            print("✓ SudachiPy tokenizer initialized")
        except (ImportError, TypeError):
            try:
                self.tokenizer_obj = dictionary.Dictionary(dict_type="full").create()
                self.mode = tokenizer.Tokenizer.SplitMode.C
                print("✓ SudachiPy tokenizer initialized")
            except Exception as e:
                raise ImportError(f"SudachiPy not available: {e}")
    
    def tokenize(self, text: str) -> List[Dict[str, str]]:
        tokens = self.tokenizer_obj.tokenize(text, self.mode)
        results = []
        
        for token in tokens:
            surface = token.surface()
            pos = token.part_of_speech()
            main_pos = pos[0] if pos else "UNKNOWN"
            results.append({"surface": surface, "pos": main_pos})
        
        return results

class JumanKnpTokenizer(BaseTokenizer):
    def __init__(self, use_neologd=True, mecab_dict_path=None):
        try:
            self.knp = KNP(jumanpp=True)
            self.use_knp = True
            print("✓ Juman++/KNP tokenizer initialized")
        except ImportError:
            print("✗ Juman++/KNP not available, falling back to MeCab")
            self.use_knp = False
            
            if mecab_dict_path:
                self.mecab = MeCab.Tagger(f"-d {mecab_dict_path}")
            elif use_neologd:
                neologd_paths = [
                    "/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd",
                    "/opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd",
                    "/usr/local/lib/mecab/dic/mecab-ipadic-neologd",
                    "/var/lib/mecab/dic/ipadic-utf8",
                ]
                
                found_dict = None
                for path in neologd_paths:
                    if os.path.exists(path):
                        self.mecab = MeCab.Tagger(f"-d {path}")
                        found_dict = path
                        print(f"Using MeCab dictionary: {path}")
                        break
                
                if not found_dict:
                    print("Warning: neologd not found, using default dictionary")
                    self.mecab = MeCab.Tagger("")
            else:
                self.mecab = MeCab.Tagger("")
    
    def tokenize(self, text: str) -> List[Dict[str, str]]:
        if self.use_knp:
            try:
                juman_result = self.knp.juman.analysis(text)
                results = []
                
                for mrph in juman_result.mrph_list():
                    surface = mrph.midasi
                    pos = mrph.hinsi
                    results.append({"surface": surface, "pos": pos})
                
                return results
            except Exception as e:
                print(f"✗ Juman++ error: {e}, falling back to MeCab")
        
        result = self.mecab.parse(text)
        lines = result.strip().split("\n")[:-1]
        results = []
        
        for line in lines:
            if "\t" not in line:
                continue
            
            surface, features = line.split("\t", 1)
            feature_parts = features.split(",")
            pos = feature_parts[0] if feature_parts else "UNKNOWN"
            
            results.append({"surface": surface, "pos": pos})
        
        return results

def get_tokenizer(tokenizer_name: str) -> BaseTokenizer:
    if tokenizer_name.lower() == "sudachipy":
        return SudachiTokenizer()
    elif tokenizer_name.lower() == "juman":
        return JumanKnpTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")

# Analytics loader
class AnalyticsLoader:
    def __init__(self, term_folder: str, tokenizer_name: str):
        self.term_folder = Path(term_folder)
        self.tokenizer_name = tokenizer_name
        self.terms: Set[str] = set()
    
    def load_terms(self) -> Set[str]:
        analytics_file = self.term_folder / f"analytics_{self.tokenizer_name}.json"
        
        if not analytics_file.exists():
            print(f"⚠ Analytics file not found: {analytics_file.absolute()}")
            print("⚠ Will rely on LLM-based term discovery.")
            return set()
        
        try:
            with open(analytics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.terms = {entry["term"] for entry in data}
            print(f"✓ Loaded {len(self.terms)} pre-filtered technical terms")
            return self.terms
        except Exception as e:
            print(f"⚠ Error loading analytics file: {e}")
            return set()

# Book chunker
class BookChunker:
    def __init__(self, tokenizer: BaseTokenizer, chunk_size: int, overlap: int):
        self.tokenizer = tokenizer
        self.chunk_name = self.tokenizer.__class__.__name__
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_bytes = 40000 if self.chunk_name == "SudachiTokenizer" else None
    
    def create_chunks_from_book(self, text: str) -> List[Dict[str, Any]]:
        print(f"Creating chunks from book ({len(text)} characters)...")
        
        # Pre-split for SudachiPy only
        if self.max_bytes:
            byte_chunks = self._split_by_bytes(text, self.max_bytes)
            all_tokens = []
            for byte_chunk in byte_chunks:
                try:
                    tokens = self.tokenizer.tokenize(byte_chunk)
                    all_tokens.extend(tokens)
                except Exception as e:
                    print(f"⚠ Error tokenizing chunk: {e}")
                    continue
        else:
            all_tokens = self.tokenizer.tokenize(text)
        
        chunks = []
        for i in range(0, len(all_tokens), self.chunk_size - self.overlap):
            chunk_tokens = all_tokens[i:i + self.chunk_size]
            chunk_text = "".join([t["surface"] for t in chunk_tokens])
            
            chunks.append({
                "id": len(chunks),
                "text": chunk_text,
                "tokens": chunk_tokens,
                "start_token": i,
                "end_token": i + len(chunk_tokens)
            })
        
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def _split_by_bytes(self, text: str, max_bytes: int) -> List[str]:
        """Split text into byte-sized chunks for tokenizers with limits"""
        byte_chunks = []
        current = ""
        
        for char in text:
            if len(current.encode('utf-8')) + len(char.encode('utf-8')) > max_bytes:
                if current:
                    byte_chunks.append(current)
                current = char
            else:
                current += char
        
        if current:
            byte_chunks.append(current)
        
        return byte_chunks

# Term matcher
class TermMatcher:
    def __init__(self, technical_terms: Set[str]):
        self.technical_terms = technical_terms
    
    def match_terms(self, tokens: List[Dict[str, str]]) -> Tuple[List[str], Dict[str, List[str]]]:
        """Return full matches and partial matches"""
        full_matches = []
        partial_matches = defaultdict(list)
        
        for token in tokens:
            surface = token["surface"]
            
            if surface in self.technical_terms:
                full_matches.append(surface)
            else:
                for term in self.technical_terms:
                    if surface in term and surface != term:
                        partial_matches[surface].append(term)
        
        full_matches = list(set(full_matches))
        partial_matches = {k: list(set(v)) for k, v in partial_matches.items()}
        
        return full_matches, partial_matches

# LLM interaction
async def vllm_complete(prompt: str, system_prompt: str = None, **kwargs) -> str:
    if system_prompt is None:
        system_prompt = "You are a helpful assistant. Respond only with requested information."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    async with GLOBAL_LLM_SEMAPHORE:
        try:
            unsupported_kwargs = ['hashing_kv', 'history_messages']
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_kwargs}
            
            extra_body = {"repetition_penalty": 1.1, "top_p": 0.9}
            if 'extra_body' in filtered_kwargs:
                extra_body.update(filtered_kwargs.pop('extra_body'))
            
            response = await vllm_client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=MAX_TOKEN,
                extra_body=extra_body,
                **filtered_kwargs
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"✗ vLLM request failed: {e}")
            return ""

async def discover_technical_terms(chunk_text: str) -> List[str]:
    """Discover power-system technical terms from text"""
    prompt = f"""
以下のテキスト断片から、電力システムの制御・監視・保護に関連する専門用語を抽出してください。

対象とする用語の特性:
- 送配電、変電所、制御装置、状態、故障、保護、周波数などの電力分野に直接関連する
- 例: 遠隔制御、周波数調整、系統連系、地絡保護、無効電力、開閉器
- 複合語（2語以上の組み合わせ）は含める
- 複合語の部分要素も含める（例：「個別制御」なら「個別」「制御」両方を検討）

対象外とする用語:
- 一般用語: 機器、装置、システム、データ、情報（これらは複合語の一部ならOK）
- 人名、地名、企業名
- 単純な動詞・形容詞: する、実施、高い、低い
- UI/一般用語: クリック、ボタン、ダイアログ、ウィンドウ
- 英字略語（例: HB, EM）は除外。ただし日本語複合語に含まれるなら対象

抽出条件:
- テキスト中で説明・定義・使用例があるもののみ
- 単なる言及は除外（「○○モード」と言及されるだけならスキップ）

テキスト断片:
{chunk_text}

出力形式: {{"terms": ["用語1", "用語2"]}}
JSONオブジェクトのみを出力してください。
"""
    
    system_prompt = """
あなたは日本語の電力システム技術文書における専門用語抽出のプロフェッショナルです。
電力システムの制御・監視・保護に関連する技術用語を抽出してください。
"terms"キーを持つ有効なJSONオブジェクトのみを出力してください。
"""
    
    response = await vllm_complete(prompt, system_prompt)
    
    if not response:
        return []
    
    try:
        result = json_repair.loads(response)
        if isinstance(result, dict) and "terms" in result and isinstance(result["terms"], list):
            return [str(t).strip() for t in result["terms"] if str(t).strip()]
        return []
    except Exception as e:
        print(f"✗ Term discovery parse error: {e}")
        return []

async def discover_compound_terms(chunk_text: str, tokens: List[Dict[str, str]]) -> List[str]:
    """
    Discover compound technical terms (multi-noun phrases with specialized meaning).
    """
    # Extract consecutive noun sequences (2-4 nouns)
    compound_candidates = []
    current_compound = []
    
    for token in tokens:
        if "名詞" in token.get("pos", ""):
            current_compound.append(token["surface"])
        else:
            if len(current_compound) >= 2:
                compound_candidates.append("".join(current_compound))
            current_compound = []
    
    # Don't forget the last compound
    if len(current_compound) >= 2:
        compound_candidates.append("".join(current_compound))
    
    # Remove duplicates and filter out very long compounds (>15 chars usually not technical)
    compound_candidates = list(set([c for c in compound_candidates if 2 <= len(c) <= 15]))
    
    if not compound_candidates:
        return []
    
    # Ask LLM to filter for actual technical compounds
    prompt = f"""
以下の候補から、電力システムに関連する複合技術用語のみを抽出してください。

複合技術用語の条件:
- 2つ以上の名詞が組み合わさった用語
- 個別の名詞とは異なる専門的意味を持つ
- 電力システムの制御・監視・保護に関連
- 例: 個別制御、遠隔監視、周波数調整、地絡保護

除外すべきもの:
- 一般的な名詞の単純な組み合わせ（例：機器設定、データ表示）
- 文脈依存の表現（例：前回実施、次回確認）

候補: {json.dumps(compound_candidates, ensure_ascii=False)}

テキスト断片（参考用）:
{chunk_text[:500]}

出力形式: {{"terms": ["用語1", "用語2"]}}
JSONオブジェクトのみを出力してください。
"""
    
    system_prompt = """
あなたは電力システム分野の複合技術用語抽出の専門家です。
複合語として意味を持つ技術用語のみを抽出してください。
"""
    
    response = await vllm_complete(prompt, system_prompt)
    
    if not response:
        return []
    
    try:
        result = json_repair.loads(response)
        if isinstance(result, dict) and "terms" in result and isinstance(result["terms"], list):
            return [str(t).strip() for t in result["terms"] if str(t).strip()]
        return []
    except Exception as e:
        print(f"✗ Compound term discovery parse error: {e}")
        return []

async def filter_technical_terms(chunk_text: str, candidate_terms: List[str]) -> List[str]:
    """Filter which terms are actually power-system technical"""
    if not candidate_terms:
        return []

    prompt = f"""
以下のテキスト断片に基づき、候補用語が電力システムに関連する技術用語かを判定してください。
判定基準:
【保持すべき用語】
- 制御系: 遠隔制御、個別制御、自動制御、手動制御
- 監視系: 周波数監視、電圧監視、電流監視
- 保護系: 地絡保護、過電流保護、周波数低下保護
- 状態系: 故障、重故障、軽故障、正常、異常
- 機器系: 開閉器、断路器、遮断器、変圧器、同調発電機
- 送配電系: 送電線、配電線、変電所、親局、子局
【除外すべき用語】
- 一般用語: 機器、装置、システム、データ、表示、設定（単体で一般的なため）
- UI用語: ボタン、クリック、ウィンドウ、ダイアログ
- 曖昧な略語: BRM, BEB など意味不明なもの
- 複合語で前置詞のみ: 「個別」「手動」が単体で含まれるなら、複合語「個別制御」「手動制御」のみ保持
判定ルール:
- テキスト中で電力制御・監視・保護の文脈で使われているか確認
- 複合語は全体で1つのドメイン用語なら保持
- 複合語の部分は、その複合語があれば別途抽出。単独で一般的なら除外
例:
「個別制御を実施する配電線上の機器」
→ 保持: 個別制御
→ 除外: 配電線上（複合表現）、機器（一般用語）
テキスト断片:
{chunk_text}
候補語:
{json.dumps(candidate_terms, ensure_ascii=False)}
出力形式: {{"terms": ["用語1", "用語2"]}}
JSONオブジェクトのみを出力してください。
"""
    
    system_prompt = """
あなたは日本語の電力システム技術文書における専門用語フィルタリングのプロフェッショナルです。
与えられた候補用語リストから、電力システムに関連する技術用語のみを抽出してください。
"terms"キーを持つ有効なJSONオブジェクトのみを出力してください。
"""
    
    response = await vllm_complete(prompt, system_prompt)
    if not response:
        return []
    
    try:
        result = json_repair.loads(response)
        if isinstance(result, dict) and "terms" in result and isinstance(result["terms"], list):
            filtered_terms = [str(t).strip() for t in result["terms"] if str(t).strip()]
            return filtered_terms
        else:
            return []
    except Exception as e:
        print(f"✗ Filter terms parse error: {e}")
        return []


async def generate_definition(term: str, context: str) -> Optional[TermDefinition]:
    """Generate definition for a term from context"""
    prompt = f"""
用語: {term}

以下の文脈から、この用語の定義を抽出・生成してください。

定義の条件:
- テキストに明示的な定義が含まれる場合（「○○とは..」「○○は..」）、それを抽出
- テキストに説明や使用例がある場合、それから定義を推論（ただし、テキストに基づいてのみ）
- テキストに該当する情報がない場合、空文字を返す（推測・外部知識は使用しない）

定義の形式:
- 1～3文程度の簡潔な説明
- 専門用語を含める（例：「66kV以上の電圧で電力を送る線路」）
- 句読点を含める

文脈:
{context}

出力形式:
成功例: {{"term": "遠隔制御", "definition": "遠隔制御とは、離れた場所から機器を制御する機能。"}}
失敗例: {{"term": "遠隔制御", "definition": ""}}

JSONオブジェクトのみを出力してください。
"""
    
    system_prompt = """
あなたは日本語の技術文書アシスタントで電力システムに詳しいです。
定義は文脈から正確に抽出・生成してください。
推測・外部知識は使用しないでください。
"""
    
    response = await vllm_complete(prompt, system_prompt)
    
    if not response:
        return None
    
    try:
        result = json_repair.loads(response)
        if isinstance(result, dict) and "term" in result and "definition" in result:
            validated = TermDefinition(
                term=result["term"],
                definition=str(result["definition"]).strip()
            )
            return validated if validated.definition else None
        return None
    except Exception as e:
        print(f"✗ Definition parse error: {e}")
        return None

# Vector DB
class VectorDB:
    def __init__(self, db_dir: str):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.passages: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.passage_index_path = self.db_dir / "passage_index.json"
        self.embeddings_path = self.db_dir / "passage_embeddings.npz"
    
    def load(self) -> bool:
        if self.passage_index_path.exists() and self.embeddings_path.exists():
            try:
                with open(self.passage_index_path, "r", encoding="utf-8") as f:
                    self.passages = json.load(f)
                
                data = np.load(self.embeddings_path)
                self.embeddings = data["embeddings"]
                
                print(f"✓ Loaded vector DB with {len(self.passages)} passages")
                return True
            except Exception as e:
                print(f"✗ Error loading vector DB: {e}")
                return False
        return False
    
    def save(self):
        with open(self.passage_index_path, "w", encoding="utf-8") as f:
            json.dump(self.passages, f, ensure_ascii=False, indent=2)
        
        if self.embeddings is not None:
            np.savez(self.embeddings_path, embeddings=self.embeddings)
        
        print(f"✓ Saved vector DB with {len(self.passages)} passages")
    
    def add_passages(self, passages: List[Dict[str, Any]]):
        new_texts = [p["text"] for p in passages]
        new_embeddings = EMBED_MODEL.encode(new_texts, normalize_embeddings=True)
        
        for passage in passages:
            passage["id"] = len(self.passages)
            passage["added_at"] = time.time()
        
        self.passages.extend(passages)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.embeddings is None or len(self.passages) == 0:
            return []
        
        query_embedding = EMBED_MODEL.encode(query, normalize_embeddings=True)
        similarities = np.dot(self.embeddings, query_embedding)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.5:
                passage = self.passages[idx].copy()
                passage["similarity"] = float(similarities[idx])
                results.append(passage)
        
        return results

# Processing state manager
class ProcessingState:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processing_dir = self.cache_dir / "processing"
        self.completed_dir = self.cache_dir / "completed"
        self.processing_dir.mkdir(exist_ok=True)
        self.completed_dir.mkdir(exist_ok=True)
    
    def get_file_hash(self, file_path: str) -> str:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_file_dir(self, file_hash: str) -> Path:
        return self.processing_dir / file_hash
    
    def get_completed_file_dir(self, file_hash: str) -> Path:
        return self.completed_dir / file_hash
    
    def get_state_path(self, file_hash: str) -> Path:
        return self.get_file_dir(file_hash) / "state.pkl"
    
    def save_file(self, file_path: str, file_hash: str = None):
        if file_hash is None:
            file_hash = self.get_file_hash(file_path)
        
        file_dir = self.get_file_dir(file_hash)
        file_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = file_dir / Path(file_path).name
        import shutil
        shutil.copy2(file_path, dest_path)
        
        return file_hash
    
    def load_state(self, file_hash: str) -> Dict[str, Any]:
        state_path = self.get_state_path(file_hash)
        if state_path.exists():
            with open(state_path, "rb") as f:
                return pickle.load(f)
        return {}
    
    def save_state(self, file_hash: str, state: Dict[str, Any]):
        state_path = self.get_state_path(file_hash)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
    
    def mark_completed(self, file_hash: str):
        src_dir = self.get_file_dir(file_hash)
        dest_dir = self.get_completed_file_dir(file_hash)
        
        if src_dir.exists():
            import shutil
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.move(str(src_dir), str(dest_dir))
    
    def get_incomplete_files(self) -> List[str]:
        incomplete_files = []
        for item in self.processing_dir.iterdir():
            if item.is_dir():
                incomplete_files.append(item.name)
        return incomplete_files
    
    def get_completed_files(self) -> List[str]:
        completed_files = []
        for item in self.completed_dir.iterdir():
            if item.is_dir():
                completed_files.append(item.name)
        return completed_files

# Definition aggregator with similarity-based filtering
class DefinitionAggregator:
    def __init__(self):
        self.fragments: Dict[str, List[str]] = defaultdict(list)
        self.sources: Dict[str, List[int]] = defaultdict(list)
        self.embeddings_cache: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.final_definitions: Dict[str, str] = {}
        self.partial_info: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    async def add_fragment(self, term: str, definition: str, chunk_id: int) -> bool:
        """
        Add definition fragment with similarity check.
        Returns True if added, False if discarded as duplicate.
        """
        if not definition or not definition.strip():
            return False
        
        new_emb = EMBED_MODEL.encode(definition, normalize_embeddings=True)
        
        # Check for very similar existing definitions
        if term in self.embeddings_cache:
            for existing_emb in self.embeddings_cache[term]:
                similarity = float(np.dot(new_emb, existing_emb))
                
                if similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                    print(f"  → Discarding duplicate definition for {term} (similarity: {similarity:.2f})")
                    return False
        
        # Add the definition
        self.fragments[term].append(definition)
        self.sources[term].append(chunk_id)
        self.embeddings_cache[term].append(new_emb)
        
        return True
    
    def add_partial_info(self, compound_term: str, source_term: str, 
                        partial_def: str, chunk_id: int):
        """Append partial info for compound terms"""
        self.partial_info[compound_term].append({
            "source": source_term,
            "partial_def": partial_def,
            "chunk_id": chunk_id
        })
    
    async def amalgamate_all(self, tokenizer: BaseTokenizer):
        """Amalgamate definitions for all terms"""
        print("Amalgamating definitions...")
        
        # Amalgamate non-compound terms
        for term in self.fragments:
            defs = self.fragments[term]
            
            if len(defs) == 1:
                self.final_definitions[term] = defs[0]
                continue
            
            # Multiple fragments - amalgamate with LLM
            defs_formatted = "".join([f"  - {d}\n" for d in defs])
            
            prompt = f"""
用語: {term}

以下の複数の定義断片を統合し、1つの包括的な定義を作成してください。

統合ルール:
1. 共通要素を特定: すべての断片に共通する本質的な特性を抽出
2. 相違点を調和: 異なる視点や側面を統合（矛盾する場合は、より詳細/正確なものを優先）
3. 簡潔性: 結果は元の断片1～2個分の長さに
4. 技術用語は保持: 「電圧」「周波数」など専門用語を含める

定義断片:
{defs_formatted}

出力形式: {{"definition": "<統合された定義>"}}

JSONオブジェクトのみを出力してください。
"""
            
            system_prompt = """
あなたは電力システム技術文書の専門編集者です。
複数の定義断片を統合し、正確で包括的な単一定義を生成してください。
"""
            
            response = await vllm_complete(prompt, system_prompt)
            if response:
                try:
                    result = json_repair.loads(response)
                    self.final_definitions[term] = result.get("definition", defs[0])
                except:
                    self.final_definitions[term] = defs[0]
            else:
                self.final_definitions[term] = defs[0]
        
        # Amalgamate compound terms with partial info
        for compound_term in self.partial_info:
            partial_list = self.partial_info[compound_term]
            
            if not partial_list:
                continue
            
            # Collect partial definitions
            partial_defs_formatted = "".join([
                f"  - {info['source']}: {info['partial_def']}\n"
                for info in partial_list
            ])
            
            # If compound also has direct definitions, include them
            direct_defs = self.fragments.get(compound_term, [])
            direct_defs_text = ""
            if direct_defs:
                direct_defs_formatted = "".join([f"  - {d}\n" for d in direct_defs])
                direct_defs_text = f"\n直接的な定義:\n{direct_defs_formatted}"
            
            prompt = f"""
複合用語: {compound_term}

この複合用語は以下の構成要素で構成されます。各要素の定義を統合し、複合用語全体の定義を作成してください。

構成要素の定義:
{partial_defs_formatted}{direct_defs_text}

統合ルール:
1. 各構成要素の定義を理解した上で、複合用語全体がどのように機能するかを定義
2. 構成要素の定義を単純に列挙しない（例：「○○とは、△△と×××の組み合わせ」は避ける）
3. 複合用語が実現する機能・目的を前面に出す
4. 例がある場合は簡潔に含める

出力形式: {{"definition": "<複合用語の統合定義>"}}

JSONオブジェクトのみを出力してください。
"""
            
            system_prompt = """
あなたは電力システム技術文書の専門編集者です。
複合語の構成要素の定義を統合し、完全で正確な定義を生成してください。
"""
            
            response = await vllm_complete(prompt, system_prompt)
            if response:
                try:
                    result = json_repair.loads(response)
                    self.final_definitions[compound_term] = result.get("definition", direct_defs[0] if direct_defs else "")
                except:
                    self.final_definitions[compound_term] = direct_defs[0] if direct_defs else ""
            else:
                self.final_definitions[compound_term] = direct_defs[0] if direct_defs else ""

# Graph manager with only COMPOUND_OF and PARENT_TERM
class GraphManager:
    def __init__(self, working_dir: str, tokenizer: BaseTokenizer):
        self.working_dir = Path(working_dir)
        self.nodes: Dict[str, Dict[str, str]] = {}
        self.edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.tokenizer = tokenizer
        self.working_dir.mkdir(parents=True, exist_ok=True)
    
    def add_terms(self, terms_with_definitions: Dict[str, str]):
        """Add terms as graph nodes"""
        for term, definition in terms_with_definitions.items():
            self.nodes[term] = {"definition": definition}
    
    def add_compound_relationships(self, partial_info: Dict[str, List[Dict[str, Any]]]):
        """
        Create COMPOUND_OF relationships from partial info.
        For each component found, create an edge: compound --[COMPOUND_OF]--> component
        """
        for compound_term, partial_list in partial_info.items():
            for info in partial_list:
                source_term = info["source"]
                
                # Create edge from compound to component
                self.edges[compound_term].append({
                    "target": source_term,
                    "type": "COMPOUND_OF",
                    "weight": 1.0
                })
    
    def build_parent_term_relationships(self):
        """
        Build PARENT_TERM relationships.
        A term is parent of another if it appears in the definition.
        Use tokenizer to identify nouns, check if they're in nodes.
        """
        print("Building parent-term relationships...")
        
        for term, node_data in self.nodes.items():
            definition = node_data["definition"]
            tokens = self.tokenizer.tokenize(definition)
            
            # Get nouns from definition
            nouns_in_def = []
            for i, token in enumerate(tokens):
                # Check if it's a noun POS
                if "名詞" in token.get("pos", ""):
                    surface = token["surface"]
                    if surface in self.nodes and surface != term:
                        # Only keep early nouns (first 30% of definition)
                        if i < len(tokens) * 0.3:
                            nouns_in_def.append(surface)
            
            # Add PARENT_TERM edges
            for parent_term in set(nouns_in_def):
                self.edges[term].append({
                    "target": parent_term,
                    "type": "PARENT_TERM",
                    "weight": 1.0
                })
    
    def query_graph(self, term: str, max_depth: int = 2) -> Dict[str, Any]:
        """Query graph for related terms using BFS"""
        if not self.nodes or term not in self.nodes:
            return {"found": False, "query_term": term}
        
        visited_terms = set()
        queue = [(term, 0)]
        related_nodes = {}
        related_edges = []
        
        while queue:
            current_term, depth = queue.pop(0)
            
            # Skip if already visited or exceeds max depth
            if current_term in visited_terms or depth > max_depth:
                continue
            
            # Mark as visited and add to results
            visited_terms.add(current_term)
            if current_term in self.nodes:
                related_nodes[current_term] = self.nodes[current_term]
            
            # Process edges from current term
            for edge in self.edges.get(current_term, []):
                target = edge["target"]
                
                # Add edge to results
                related_edges.append({
                    "source": current_term,
                    "target": target,
                    "type": edge["type"],
                    "weight": edge["weight"]
                })
                
                # Add target to queue if not visited and within depth limit
                if target not in visited_terms and depth + 1 <= max_depth:
                    queue.append((target, depth + 1))
        
        return {
            "found": True,
            "query_term": term,
            "nodes": related_nodes,
            "edges": related_edges,
            "total_nodes": len(related_nodes),
            "total_edges": len(related_edges)
        }
    
    def save(self):
        """Save graph data"""
        nodes_path = self.working_dir / "graph_nodes.json"
        with open(nodes_path, "w", encoding="utf-8") as f:
            json.dump(self.nodes, f, indent=2, ensure_ascii=False)
        
        edges_path = self.working_dir / "graph_edges.json"
        with open(edges_path, "w", encoding="utf-8") as f:
            json.dump(dict(self.edges), f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved graph: {len(self.nodes)} nodes, {sum(len(v) for v in self.edges.values())} edges")
    
    def load(self) -> bool:
        """Load graph data"""
        nodes_path = self.working_dir / "graph_nodes.json"
        edges_path = self.working_dir / "graph_edges.json"
        
        if not nodes_path.exists() or not edges_path.exists():
            return False
        
        with open(nodes_path, "r", encoding="utf-8") as f:
            self.nodes = json.load(f)
        
        with open(edges_path, "r", encoding="utf-8") as f:
            edges_data = json.load(f)
            self.edges = defaultdict(list, {k: v for k, v in edges_data.items()})
        
        print(f"✓ Loaded graph: {len(self.nodes)} nodes, {sum(len(v) for v in self.edges.values())} edges")
        return True

# Dictionary manager
class DictionaryManager:
    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.dictionary: Dict[str, str] = {}
        self.working_dir.mkdir(parents=True, exist_ok=True)
    
    def add_terms(self, terms_with_definitions: Dict[str, str]):
        """Add or update terms"""
        for term, definition in terms_with_definitions.items():
            self.dictionary[term.lower()] = definition
    
    def lookup_exact(self, term: str) -> Optional[str]:
        """Exact lookup"""
        return self.dictionary.get(term.lower())
    
    def lookup_fuzzy(self, term: str, min_similarity: float = 0.6, top_n: int = 10) -> List[Dict]:
        """Fuzzy lookup using character similarity"""
        from difflib import SequenceMatcher
        
        similarities = []
        term_normalized = term.lower()
        
        for dict_term, definition in self.dictionary.items():
            similarity = SequenceMatcher(None, term_normalized, dict_term).ratio()
            if similarity < min_similarity:
                continue
            
            similarities.append({
                "term": dict_term,
                "definition": definition,
                "similarity": similarity
            })
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_n]
    
    def save(self):
        """Save dictionary to disk"""
        dict_path = self.working_dir / "dictionary.json"
        with open(dict_path, "w", encoding="utf-8") as f:
            json.dump(self.dictionary, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(self.dictionary)} terms")
    
    def load(self) -> bool:
        """Load dictionary from disk"""
        dict_path = self.working_dir / "dictionary.json"
        if dict_path.exists():
            with open(dict_path, "r", encoding="utf-8") as f:
                self.dictionary = json.load(f)
            print(f"✓ Loaded {len(self.dictionary)} terms")
            return True
        return False

# Chunk processor
class ChunkProcessor:
    def __init__(self, matcher: TermMatcher, tokenizer: BaseTokenizer):
        self.matcher = matcher
        self.tokenizer = tokenizer
    
    async def process_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single chunk and extract terms/definitions"""
        chunk_id = chunk["id"]
        chunk_text = chunk["text"]
        tokens = chunk["tokens"]
        
        print(f"{'='*60}")
        print(f"Processing Chunk {chunk_id}")
        print(f"{'='*60}")
        
        # Match terms against pre-filtered technical terms
        full_matches, partial_matches = self.matcher.match_terms(tokens)
        matched_terms = set(full_matches + list(partial_matches.keys()))
        
        print(f"+ Found {len(matched_terms)} terms from pre-filtered list")
        
        # Discover additional terms with LLM
        print("+ Discovering additional terms with LLM...")
        llm_terms = set(await discover_technical_terms(chunk_text))
        print(f"+ LLM found {len(llm_terms)} new terms")
        
        # Discover compound terms
        print("+ Discovering compound technical terms...")
        compound_terms = set(await discover_compound_terms(chunk_text, tokens))
        print(f"+ Found {len(compound_terms)} compound terms")
        
        # Filter all candidate terms
        all_candidate_terms = list(set(matched_terms) | llm_terms | compound_terms)
        if all_candidate_terms:
            print("+ Filtering candidate terms with LLM...")
            filtered_terms = set(await filter_technical_terms(chunk_text, all_candidate_terms))
            print(f"+ After filtering: {len(filtered_terms)} terms")
        else:
            filtered_terms = set()
        
        # Generate definitions
        definitions = {}
        if filtered_terms:
            print(f"+ Generating definitions for {len(filtered_terms)} terms...")
            definitions = await self._generate_definitions_concurrent(
                list(filtered_terms), chunk_text
            )
            print(f"+ Generated {len(definitions)} definitions")
        
        return {
            "chunk_id": chunk_id,
            "terms": list(definitions.keys()),
            "definitions": definitions,
            "partial_matches": partial_matches,
            "compound_terms": list(compound_terms & filtered_terms)
        }
    
    async def _generate_definitions_concurrent(self, terms: List[str], 
                                              context: str) -> Dict[str, str]:
        """Generate definitions in batches"""
        definitions = {}
        batch_size = MAX_CONCURRENT_DEFINITIONS
        
        for i in range(0, len(terms), batch_size):
            batch_terms = terms[i:i + batch_size]
            print(f"+ Batch {i//batch_size + 1}/{(len(terms) - 1)//batch_size + 1}")
            
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_DEFINITIONS)
            
            async def generate_with_semaphore(term: str) -> Tuple[str, Optional[str]]:
                async with semaphore:
                    result = await generate_definition(term, context)
                    if result:
                        return result.term, result.definition
                    return term, None
            
            tasks = [generate_with_semaphore(term) for term in batch_terms]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for res in results:
                if isinstance(res, Exception):
                    print(f"✗ Definition generation failed: {res}")
                    continue
                
                term, defn = res
                if defn is not None and defn.strip():
                    definitions[term] = defn
        
        return definitions

# Main adaptive dictionary builder
class AdaptiveDictionaryBuilder:
    def __init__(self, tokenizer_name: str):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.analytics_loader = AnalyticsLoader(TERM_FOLDER, tokenizer_name)
        self.dictionary = DictionaryManager(WORKING_DIR)
        self.graph = GraphManager(WORKING_DIR, self.tokenizer)
        self.vector_db = VectorDB(VECTOR_DB_DIR)
        self.state = ProcessingState(CACHE_DIR)
        self.technical_terms: Set[str] = set()
    
    def _load_existing_data(self) -> bool:
        """Load existing dictionary, graph, and vector DB"""
        dict_loaded = self.dictionary.load()
        graph_loaded = self.graph.load()
        vector_db_loaded = self.vector_db.load()
        
        if dict_loaded or graph_loaded:
            print(f"✓ Loaded existing data")
            return True
        return False
    
    async def build(self, book_path: str, continue_build: bool = False) -> bool:
        """Main build process with continuation support"""
        print(f"{'='*60}")
        print("ADAPTIVE DICTIONARY BUILDER")
        print(f"{'='*60}")
        print(f"Tokenizer: {self.tokenizer_name}")
        print(f"Book: {book_path}")
        print(f"Continue: {continue_build}")
        print(f"{'='*60}")
        
        if not Path(book_path).exists():
            print(f"✗ File not found: {book_path}")
            return False
        
        # Calculate file hash
        file_hash = self.state.get_file_hash(book_path)
        completed_files = self.state.get_completed_files()
        incomplete_files = self.state.get_incomplete_files()
        
        # Check if already completed
        if file_hash in completed_files:
            print(f"✓ File already processed: {book_path}")
            return True
        
        # Check for incomplete processing
        resume_mode = False
        if file_hash in incomplete_files:
            if continue_build:
                print(f"✓ Resuming processing for: {book_path}")
                resume_mode = True
            else:
                print(f"✗ Processing already started. Use --continue to resume.")
                return False
        else:
            print(f"✓ Starting new processing for: {book_path}")
            self.state.save_file(book_path, file_hash)
        
        # Load or initialize state
        if resume_mode:
            state_data = self.state.load_state(file_hash)
            processed_chunks = state_data.get("processed_chunks", [])
            aggregator = state_data.get("aggregator", DefinitionAggregator())
        else:
            processed_chunks = []
            aggregator = DefinitionAggregator()
        
        # Load book
        with open(book_path, "r", encoding="utf-8") as f:
            book_text = f.read()
        
        print(f"✓ Loaded book: {len(book_text)} characters")
        
        # Load technical terms from analytics (if available)
        self.technical_terms = self.analytics_loader.load_terms()
        
        # Create chunks
        chunker = BookChunker(self.tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)
        chunks = chunker.create_chunks_from_book(book_text)
        
        # Add chunks to vector DB
        passages = [
            {"text": chunk["text"], "source": book_path, "chunk_id": chunk["id"]}
            for chunk in chunks
        ]
        self.vector_db.add_passages(passages)
        
        # Process chunks
        matcher = TermMatcher(self.technical_terms)
        processor = ChunkProcessor(matcher, self.tokenizer)
        
        chunks_to_process = [c for c in chunks if c["id"] not in processed_chunks]
        
        if not chunks_to_process:
            print("✓ All chunks already processed")
        else:
            print(f"✓ Processing {len(chunks_to_process)} remaining chunks")
            
            for chunk in chunks_to_process:
                result = await processor.process_chunk(chunk)
                
                # Add definitions to aggregator
                for term, definition in result["definitions"].items():
                    added = await aggregator.add_fragment(
                        term, definition, result["chunk_id"]
                    )
                    
                    # For compound terms, try to decompose and link to components
                    if added and term in result.get("compound_terms", []):
                        # Tokenize the term itself to find component nouns
                        term_tokens = self.tokenizer.tokenize(term)
                        components = [t["surface"] for t in term_tokens if "名詞" in t.get("pos", "")]
                        
                        # Link to component terms that have definitions
                        for component in components:
                            if component != term and component in aggregator.fragments:
                                component_def = aggregator.fragments[component][0]
                                aggregator.add_partial_info(
                                    term, component, component_def, 
                                    result["chunk_id"]
                                )
                    
                    # Also detect compound relationships from partial matches
                    elif added and term in result["partial_matches"]:
                        # Tokenize definition to find components
                        def_tokens = self.tokenizer.tokenize(definition)
                        for token in def_tokens:
                            if token["surface"] in self.technical_terms:
                                component = token["surface"]
                                
                                # Check if we have a definition for the component
                                if component in aggregator.fragments:
                                    component_def = aggregator.fragments[component][0]
                                    aggregator.add_partial_info(
                                        term, component, component_def, 
                                        result["chunk_id"]
                                    )
                
                # Save state
                processed_chunks.append(result["chunk_id"])
                state_data = {
                    "processed_chunks": processed_chunks,
                    "aggregator": aggregator,
                    "input_file": book_path
                }
                self.state.save_state(file_hash, state_data)
        
        # Amalgamate definitions
        await aggregator.amalgamate_all(self.tokenizer)
        
        print(f"{'='*60}")
        print(f"AGGREGATION COMPLETE: {len(aggregator.final_definitions)} terms")
        print(f"{'='*60}")
        
        # Build dictionary
        print("Building dictionary...")
        self.dictionary.add_terms(aggregator.final_definitions)
        
        # Build graph
        print("Building graph...")
        self.graph.add_terms(aggregator.final_definitions)
        self.graph.add_compound_relationships(aggregator.partial_info)
        self.graph.build_parent_term_relationships()
        
        # Save all
        print("Saving data...")
        self.dictionary.save()
        self.graph.save()
        self.vector_db.save()
        
        # Mark as completed
        self.state.mark_completed(file_hash)
        
        print(f"{'='*60}")
        print("BUILD COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
        return True
    
    async def process_terms_from_json(self, terms_json_path: str, 
                                     text_files: Optional[List[str]] = None) -> bool:
        """Process specific terms from JSON with optional text files"""
        print(f"{'='*60}")
        print("PROCESSING TERMS FROM JSON")
        print(f"{'='*60}")
        
        if not Path(terms_json_path).exists():
            print(f"✗ Terms file not found: {terms_json_path}")
            return False
        
        # Load terms from JSON
        with open(terms_json_path, "r", encoding="utf-8") as f:
            terms_data = json.load(f)
        
        if isinstance(terms_data, dict) and "terms" in terms_data:
            terms = terms_data["terms"]
        elif isinstance(terms_data, list):
            terms = terms_data
        else:
            print("✗ Invalid JSON format")
            return False
        
        print(f"✓ Loaded {len(terms)} terms from JSON")
        
        aggregator = DefinitionAggregator()
        
        # If text files provided, process them
        if text_files:
            for text_file in text_files:
                if not Path(text_file).exists():
                    print(f"✗ File not found: {text_file}")
                    continue
                
                file_hash = self.state.get_file_hash(text_file)
                self.state.save_file(text_file, file_hash)
                
                with open(text_file, "r", encoding="utf-8") as f:
                    text = f.read()
                
                print(f"✓ Processing: {text_file}")
                
                # Create chunks
                chunker = BookChunker(self.tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)
                chunks = chunker.create_chunks_from_book(text)
                
                # Add to vector DB
                passages = [
                    {"text": chunk["text"], "source": text_file, "chunk_id": chunk["id"]}
                    for chunk in chunks
                ]
                self.vector_db.add_passages(passages)
                
                # Process chunks to find terms
                matcher = TermMatcher(set(terms))
                processor = ChunkProcessor(matcher, self.tokenizer)
                
                for chunk in chunks:
                    result = await processor.process_chunk(chunk)
                    
                    for term, definition in result["definitions"].items():
                        await aggregator.add_fragment(term, definition, chunk["id"])
        
        # Search vector DB for each term to find definitions
        print("Searching vector DB for term definitions...")
        for term in terms:
            relevant_passages = self.vector_db.search(term, top_k=3)
            
            if relevant_passages:
                context = "\n\n".join([p["text"] for p in relevant_passages])
                result = await generate_definition(term, context)
                
                if result and result.definition:
                    await aggregator.add_fragment(term, result.definition, -1)
                    print(f"  ✓ Found definition for {term}")
        
        # Amalgamate
        await aggregator.amalgamate_all(self.tokenizer)
        
        # Save
        self.dictionary.add_terms(aggregator.final_definitions)
        self.graph.add_terms(aggregator.final_definitions)
        self.graph.add_compound_relationships(aggregator.partial_info)
        self.graph.build_parent_term_relationships()
        
        self.dictionary.save()
        self.graph.save()
        self.vector_db.save()
        
        print(f"{'='*60}")
        print("TERMS PROCESSING COMPLETED")
        print(f"{'='*60}")
        
        return True

# Query system
class QuerySystem:
    def __init__(self):
        self.dictionary = DictionaryManager(WORKING_DIR)
        self.graph = GraphManager(WORKING_DIR, get_tokenizer("sudachipy"))
    
    def load_data(self) -> bool:
        dict_loaded = self.dictionary.load()
        graph_loaded = self.graph.load()
        
        if not dict_loaded:
            print("✗ No dictionary data found")
            return False
        
        return True
    
    def dict_lookup(self, term: str, exact: bool = False, top_n: int = 10) -> Dict[str, Any]:
        exact_match = self.dictionary.lookup_exact(term)
        
        if exact_match:
            return {
                "found": True,
                "exact_match": term,
                "definition": exact_match
            }
        
        if exact:
            return {
                "found": False,
                "query": term,
                "message": "No exact match found"
            }
        
        fuzzy_matches = self.dictionary.lookup_fuzzy(term, top_n=top_n)
        
        return {
            "found": False,
            "query": term,
            "fuzzy_matches": fuzzy_matches,
            "message": f"Found {len(fuzzy_matches)} similar terms"
        }
    
    async def full_query(self, term: str, max_depth: int = 2) -> Dict[str, Any]:
        dict_result = self.dict_lookup(term)
        graph_result = None
        
        if dict_result["found"]:
            # Exact match found - query graph directly
            graph_result = self.graph.query_graph(term, max_depth=max_depth)
        else:
            # No exact match - try fuzzy matching
            fuzzy_matches = dict_result.get("fuzzy_matches", [])
            
            if fuzzy_matches and fuzzy_matches[0]["similarity"] >= 0.1:
                # Use best fuzzy match if similarity >= 0.1
                best_match = fuzzy_matches[0]
                graph_result = self.graph.query_graph(best_match["term"], max_depth=max_depth)
                
                # Update dict_result to indicate we're using fuzzy match
                dict_result["used_fuzzy_match"] = best_match["term"]
                dict_result["fuzzy_similarity"] = best_match["similarity"]
        
        return {
            "query": term,
            "dict_result": dict_result,
            "graph_result": graph_result
        }

# Main function
async def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [options]")
        print("Commands:")
        print("  build <book_file> <tokenizer_name> [--continue]")
        print("  terms <terms_json> [text_file1] [text_file2] ...")
        print("  dict <term>")
        print("  query <term>")
        print("  resume")
        print("\nExamples:")
        print("  python main.py build power_systems.txt sudachipy")
        print("  python main.py build power_systems.txt sudachipy --continue")
        print("  python main.py terms terms.json")
        print("  python main.py terms terms.json doc1.txt doc2.txt")
        print("  python main.py dict 遠隔制御")
        print("  python main.py query 遠隔制御")
        print("  python main.py resume")
        return
    
    command = sys.argv[1]
    
    if command == "build":
        if len(sys.argv) < 4:
            print("Usage: python main.py build <book_file> <tokenizer_name> [--continue]")
            return
        
        book_file = sys.argv[2]
        tokenizer_name = sys.argv[3]
        continue_build = "--continue" in sys.argv
        
        builder = AdaptiveDictionaryBuilder(tokenizer_name)
        builder._load_existing_data()
        
        success = await builder.build(book_file, continue_build)
        
        if not success:
            sys.exit(1)
    
    elif command == "terms":
        if len(sys.argv) < 3:
            print("Usage: python main.py terms <terms_json> [text_file1] [text_file2] ...")
            return
        
        terms_json = sys.argv[2]
        text_files = sys.argv[3:] if len(sys.argv) > 3 else None
        
        builder = AdaptiveDictionaryBuilder("sudachipy")
        builder._load_existing_data()
        
        success = await builder.process_terms_from_json(terms_json, text_files)
        
        if not success:
            sys.exit(1)
    
    elif command == "dict":
        if len(sys.argv) < 3:
            print("Usage: python main.py dict <term>")
            return
        
        term = sys.argv[2]
        query_system = QuerySystem()
        
        if not query_system.load_data():
            print("✗ No dictionary data found")
            return
        
        result = query_system.dict_lookup(term)
        
        print(f"{'='*60}")
        print("DICTIONARY LOOKUP")
        print(f"{'='*60}")
        
        if result["found"]:
            print("✓ EXACT MATCH")
            print(f"Term: {result['exact_match']}")
            print(f"Definition: {result['definition']}")
        else:
            print(f"Query: {result['query']}")
            if "fuzzy_matches" in result and result["fuzzy_matches"]:
                print(f"\n{result['message']}")
                for i, match in enumerate(result["fuzzy_matches"], 1):
                    print(f"\n{i}. {match['term']} (similarity: {match['similarity']:.2f})")
                    print(f"   {match['definition'][:100]}...")
            else:
                print("No matches found")
    
    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python main.py query <term>")
            return
        
        term = sys.argv[2]
        query_system = QuerySystem()
        
        if not query_system.load_data():
            print("✗ No dictionary data found")
            return
        
        result = await query_system.full_query(term)
        
        print(f"{'='*60}")
        print("FULL QUERY RESULTS")
        print(f"{'='*60}")
        
        dict_result = result["dict_result"]
        
        if dict_result["found"]:
            print("✓ DEFINITION FOUND")
            print(f"Term: {dict_result['exact_match']}")
            print(f"Definition: {dict_result['definition']}")
        elif dict_result.get("used_fuzzy_match"):
            print("✓ FUZZY MATCH USED")
            print(f"Query: {result['query']}")
            print(f"Matched: {dict_result['used_fuzzy_match']} (similarity: {dict_result['fuzzy_similarity']:.2f})")
            
            # Show definition of matched term
            matched_def = query_system.dictionary.lookup_exact(dict_result['used_fuzzy_match'])
            if matched_def:
                print(f"Definition: {matched_def}")
        
        if result["graph_result"]:
            graph = result["graph_result"]
            if graph["found"] and (graph["total_nodes"] > 1 or graph["total_edges"] > 0):
                print(f"\n{'-'*60}")
                print(f"Related Terms ({graph['total_nodes']} nodes, {graph['total_edges']} edges)")
                print(f"{'-'*60}")
                
                edge_types = defaultdict(list)
                for edge in graph["edges"]:
                    edge_types[edge["type"]].append(edge)
                
                for edge_type in sorted(edge_types.keys()):
                    edges = edge_types[edge_type]
                    print(f"\n{edge_type}:")
                    for edge in edges[:10]:  # Show up to 10 edges per type
                        print(f"  {edge['source']} → {edge['target']}")
            else:
                print("\nNo related terms found in graph")
        
        if not dict_result["found"] and not dict_result.get("used_fuzzy_match"):
            print(f"No exact match for: {dict_result['query']}")
            
            if "fuzzy_matches" in dict_result and dict_result["fuzzy_matches"]:
                print(f"\n{dict_result['message']}")
                for i, match in enumerate(dict_result["fuzzy_matches"][:3], 1):
                    print(f"\n{i}. {match['term']} (similarity: {match['similarity']:.2f})")
                    print(f"   {match['definition'][:100]}...")
    
    elif command == "resume":
        print("Checking for incomplete processing...")
        
        state = ProcessingState(CACHE_DIR)
        incomplete_files = state.get_incomplete_files()
        
        if not incomplete_files:
            print("✓ No incomplete files")
            return
        
        print(f"Found {len(incomplete_files)} incomplete files:")
        
        builder = AdaptiveDictionaryBuilder("sudachipy")
        builder._load_existing_data()
        
        for file_hash in incomplete_files:
            file_dir = state.get_file_dir(file_hash)
            original_files = list(file_dir.glob("*"))
            
            if original_files:
                original_file = original_files[0]
                print(f"  - {original_file.name}")
                
                success = await builder.build(str(original_file), continue_build=True)
                
                if success:
                    print(f"✓ Resumed {original_file.name}")
        
        print(f"{'='*60}")
        print("RESUME COMPLETED")
        print(f"{'='*60}")
    
    else:
        print(f"✗ Unknown command: {command}")

if __name__ == "__main__":
    asyncio.run(main())
