import asyncio
import json
from pathlib import Path
from tqdm.asyncio import tqdm
import time
import re
from collections import Counter
from sudachipy import tokenizer
from sudachipy import dictionary
import requests
import aiohttp

# ============ CONFIGURATION ============
# vLLM server configuration
VLLM_SERVER_URL = "http://localhost:8000/v1/completions"  # Change if server is remote
MODEL_NAME = "Qwen/Qwen3-8B-AWQ"  # Must match server model

# Sampling parameters
SAMPLING_PARAMS = {
    "temperature": 0.1,
    "top_p": 0.95,
    "max_tokens": 1024,
    "stop": ["</output>", "\n\n\n"],
}

MAX_CONCURRENT_REQUESTS = 50
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Initialize Sudachi
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

# Wikipedia cache to avoid redundant API calls
wiki_cache = {}

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
print(f"\n✓ Output directory: {output_dir.absolute()}")

# ============ vLLM CLIENT ============
async def call_vllm_server(session, prompt, request_id="", max_retries=3):
    """Call vLLM server via OpenAI-compatible API with retries"""
    for attempt in range(max_retries + 1):
        async with semaphore:
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                **SAMPLING_PARAMS
            }
            
            try:
                timeout = aiohttp.ClientTimeout(total=600 + attempt * 60)  # Increase timeout per attempt
                async with session.post(VLLM_SERVER_URL, json=payload, timeout=timeout) as response:
                    if response.status != 200:
                        print(f"    Warning: Server returned {response.status} for request {request_id} (attempt {attempt+1})")
                        if attempt < max_retries:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return ""
                    
                    result = await response.json()
                    return result["choices"][0]["text"].strip()
            except asyncio.TimeoutError:
                print(f"    Warning: Timeout for request {request_id} (attempt {attempt+1})")
            except Exception as e:
                print(f"    Warning: Request {request_id} failed (attempt {attempt+1}): {e}")
            
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return ""

# ============ CORPUS CONTEXT FALLBACK ============
def extract_corpus_context(term, corpus_text):
    """Extract usage examples from corpus when Wikipedia fails"""
    sentences = re.split('[。\n]', corpus_text)
    contexts = [s.strip() for s in sentences if term in s and len(s.strip()) > 10]
    return '\n'.join(contexts[:3]) if contexts else ""

# ============ SUDACHI TOKENIZATION ============
def extract_tokens_from_text(text_lines, existing_dict=None):
    """Extract relevant tokens using Sudachi, processing line by line to avoid memory issues"""
    relevant_pos = ['名詞', '固有名詞']
    all_tokens = []
    BATCH_SIZE = 50
    
    for i in range(0, len(text_lines), BATCH_SIZE):
        batch_lines = text_lines[i:i+BATCH_SIZE]
        # Dynamically split batch if too large
        sub_batch_size = len(batch_lines)
        while sub_batch_size > 0:
            sub_lines = batch_lines[:sub_batch_size]
            sub_text = '\n'.join(sub_lines)
            sub_bytes = len(sub_text.encode('utf-8'))
            
            if sub_bytes <= 49149:
                try:
                    tokens = tokenizer_obj.tokenize(sub_text, mode)
                    
                    for token in tokens:
                        pos = token.part_of_speech()[0]
                        surface = token.surface()
                        normalized = token.normalized_form()
                        
                        if pos in relevant_pos and len(surface) > 1:
                            if existing_dict and normalized.lower() in existing_dict:
                                continue
                            
                            all_tokens.append({
                                'surface': surface,
                                'normalized': normalized,
                                'pos': pos
                            })
                    break  # Success, no need for smaller sub-batch
                except Exception as sub_e:
                    print(f"    Warning: Sudachi error on sub-batch {i}.{len(batch_lines)-sub_batch_size}: {sub_e}")
                    sub_batch_size = max(1, sub_batch_size // 2)  # Halve size
                    continue
            else:
                sub_batch_size = max(1, sub_batch_size // 2)
    
    return all_tokens

def load_existing_definitions(filepath):
    """Load existing dictionary to avoid re-defining terms"""
    if not Path(filepath).exists():
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        existing = {}
        for entry in data:
            term = entry.get('term', '')
            definition = entry.get('definition', '')
            if term and definition:
                existing[term.lower().strip()] = definition
        
        print(f"  Loaded {len(existing)} existing definitions")
        return existing
    except Exception as e:
        print(f"  Warning: Could not load existing definitions: {e}")
        return {}

# ============ MINIMAL FILTERING ============
BLACKLIST_PATTERNS = [
    r'^[0-9]+$',  # Pure numbers
    r'^[a-zA-Z]$',  # Single letters
    r'^(は|が|を|に|へ|と|で|から|より|まで|の|も|や|など)$',  # Particles only
]

def is_obvious_noise(term):
    """Very minimal filter - only obvious garbage"""
    if len(term) <= 1:
        return True
    for pattern in BLACKLIST_PATTERNS:
        if re.search(pattern, term, re.IGNORECASE):
            return True
    return False

# ============ STEP 1: EXTRACT TERMS ============
async def extract_terms_from_corpus(session, corpus_path, existing_dict=None):
    """Extract terms from corpus using multi-pass approach"""
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    text = ''.join(lines)
    print(f"  Corpus length: {len(text)} characters, {len(lines)} lines")
    
    # First pass: Tokenize with Sudachi
    print("  Pass 1: Tokenizing corpus with Sudachi...")
    sudachi_tokens = extract_tokens_from_text(lines, existing_dict)
    print(f"    Found {len(sudachi_tokens)} candidate tokens from Sudachi")
    
    sudachi_terms = set(token_info['normalized'] for token_info in sudachi_tokens)
    print(f"    Unique Sudachi terms: {len(sudachi_terms)}")
    
    # Second pass: LLM extraction for better coverage
    print("  Pass 2: LLM extraction...")
    
    CHUNK_SIZE = 1000
    chunks = [text[i:i+CHUNK_SIZE*4] for i in range(0, len(text), CHUNK_SIZE*4)]
    
    # Build existing terms list for prompt
    existing_terms_str = ""
    if existing_dict:
        sample_existing = list(existing_dict.keys())[:50]
        if sample_existing:
            existing_terms_str = f"\n\n既に定義済みの用語（これらは抽出不要）:\n" + ", ".join(sample_existing)
            if len(existing_dict) > 50:
                existing_terms_str += f"\n（他 {len(existing_dict)-50} 件）"
    
    prompt_template = """あなたは電力工学の専門家です。以下のテキストから電力系統に関連する技術用語を抽出してください。{existing_terms}

ステップ1: テキストを読み、電力系統関連のキーワード（設備、概念、単位など）をリストアップ。
ステップ2: 各用語のカテゴリ（発電/送配電/保護制御/計測/機器/一般）を判断。
ステップ3: 重複を除き、形式で出力。

テキスト:
{text}

形式: 用語|カテゴリ

例:
負荷曲線|送配電
変圧器|機器
周波数制御|保護制御"""

    tasks = [
        call_vllm_server(
            session, 
            prompt_template.format(text=chunk, existing_terms=existing_terms_str),
            f"extract_{i}"
        )
        for i, chunk in enumerate(chunks)
    ]
    
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="  LLM Extracting"):
        result = await coro
        results.append(result)
    
    # Collect terms
    llm_terms = {}
    filtered_count = 0
    
    for result in results:
        for line in result.strip().split('\n'):
            if '|' in line and not line.startswith('形式:') and not line.startswith('カテゴリ:') and not line.startswith('例:') and not line.startswith('ステップ'):
                parts = line.split('|')
                if len(parts) >= 2:
                    term = parts[0].strip()
                    category = parts[1].strip() if len(parts) > 1 else '一般'
                    
                    if is_obvious_noise(term):
                        filtered_count += 1
                        continue
                    
                    if existing_dict and term.lower() in existing_dict:
                        continue
                    
                    if term not in llm_terms:
                        llm_terms[term] = category
    
    print(f"    LLM extracted {len(llm_terms)} unique terms")
    print(f"    Filtered as noise: {filtered_count}")
    
    # Combine Sudachi and LLM terms
    combined_terms = {term: category for term, category in llm_terms.items()}
    for term in sudachi_terms:
        if term not in combined_terms:
            combined_terms[term] = '一般'
    
    print(f"\n  Combined total: {len(combined_terms)} unique terms")
    print(f"    From LLM: {len(llm_terms)}")
    print(f"    From Sudachi only: {len(sudachi_terms - set(llm_terms.keys()))}")
    
    return combined_terms, text

# ============ STEP 2: BASIC CLEANUP + FREQUENCY FILTER ============
def cleanup_terms(terms_dict, corpus_text, min_freq=2):
    """Basic cleanup - remove obvious junk and low-frequency terms"""
    
    print(f"\n  Cleaning up {len(terms_dict)} extracted terms...")
    
    cleaned = {}
    removed = {
        'too_short': [],
        'pure_numbers': [],
        'particles': [],
        'low_freq': []
    }
    
    particles = ['は', 'が', 'を', 'に', 'へ', 'と', 'で', 'から', 'より', 'まで', 'の', 'も', 'や', 'など', 'として', 'について', 'における']
    
    for term, category in terms_dict.items():
        if len(term) <= 1:
            removed['too_short'].append(term)
            continue
        
        if re.match(r'^[0-9]+$', term):
            removed['pure_numbers'].append(term)
            continue
        
        if term in particles:
            removed['particles'].append(term)
            continue
        
        freq = corpus_text.count(term)
        if freq < min_freq:
            removed['low_freq'].append(term)
            continue
        
        cleaned[term] = category
    
    total_removed = sum(len(v) for v in removed.values())
    print(f"    Removed {total_removed} items:")
    for k, v in removed.items():
        print(f"      {k}: {len(v)}")
    print(f"    Kept: {len(cleaned)}/{len(terms_dict)} ({100*len(cleaned)/len(terms_dict):.1f}%)")
    
    return cleaned

# ============ STEP 3: GENERATE DEFINITIONS ============
async def generate_definitions(session, terms_dict):
    """Generate definitions for all terms"""
    
    print(f"\n  Generating definitions for {len(terms_dict)} terms...")
    
    BATCH_SIZE = 10
    terms_list = list(terms_dict.keys())
    
    definition_prompt = """以下の電力系統用語について、簡潔な定義を提供してください。

用語リスト:
{terms}

各用語について「用語: 定義」の形式で1行ずつ出力してください。定義は1文で簡潔に。

例:
負荷曲線: 時間帯別の電力需要の変化を示すグラフ
変圧器: 電圧を変換する電力機器
GIS: ガス絶縁開閉装置
周波数: 交流電力の1秒あたりの波の数

必ず上記の形式で出力してください。"""

    definitions = {}
    raw_outputs = []
    
    batches = [terms_list[i:i+BATCH_SIZE] for i in range(0, len(terms_list), BATCH_SIZE)]
    
    tasks = [
        call_vllm_server(
            session,
            definition_prompt.format(terms='\n'.join(batch)),
            f"define_{i}"
        )
        for i, batch in enumerate(batches)
    ]
    
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="  Generating defs"):
        result = await coro
        raw_outputs.append(result)
        
        for line in result.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if any(skip in line for skip in ['形式:', '例:', 'ステップ', '用語リスト', '電力系統']):
                continue
            
            if ':' in line or '：' in line:
                line = line.replace('：', ':')
                parts = line.split(':', 1)
                
                if len(parts) == 2:
                    term = parts[0].strip()
                    definition = parts[1].strip()
                    
                    term = re.sub(r'^[-•\*\d+\.\)]+\s*', '', term)
                    
                    if term in terms_dict and definition and len(definition) >= 5:
                        definitions[term] = definition
    
    print(f"    Generated {len(definitions)} definitions ({100*len(definitions)/len(terms_dict):.1f}%)")
    
    with open(output_dir / "03_raw_llm_outputs.txt", "w", encoding="utf-8") as f:
        for i, output in enumerate(raw_outputs):
            f.write(f"=== Batch {i} ===\n{output}\n\n")
    
    for term in terms_dict:
        if term not in definitions:
            definitions[term] = ""
    
    return definitions

# ============ STEP 4: VERIFY & IMPROVE WITH RAG + CONSENSUS (OPTIMIZED) ============
async def verify_definitions(session, terms_dict, definitions, corpus_text):
    """Verify with RAG - BATCHED for massive speedup"""
    
    print(f"\n  Verifying definitions with RAG & consensus...")
    
    needs_improvement = [
        term for term in terms_dict.keys()
        if not definitions.get(term) or len(definitions.get(term, '')) < 10
    ]
    print(f"    Processing {len(needs_improvement)}/{len(terms_dict)} terms needing improvement")
    
    if not needs_improvement:
        return definitions
    
    improved_definitions = definitions.copy()
    
    verification_prompt_template = """以下の電力系統用語について、提供された文脈に基づき、適切な定義を1文で生成してください。

参考文脈:
{context}

用語: {term}

形式: {term}: 定義

簡潔かつ正確に。"""

    async def verify_single_term(term):
        """Verify a single term with consensus"""
        corpus_context = extract_corpus_context(term, corpus_text)
        
        if not corpus_context:
            return term, definitions.get(term, "")
        
        # Generate 3 candidates in parallel
        gen_tasks = []
        for j in range(3):
            prompt = verification_prompt_template.format(
                context=corpus_context[:800],
                term=term
            )
            gen_tasks.append(
                call_vllm_server(session, prompt, f"verify_{term}_{j}")
            )
        
        try:
            results = await asyncio.gather(*gen_tasks, return_exceptions=True)
            
            gens = []
            for gen in results:
                if isinstance(gen, str) and ':' in gen:
                    parts = gen.split(':', 1)
                    if len(parts) == 2:
                        gens.append(parts[1].strip())
            
            if gens:
                consensus = Counter(gens).most_common(1)[0][0]
                
                # Simple quality check
                if len(consensus) >= 10:
                    return term, consensus
            
            return term, definitions.get(term, "")
            
        except Exception as e:
            print(f"    Warning: Verification failed for {term}: {e}")
            return term, definitions.get(term, "")
    
    # Process all terms concurrently
    all_tasks = [verify_single_term(term) for term in needs_improvement]
    
    for coro in tqdm(asyncio.as_completed(all_tasks), 
                     total=len(all_tasks), 
                     desc="  Verifying"):
        term, definition = await coro
        improved_definitions[term] = definition
    
    improvements = sum(
        1 for term in needs_improvement 
        if improved_definitions[term] and improved_definitions[term] != definitions.get(term, "")
    )
    print(f"    Improved {improvements} definitions via RAG/consensus")
    
    return improved_definitions

# ============ STEP 5: QUALITY REVIEW ============
async def quality_review(dictionary):
    """Quality review of final dictionary"""
    
    print("\n" + "="*60)
    print("Quality Review")
    print("="*60)
    
    issues = {
        'no_definition': [],
        'very_short_definition': [],
    }
    
    for entry in dictionary:
        term = entry['term']
        definition = entry['definition']
        
        if not definition or definition == '':
            issues['no_definition'].append(term)
        elif len(definition) < 8:
            issues['very_short_definition'].append(term)
    
    print(f"\n  Quality Checks:")
    print(f"    Total entries: {len(dictionary)}")
    print(f"    Missing definitions: {len(issues['no_definition'])}")
    print(f"    Very short definitions: {len(issues['very_short_definition'])}")
    
    if len(dictionary) > 0:
        coverage = (len(dictionary) - len(issues['no_definition'])) / len(dictionary) * 100
        print(f"    Definition coverage: {coverage:.1f}%")
    
    return issues

# ============ MAIN ============
async def main():
    start_time = time.time()
    
    print("="*60)
    print("Technical Dictionary Builder - vLLM Server Mode")
    print("="*60)
    print(f"\nConnecting to vLLM server at: {VLLM_SERVER_URL}")
    
    # Create persistent aiohttp session
    async with aiohttp.ClientSession() as session:
        # Test connection
        try:
            test_result = await call_vllm_server(session, "Test", "connection_test")
            print(f"✓ Successfully connected to vLLM server")
        except Exception as e:
            print(f"ERROR: Cannot connect to vLLM server: {e}")
            print(f"\nMake sure vLLM server is running:")
            print(f"  python -m vllm.entrypoints.openai.api_server \\")
            print(f"    --model {MODEL_NAME} \\")
            print(f"    --quantization awq \\")
            print(f"    --max-model-len 16384 \\")
            print(f"    --gpu-memory-utilization 0.9 \\")
            print(f"    --port 8000")
            return
        
        # Load existing dictionary
        existing_dict_path = output_dir / "power_system_dictionary.json"
        existing_dict = load_existing_definitions(existing_dict_path)
        
        # Step 1: Multi-pass extraction
        print("\n" + "="*60)
        print("Step 1: Multi-pass term extraction")
        print("="*60)
        step_start = time.time()
        
        corpus_file = "wiki.txt"
        if Path(corpus_file).exists():
            terms_with_categories, corpus_text = await extract_terms_from_corpus(session, corpus_file, existing_dict)
            step_time = time.time() - step_start
            print(f"✓ Extracted {len(terms_with_categories)} terms in {step_time:.2f}s")
            
            with open(output_dir / "01_raw_extraction.json", "w", encoding="utf-8") as f:
                json.dump(terms_with_categories, f, ensure_ascii=False, indent=2)
        else:
            print(f"ERROR: {corpus_file} not found!")
            return
        
        # Step 2: Basic cleanup + frequency filter
        print("\n" + "="*60)
        print("Step 2: Basic cleanup + frequency filter")
        print("="*60)
        step_start = time.time()
        
        cleaned = cleanup_terms(terms_with_categories, corpus_text, min_freq=2)
        step_time = time.time() - step_start
        print(f"✓ Cleaned to {len(cleaned)} terms in {step_time:.2f}s")
        
        with open(output_dir / "02_cleaned.json", "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        
        # Step 3: Generate definitions
        print("\n" + "="*60)
        print("Step 3: Generate definitions")
        print("="*60)
        step_start = time.time()
        
        definitions = await generate_definitions(session, cleaned)
        step_time = time.time() - step_start
        print(f"✓ Generated definitions in {step_time:.2f}s")
        
        with open(output_dir / "03_definitions.json", "w", encoding="utf-8") as f:
            json.dump(definitions, f, ensure_ascii=False, indent=2)
        
        # Step 4: Verify with RAG & consensus (OPTIMIZED)
        print("\n" + "="*60)
        print("Step 4: Verify with RAG & consensus (Optimized)")
        print("="*60)
        step_start = time.time()
        
        final_definitions = await verify_definitions(session, cleaned, definitions, corpus_text)
        step_time = time.time() - step_start
        print(f"✓ Verified in {step_time:.2f}s")
        
        # Compile final dictionary
        dictionary = [
            {
                "term": term,
                "category": category,
                "definition": final_definitions.get(term, "")
            }
            for term, category in cleaned.items()
        ]
        
        dictionary = sorted(dictionary, key=lambda x: x['term'])
        
        # Step 5: Quality Review
        issues = await quality_review(dictionary)
        
        # Save outputs
        with open(output_dir / "power_system_dictionary.json", "w", encoding="utf-8") as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=2)
        
        high_quality = [
            entry for entry in dictionary 
            if entry['definition'] and len(entry['definition']) >= 10
        ]
        
        with open(output_dir / "power_system_dictionary_high_quality.json", "w", encoding="utf-8") as f:
            json.dump(high_quality, f, ensure_ascii=False, indent=2)
        
        with open(output_dir / "quality_issues.json", "w", encoding="utf-8") as f:
            json.dump(issues, f, ensure_ascii=False, indent=2)
        
        # Statistics
        category_stats = Counter([e['category'] for e in dictionary])
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ COMPLETE!")
        print(f"{'='*60}")
        print(f"\nTotal entries: {len(dictionary)}")
        print(f"High quality entries: {len(high_quality)} ({100*len(high_quality)/len(dictionary):.1f}%)")
        print(f"\nCategory breakdown:")
        for cat, count in category_stats.most_common():
            print(f"  {cat}: {count} ({100*count/len(dictionary):.1f}%)")
        print(f"\nTotal time: {total_time/60:.2f} minutes")
        print(f"\nFiles saved in '{output_dir}/':")
        print(f"  - power_system_dictionary.json (all {len(dictionary)} entries)")
        print(f"  - power_system_dictionary_high_quality.json ({len(high_quality)} entries)")
        print(f"  - quality_issues.json (terms needing review)")

if __name__ == "__main__":
    asyncio.run(main())
