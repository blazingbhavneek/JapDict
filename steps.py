import asyncio
import json
import os
import re
from asyncio import Semaphore
from pathlib import Path
from string import Template
from typing import Any, Dict, List

import httpx
import openai
import pandas as pd
from dotenv import load_dotenv
from json_repair import repair_json

# Switch between clients by changing this variable
USE_VLLM = True  # Set to False to use OpenAI (ChatGPT) client
MAX_CONCURRENCY = 20  # Adjust as needed; if 1, runs sequentially
semaphore = Semaphore(MAX_CONCURRENCY)


class LLMClient:
    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = 1000,
        endpoint_env: str = None,
    ):
        load_dotenv()
        if model is None:
            model = "kaitchup/Phi-4-AutoRound-GPTQ-4bit" if USE_VLLM else "gpt-4o-mini"
        if temperature is None:
            temperature = 0.1 if USE_VLLM else 0.0
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_vllm = USE_VLLM
        if USE_VLLM:
            endpoint_env = endpoint_env or "VLLM_ENDPOINT"
            self.base_url = os.getenv(endpoint_env)
        else:
            openai.api_type = "azure"
            openai.api_base = os.getenv("AOAI_ENDPOINT")
            openai.api_key = os.getenv("AOAI_API_KEY")
            openai.api_version = os.getenv("AOAI_API_VERSION")

    async def make_generic_request_with_prompt(
        self,
        system_content: str,
        user_content: str,
        extra_body: dict = None,
        **filtered_kwargs,
    ) -> str:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        async with semaphore:
            if self.is_vllm:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    **filtered_kwargs,
                }
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        f"{self.base_url}/v1/chat/completions", json=payload
                    )
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
            else:  # OpenAI

                def sync_openai_call():
                    response = openai.ChatCompletion.create(
                        engine=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        **filtered_kwargs,
                    )
                    return response.choices[0].message.content

                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, sync_openai_call)


class DocumentMatcher:
    def __init__(self, json_file: str, review_excel_file: str):
        self.json_file = json_file
        self.review_excel_file = review_excel_file
        self.extracted_data = None
        self.df = None
        self.llm_client = LLMClient()

    def load_data(self):
        print(f"Loading JSON from: {self.json_file}")
        print(f"Loading Excel from: {self.review_excel_file}")
        if not os.path.exists(self.json_file):
            raise FileNotFoundError(f"JSON file not found: {self.json_file}")
        if not os.path.exists(self.review_excel_file):
            raise FileNotFoundError(f"Excel file not found: {self.review_excel_file}")
        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                self.extracted_data = json.load(f)
            self.df = pd.read_excel(
                self.review_excel_file,
                skiprows=12,
                usecols="B:D",
                sheet_name="【様式】 DRシート",
            )
        except Exception as e:
            print(f"Error in loading {e}")
            raise
        print("COLUMNS OF THE HUMAN REVIEW FILE: ", self.df.columns.tolist())

    def isconvertible(self, num):
        """Check if a number is convertible to int."""
        try:
            i = int(num)
        except Exception:
            return False, 0
        return True, i

    def get_subsection(self, row):
        """Convert Excel title to subsection number (fullwidth japanese)."""
        title = str(row["項目"])
        num_str = title.split("）")[1] if "）" in title else title
        parts = num_str.split(".")
        if len(parts) > 2:
            return parts[0] + "." + parts[1]
        return num_str

    async def find_relevant_content(
        self, feedback: str, target_sub: str, page: int, extracted_data: List[Dict]
    ) -> tuple[str, str, int]:
        """LLM-based finder for the most relevant content chunk from extracted data based on feedback."""
        candidates = []
        major = target_sub.split(".")[0] if "." in target_sub else target_sub

        for i, entry in enumerate(extracted_data):
            if entry.get("subsection_name") is None:
                continue
            entry_sub_name = entry["subsection_name"]
            entry_sub = (
                entry_sub_name.split("）")[1]
                if "）" in entry_sub_name
                else entry_sub_name
            )
            entry_major = entry_sub.split(".")[0] if "." in entry_sub else entry_sub
            content_trunc = entry.get("content", "")[:800]

            if entry_major == major:
                candidates.append((i, entry_sub, content_trunc))
            if len(candidates) >= 10:
                break

        if len(candidates) < 2:
            # Add more if needed
            for i, entry in enumerate(extracted_data):
                if any(i == c[0] for c in candidates):
                    continue
                if len(candidates) >= 10:
                    break
                entry_sub_name = entry.get("subsection_name")
                if entry_sub_name:
                    entry_sub = (
                        entry_sub_name.split("）")[1]
                        if "）" in entry_sub_name
                        else entry_sub_name
                    )
                    content_trunc = entry.get("content", "")[:800]
                    candidates.append((i, entry_sub, content_trunc))

        candidates_str = "\n".join(
            [
                f"Candidate {j+1} (Sub: {sub}): {cont}"
                for j, (i, sub, cont) in enumerate(candidates)
            ]
        )

        pick_prompt = Template(
            """あなたは文書マッチング専門家です。
            以下のフィードバックに最も関連する候補を選んでください。
            フィードバック: $feedback
            候補: $candidates_str
            最も関連する候補の番号 (1から)を出力してください。理由も簡単に。"""
        )
        prompt = pick_prompt.safe_substitute(
            feedback=feedback, candidates_str=candidates_str
        )

        try:
            response = await self.llm_client.make_generic_request_with_prompt(
                system_content="出力は候補番号と簡単理由のみ。", user_content=prompt
            )
            num_match = re.search(r"(\d+)", response)
            if num_match:
                index = int(num_match.group(1)) - 1
                if 0 <= index < len(candidates):
                    cand_i, sub, _ = candidates[index]
                    full_content = extracted_data[cand_i].get("content", "")
                    return full_content, sub, cand_i
        except Exception as e:
            print(f"Error in finding relevant content: {e}")
            # Fallback to first candidate
            if candidates:
                cand_i, sub, _ = candidates[0]
                full_content = extracted_data[cand_i].get("content", "")
                return full_content, sub, cand_i
        # Ultimate fallback
        return "", "", -1


class DeepResearcher:
    """
    Analyzes human feedback to understand what errors exist and their technical context.
    """

    def __init__(self):
        self.llm_client = LLMClient(max_tokens=2000)

    async def analyze_feedback(self, content: str, feedback: str) -> Dict:
        """
        Deep analysis of human feedback to understand the error.

        Returns:
            research_report: Dict containing problem analysis
        """
        analyze_prompt = """あなたは電力システム文書のエラー分析専門家です。

## タスク
人間のレビュアーが指摘したフィードバックを深く分析し、以下を明確にしてください：

1. **何が問題か**: レビュアーが指摘している具体的な問題点
2. **なぜ問題か**: その問題が技術的になぜ重要なのか
3. **どこに問題があるか**: 文書内のどの部分・要素に問題があるか
4. **エラーのタイプ**: 欠落、不整合、誤記、不明瞭、等

## 入力
文書内容: 
$content

人間のフィードバック: 
$feedback

## 出力形式 (JSON)
{
  "problem_description": "問題の説明（自然言語）",
  "why_important": "なぜこれが問題なのか（技術的理由）",
  "location_in_content": "問題がある箇所の説明",
  "error_types": ["欠落", "不整合", "誤記"],
  "affected_elements": ["影響を受ける要素のリスト"],
  "needs_translation": false
}

**重要**: 日本語で回答してください。JSON形式のみ出力。"""

        template = Template(analyze_prompt)
        prompt = template.safe_substitute(
            content=content[:3000] if len(content) > 3000 else content,
            feedback=feedback,
        )

        try:
            response = await self.llm_client.make_generic_request_with_prompt(
                system_content="You are an error analysis expert. Output only valid JSON in Japanese.",
                user_content=prompt,
            )
            repaired = repair_json(response)
            research_report = json.loads(repaired)
            return research_report
        except Exception as e:
            print(f"Error in analyze_feedback: {e}")
            return {
                "problem_description": feedback,
                "why_important": "分析失敗",
                "location_in_content": "不明",
                "error_types": ["不明"],
                "affected_elements": [],
                "needs_translation": False,
            }


class RootCauseAnalyzer:
    """
    Executes 7 micro-tasks to generate validation workflows from error patterns.
    """

    def __init__(self):
        self.llm_client = LLMClient(max_tokens=3000)

    async def decompose_into_error_patterns(
        self, research_report: Dict, feedback: str
    ) -> List[Dict]:
        """
        Decompose feedback into individual error patterns (max 3).

        Returns:
            List of error pattern descriptions
        """
        decompose_prompt = """あなたはエラーパターン分解の専門家です。

## タスク
DeepResearcherの分析結果を受け取り、個別の検証可能なエラーパターンに分解してください。

**重要ルール**:
- 1つのフィードバックから複数のエラーパターンが見つかる場合、それぞれを独立させる
- 各エラーパターンは独立して検証可能であること
- 最大3つまでのエラーパターンに分解（それ以上は統合）

## 入力
分析レポート: 
$research_report

元のフィードバック:
$feedback

## 出力形式 (JSON)
{
  "error_patterns": [
    {
      "pattern_id": 1,
      "pattern_description": "エラーパターンの説明（自然言語）",
      "detection_focus": "何を検出すべきか"
    }
  ]
}

**日本語で回答。最大3パターンまで。JSON形式のみ出力。**"""

        template = Template(decompose_prompt)
        prompt = template.safe_substitute(
            research_report=json.dumps(research_report, ensure_ascii=False, indent=2),
            feedback=feedback,
        )

        try:
            response = await self.llm_client.make_generic_request_with_prompt(
                system_content="You are an error pattern decomposition expert. Output only valid JSON with max 3 patterns.",
                user_content=prompt,
            )
            repaired = repair_json(response)
            result = json.loads(repaired)
            patterns = result.get("error_patterns", [])

            # Enforce max 3 patterns
            if len(patterns) > 3:
                patterns = patterns[:3]

            return patterns
        except Exception as e:
            print(f"Error in decompose_into_error_patterns: {e}")
            # Fallback: create single pattern from research report
            return [
                {
                    "pattern_id": 1,
                    "pattern_description": research_report.get(
                        "problem_description", "不明なエラー"
                    ),
                    "detection_focus": research_report.get(
                        "location_in_content", "文書全体"
                    ),
                }
            ]

    def validate_natural_language_step(self, step: Dict) -> bool:
        """
        Check that validation step contains no code/programmatic elements.
        """
        forbidden_patterns = [
            r"regex",
            r"\.find\(",
            r"\.get\(",
            r"!=",
            r"==",
            r"if\s+.*\s+then",
            r"SELECT",
            r"WHERE",
            r"function",
            r"lambda",
            r"=>",
            r"\bdef\b",
            r"\bclass\b",
        ]

        full_text = f"{step.get('action', '')} {step.get('instruction', '')} {step.get('expected_output', '')}"

        for pattern in forbidden_patterns:
            if re.search(pattern, full_text, re.IGNORECASE):
                return False
        return True

    def enforce_step_limit(self, steps: List[Dict], max_steps: int = 10) -> List[Dict]:
        """
        Limit to max 10 steps.
        """
        if len(steps) <= max_steps:
            return steps

        # If over limit, truncate and warn
        print(f"Warning: Generated {len(steps)} steps, truncating to {max_steps}")
        return steps[:max_steps]

    async def generate_validation_workflow(
        self, error_pattern: Dict, content: str, research_report: Dict
    ) -> Dict:
        """
        Execute 7 micro-tasks chain and generate validation workflow.

        Returns:
            validation_workflow: Dict with validation_steps
        """
        micro_tasks_prompt = """あなたは文書エラー検出ワークフロー生成の専門家です。

## タスク
エラーパターンから、小型LLMが実行可能な検証ステップを生成してください。

### 7つのマイクロタスク実行順序:

1. **証拠抽出**: 文書内容から問題の証拠を特定
2. **矛盾発見**: 不整合や矛盾点を見つける
3. **パターン分類**: エラーの種類を分類
4. **原因仮説**: なぜこのエラーが発生したか
5. **前提条件検出**: どのような状況でこのエラーが起こるか
6. **検証ステップ生成**: 小型LLMが実行できる自然言語の検証手順
7. **ステップ順序付け**: 論理的な順序に並べる

## 入力
エラーパターン: 
$error_pattern

文書内容: 
$content

分析レポート: 
$research_report

## 重要な制約
- 検証ステップは**自然言語のみ**
- プログラムコード、正規表現、SQLなど一切禁止
- 人間が読んで理解できる指示のみ
- **最大10ステップ**まで（それ以上は統合）
- 各ステップは独立して実行可能であること

## 出力形式 (JSON)
{
  "validation_steps": [
    {
      "step_number": 1,
      "action": "実行するアクション（動詞で始まる簡潔な文）",
      "instruction": "詳細な指示（小型LLMが理解できる自然言語）。何を探すか、どこを見るか、何に注意するかを明確に記述。",
      "expected_output": "このステップの期待される出力（自然言語で説明）",
      "pass_condition": "このステップが成功したと判断する条件（自然言語）"
    }
  ]
}

### 良い例:
{
  "step_number": 1,
  "action": "変圧器仕様表を特定する",
  "instruction": "文書内で変圧器の仕様が記載されている表または構造化されたリストを探してください。'変圧器'、'定格容量'、'電圧比'、'インピーダンス'などのキーワードに注目してください。",
  "expected_output": "変圧器仕様を含む表またはリストの位置を特定し、その表に含まれる主要な列名をリストアップ",
  "pass_condition": "変圧器に関連する表またはリストが少なくとも1つ見つかり、定格容量または電圧比の列が存在する"
}

### 悪い例（避けるべき）:
- "regex /変圧器.*定格/ を実行" ❌ (プログラム的)
- "Table.find('transformer')" ❌ (コード)
- "Check if transformer_specs != null" ❌ (プログラム構文)

**日本語のみ。最大10ステップ。自然言語のみ。JSON形式のみ出力。**"""

        template = Template(micro_tasks_prompt)
        prompt = template.safe_substitute(
            error_pattern=json.dumps(error_pattern, ensure_ascii=False, indent=2),
            content=content[:2000] if len(content) > 2000 else content,
            research_report=json.dumps(research_report, ensure_ascii=False, indent=2),
        )

        try:
            response = await self.llm_client.make_generic_request_with_prompt(
                system_content="You are a validation workflow generation expert. Output only valid JSON with natural language steps (max 10).",
                user_content=prompt,
            )
            repaired = repair_json(response)
            result = json.loads(repaired)
            steps = result.get("validation_steps", [])

            # Validate each step is natural language
            validated_steps = []
            for step in steps:
                if self.validate_natural_language_step(step):
                    validated_steps.append(step)
                else:
                    print(
                        f"Warning: Step {step.get('step_number')} contains programmatic elements, skipping"
                    )

            # Enforce step limit
            validated_steps = self.enforce_step_limit(validated_steps, max_steps=10)

            # Renumber steps
            for i, step in enumerate(validated_steps, 1):
                step["step_number"] = i

            return {"validation_steps": validated_steps}
        except Exception as e:
            print(f"Error in generate_validation_workflow: {e}")
            # Fallback: create basic validation workflow
            return {
                "validation_steps": [
                    {
                        "step_number": 1,
                        "action": "文書内容を確認する",
                        "instruction": f"{error_pattern.get('detection_focus', '問題箇所')}を文書内で探してください。",
                        "expected_output": "該当箇所の特定",
                        "pass_condition": "該当箇所が見つかる",
                    }
                ]
            }


async def main():
    # Define input and output folders
    input_folder = Path("input")
    if USE_VLLM:
        output_folder = Path("output_vllm")
    else:
        output_folder = Path("output")

    # Create folders if they don't exist
    input_folder.mkdir(exist_ok=True)
    output_folder.mkdir(exist_ok=True)

    # Define multiple input pairs as list of tuples (json file, excel_file)
    input_pairs = [
        # (input_folder / "test_doc_with_tables.json", input_folder / "human.xlsx"),
        (input_folder / "test_doc2_with_tables.json", input_folder / "human2.xlsx"),
    ]

    # Initialize components
    researcher = DeepResearcher()
    analyzer = RootCauseAnalyzer()

    # Global rules library and counter
    rules_library = []
    rule_counter = 1

    # Process each input pair
    for pair_idx, pair in enumerate(input_pairs):
        json_file, excel_input = pair
        print(f"\n{'='*60}")
        print(f"Processing pair {pair_idx+1}: {json_file.name} and {excel_input.name}")
        print(f"{'='*60}\n")

        # Initialize document matcher
        matcher = DocumentMatcher(json_file=json_file, review_excel_file=excel_input)
        matcher.load_data()

        # Prepare feedback map
        feedback_map = {}
        sub_to_page = {}
        for row in matcher.df.iterrows():
            subsection = matcher.get_subsection(row[1])
            feedback = row[1]["指摘事項 (代替案)"]
            page_tuple = matcher.isconvertible(row[1]["ページ"])
            page = page_tuple[1] if page_tuple[0] else 0
            if subsection not in feedback_map:
                feedback_map[subsection] = []
                sub_to_page[subsection] = page
            if feedback and str(feedback).strip():
                feedback_map[subsection].append(feedback)

        # Initialize updated data with validation_workflow_ids
        updated_data = []
        for entry in matcher.extracted_data:
            entry_copy = entry.copy()
            entry_copy["human_feedback"] = ""
            entry_copy["validation_workflow_ids"] = []
            updated_data.append(entry_copy)

        # Prepare list of (target_sub, feedback, page, entry_index)
        processing_tasks = []
        for target_sub, feedbacks in feedback_map.items():
            page = sub_to_page[target_sub]

            # Try exact match first
            exact_entry = None
            entry_index = -1
            for idx, entry in enumerate(matcher.extracted_data):
                entry_sub_name = entry.get("subsection_name")
                entry_sub = (
                    entry_sub_name.split("）")[1]
                    if entry_sub_name and "）" in entry_sub_name
                    else entry_sub_name
                )
                if entry_sub == target_sub:
                    exact_entry = entry
                    entry_index = idx
                    break

            for feedback in feedbacks:
                if not feedback or not str(feedback).strip():
                    continue

                processing_tasks.append(
                    {
                        "target_sub": target_sub,
                        "feedback": feedback,
                        "page": page,
                        "exact_entry": exact_entry,
                        "entry_index": entry_index,
                    }
                )

        print(f"Found {len(processing_tasks)} feedback items to process\n")

        # Process each feedback
        for task_idx, task in enumerate(processing_tasks, 1):
            print(f"Processing feedback {task_idx}/{len(processing_tasks)}")
            print(f"  Subsection: {task['target_sub']}")
            print(f"  Feedback: {task['feedback'][:100]}...")

            # Get content
            if (
                task["exact_entry"]
                and task["exact_entry"].get("content")
                and task["exact_entry"]["content"].strip()
                not in ["The page number is not defined", "No matching content found"]
            ):
                content = task["exact_entry"]["content"]
                entry_index = task["entry_index"]
            else:
                content, _, entry_index = await matcher.find_relevant_content(
                    task["feedback"],
                    task["target_sub"],
                    task["page"],
                    matcher.extracted_data,
                )

            if not content or content.strip() in [
                "The page number is not defined",
                "No matching content found",
                "",
            ]:
                print("  ⚠ No valid content found, skipping")
                continue

            # Step 1: Deep Research
            print("  → Deep Research...")
            research_report = await researcher.analyze_feedback(
                content, task["feedback"]
            )

            # Step 2: Decompose into error patterns
            print("  → Decomposing error patterns...")
            error_patterns = await analyzer.decompose_into_error_patterns(
                research_report, task["feedback"]
            )
            print(f"  → Found {len(error_patterns)} error pattern(s)")

            # Step 3: Generate validation workflow for each pattern
            for pattern_idx, pattern in enumerate(error_patterns, 1):
                print(
                    f"    → Generating workflow for pattern {pattern_idx}/{len(error_patterns)}..."
                )

                workflow = await analyzer.generate_validation_workflow(
                    error_pattern=pattern,
                    content=content,
                    research_report=research_report,
                )

                # Assign rule_id
                rule_id = f"RULE_{rule_counter:03d}"
                rule_counter += 1

                workflow["rule_id"] = rule_id
                rules_library.append(workflow)

                # Link to chunk
                if entry_index >= 0:
                    updated_data[entry_index]["validation_workflow_ids"].append(rule_id)
                    if not updated_data[entry_index]["human_feedback"]:
                        updated_data[entry_index]["human_feedback"] = task["feedback"]

                print(
                    f"      ✓ Generated {rule_id} with {len(workflow['validation_steps'])} steps"
                )

            print()

        # Save the updated JSON for this pair (chunks with workflow IDs)
        output_json_path = (
            output_folder / f"chunks_with_workflow_ids_{json_file.stem}.json"
        )
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved chunks with workflow IDs to: {output_json_path}\n")

    # Save the global rules library (standalone validation workflows)
    rules_library_path = output_folder / "validation_workflows_library.json"
    with open(rules_library_path, "w", encoding="utf-8") as f:
        json.dump(rules_library, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*60}")
    print(f"✓ Saved validation workflows library to: {rules_library_path}")
    print(f"  Total workflows generated: {len(rules_library)}")
    print(f"{'='*60}\n")

    print("✓ Pipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
