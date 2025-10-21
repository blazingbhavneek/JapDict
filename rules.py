import asyncio
import json
import os
from asyncio import Semaphore
from pathlib import Path
from string import Template
from typing import Any, Dict, List

import httpx
import openai
import pandas as pd
from dotenv import load_dotenv
from json_repair import repair_json
from tqdm import tqdm

# Switch between clients by changing this variable
USE_VLLM = False  # Set to False to use OpenAI (ChatGPT) client
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
            endpoint_env = endpoint_env or "VLLM_HOST"
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
            else:

                def sync_openai_call():
                    response = openai.ChatCompletion.create(
                        engine=self.model,
                        messages=messages,
                        # temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        **filtered_kwargs,
                    )
                    return response.choices[0].message.content

                loop = asyncio.get_event_loop()
                try:
                    return await asyncio.wait_for(
                        loop.run_in_executor(None, sync_openai_call), timeout=300.0
                    )
                except asyncio.TimeoutError:
                    raise Exception("OpenAI API call timed out after 300 seconds")


class DocumentMatcher:
    def __init__(self, json_file: str, review_excel_file: str):
        self.json_file = json_file
        self.review_excel_file = review_excel_file
        self.extracted_data = None
        self.df = None

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
                sheet_name="【様式】DRシート",
            )
        except Exception as e:
            print(f"Error in loading: {e}")
            raise

        print("COLUMNS OF THE HUMAN REVIEW FILE:\n" + str(self.df.columns.tolist()))

    def is_convertible(self, num):
        """Check if a number is convertible to int."""
        try:
            i = int(num)
        except Exception:
            return (False, -1)
        return (True, i)

    def get_subsection(self, row):
        """Convert Excel title to subsection number (fullwidth japanese)."""
        title = str(row["項目"])
        num_str = title.split(" ")[0] if " " in title else title
        parts = num_str.split(".")
        if len(parts) > 2:
            return parts[0] + "." + parts[1]
        return num_str

    def get_content(self, row):
        """Get matching content from JSON for a given row (first match)."""
        page = row["ページ"]
        if not self.is_convertible(page)[0]:
            return "The page number is not defined"
        page = self.is_convertible(page)[1]
        target_sub = self.get_subsection(row)
        print(
            "Finding matching content for the row with page number:", page, type(page)
        )

        for entry in self.extracted_data:
            if entry["subsection_name"] is None:
                continue
            entry_sub_num = (
                entry["subsection_name"].split(" ")[0]
                if " " in entry["subsection_name"]
                else entry["subsection_name"]
            )
            if entry_sub_num == target_sub:
                print(
                    "Found content with page number",
                    page,
                    "with subsection number(From row):",
                    entry_sub_num,
                    "Target Sub:",
                    target_sub,
                )
                return entry["content"]
        return "No matching content found"

    def update_json_with_human_feedback(self) -> List[Dict]:
        """Update JSON entries with human feedback from Excel."""
        if self.extracted_data is None or self.df is None:
            self.load_data()

        # Create a mapping from subsection to human feedback
        feedback_map = {}
        for _, row in self.df.iterrows():
            subsection = self.get_subsection(row)
            feedback = row["指摘事項(代替案)"]
            feedback_map[subsection] = feedback

        # Update the JSON data with human feedback and fix fields
        updated_data = []
        for entry in self.extracted_data:
            entry_copy = entry.copy()
            entry_sub_num = (
                (
                    entry["subsection_name"].split(" ")[0]
                    if entry["subsection_name"] and " " in entry["subsection_name"]
                    else entry["subsection_name"]
                )
                if entry["subsection_name"]
                else None
            )

            # Add human feedback and fix fields
            if entry_sub_num and entry_sub_num in feedback_map:
                entry_copy["human_feedback"] = feedback_map[entry_sub_num]
                entry_copy["fixed"] = feedback_map[
                    entry_sub_num
                ]  # Set fix to human feedback
            else:
                entry_copy["human_feedback"] = ""
                entry_copy["fixed"] = ""  # Empty if no human feedback

            # Add rule field (will be filled later)
            entry_copy["rule"] = ""

            updated_data.append(entry_copy)

        return updated_data


class RuleGenerator:
    def __init__(
        self,
        prompt_file: str = "prompt/rule_refinement_prompt.txt",
        meta_prompt_file: str = "prompt/meta_prompt_for_processing.txt",
    ):
        self.PROMPT_FILE = prompt_file
        self.META_PROMPT_FILE = meta_prompt_file
        self.llmObj = LLMClient()
        # Start empty; categories will be generated later
        self.categories_json = []

        if not Path(self.PROMPT_FILE).exists():
            print("Error: Prompt file rule_refinement_prompt.txt doesn't exist.")
            return
        elif not Path(self.META_PROMPT_FILE).exists():
            print(
                "Error: The prompt file for generating the combined prompt file doesn't exist."
            )
            return

    def load_prompt_template(self, template_file):
        with open(template_file, "r", encoding="utf-8") as f:
            return Template(f.read())

    def clean_llm_suggestion(self, raw: str) -> Dict[str, List[str]]:
        raw = raw.strip()
        if raw == "提案なし":
            return {"proposals": []}
        try:
            # Repair the JSON using json_repair
            repaired = repair_json(raw)
            parsed = json.loads(repaired)
            if "proposals" in parsed and isinstance(parsed["proposals"], list):
                # Filter out None/empty proposals
                valid_proposals = [
                    p for p in parsed["proposals"] if p and str(p).strip()
                ]
                return {"proposals": valid_proposals}
            else:
                return {"proposals": []}
        except Exception:
            # Fallback: treat as single proposal
            return {"proposals": [raw]}

    async def infer_content_type(
        self, section: str, subsection: str, content: str
    ) -> str:
        """First pass: Infer what type of technical content this is."""
        content_inference_prompt = """
あなたは電力システム文書の技術的内容分析専門家です。
以下の情報をもとに、この文書セクションがどのような技術的カテゴリに属するかを分析してください。

### 指示
- 日本語のみで回答してください。
- セクション名、サブセクション名、および実際のテキスト内容を分析し、高度に具体的な技術的サブカテゴリを特定してください。
- 出力は2-3語の簡潔な技術カテゴリ名を返してください。(例: 「変圧器地絡保護設定」、「周波数変動監視条件」、「保護リレー動作値表」、「系制装置信号伝送パターン」、「地絡検知閾値仕様」など)
- 例:
  - 「変圧器地絡保護設定」: 変圧器の地絡保護に関する具体的な設定値
  - 「周波数変動監視条件」: 周波数変動時の監視と制御条件
  - 「保護リレー動作値表」: 保護リレーの動作値が記載された詳細表
  - 「系制装置信号伝送パターン」: 系制装置の信号伝送パターンとタイミング
  - 「地絡検知閾値仕様」: 地絡検知の閾値と関連仕様
- 具体的で技術的なサブカテゴリ名を優先し、「表問題」や「スペル問題」のような一般的すぎるカテゴリ名ではなく、電力システムの継かな技術領域を反映したものを。
- セクション名: $section
- サブセクション名: $subsection
- 実際のテキスト内容: $content
- 出力形式: 2-3語の具体的な技術カテゴリ名のみ (例: 変圧器地絡保護設定)
        """

        template = Template(content_inference_prompt)
        prompt = template.safe_substitute(
            section=section or "N/A",
            subsection=subsection or "N/A",
            content=content or "N/A",
        )

        try:
            category = await self.llmObj.make_generic_request_with_prompt(
                system_content="You are a technical content analysis expert. Output only the category name in Japanese. 2-3 words maximum. Highly specific technical subcategory.",
                user_content=prompt,
            )
            # Clean the output to ensure it's concise
            category = category.strip()
            # Limit to first 2-3 words if it's too long
            words = category.split()
            if len(words) > 3:
                category = " ".join(words[:3])
            return category
        except Exception as e:
            return f"Error in content inference: {str(e)}"

    async def generate_rule_from_case(
        self,
        content: str,
        feedback: str,
        existing_rules: str = "",
        section: str = "",
        content_type: str = None,
        page=None,
    ) -> Dict[str, List[str]]:
        if (
            not content
            or content
            in ["The page number is not defined", "No matching content found"]
            or content.strip() == ""
        ):
            return {"proposals": []}

        # First Pass: Infer the technical content type
        if content_type is None:
            content_type = await self.infer_content_type(section, section, content)

        # Second Pass: Generate rules based on the inferred content type and human feedback
        rule_prompt_str = """
あなたは電気工学 (特に電力システム) の文書処理システムの専門家です。
以下の情報をもとに、既存の検証・抽出ルールに追加すべき新しいルールを提案してください。

### 指示
- 日本語のみで回答してください。
- 以下の技術カテゴリの内容に対するルールを生成してください: $content_type
- ヒューマンフィードバックに基づき、改善が必要な場合のみ、具体的で実行可能なルールを提案してください。
- 現在の文書内容とヒューマンフィードバックを比較・分析し、フィードバックが示す問題点を解決するルールを生成してください。
- 各ルールは自然言語で明確に記述し、LLMが理解・適用しやすいように、必要な条件、検出方法、修正案を含めてください。
- 提案するルールは一般化しつつ、技術カテゴリの特性を反映した具体的なパターンを記述してください

(例: 「変圧器地絡保護設定」の場合は、地絡保護の閾値、遅延時間、リセット条件などの項目に関するルール)
- ルール記述の要件:
  - 電力システム文書作成時の背景を簡潔に含め (例: 「変圧器の地絡保護設定が記述される場面では…」)
  - 検出・修正方法を詳細に記述。特定の表番号、ページ、または文書固有情報を参照せず、一般化せよ。
- 重要: ヒューマンフィードバックが「提案なし」や改善点がない場合は、「提案なし」とだけ出力してください。
- 出力形式: JSONオブジェクトのみを出力
  - ルール提案あり:
    json
    {
      "proposals": [
        "ルール案1",
        "ルール案2"
      ]
    }
  - 提案なしの場合:
    json
    {
      "proposals": []
    }
- 重要: 上記のJSON構造に正確に従ってください。独自の構造を作成しないでください。
- 現在の文書テキスト: $content
- ヒューマンレビューの指摘・修正提案: $feedback
- 現在のルール (ある場合): $existing_rules
- 推定された技術カテゴリ: $content_type
        """

        rule_template = Template(rule_prompt_str)
        prompt = rule_template.safe_substitute(
            content=content or "N/A",
            feedback=feedback or "No feedback provided.",
            existing_rules=existing_rules.strip()
            or "No existing rules you need to start generating from this one.",
            content_type=content_type,
        )

        try:
            suggestion = await self.llmObj.make_generic_request_with_prompt(
                system_content="You are a precise rule-generation assistant for electrical engineering document processing. Output only valid JSON as specified. Generate rules ONLY if human feedback indicates a problem that needs addressing.",
                user_content=prompt,
            )
            cleaned_suggestion = self.clean_llm_suggestion(suggestion)
            return cleaned_suggestion
        except Exception as e:
            return {"proposals": ["Error occurred for this entry.\n" + str(e)]}

    async def infer_all_technical_categories(
        self, all_proposals_with_metadata: List[Dict]
    ) -> List[str]:
        """First pass of clustering: Infer all technical categories we have rules for."""
        if not all_proposals_with_metadata:
            return []

        # Extract unique content types from the metadata
        content_types = set()
        for item in all_proposals_with_metadata:
            if "content_type" in item:
                content_types.add(item["content_type"])
        return list(content_types)

    async def cluster_rules(
        self, all_proposals_with_metadata: List[Dict]
    ) -> List[Dict]:
        """Two-pass clustering: First infer categories, then group rules by category."""
        if not all_proposals_with_metadata:
            return []

        # First pass: Infer all technical categories
        print("First pass: Inferring technical categories...")
        all_content_types = await self.infer_all_technical_categories(
            all_proposals_with_metadata
        )
        print(
            f"Identified {len(all_content_types)} technical categories: {all_content_types}"
        )

        # Second pass: Group rules by their content type
        print("Second pass: Grouping rules by technical categories...")
        categories_dict = {}
        for item in all_proposals_with_metadata:
            content_type = item.get("content_type", "Unknown Category")
            proposal = item["proposal"]

            if content_type not in categories_dict:
                categories_dict[content_type] = []
            categories_dict[content_type].append(proposal)

        # Convert to the expected format
        categories = []
        for content_type, rules in categories_dict.items():
            categories.append({"type": content_type, "rules": rules})

        self.categories_json = categories
        print(f"Clustered into {len(categories)} categories.")
        return categories

    async def refine_rules_per_category(
        self, llm_obj: LLMClient, categorized_rules: List[Dict]
    ) -> List[Dict]:
        """Refine rules category by category: deduplicate, refine, and generalize."""
        refined_categories = []
        refinement_prompt = """
意味的に重複または非常に類似したルールを1つに統合。
重複ルール (意味的に同じまたは非常に類似) を12億統合。
多ルールを洗練：文法修正、冗長除去、明確化、技術的詳細追加 (例：変電所関連の監視表ではA列のNWCを明記)
出力：JSON{"refined_rules": ["洗練されたルール1", "洗練されたルール2", ...]}のみ。
ルールが空の場合：{"refined_rules": []}。
カテゴリ: $category
元のルールリスト: $rules_list
        """

        template = Template(refinement_prompt)
        for category in tqdm(categorized_rules, desc="Refining categories"):
            if len(category["rules"]) == 0:
                refined_categories.append({"type": category["type"], "rules": []})
                continue

            # Ensure all rules are strings
            rules_list = "\n".join([f"- {str(rule)}" for rule in category["rules"]])
            prompt = template.safe_substitute(
                category=category["type"], rules_list=rules_list
            )

            try:
                refined_raw = await llm_obj.make_generic_request_with_prompt(
                    system_content="You are an expert rule refiner. Output only valid JSON.",
                    user_content=prompt,
                )
                repaired = repair_json(refined_raw)
                parsed = json.loads(repaired)
                refined_rules = parsed.get("refined_rules", [])
                refined_categories.append(
                    {"type": category["type"], "rules": refined_rules}
                )
                print(
                    f"Refined {len(refined_rules)} rules for category {category['type']}."
                )
            except Exception as e:
                print(f"Error refining category {category['type']}: {e}")
                refined_categories.append(category)  # Fallback to original
        return refined_categories


def load_existing_rules():
    current_rules_path = Path("rules/current_rules.txt")
    if os.path.exists(current_rules_path):
        with open(current_rules_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


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

    # Define multiple input pairs as list of tuples (json_file, excel_file)
    input_pairs = [
        (input_folder / "test_doc_with_tables.json", input_folder / "human.xlsx"),
        (input_folder / "test_doc2_with_tables.json", input_folder / "human2.xlsx"),
    ]

    # Load existing rules
    existing_rules = load_existing_rules()
    print("Loaded the existing rules if any.")

    # Generate rules for entries that have human feedback
    cumulative_context = existing_rules.strip() + "\n\n" if existing_rules else ""
    gen_rules = RuleGenerator()

    # Collect all proposals (rules without categories) with metadata
    all_proposals_with_metadata = []

    # Process each input pair
    for i, pair in enumerate(tqdm(input_pairs, desc="Processing input pairs")):
        json_file, excel_input = pair
        print(f"Processing pair {i+1}: {json_file} and {excel_input}")

        # Initialize document matcher and update JSON with human feedback
        matcher = DocumentMatcher(json_file=json_file, review_excel_file=excel_input)
        updated_json_data = matcher.update_json_with_human_feedback()
        print(f"Pair {i+1}: Completed mapping human review with JSON content")

        # Collect tasks for concurrent execution
        tasks = []
        for entry in updated_json_data:
            feedback = entry.get("human_feedback", "")
            content = entry.get("content", "")
            section = entry.get("subsection_name", "")

            if (
                feedback and feedback.strip()
            ):  # Only process entries that have human feedback
                content_type = await gen_rules.infer_content_type(
                    section, section, content
                )
                task = gen_rules.generate_rule_from_case(
                    content=content,
                    feedback=feedback,
                    existing_rules=cumulative_context,
                    section=section,
                    content_type=content_type,
                    page=entry.get("page_number", None),
                )
                tasks.append((task, entry, content, section, content_type))
            else:
                entry["rules"] = []
                entry["rule"] = ""
                entry["rule_categories"] = []

        # Execute tasks concurrently
        results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)

        # Process results with tqdm
        for j in tqdm(range(len(tasks)), desc=f"Processing results for pair {i+1}"):
            task, entry, content, section, content_type = tasks[j]
            try:
                result = results[j]
                if isinstance(result, Exception):
                    print(f"Error in task {j}: {result}")
                    continue

                proposals = result.get("proposals", [])

                # Ensure proposals are strings
                valid_proposals = [str(p) for p in proposals if p and str(p).strip()]
                if valid_proposals:
                    entry["rules"] = valid_proposals  # Store as list of strings
                    entry["rule"] = ";".join(
                        valid_proposals
                    )  # Backward compatible single string
                    entry["rule_categories"] = []  # No per-entry categories

                    for p in valid_proposals:
                        all_proposals_with_metadata.append(
                            {
                                "proposal": p,
                                "content_type": content_type,
                                "section": section,
                                "content": content,
                            }
                        )

                    # Only add string rules to cumulative context
                    cumulative_context += f"Previous Rule: {p}\n\n"
                else:
                    entry["rules"] = []
                    entry["rule"] = ""
                    entry["rule_categories"] = []
            except Exception as e:
                print(f"Error processing entry {j}: {e}")
                entry["rules"] = []
                entry["rule"] = ""
                entry["rule_categories"] = []

        # Save the updated JSON for this pair
        output_json_path = (
            output_folder / f"updated_doc_with_rules_{json_file.stem}.json"
        )
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(updated_json_data, f, ensure_ascii=False, indent=2)
        print(f"Pair {i+1}: Updated JSON with rules saved to: {output_json_path}")

    # Two-pass clustering: Cluster rules into categories
    print("Clustering rules into categories (Two-pass approach)...")
    categorized_rules = await gen_rules.cluster_rules(all_proposals_with_metadata)

    # Refine rules category by category
    llm_obj = LLMClient()
    refined_categorized_rules = await gen_rules.refine_rules_per_category(
        llm_obj, categorized_rules
    )

    # Save the refined categorized rules JSON (single JSON with categories and rules)
    rules_json_path = output_folder / "categorized_rules.json"
    with open(rules_json_path, "w", encoding="utf-8") as f:
        json.dump(categorized_rules, f, ensure_ascii=False, indent=2)

    rules_json_path = output_folder / "refined_categorized_rules.json"
    with open(rules_json_path, "w", encoding="utf-8") as f:
        json.dump(refined_categorized_rules, f, ensure_ascii=False, indent=2)

    print(f"Categorized rules JSON saved to: {rules_json_path}")
    print("Pipeline completed.")


if __name__ == "__main__":
    asyncio.run(main())
