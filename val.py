import asyncio
import json
import os
import random
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

# Configuration
USE_VLLM = False  # Set to False to use OpenAI (ChatGPT) client for rule generation
USE_VLLM_VALIDATION = False  # Set to False to use OpenAI for validation
MAX_CONCURRENCY = 20  # Adjust as needed; if 1, runs sequentially
MAX_RULE_SELECTIONS = (
    5  # Maximum number of rule categories the validation model can select
)
VALIDATION_SPLIT = 0.1  # 10% for validation
MODE = "all"  # Options: "generate", "validate", "all"

# Model configurations
RULE_GENERATION_MODEL = None  # Will use default based on USE_VLLM
VALIDATION_MODEL = None  # Will use default based on USE_VLLM_VALIDATION

semaphore = Semaphore(MAX_CONCURRENCY)


class LLMClient:
    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = 1000,
        endpoint_env: str = None,
        use_vllm: bool = None,  # New parameter
    ):
        load_dotenv()

        # Determine which backend to use
        if use_vllm is None:
            use_vllm = USE_VLLM  # Default to global setting

        self.is_vllm = use_vllm

        if model is None:
            model = (
                "kaitchup/Phi-4-AutoRound-GPTQ-4bit" if self.is_vllm else "gpt-5-nano"
            )
        if temperature is None:
            temperature = 0.1 if self.is_vllm else 0.0

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.is_vllm:
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


class SelfContainedFilter:
    def __init__(self):
        self.llmObj = LLMClient(model=RULE_GENERATION_MODEL, use_vllm=USE_VLLM)

    async def is_self_contained(self, content: str, feedback: str) -> bool:
        """
        Determine if the error/issue in the feedback can be detected from the content text alone.
        Returns True if self-contained, False if requires external context.
        """
        filter_prompt = """
あなたは文書品質分析の専門家です。以下のテキストチャンクと人間のフィードバックを分析してください。

### タスク
人間のフィードバックで指摘された問題が、提供されたテキストチャンク**だけ**から検出可能かどうかを判断してください。

### 判断基準
**自己完結型（True）の条件:**
- 誤字、スペルミス、文法エラーがテキスト内で明確
- 数値の矛盾や不一致がテキスト内で確認可能
- フォーマットの問題がテキストから明らか
- 技術用語の誤用がテキスト内で判断可能

**外部コンテキスト必要（False）の条件:**
- 図表や画像の参照が必要
- 他のページやセクションとの照合が必要
- 文書全体の構造理解が必要
- このチャンクに含まれていない情報への参照が必要

### 入力
テキストチャンク: $content

人間のフィードバック: $feedback

### 出力形式
JSONのみを出力してください:
{
  "self_contained": true または false,
  "reasoning": "判断理由を簡潔に"
}
        """

        template = Template(filter_prompt)
        prompt = template.safe_substitute(content=content, feedback=feedback)

        try:
            response = await self.llmObj.make_generic_request_with_prompt(
                system_content="You are a document analysis expert. Output only valid JSON.",
                user_content=prompt,
            )
            repaired = repair_json(response)
            parsed = json.loads(repaired)
            return parsed.get("self_contained", False)
        except Exception as e:
            print(f"Error in self-contained check: {e}")
            return False


class RuleGenerator:
    def __init__(
        self,
        prompt_file: str = "prompt/rule_refinement_prompt.txt",
        meta_prompt_file: str = "prompt/meta_prompt_for_processing.txt",
    ):
        self.PROMPT_FILE = prompt_file
        self.META_PROMPT_FILE = meta_prompt_file
        self.llmObj = LLMClient(model=RULE_GENERATION_MODEL, use_vllm=USE_VLLM)
        # Start empty; categories will be generated later
        self.categories_json = []

        if not Path(self.PROMPT_FILE).exists():
            print("Warning: Prompt file rule_refinement_prompt.txt doesn't exist.")
        elif not Path(self.META_PROMPT_FILE).exists():
            print(
                "Warning: The prompt file for generating the combined prompt file doesn't exist."
            )

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
        """First pass: Infer what type of technical content this is with rich context."""
        content_inference_prompt = """
あなたは電力システム文書の技術的内容分析専門家です。
以下の情報をもとに、この文書セクションがどのような技術的カテゴリに属するかを分析してください。

### 指示
    - 日本語のみで回答してください。
    - セクション名、サブセクション名、および実際のテキスト内容を分析し、高度に具体的な技術的サブカテゴリを特定してください。
    - **重要**: カテゴリ名には技術的コンテキストを含めてください。単なるタスク名（「表問題」「スペルチェック」）ではなく、文書のどの技術領域・部分に関するものかを明示してください。
    - 出力は2-5語の簡潔だが情報豊富な技術カテゴリ名を返してください。
    - 例:
    - 良い例: 「変圧器地絡保護設定値および動作条件」（技術的コンテキスト豊富）
    - 悪い例: 「表の誤り」（タスクのみ、技術コンテキストなし）
    - 良い例: 「周波数変動監視の閾値仕様」
    - 悪い例: 「数値チェック」
    - 良い例: 「保護リレー動作値表の整合性」
    - 悪い例: 「フォーマット問題」

- セクション名: $section
- サブセクション名: $subsection
- 実際のテキスト内容: $content

### 出力形式
2-5語の具体的で技術コンテキストを含むカテゴリ名のみを出力してください。
例: 変圧器地絡保護設定値および動作条件
        """

        template = Template(content_inference_prompt)
        prompt = template.safe_substitute(
            section=section or "N/A",
            subsection=subsection or "N/A",
            content=content or "N/A",
        )

        try:
            category = await self.llmObj.make_generic_request_with_prompt(
                system_content="You are a technical content analysis expert. Output only the category name in Japanese. 2-5 words maximum. Must include technical context, not just task type.",
                user_content=prompt,
            )
            # Clean the output
            category = category.strip()
            # Limit to first 5 words if too long
            words = category.split()
            if len(words) > 5:
                category = " ".join(words[:5])
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
        """LLM-based many-to-one clustering of content types into broader technical categories."""
        if not all_proposals_with_metadata:
            return []

        # Extract all unique content types
        content_types = list(
            set(item["content_type"] for item in all_proposals_with_metadata)
        )

        print(
            f"Starting LLM-based clustering of {len(content_types)} unique content types..."
        )

        # Prepare content types with sample rules for context
        content_type_details = []
        for ct in content_types:
            # Get sample rules for this content type
            sample_rules = [
                item["proposal"]
                for item in all_proposals_with_metadata
                if item["content_type"] == ct
            ][
                :3
            ]  # Take up to 3 sample rules

            content_type_details.append(
                {"content_type": ct, "sample_rules": sample_rules}
            )

        # LLM clustering prompt
        clustering_prompt = """
あなたは電力システム文書の技術領域分類の専門家です。

### タスク
以下のコンテンツタイプのリストを分析し、技術的に関連するものを少数の広範なカテゴリにグループ化してください。

### 重要な指示
1. **技術的概念に基づく分類**: 文書の技術的部分・システム・コンポーネントに基づいてグループ化してください
2. **タスクではなく技術領域**: 「表の問題」「スペルミス」ではなく、「変圧器保護システム」「周波数制御機構」のような技術領域でグループ化
3. **多対一マッピング**: 複数の詳細なコンテンツタイプを1つの広範なカテゴリにマッピング
4. **カテゴリ数削減**: 元のコンテンツタイプ数よりも大幅に少ない数の新カテゴリを作成（目標：元の30-50%）
5. **技術的意味の保持**: グループ化しても技術的意味が失われないように

### 入力コンテンツタイプ（サンプルルール付き）
$content_types_with_samples

### 出力形式
JSONのみを出力してください：
{
"mapping": {
    "元のコンテンツタイプ1": "新しい広範なカテゴリA",
    "元のコンテンツタイプ2": "新しい広範なカテゴリA",
    "元のコンテンツタイプ3": "新しい広範なカテゴリB",
    ...
},
"category_descriptions": {
    "新しい広範なカテゴリA": "このカテゴリの技術的説明",
    "新しい広範なカテゴリB": "このカテゴリの技術的説明",
    ...
},
"reasoning": "クラスタリングの判断理由"
}

### 良いクラスタリングの例
入力:
- "変圧器地絡保護設定値"
- "変圧器過電流保護条件"
- "変圧器温度監視閾値"
- "周波数変動監視条件"
- "周波数制御ゲイン設定"

出力マッピング:
- "変圧器地絡保護設定値" → "変圧器保護システム全般"
- "変圧器過電流保護条件" → "変圧器保護システム全般"
- "変圧器温度監視閾値" → "変圧器保護システム全般"
- "周波数変動監視条件" → "周波数制御システム"
- "周波数制御ゲイン設定" → "周波数制御システム"

5つのタイプ → 2つの広範なカテゴリ
        """

        # Format content types with samples
        formatted_types = []
        for detail in content_type_details:
            rules_str = "\n    ".join([f"- {rule}" for rule in detail["sample_rules"]])
            formatted_types.append(
                f"- コンテンツタイプ: {detail['content_type']}\n  サンプルルール:\n    {rules_str}"
            )

        content_types_str = "\n\n".join(formatted_types)

        template = Template(clustering_prompt)
        prompt = template.safe_substitute(content_types_with_samples=content_types_str)

        try:
            response = await self.llmObj.make_generic_request_with_prompt(
                system_content="You are an expert in technical domain clustering for power system documents. Output only valid JSON.",
                user_content=prompt,
            )

            repaired = repair_json(response)
            parsed = json.loads(repaired)

            mapping = parsed.get("mapping", {})
            category_descriptions = parsed.get("category_descriptions", {})

            print(
                f"LLM clustered {len(content_types)} types into {len(set(mapping.values()))} broader categories"
            )
            print(f"Category descriptions: {category_descriptions}")

            # Now group rules by the NEW broader categories
            categories_dict = {}
            for item in all_proposals_with_metadata:
                original_type = item["content_type"]
                new_category = mapping.get(
                    original_type, original_type
                )  # Fallback to original if not mapped

                if new_category not in categories_dict:
                    categories_dict[new_category] = {
                        "rules": [],
                        "description": category_descriptions.get(new_category, ""),
                    }

                categories_dict[new_category]["rules"].append(item["proposal"])

            # Convert to expected format
            categories = []
            for cat_name, cat_data in categories_dict.items():
                categories.append(
                    {
                        "type": cat_name,
                        "rules": cat_data["rules"],
                        "description": cat_data["description"],
                    }
                )

            self.categories_json = categories
            print(f"Final clustered categories: {len(categories)}")
            for cat in categories:
                print(f"  - {cat['type']}: {len(cat['rules'])} rules")

            return categories

        except Exception as e:
            print(f"Error in LLM-based clustering: {e}")
            print("Falling back to original grouping...")
            # Fallback: group by original content_type
            categories_dict = {}
            for item in all_proposals_with_metadata:
                content_type = item.get("content_type", "Unknown Category")
                if content_type not in categories_dict:
                    categories_dict[content_type] = []
                categories_dict[content_type].append(item["proposal"])

            categories = []
            for content_type, rules in categories_dict.items():
                categories.append(
                    {"type": content_type, "rules": rules, "description": ""}
                )

            self.categories_json = categories
            return categories

    async def refine_rules_per_category(
        self, llm_obj: LLMClient, categorized_rules: List[Dict]
    ) -> List[Dict]:
        """Refine rules category by category: deduplicate, refine, and generalize."""
        refined_categories = []
        refinement_prompt = """
意味的に重複または非常に類似したルールを1つに統合。
重複ルール (意味的に同じまたは非常に類似) を1つに統合。
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


class ValidationCategoryAssigner:
    def __init__(self, categorized_rules: List[Dict]):
        self.categorized_rules = categorized_rules
        self.llmObj = LLMClient(model=RULE_GENERATION_MODEL, use_vllm=USE_VLLM)

    async def assign_category(self, content: str, feedback: str) -> str:
        """
        Assign a validation chunk to one of the generated rule categories.
        Returns the category name or "NO_MATCH" if it doesn't fit any category.
        """
        categories_list = "\n".join(
            [f"- {cat['type']}" for cat in self.categorized_rules]
        )

        assignment_prompt = """
あなたは文書品質ルール分類の専門家です。

### タスク
以下のテキストチャンクと人間のフィードバックを分析し、生成されたルールカテゴリのうち、最も適合するものを1つ選択してください。

### 利用可能なカテゴリ
$categories

### 入力
テキストチャンク: $content

人間のフィードバック: $feedback

### 指示
- フィードバックで指摘された問題の種類を分析
- 最も適合するカテゴリを1つ選択
- どのカテゴリにも適合しない場合は "NO_MATCH" を返す

### 出力形式
JSONのみを出力:
{
  "category": "選択されたカテゴリ名" または "NO_MATCH",
  "reasoning": "選択理由を簡潔に"
}
        """

        template = Template(assignment_prompt)
        prompt = template.safe_substitute(
            categories=categories_list, content=content, feedback=feedback
        )

        try:
            response = await self.llmObj.make_generic_request_with_prompt(
                system_content="You are a rule category assignment expert. Output only valid JSON.",
                user_content=prompt,
            )
            repaired = repair_json(response)
            parsed = json.loads(repaired)
            return parsed.get("category", "NO_MATCH")
        except Exception as e:
            print(f"Error in category assignment: {e}")
            return "NO_MATCH"


class ValidationEvaluator:
    def __init__(self, categorized_rules: List[Dict], use_vllm_validation: bool = None):
        self.categorized_rules = categorized_rules
        # Use separate validation LLM client
        if use_vllm_validation is None:
            use_vllm_validation = USE_VLLM_VALIDATION

        validation_model = VALIDATION_MODEL
        self.llmObj = LLMClient(model=validation_model, use_vllm=use_vllm_validation)

    async def evaluate_single_category(
        self, content: str, category_name: str, category_rules: List[str]
    ) -> Dict:
        """
        Evaluate content against a SINGLE category's rules.
        Returns detection result and confidence.
        """
        rules_text = "\n".join([f"  - {rule}" for rule in category_rules])

        eval_prompt = """
あなたは文書品質検証の専門家です。

### タスク
以下のテキストチャンクを分析し、提供されたルールカテゴリに違反があるかどうかを判定してください。

### ルールカテゴリ
カテゴリ名: $category_name

ルール:
$rules_text

### テキストチャンク
$content

### 指示
- テキストを注意深く分析
- このカテゴリのルールに違反があれば detected: true
- 違反がなければ detected: false
- 判定の確信度を0.0-1.0で評価

### 出力形式
JSONのみを出力:
{
  "detected": true または false,
  "confidence": 0.0から1.0の数値,
  "reasoning": "判定理由を簡潔に",
  "violated_rules": ["違反したルール番号または説明"] (detected=trueの場合のみ)
}
        """

        template = Template(eval_prompt)
        prompt = template.safe_substitute(
            category_name=category_name, rules_text=rules_text, content=content
        )

        try:
            response = await self.llmObj.make_generic_request_with_prompt(
                system_content="You are a document quality validation expert. Output only valid JSON.",
                user_content=prompt,
            )
            repaired = repair_json(response)
            parsed = json.loads(repaired)

            return {
                "category": category_name,
                "detected": parsed.get("detected", False),
                "confidence": parsed.get("confidence", 0.0),
                "reasoning": parsed.get("reasoning", ""),
                "violated_rules": parsed.get("violated_rules", []),
            }
        except Exception as e:
            print(f"Error evaluating category {category_name}: {e}")
            return {
                "category": category_name,
                "detected": False,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "violated_rules": [],
            }

    async def evaluate_chunk(
        self, content: str, has_error: bool, true_category: str = None
    ) -> Dict:
        """
        Evaluate a validation chunk by running it against ALL categories separately,
        then ranking and selecting top-K.
        """
        # Run evaluation against each category separately
        eval_tasks = []
        for category in self.categorized_rules:
            task = self.evaluate_single_category(
                content=content,
                category_name=category["type"],
                category_rules=category["rules"],
            )
            eval_tasks.append(task)

        # Execute all category evaluations concurrently
        category_results = await asyncio.gather(*eval_tasks, return_exceptions=True)

        # Filter and process results
        valid_results = []
        for result in category_results:
            if isinstance(result, Exception):
                print(f"Error in category evaluation: {result}")
                continue
            valid_results.append(result)

        # Sort by confidence (descending) and filter detected=True
        detected_results = [r for r in valid_results if r["detected"]]
        detected_results.sort(key=lambda x: x["confidence"], reverse=True)

        # Select top-K categories
        selected_categories = [
            r["category"] for r in detected_results[:MAX_RULE_SELECTIONS]
        ]

        # Evaluate correctness
        is_correct = False
        if has_error:
            # Chunk has error - model should select the correct category
            is_correct = (
                true_category in selected_categories
                if true_category != "NO_MATCH"
                else False
            )
        else:
            # Chunk has no error - model should select nothing
            is_correct = len(selected_categories) == 0

        return {
            "selected_categories": selected_categories,
            "all_category_results": valid_results,  # Full details for analysis
            "top_detections": detected_results[
                :MAX_RULE_SELECTIONS
            ],  # Top detections with confidence
            "is_correct": is_correct,
            "true_category": true_category,
            "has_error": has_error,
        }


def load_existing_rules():
    current_rules_path = Path("rules/current_rules.txt")
    if os.path.exists(current_rules_path):
        with open(current_rules_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def get_timestamp():
    """Get current timestamp for unique filenames."""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json_with_timestamp(data: Any, base_path: Path, base_filename: str):
    """Save JSON with timestamp to avoid overwriting."""
    timestamp = get_timestamp()
    name_parts = base_filename.rsplit(".", 1)
    if len(name_parts) == 2:
        filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
    else:
        filename = f"{base_filename}_{timestamp}.json"

    filepath = base_path / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {filepath}")
    return filepath


def load_latest_json(base_path: Path, pattern: str) -> tuple:
    """Load the most recent JSON file matching pattern. Returns (data, filepath)."""
    import glob

    matching_files = glob.glob(str(base_path / pattern))
    if not matching_files:
        raise FileNotFoundError(f"No files matching pattern: {pattern}")

    latest_file = max(matching_files, key=os.path.getctime)

    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded: {latest_file}")
    return data, Path(latest_file)


async def run_validation_only(output_folder: Path):
    """
    Run validation on pre-generated rules and validation set.
    This function can be called independently after rules are generated.
    """
    print("\n" + "=" * 60)
    print("RUNNING VALIDATION-ONLY MODE")
    print("=" * 60)

    # Load the latest refined categorized rules
    try:
        refined_categorized_rules, rules_file = load_latest_json(
            output_folder, "refined_categorized_rules_*.json"
        )
        print(f"Loaded rules from: {rules_file}")
    except FileNotFoundError:
        print("ERROR: No refined_categorized_rules file found!")
        print("Please run with --mode generate first to create rules.")
        return

    # Load the latest validation set
    try:
        validation_set, val_file = load_latest_json(
            output_folder, "validation_set_with_categories_*.json"
        )
        print(f"Loaded validation set from: {val_file}")
    except FileNotFoundError:
        print("ERROR: No validation_set_with_categories file found!")
        print("Please run with --mode generate first to create validation set.")
        return

    print(f"Using validation model: {VALIDATION_MODEL or 'default'}")
    print(f"Validation backend: {'VLLM' if USE_VLLM_VALIDATION else 'OpenAI'}")

    # Run evaluation
    print("\n" + "=" * 60)
    print("Evaluating validation set with model")
    print("=" * 60)

    evaluator = ValidationEvaluator(refined_categorized_rules, USE_VLLM_VALIDATION)

    eval_tasks = []
    for entry in validation_set:
        content = entry.get("content", "")
        has_error = entry.get("has_error", False)
        true_category = entry.get("assigned_category", None)

        eval_tasks.append(
            (entry, evaluator.evaluate_chunk(content, has_error, true_category))
        )

    # Execute evaluation tasks
    eval_results = await asyncio.gather(
        *[t[1] for t in eval_tasks], return_exceptions=True
    )

    # Process evaluation results
    evaluation_results = []
    correct_count = 0
    no_match_count = sum(
        1 for e in validation_set if e.get("assigned_category") == "NO_MATCH"
    )

    for i, (entry, _) in enumerate(tqdm(eval_tasks, desc="Evaluating chunks")):
        try:
            result = eval_results[i]
            if isinstance(result, Exception):
                print(f"Error evaluating entry: {result}")
                continue

            entry["evaluation"] = result
            evaluation_results.append(
                {
                    "content": entry.get("content", "")[:200] + "...",
                    "has_error": result["has_error"],
                    "true_category": result["true_category"],
                    "selected_categories": result["selected_categories"],
                    "top_detections": result.get("top_detections", []),
                    "is_correct": result["is_correct"],
                }
            )

            if result["is_correct"]:
                correct_count += 1
        except Exception as e:
            print(f"Error processing evaluation result: {e}")

    # Calculate metrics
    total = len(validation_set)
    accuracy = correct_count / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total validation chunks: {total}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"NO_MATCH chunks (automatic misses): {no_match_count}")

    # Save evaluation results with timestamp
    save_json_with_timestamp(
        validation_set, output_folder, "validation_set_evaluated.json"
    )

    save_json_with_timestamp(
        {
            "validation_model": VALIDATION_MODEL or "default",
            "use_vllm_validation": USE_VLLM_VALIDATION,
            "total_chunks": total,
            "correct_predictions": correct_count,
            "accuracy": accuracy,
            "no_match_count": no_match_count,
            "max_rule_selections": MAX_RULE_SELECTIONS,
            "results": evaluation_results,
        },
        output_folder,
        "evaluation_summary.json",
    )

    print(f"\nValidation results saved to: {output_folder}")


async def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Rule Generation and Validation Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "validate", "all"],
        default=MODE,
        help="Mode: 'generate' (rules only), 'validate' (validation only), 'all' (both)",
    )
    args = parser.parse_args()

    mode = args.mode
    print(f"\n{'='*60}")
    print(f"RUNNING IN MODE: {mode.upper()}")
    print(f"{'='*60}\n")

    # Define output folder
    if USE_VLLM:
        output_folder = Path("output_vllm")
    else:
        output_folder = Path("output")

    output_folder.mkdir(exist_ok=True)

    # If validate-only mode, skip to validation
    if mode == "validate":
        await run_validation_only(output_folder)
        return

    # Otherwise, run generation (for "generate" or "all" modes)
    input_folder = Path("input")
    input_folder.mkdir(exist_ok=True)

    input_pairs = [
        (input_folder / "test_doc_with_tables.json", input_folder / "human.xlsx"),
        (input_folder / "test_doc2_with_tables.json", input_folder / "human2.xlsx"),
    ]

    # ============= STEP 1: Filter self-contained chunks =============
    print("\n" + "=" * 60)
    print("STEP 1: Filtering self-contained chunks")
    print("=" * 60)

    all_chunks = []
    filter_obj = SelfContainedFilter()

    for json_file, excel_input in tqdm(
        input_pairs, desc="Loading and filtering chunks"
    ):
        matcher = DocumentMatcher(json_file=json_file, review_excel_file=excel_input)
        updated_json_data = matcher.update_json_with_human_feedback()

        filter_tasks = []
        for entry in updated_json_data:
            content = entry.get("content", "")
            feedback = entry.get("human_feedback", "")

            if feedback and feedback.strip():
                filter_tasks.append(
                    (entry, filter_obj.is_self_contained(content, feedback))
                )
            elif not feedback or not feedback.strip():
                entry["self_contained"] = True
                entry["has_error"] = False
                all_chunks.append(entry)

        results = await asyncio.gather(
            *[t[1] for t in filter_tasks], return_exceptions=True
        )

        for i, (entry, _) in enumerate(filter_tasks):
            try:
                is_self_contained = results[i]
                if isinstance(is_self_contained, Exception):
                    print(f"Error filtering entry: {is_self_contained}")
                    continue

                entry["self_contained"] = is_self_contained
                entry["has_error"] = True
                if is_self_contained:
                    all_chunks.append(entry)
            except Exception as e:
                print(f"Error processing filter result: {e}")

    print(f"Total self-contained chunks: {len(all_chunks)}")

    # ============= STEP 2: Split validation set =============
    print("\n" + "=" * 60)
    print("STEP 2: Splitting validation set")
    print("=" * 60)

    chunks_with_errors = [c for c in all_chunks if c.get("has_error", False)]
    chunks_without_errors = [c for c in all_chunks if not c.get("has_error", False)]

    print(f"Chunks with errors: {len(chunks_with_errors)}")
    print(f"Chunks without errors: {len(chunks_without_errors)}")

    random.seed(42)
    n_val_errors = max(1, int(len(chunks_with_errors) * VALIDATION_SPLIT))
    n_val_no_errors = max(1, int(len(chunks_without_errors) * VALIDATION_SPLIT))

    val_chunks_with_errors = random.sample(chunks_with_errors, n_val_errors)
    val_chunks_without_errors = random.sample(chunks_without_errors, n_val_no_errors)

    validation_set = val_chunks_with_errors + val_chunks_without_errors
    train_chunks_with_errors = [
        c for c in chunks_with_errors if c not in val_chunks_with_errors
    ]
    train_chunks_without_errors = [
        c for c in chunks_without_errors if c not in val_chunks_without_errors
    ]
    training_set = train_chunks_with_errors + train_chunks_without_errors

    print(
        f"Validation set size: {len(validation_set)} ({len(val_chunks_with_errors)} with errors, {len(val_chunks_without_errors)} without)"
    )
    print(f"Training set size: {len(training_set)}")

    # Save with timestamps
    save_json_with_timestamp(validation_set, output_folder, "validation_set.json")
    save_json_with_timestamp(training_set, output_folder, "training_set.json")

    # ============= STEP 3: Generate rules on training set =============
    print("\n" + "=" * 60)
    print("STEP 3: Running rule generation pipeline on training set")
    print("=" * 60)

    existing_rules = load_existing_rules()
    cumulative_context = existing_rules.strip() + "\n\n" if existing_rules else ""
    gen_rules = RuleGenerator()

    all_proposals_with_metadata = []

    print(f"Generating rules from {len(training_set)} training chunks...")

    tasks = []
    for entry in training_set:
        feedback = entry.get("human_feedback", "")
        content = entry.get("content", "")
        section = entry.get("subsection_name", "")

        if feedback and feedback.strip():
            content_type = await gen_rules.infer_content_type(section, section, content)
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

    results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)

    for j in tqdm(range(len(tasks)), desc="Processing training results"):
        task, entry, content, section, content_type = tasks[j]
        try:
            result = results[j]
            if isinstance(result, Exception):
                print(f"Error in task {j}: {result}")
                continue

            proposals = result.get("proposals", [])

            valid_proposals = [str(p) for p in proposals if p and str(p).strip()]
            if valid_proposals:
                entry["rules"] = valid_proposals
                entry["rule"] = ";".join(valid_proposals)

                for p in valid_proposals:
                    all_proposals_with_metadata.append(
                        {
                            "proposal": p,
                            "content_type": content_type,
                            "section": section,
                            "content": content,
                        }
                    )

                cumulative_context += f"Previous Rule: {p}\n\n"
            else:
                entry["rules"] = []
                entry["rule"] = ""
        except Exception as e:
            print(f"Error processing entry {j}: {e}")
            entry["rules"] = []
            entry["rule"] = ""

    # LLM-based clustering
    print("\nClustering rules into categories using LLM...")
    categorized_rules = await gen_rules.cluster_rules(all_proposals_with_metadata)

    print("\nRefining rules category by category...")
    llm_obj = LLMClient(use_vllm=USE_VLLM)
    refined_categorized_rules = await gen_rules.refine_rules_per_category(
        llm_obj, categorized_rules
    )

    # Save categorized rules with timestamps
    save_json_with_timestamp(categorized_rules, output_folder, "categorized_rules.json")
    save_json_with_timestamp(
        refined_categorized_rules, output_folder, "refined_categorized_rules.json"
    )

    print(f"Generated {len(refined_categorized_rules)} rule categories")

    # ============= STEP 3b: Assign categories to validation set =============
    print("\n" + "=" * 60)
    print("STEP 3b: Assigning categories to validation chunks")
    print("=" * 60)

    category_assigner = ValidationCategoryAssigner(refined_categorized_rules)

    assignment_tasks = []
    for entry in validation_set:
        if entry.get("has_error", False):
            content = entry.get("content", "")
            feedback = entry.get("human_feedback", "")
            assignment_tasks.append(
                (entry, category_assigner.assign_category(content, feedback))
            )
        else:
            entry["assigned_category"] = "NO_ERROR"

    assignment_results = await asyncio.gather(
        *[t[1] for t in assignment_tasks], return_exceptions=True
    )

    for i, (entry, _) in enumerate(tqdm(assignment_tasks, desc="Assigning categories")):
        try:
            category = assignment_results[i]
            if isinstance(category, Exception):
                print(f"Error assigning category: {category}")
                entry["assigned_category"] = "ERROR"
                continue

            entry["assigned_category"] = category
        except Exception as e:
            print(f"Error processing assignment: {e}")
            entry["assigned_category"] = "ERROR"

    no_match_count = sum(
        1 for e in validation_set if e.get("assigned_category") == "NO_MATCH"
    )
    print(
        f"Validation chunks with NO_MATCH category: {no_match_count} (automatic misses)"
    )

    # Save validation set with categories (with timestamp)
    save_json_with_timestamp(
        validation_set, output_folder, "validation_set_with_categories.json"
    )

    print("\n" + "=" * 60)
    print("RULE GENERATION COMPLETED")
    print("=" * 60)
    print(f"All outputs saved to: {output_folder}")

    # If mode is "all", continue to validation
    if mode == "all":
        print("\n" + "=" * 60)
        print("Continuing to validation phase...")
        print("=" * 60)

        # ============= STEP 4: Evaluate validation set =============
        print("\n" + "=" * 60)
        print("STEP 4: Evaluating validation set with model")
        print("=" * 60)

        print(f"Using validation model: {VALIDATION_MODEL or 'default'}")
        print(f"Validation backend: {'VLLM' if USE_VLLM_VALIDATION else 'OpenAI'}")

        evaluator = ValidationEvaluator(refined_categorized_rules, USE_VLLM_VALIDATION)

        eval_tasks = []
        for entry in validation_set:
            content = entry.get("content", "")
            has_error = entry.get("has_error", False)
            true_category = entry.get("assigned_category", None)

            eval_tasks.append(
                (entry, evaluator.evaluate_chunk(content, has_error, true_category))
            )

        eval_results = await asyncio.gather(
            *[t[1] for t in eval_tasks], return_exceptions=True
        )

        evaluation_results = []
        correct_count = 0

        for i, (entry, _) in enumerate(tqdm(eval_tasks, desc="Evaluating chunks")):
            try:
                result = eval_results[i]
                if isinstance(result, Exception):
                    print(f"Error evaluating entry: {result}")
                    continue

                entry["evaluation"] = result
                evaluation_results.append(
                    {
                        "content": entry.get("content", "")[:200] + "...",
                        "has_error": result["has_error"],
                        "true_category": result["true_category"],
                        "selected_categories": result["selected_categories"],
                        "top_detections": result.get("top_detections", []),
                        "is_correct": result["is_correct"],
                    }
                )

                if result["is_correct"]:
                    correct_count += 1
            except Exception as e:
                print(f"Error processing evaluation result: {e}")

        total = len(validation_set)
        accuracy = correct_count / total if total > 0 else 0

        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        print(f"Total validation chunks: {total}")
        print(f"Correct predictions: {correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"NO_MATCH chunks (automatic misses): {no_match_count}")

        # Save evaluation results with timestamps
        save_json_with_timestamp(
            validation_set, output_folder, "validation_set_evaluated.json"
        )

        save_json_with_timestamp(
            {
                "validation_model": VALIDATION_MODEL or "default",
                "use_vllm_validation": USE_VLLM_VALIDATION,
                "total_chunks": total,
                "correct_predictions": correct_count,
                "accuracy": accuracy,
                "no_match_count": no_match_count,
                "max_rule_selections": MAX_RULE_SELECTIONS,
                "results": evaluation_results,
            },
            output_folder,
            "evaluation_summary.json",
        )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)
    print(f"All outputs saved to: {output_folder}")


if __name__ == "__main__":
    asyncio.run(main())
