"""
===========================================================================
文件角色：写作器 Agent —— 调用 LLM 生成训练计划文本和动作分析报告
===========================================================================
- 被谁调用：Orchestrator 在流水线的第 4 步调用 WriterAgent 的方法
- 调用谁：
    LLMProvider.chat_with_json_mode() → JSON 模式生成（同步版）
    LLMProvider.chat_stream()         → 流式生成（流式版）
    build_writer_messages()           → 构造训练计划生成的提示词（来自 prompts 模块）
- 核心职责：
    1. write_plan / write_plan_stream: 根据检索到的动作数据+用户画像，生成一周训练计划
    2. write_analysis / write_analysis_stream: 对比用户描述和标准规范，分析动作问题
    3. 为生成的计划附加 plan_id 和 user_id 用于追踪
- 输出格式：
    训练计划 JSON 结构含 days 列表，每天含 exercises 列表；
    动作分析 JSON 结构含 issues_found、severity、suggestions、confidence
===========================================================================
"""

import uuid
import json
from typing import Generator
from src.llm.provider import LLMProvider
from src.llm.prompts.writer import build_writer_messages


class WriterAgent:
    """写作器 Agent：在 Orchestrator 流水线的第 4 步被调用。
    职责：将检索到的数据和用户画像"翻译"为人类可读的训练计划或动作分析。
    提供同步版（write_plan, write_analysis）和流式版（write_plan_stream, write_analysis_stream）。
    流式版的核心差异：用 chat_stream 替代 chat_with_json_mode，手动解析 JSON 输出。"""

    def __init__(self):
        self.llm = LLMProvider()

    def write_plan(self, retrieved: dict, profile: dict, plan_config: dict) -> dict:
        """【同步版】根据检索结果和用户画像生成训练计划。

        输入：
            retrieved: dict — RetrieverAgent 返回的检索结果（含 exercises 列表）
            profile: dict — 用户画像字典
            plan_config: dict — Planner 匹配的技能模板配置
        输出：
            dict — 训练计划 JSON（含 "days"/"plan_id"/"user_id" 等字段）

        流程：
            1. 提取用户训练目标（兜底 "增肌"）
            2. 构建 LLM 提示词（含动作库数据 + 用户画像 + 生成指令）
            3. 调用 chat_with_json_mode 强制 LLM 以 JSON 格式输出
            4. 附加 plan_id（UUID 前 8 位）和 user_id 用于标识和追踪
        """
        goal = profile.get("goal", "增肌")
        messages = build_writer_messages(
            retrieved.get("exercises", []), profile, goal
        )
        # chat_with_json_mode：在 system prompt 中注入 JSON 格式约束，
        # 并在 API 调用时设置 response_format={"type": "json_object"}
        plan_json = self.llm.chat_with_json_mode(messages, model="reasoner")
        plan_json["plan_id"] = str(uuid.uuid4())[:8]
        plan_json["user_id"] = profile.get("id", 0)
        return plan_json

    def write_plan_stream(self, retrieved: dict, profile: dict, plan_config: dict,
                          plan_context: str = "", user_query: str = "") -> Generator:
        """【流式版】逐 token 产出训练计划。每次产出 (event_type, data) 元组。

        输入：同 write_plan
            plan_context: str — 上一轮计划摘要（多轮对话中用户要修改的计划）
            user_query: str — 用户当前的修改请求
        产出（Generator）：
            ("chunk", str) — LLM 生成的文本片段（逐 token）
            ("done", dict) — 解析完成后的训练计划 JSON

        流式版的 JSON 解析策略：
            由于 chat_stream 无法强制 JSON 格式（流式 API 通常不支持 json_object mode），
            LLM 可能用 markdown 代码块包裹 JSON。解析时先剥离 ``` 标记再 json.loads。
            如果解析失败（如 LLM 格式严重偏离），则返回 {"raw": full_text} 作为兜底。
        """
        goal = profile.get("goal", "增肌")
        messages = build_writer_messages(
            retrieved.get("exercises", []), profile, goal
        )
        # 多轮对话：注入已有计划上下文，让 LLM 基于原计划做修改而非从零生成
        if plan_context:
            context_hint = (
                f"\n\n【重要】用户之前已经有一个训练计划：\n{plan_context}\n"
                f"用户现在的修改请求是：{user_query}\n"
                f"请在此计划基础上进行修改，保留未涉及的部分，只调整用户要求改的部分。"
            )
            messages[-1]["content"] += context_hint
        full_text = ""
        for chunk in self.llm.chat_stream(messages, temperature=0.3, model="reasoner"):
            full_text += chunk
            yield ("chunk", chunk)
        # === 解析累积的流式输出文本 ===
        content = full_text
        # 剥离可能的 markdown 代码块包裹：```json ... ``` 或 ``` ... ```
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        try:
            result = json.loads(content.strip())
        except json.JSONDecodeError:
            # 兜底：LLM 输出格式不可解析时，保留原始文本供调试
            result = {"raw": full_text}
        result["plan_id"] = str(uuid.uuid4())[:8]
        result["user_id"] = profile.get("id", 0)
        yield ("done", result)

    def _parse_json_output(self, content: str) -> dict:
        """从 LLM 流式输出中解析 JSON，剥离 markdown 代码块。"""
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        try:
            result = json.loads(content.strip())
        except json.JSONDecodeError:
            result = {"raw": content}
        result["plan_id"] = str(uuid.uuid4())[:8]
        result["user_id"] = 0
        return result

    def rewrite_plan(self, original_plan: dict, issues: list[dict],
                     retrieved: dict, profile: dict) -> dict:
        """根据 FactChecker 的反馈，针对性地重写训练计划中有问题的部分。

        输入：
            original_plan: dict — 上一次生成的计划（需要修正）
            issues: list[dict] — FactChecker 发现的问题列表
            retrieved: dict — 检索到的动作库数据
            profile: dict — 用户画像
        输出：
            dict — 修正后的训练计划 JSON

        与 write_plan 的区别：
            - 不是从零生成，而是在原计划基础上"修修补补"
            - 只改动 FactChecker 指出的问题部分，保留其他内容不变
            - 如果某个动作被标记为危险，从动作库中找安全的替代
        """
        goal = profile.get("goal", "增肌")
        issues_text = "\n".join(
            f"- {i.get('issue', str(i))}（严重程度: {i.get('severity', '未知')}）"
            for i in issues
        )

        fix_prompt = f"""你之前生成的训练计划存在以下安全问题，请修正：

{issues_text}

原计划内容：
{json.dumps(original_plan, ensure_ascii=False, indent=2)}

修正要求：
1. 只改动有问题的地方，其他部分原样保留
2. 如果某个动作有安全风险，从可用动作库里找安全的替代动作
3. 如果用户的伤病涉及某些部位，完全避开相关动作
4. 修正后的计划必须仍然是 {profile.get('days_per_week', original_plan.get('days_per_week', 4))} 天
5. 输出完整的修正后计划 JSON（不是只输出修改的部分）

目标：{goal}
用户画像：{profile}
可用动作：{retrieved.get('exercises', [])}"""

        messages = [
            {"role": "system", "content": "你是训练计划修正专家。根据安全检查的反馈，修正计划中的问题。"},
            {"role": "user", "content": fix_prompt}
        ]
        result = self.llm.chat_with_json_mode(messages, model="reasoner")
        result["plan_id"] = original_plan.get("plan_id", str(uuid.uuid4())[:8])
        result["user_id"] = profile.get("id", 0)
        return result

    def write_analysis(self, exercise_name: str, user_desc: str,
                       retrieved: dict, profile: dict) -> dict:
        """【同步版】分析单个动作的规范性和问题。

        输入：
            exercise_name: str — 动作名称
            user_desc: str — 用户对自己动作的描述
            retrieved: dict — 从 Retriever 获取的标准动作规范
            profile: dict — 用户画像
        输出：
            dict — 含 "exercise_name"、"issues_found"、"severity"、"suggestions"、"confidence"

        与 write_plan 的区别：不依赖外部 prompts 模块，直接在方法内构造内联 prompt，
        因为动作分析的提示词模板固定且简短，无需单独管理。
        setdefault 确保即使 LLM 漏掉了某个字段，返回结构也是完整的。
        """
        prompt = f"""分析动作：{exercise_name}
用户描述：{user_desc}
用户水平：{profile.get('training_years', 1)}年经验
标准动作规范：{retrieved}

输出 JSON：
{{
  "exercise_name": "{exercise_name}",
  "issues_found": ["问题1", "问题2"],
  "severity": "安全" | "注意" | "警告",
  "suggestions": ["改进1", "改进2"],
  "confidence": 0.0-1.0
}}"""
        result = self.llm.chat_with_json_mode([{"role": "user", "content": prompt}])
        # 强制覆盖 exercise_name（防止 LLM 自行修改）
        result["exercise_name"] = exercise_name
        # setdefault：如果 LLM 输出中缺少这些字段，用默认值填充
        result.setdefault("issues_found", [])
        result.setdefault("severity", "安全")
        result.setdefault("suggestions", [])
        result.setdefault("confidence", 0.5)
        return result

    def write_analysis_stream(self, exercise_name: str, user_desc: str,
                              retrieved: dict, profile: dict,
                              conv_context: str = "") -> Generator:
        """【流式版】逐块输出动作分析结果。产出 (event_type, data) 元组。

        与同步版的区别和 write_plan_stream 类似：
        - 使用 chat_stream 逐 token 产出
        - 手动剥离 markdown 代码块后解析 JSON
        - JSONDecodeError 时返回 {"raw": full_text} 兜底
        """
        prompt = f"""分析动作：{exercise_name}
用户描述：{user_desc}
用户水平：{profile.get('training_years', 1)}年经验
标准动作规范：{retrieved}

输出 JSON：
{{
  "exercise_name": "{exercise_name}",
  "issues_found": ["问题1", "问题2"],
  "severity": "安全" | "注意" | "警告",
  "suggestions": ["改进1", "改进2"],
  "confidence": 0.0-1.0
}}"""
        if conv_context:
            prompt = f"{conv_context}\n\n{prompt}"
        full_text = ""
        for chunk in self.llm.chat_stream([{"role": "user", "content": prompt}], temperature=0.3):
            full_text += chunk
            yield ("chunk", chunk)
        content = full_text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        try:
            result = json.loads(content.strip())
        except json.JSONDecodeError:
            result = {"raw": full_text}
        result["exercise_name"] = exercise_name
        result.setdefault("issues_found", [])
        result.setdefault("severity", "安全")
        result.setdefault("suggestions", [])
        result.setdefault("confidence", 0.5)
        yield ("done", result)
