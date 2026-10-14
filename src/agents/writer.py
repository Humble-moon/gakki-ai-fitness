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
    动作分析 JSON 结构含 issues_found、possible_causes、severity、
    suggestions、alternatives、confidence
===========================================================================
"""

import uuid
import json
from typing import Generator
from src.config import REWRITE_MODEL
from src.llm.provider import LLMProvider
from src.llm.prompts.writer import build_writer_messages


# 动作分析保留的知识片段条数与单块截断长度。
# 知识块动辄上千字，全量塞进去会让背景知识盖过用户自己的描述——
# 用户说"深蹲时膝盖内侧疼"，主体应该是这句话，文档是佐证而不是主角。
_KNOWLEDGE_TOP_K = 3
_KNOWLEDGE_CHARS = 400


def build_standard_reference(exercise_name: str, retrieved: dict) -> tuple[str, list[str]]:
    """把检索结果整理成「标准动作规范」文本。返回 (文本, 实际引用的动作名)。

    为什么要挑而不是全用：
        检索器是给「生成训练计划」服务的——搜「杠铃深蹲」会返回 34 个深蹲变体
        （泽奇深蹲、安德森深蹲、箱式深蹲……），排课时这是优点（有得挑），
        诊断时是噪音：用户问的是"我这个深蹲哪里不对"，另外 33 个变体跟他没关系，
        还会把真正要对比的那一条挤出注意力。

    为什么保留 similarity/rrf_score 之外的信息但不保留它们本身：
        那些字段是检索排序用的，对"诊断动作"没有任何价值，只会稀释 prompt。

    匹配不到时：退回相似度最高的一条并显式标注，让模型知道自己在拿别的动作
    作参照，而不是不声不响地拿它硬套。
    """
    exercises = retrieved.get("exercises") if isinstance(retrieved, dict) else None
    rows = [e for e in exercises if isinstance(e, dict)] if isinstance(exercises, list) else []

    exact = [e for e in rows if e.get("name") == exercise_name]
    if exact:
        picked, matched = exact[:1], True
    else:
        picked = sorted(rows, key=lambda e: e.get("similarity") or 0, reverse=True)[:1]
        matched = False

    if not picked:
        return "（动作库中没有检索到该动作的资料）", []

    lines = []
    if not matched:
        lines.append(f"注意：动作库中没有「{exercise_name}」，以下是最接近的动作，仅供参考。")
    for item in picked:
        lines.append(f"【{item.get('name', '?')}】")
        if item.get("equipment"):
            lines.append(f"器械：{item['equipment']}")
        muscles = item.get("target_muscles")
        if muscles:
            lines.append(f"目标肌群：{'、'.join(muscles)}")
        if item.get("description"):
            lines.append(f"动作要领：{item['description']}")
        errors = item.get("common_errors")
        if errors:
            lines.append(f"常见错误：{'、'.join(errors)}")
    names = [i.get("name", "") for i in picked if i.get("name")]
    return "\n".join(lines), names


def build_knowledge_context(retrieved: dict) -> str:
    """把知识库片段整理成「相关专业知识」文本。

    这些片段来自 `data/knowledge/`——动作技术文档、伤病管理、FAQ。
    此前动作分析完全不检索知识库（knowledge 恒为空），于是 LLM 手里
    只有动作库里那句一句话描述和三个错误词，只能给出放之四海皆准的建议。
    """
    chunks = retrieved.get("knowledge") if isinstance(retrieved, dict) else None
    if not isinstance(chunks, list) or not chunks:
        return "（未检索到相关专业知识）"
    lines = []
    for chunk in chunks[:_KNOWLEDGE_TOP_K]:
        if not isinstance(chunk, dict):
            continue
        title = chunk.get("title") or chunk.get("source_file") or "知识片段"
        content = str(chunk.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        lines.append(f"【{title}】{content[:_KNOWLEDGE_CHARS]}")
    return "\n".join(lines) if lines else "（未检索到相关专业知识）"


def build_analysis_prompt(exercise_name: str, user_desc: str, retrieved: dict,
                          profile: dict, conv_context: str = "") -> tuple[str, list[str]]:
    """构造动作分析提示词。同步版与流式版共用，避免两处各写一份而漂移。

    返回 (prompt, 实际引用的动作名列表)。
    """
    standard_reference, ref_names = build_standard_reference(exercise_name, retrieved)
    knowledge = build_knowledge_context(retrieved)
    prompt = f"""分析动作：{exercise_name}
用户描述：{user_desc}
用户水平：{profile.get('training_years', 1)}年经验

标准动作规范：
{standard_reference}

相关专业知识：
{knowledge}

要求：
- issues_found 必须扣住用户描述里的具体感受，不要写成通用清单
- possible_causes 写可能导致该问题的原因，与 issues_found 一一对应
- suggestions 要可执行（具体怎么调整），不要写"注意安全"这类空话
- alternatives 找不到合适替代动作时给空数组

输出 JSON：
{{
  "exercise_name": "{exercise_name}",
  "issues_found": ["问题1", "问题2"],
  "possible_causes": ["原因1", "原因2"],
  "severity": "安全" | "注意" | "警告",
  "suggestions": ["改进1", "改进2"],
  "alternatives": ["替代动作1", "替代动作2"],
  "confidence": 0.0-1.0
}}"""
    if conv_context:
        prompt = f"{conv_context}\n\n{prompt}"
    return prompt, ref_names


class WriterAgent:
    """写作器 Agent：在 Orchestrator 流水线的第 4 步被调用。
    职责：将检索到的数据和用户画像"翻译"为人类可读的训练计划或动作分析。
    提供同步版（write_plan, write_analysis）和流式版（write_plan_stream, write_analysis_stream）。
    流式版的核心差异：用 chat_stream 替代 chat_with_json_mode，手动解析 JSON 输出。"""

    def __init__(self):
        self.llm = LLMProvider()

    def write_plan(self, retrieved: dict, profile: dict, plan_config: dict,
                   training_context: str = "",
                   long_term_context: str = "") -> dict:
        """【同步版】根据检索结果和用户画像生成训练计划。

        输入：
            retrieved: dict — RetrieverAgent 返回的检索结果（含 exercises 列表）
            profile: dict — 用户画像字典
            plan_config: dict — Planner 匹配的技能模板配置
            training_context: str — 训练执行历史与已确认调整（可空）
            long_term_context: str — 跨会话画像记忆：目标/伤病/运动习惯/器械（可空）
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
            retrieved.get("exercises", []), profile, goal,
            training_context=training_context,
            long_term_context=long_term_context,
        )
        # chat_with_json_mode：在 system prompt 中注入 JSON 格式约束，
        # 并在 API 调用时设置 response_format={"type": "json_object"}
        plan_json = self.llm.chat_with_json_mode(messages, model="reasoner")
        plan_json["plan_id"] = str(uuid.uuid4())[:8]
        plan_json["user_id"] = profile.get("id", 0)
        plan_json.setdefault("weeks", plan_config.get("weeks"))
        return plan_json

    def write_plan_stream(self, retrieved: dict, profile: dict, plan_config: dict,
                          plan_context: str = "", user_query: str = "",
                          training_context: str = "",
                          long_term_context: str = "") -> Generator:
        """【流式版】逐 token 产出训练计划。每次产出 (event_type, data) 元组。

        输入：同 write_plan
            plan_context: str — 上一轮计划摘要（多轮对话中用户要修改的计划）
            user_query: str — 用户当前的修改请求
            training_context: str — 训练执行历史与已确认调整（可空）
            long_term_context: str — 跨会话画像记忆（可空）
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
            retrieved.get("exercises", []), profile, goal,
            training_context=training_context,
            long_term_context=long_term_context,
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
        stream = self.llm.chat_stream(messages, temperature=0.3, model="reasoner")
        for chunk in stream:
            full_text += chunk
            yield ("chunk", chunk)
        # Provider metadata is updated when the stream is exhausted. Preserve the
        # degraded state so the orchestrator can fail closed before persistence.
        provider_degraded = bool(getattr(stream, "metadata", None) and stream.metadata.degraded)
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
        result.setdefault("weeks", plan_config.get("weeks"))
        result["_degraded"] = provider_degraded
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
4. 修正后的计划必须仍然是 {profile.get('days_per_week', original_plan.get('sessions_per_week'))} 天
5. 输出完整的修正后计划 JSON（不是只输出修改的部分），并保留原计划的 weeks 和 sessions_per_week；未知的 weeks 保持 null，不得猜测
6. 输出 JSON 的 goal 必须严格等于用户的 canonical goal：{goal}；不得输出其他 goal
7. 用户平时的运动习惯是 {profile.get('habits') or '无'}，修正后仍要与这些习惯配合

目标：{goal}
用户画像：{profile}
可用动作：{retrieved.get('exercises', [])}"""

        messages = [
            {"role": "system", "content": "你是训练计划修正专家。根据安全检查的反馈，修正计划中的问题。"},
            {"role": "user", "content": fix_prompt}
        ]
        # 用 REWRITE_MODEL 而不是写死 reasoner。
        #
        # 重写是"按明确指令改几处"的任务，不是需要长链推理的任务——实测
        # 用 reasoner 单次重写要 42~58s，是整个计划生成里最慢的一段
        #（三次重写合计约 120s，占 226s 总时长的一半以上）。
        #
        # 换快模型是安全的：**每次重写之后紧跟一次 FactChecker 检查**，
        # 改得不好会在下一轮被抓出来继续改，安全网没有变薄。
        # 需要复现旧行为时设 REWRITE_MODEL=deepseek-reasoner 即可。
        result = self.llm.chat_with_json_mode(messages, model=REWRITE_MODEL)
        result["plan_id"] = original_plan.get("plan_id", str(uuid.uuid4())[:8])
        result["user_id"] = profile.get("id", 0)
        result.setdefault("weeks", original_plan.get("weeks"))
        result.setdefault(
            "sessions_per_week",
            profile.get("days_per_week", original_plan.get("sessions_per_week")),
        )
        return result

    def write_analysis(self, exercise_name: str, user_desc: str,
                       retrieved: dict, profile: dict) -> dict:
        """【同步版】分析单个动作的规范性和问题。

        输入：
            exercise_name: str — 动作名称
            user_desc: str — 用户对自己动作的描述
            retrieved: dict — 从 Retriever 获取的标准动作规范与知识片段
            profile: dict — 用户画像
        输出：
            dict — 含 "exercise_name"、"issues_found"、"possible_causes"、
                   "severity"、"suggestions"、"alternatives"、"confidence"

        与 write_plan 的区别：不依赖外部 prompts 模块，直接在方法内构造内联 prompt，
        因为动作分析的提示词模板固定且简短，无需单独管理。
        setdefault 确保即使 LLM 漏掉了某个字段，返回结构也是完整的。
        """
        prompt, ref_names = build_analysis_prompt(exercise_name, user_desc, retrieved, profile)
        result = self.llm.chat_with_json_mode([{"role": "user", "content": prompt}])
        # 强制覆盖 exercise_name（防止 LLM 自行修改）
        result["exercise_name"] = exercise_name
        # setdefault：如果 LLM 输出中缺少这些字段，用默认值填充
        result.setdefault("issues_found", [])
        result.setdefault("possible_causes", [])
        result.setdefault("severity", "安全")
        result.setdefault("suggestions", [])
        result.setdefault("alternatives", [])
        result.setdefault("confidence", 0.5)
        # 由 writer 给出实际引用的动作名，而不是让调用方猜——
        # 它才知道 prompt 里到底喂了哪几条。
        result["reference_exercises"] = ref_names
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
        prompt, ref_names = build_analysis_prompt(
            exercise_name, user_desc, retrieved, profile, conv_context=conv_context
        )
        full_text = ""
        stream = self.llm.chat_stream([{"role": "user", "content": prompt}], temperature=0.3)
        for chunk in stream:
            full_text += chunk
            yield ("chunk", chunk)
        provider_degraded = bool(getattr(stream, "metadata", None) and stream.metadata.degraded)
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
        result.setdefault("possible_causes", [])
        result.setdefault("severity", "安全")
        result.setdefault("suggestions", [])
        result.setdefault("alternatives", [])
        result.setdefault("confidence", 0.5)
        result["reference_exercises"] = ref_names
        result["_degraded"] = provider_degraded
        yield ("done", result)
