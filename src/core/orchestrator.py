"""
===========================================================================
文件角色：总调度器 (Orchestrator) —— 整个 AI 健身教练系统的"大脑"
===========================================================================
- 被谁调用：FastAPI 路由层 (app/ 目录) 通过 Orchestrator 实例调用其公开方法
- 调用谁：
    PlannerAgent   → 将用户输入拆解为子任务，匹配技能模板
    RetrieverAgent → 根据子任务检索动作库和知识图谱
    WriterAgent    → 用 LLM 生成训练计划 / 动作分析文本
    FactCheckerAgent → 对生成结果进行安全审查 + HITL 升级判定
    SemanticCache  → 语义缓存，避免相同输入重复调用 LLM
    KnowledgeSearch → 知识库搜索（向量 + 关键词 → RRF 融合 → 重排序）
    ConversationManager → 多轮对话上下文管理（滑动窗口 + 摘要）
    SkillRegistry  → 根据用户输入匹配合适的训练技能模板
    MessageBus     → A2A (Agent-to-Agent) 消息总线，记录任务流转
- 核心职责：
    1. 编排完整的"用户输入 → 规划 → 检索 → 生成 → 核查 → 缓存"流水线
    2. 提供同步版和流式版两种接口，分别适配轮询和 SSE 场景
    3. 归一化 LLM 输出格式（兼容不同模型返回的键名差异）
    4. 协调多轮问答的上下文注入与引用来源构建
===========================================================================
"""

import logging
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.writer import WriterAgent
from src.agents.fact_checker import FactCheckerAgent
from src.rag.semantic_cache import SemanticCache
from src.rag.knowledge_search import KnowledgeSearch
from src.memory.conversation import ConversationManager
from src.skills.registry import SkillRegistry
from src.a2a.messaging import MessageBus, Task, Artifact
from src.models.schemas import UserProfileInput
from src.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)


class Orchestrator:
    """总调度器：串联 Planner → Retriever → Writer → FactChecker 的流水线。
    在项目流程中的位置：HTTP 请求 → Orchestrator → 各 Agent → 返回结果。
    提供同步版 (generate_plan) 和流式版 (generate_plan_stream / answer_question_stream) 两种接口。"""

    def __init__(self):
        """初始化所有子模块。
        在应用启动时创建一次 Orchestrator 实例（单例模式），
        后续每个请求复用同一实例中的各 Agent 和缓存/搜索组件。
        各子模块职责：
        - planner:      拆解用户需求为子任务 + 匹配训练技能模板
        - retriever:    从动作库/知识图谱检索相关数据
        - writer:       调用 LLM 生成训练计划或动作分析
        - fact_checker: 安全审查 + HITL 人工复核升级
        - cache:        语义缓存，避免相同输入重复调用 LLM，降低延迟和成本
        - skills:       技能注册表，根据用户输入自动匹配训练模板
        - bus:          A2A 消息总线，记录任务在 Agent 间的流转日志
        - knowledge:    知识库搜索器（向量检索 + BM25 关键词检索 → RRF 融合 → LLM 重排序）
        - conversation: 多轮对话管理器（滑动窗口 + LLM 摘要压缩历史）"""
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.writer = WriterAgent()
        self.fact_checker = FactCheckerAgent()
        self.cache = SemanticCache()
        self.skills = SkillRegistry()
        self.bus = MessageBus()
        self.knowledge = KnowledgeSearch()
        self.conversation = ConversationManager()
        self.documents = DocumentStore()

    def generate_plan(self, profile: UserProfileInput, query: str = "") -> dict:
        """【同步版】根据用户画像和查询生成完整的训练计划。

        输入：
            profile: UserProfileInput — 用户画像（含身高/体重/训练年限/目标/伤病等）
            query: str — 用户额外补充的文本描述，可为空字符串
        输出：
            dict — 包含以下关键字段的训练计划：
                - "days": list[dict] — 每日训练安排（含动作列表）
                - "warnings": list[str] — 安全检查发现的问题
                - "requires_review": bool — 是否需要人工复核
                - "confidence": float — 整体置信度 (0~1)

        核心流水线（6 步）：
            1. 语义缓存查询 → 命中则直接返回，节省 LLM 调用成本
            2. Planner 拆解任务 → 产出子任务列表 + 匹配的训练技能模板
            3. Retriever 检索 → 根据子任务从动作库和知识图谱获取数据
            4. Writer 调用 LLM 生成计划 → A2A 消息总线记录任务流转
            5. FactChecker 安全审查 → 产出警告列表 + HITL 升级判定
            6. 写入语义缓存 → 下次相同输入可直接命中
        """
        profile_dict = profile.model_dump()
        # 1. 检查缓存 —— 语义缓存通过向量相似度匹配，不要求精确相等
        cached = self.cache.get(profile_dict, query)
        if cached:
            logger.info("Cache hit for plan generation")
            return cached

        # 2. 规划器 —— 如果没有 query，用用户目标构造默认查询
        plan = self.planner.plan(query or f"为{profile.goal}目标生成训练计划", profile_dict)

        # 3. 检索器 —— 根据 Planner 产出的子任务逐条检索动作数据
        retrieved = self.retriever.retrieve(plan)

        # 4. Writer（首次生成）→ 归一化
        task = Task(
            task_id=f"write_{profile_dict.get('id', 0)}",
            from_agent="orchestrator", to_agent="writer",
            task_type="generate_plan", payload={
                "retrieved": retrieved, "profile": profile_dict,
                "plan_config": plan.get("skill_config", {})
            }
        )
        self.bus.send(task)
        result = self.writer.write_plan(
            retrieved, profile_dict, plan.get("skill_config", {})
        )
        result = self._normalize_plan(result)
        task.add_artifact(Artifact(
            artifact_id=task.task_id, artifact_type="training_plan", content=result
        ))

        # 5. FactChecker 审查 + Writer 修正回路
        # 设计意图：FactChecker 发现问题后，不直接返回带警告的计划给用户，
        # 而是把问题退回给 Writer 重写。Writer 根据反馈有针对性地替换危险动作、
        # 调整不合理配置，然后 FactChecker 再次审查。最多重试 3 次。
        MAX_RETRIES = 3
        rewrite_count = 0
        all_checks = []  # 记录每次审查结果，用于最终汇总

        check = self.fact_checker.check(result, profile_dict)
        all_checks.append(check)

        while (not check.get("is_safe", True) or check.get("issues")) and rewrite_count < MAX_RETRIES:
            logger.info(
                f"FactChecker found {len(check.get('issues', []))} issue(s), "
                f"rewrite attempt {rewrite_count + 1}/{MAX_RETRIES}"
            )
            # Writer 根据 FactChecker 反馈重写
            result = self.writer.rewrite_plan(
                result, check.get("issues", []), retrieved, profile_dict
            )
            result = self._normalize_plan(result)
            rewrite_count += 1

            # 再次审查
            check = self.fact_checker.check(result, profile_dict)
            all_checks.append(check)

        # 汇总所有轮次的警告（去重）
        all_issues = []
        seen = set()
        for c in all_checks:
            for issue in c.get("issues", []):
                key = issue.get("issue", str(issue))
                if key not in seen:
                    seen.add(key)
                    all_issues.append(key)

        result["warnings"] = all_issues
        result["requires_review"] = any(c.get("requires_human_review", False) for c in all_checks)
        result["confidence"] = min(c.get("confidence", 0) for c in all_checks)
        result["rewrite_count"] = rewrite_count

        # 6. 写入缓存并返回 —— 下次相同/相似输入可直接命中，减少 LLM 调用
        self.cache.set(profile_dict, query, result)
        task.complete()
        return result

    def _normalize_plan(self, result: dict) -> dict:
        """归一化 LLM 输出的训练计划 JSON。

        不同 LLM 可能返回不同的键名（weekly_plan / schedule / plan / days），
        这里统一映射为 "days"。同时归一化动作内部字段名。
        """
        for key in ("weekly_plan", "weekly_schedule", "days", "schedule", "plan"):
            if key in result:
                result["days"] = result.pop(key)
                break
        for day in result.get("days", []):
            for ex in day.get("exercises", []):
                if "rest_seconds" in ex and "rest" not in ex:
                    ex["rest"] = f"{ex.pop('rest_seconds')}s"
                if "exercise" in ex and "name" not in ex:
                    ex["name"] = ex.pop("exercise")
                if "movement" in ex and "name" not in ex:
                    ex["name"] = ex.pop("movement")
        return result

    def generate_plan_stream(self, profile: UserProfileInput, query: str = "",
                             session_id: str = None):
        """【流式版】逐阶段产出 (stage, data) 元组，供前端通过 SSE 实时更新 UI。

        输入：
            profile: UserProfileInput — 用户画像
            query: str — 用户补充描述
            session_id: str | None — 会话 ID，非 None 时启用多轮对话
        产出（Generator）：
            Generator[(str, any)] — 每个阶段产出一个或多个元组：
                - ("stage", str)          → 当前阶段描述，前端显示为进度提示
                - ("advice_chunk", str)   → 教练口头建议的流式文本片段
                - ("advice_done", str)    → 口头建议完成，附完整文本
                - ("planner_done", dict)  → 规划阶段完成，附技能名和子任务列表
                - ("retriever_done", dict)→ 检索完成，附动作数量和名称预览
                - ("writer_chunk", str)   → 计划生成的流式文本片段
                - ("writer_done_raw", str)→ Writer 原始输出全文
                - ("factcheck_done", dict)→ 安全检查完成，附安全状态和置信度
                - ("cache_hit", dict)     → 缓存命中，直接返回完整结果
                - ("done", dict)          → 最终完成，附完整训练计划结果

        与同步版的区别：
            1. 新增口头建议阶段 —— 在正式流水线前给出教练式的初步分析，
               让用户在等待时不感到空白，提升体验
            2. 每个阶段通过 yield 实时推送进度，前端可渐进式渲染
            3. Writer 使用 chat_stream 而非 chat_with_json_mode，
               实现逐 token 输出，让用户看到计划"被写出来"的过程
        """
        profile_dict = profile.model_dump()

        # === 多轮对话：注入历史上下文 ===
        conv_context = ""
        plan_context = ""
        if session_id:
            user_turn_preview = query[:200] if query else "生成训练计划"
            self.conversation.add_turn(session_id, "user", user_turn_preview)
            conv_context = self.conversation.build_context_for_prompt(session_id, query or "")
            # 获取上一轮生成的计划，用于"把第二天改成哑铃"这类修改请求
            plan_context = self.conversation.get_plan_state(session_id) or ""

        # 1. 检查缓存（多轮对话场景跳过缓存，因为每次修改都需要重新生成）
        if not session_id:
            cached = self.cache.get(profile_dict, query)
            if cached:
                yield ("cache_hit", cached)
                yield ("done", cached)
                return

        # 2. 初步分析 —— 在流水线开始前给出教练式的口头建议
        # 设计意图：LLM 调用耗时较长，先推送一个简短的分析结果让用户有内容可看，
        # 避免用户面对空白页面等待，同时建立教练对话感
        yield ("stage", "[分析] 正在分析你的情况...")
        advice_context = ""
        if plan_context:
            advice_context = f"\n\n用户之前已经有了一个训练计划：\n{plan_context}\n用户现在说：{query}\n请结合这个上下文给出建议。"
        advice_prompt = self._build_advice_prompt(profile_dict, query) + advice_context
        advice_text = ""
        for chunk in self.writer.llm.chat_stream(
            [{"role": "user", "content": advice_prompt}], temperature=0.5
        ):
            advice_text += chunk
            yield ("advice_chunk", chunk)
        yield ("advice_done", advice_text)

        # 3. 规划器 —— 多轮场景下注入历史上下文和已有计划
        yield ("stage", "[规划] Planner 正在拆解任务...")
        plan = self.planner.plan(
            query or f"为{profile.goal}目标生成训练计划",
            profile_dict,
            conv_context=conv_context,
            plan_context=plan_context,
        )
        yield ("planner_done", {"skill": plan.get("skill", "unknown"),
                                 "subtasks": plan.get("subtasks", [])})

        # 3. 检索器
        yield ("stage", "[检索] Retriever 正在检索动作库...")
        retrieved = self.retriever.retrieve(plan)
        exercises = retrieved.get("exercises", [])
        yield ("retriever_done", {"count": len(exercises),
                                   "names": [e.get("name", "?") for e in exercises[:8]]})

        # 4. Writer —— 流式输出，多轮场景下注入已有计划上下文
        yield ("stage", "[生成] Writer 正在生成训练计划...")
        writer_extra = {}
        if plan_context:
            writer_extra["plan_context"] = plan_context
            writer_extra["user_query"] = query
        full_text = ""
        for event, data in self.writer.write_plan_stream(
            retrieved, profile_dict, plan.get("skill_config", {}),
            **writer_extra,
        ):
            if event == "chunk":
                full_text += data
                yield ("writer_chunk", data)
            elif event == "done":
                result = data
        yield ("writer_done_raw", full_text)
        result = self._normalize_plan(result)

        # 5. FactChecker 审查 + Writer 修正回路（流式版）
        MAX_RETRIES = 3
        rewrite_count = 0
        all_checks = []

        check = self.fact_checker.check(result, profile_dict)
        all_checks.append(check)
        yield ("factcheck_done", {"safe": check.get("is_safe", True),
                                   "issues": len(check.get("issues", [])),
                                   "confidence": check.get("confidence", 0)})

        while (not check.get("is_safe", True) or check.get("issues")) and rewrite_count < MAX_RETRIES:
            yield ("stage", f"[修正] 安全检查发现 {len(check.get('issues', []))} 个问题，第 {rewrite_count + 1} 次重写...")
            result = self.writer.rewrite_plan(
                result, check.get("issues", []), retrieved, profile_dict
            )
            result = self._normalize_plan(result)
            rewrite_count += 1

            check = self.fact_checker.check(result, profile_dict)
            all_checks.append(check)
            yield ("factcheck_done", {"safe": check.get("is_safe", True),
                                       "issues": len(check.get("issues", [])),
                                       "confidence": check.get("confidence", 0)})

        # 汇总所有轮次警告
        all_issues = []
        seen = set()
        for c in all_checks:
            for issue in c.get("issues", []):
                key = issue.get("issue", str(issue))
                if key not in seen:
                    seen.add(key)
                    all_issues.append(key)

        result["warnings"] = all_issues
        result["requires_review"] = any(c.get("requires_human_review", False) for c in all_checks)
        result["confidence"] = min(c.get("confidence", 0) for c in all_checks)
        result["rewrite_count"] = rewrite_count

        # === 多轮对话：保存计划状态 + 助手回复到对话历史 ===
        if session_id:
            plan_summary = self._summarize_plan_for_context(result)
            self.conversation.set_plan_state(session_id, plan_summary)
            assistant_preview = self._summarize_plan_for_context(result)
            self.conversation.add_turn(session_id, "assistant", assistant_preview[:500])

        # 6. 写入缓存（多轮对话跳过缓存）
        if not session_id:
            self.cache.set(profile_dict, query, result)
        yield ("done", result)

    def _build_advice_prompt(self, profile: dict, query: str) -> str:
        """【私有方法】构建"教练口头建议"的 LLM 提示词。

        输入：
            profile: dict — 用户画像字典（已通过 model_dump() 转换）
            query: str — 用户补充描述
        输出：
            str — 一段精心设计的提示词，引导 LLM 以健身教练的口吻
                  对用户数据做初步分析（BMI、训练阶段、伤病关注、训练原则），
                  并给出 2-3 条实用建议。用于流式版生成前的"暖场"阶段。
        """
        eq = "、".join(profile.get("available_equipment", []))
        injuries_raw = profile.get("injuries", [])
        injuries = injuries_raw[0] if injuries_raw else "无"
        return f"""你是一位经验丰富的健身教练。用户刚刚输入了以下信息，请先给一个简短、友好、专业的初步分析。

用户信息：
- 身高 {profile.get('height')}cm，体重 {profile.get('weight')}kg
- 训练 {profile.get('training_years')} 年
- 目标：{profile.get('goal')}
- 训练场景对应的可用器械：{eq}
- 每周 {profile.get('days_per_week')} 练
- 伤病情况：{injuries}
- 补充说明：{query or '无'}

要求：
1. 先打招呼，认可用户的基础和目标
2. 点评用户的数据（BMI是否合理、训练年限处于什么阶段、器械是否够用）
3. 如果用户有伤病，认真分析并给出规避建议；如果没有伤病也要提一下注意预防
4. 给 2-3 条针对该场景的实用训练原则
5. 最后用一句"接下来我为你生成具体的训练计划"过渡
6. 总共 150-200 字，用口语化、有温度的语气，像一个真正的教练在对话
7. 用中文回复，不要用 markdown 格式，就是纯文字段落"""

    def analyze_exercise(self, exercise_name: str, user_desc: str,
                         profile: UserProfileInput) -> dict:
        """【同步版】分析单个动作的规范性和问题。

        输入：
            exercise_name: str — 动作名称，如 "深蹲"、"卧推"
            user_desc: str — 用户对自己做这个动作时的描述（可能包含问题描述）
            profile: UserProfileInput — 用户画像
        输出：
            dict — 包含 "exercise_name"、"issues_found"、"severity"、"suggestions"、"confidence"
        流程：Retriever 检索动作标准规范 → Writer 对比分析 → 返回诊断结果
        """
        profile_dict = profile.model_dump()
        retrieved = self.retriever.retrieve({"subtasks": [exercise_name], "skill_config": {}})
        return self.writer.write_analysis(exercise_name, user_desc, retrieved, profile_dict)

    def analyze_exercise_stream(self, exercise_name: str, user_desc: str,
                                profile: UserProfileInput, session_id: str = None):
        """【流式版】动作分析 —— 逐阶段产出进度事件，供前端 SSE 实时渲染。
        产出 (event_type, data) 元组，与 generate_plan_stream 模式一致。"""
        profile_dict = profile.model_dump()

        # === 多轮对话：注入历史上下文 ===
        conv_context = ""
        if session_id:
            self.conversation.add_turn(session_id, "user",
                                       f"分析动作：{exercise_name} — {user_desc}"[:200])
            conv_context = self.conversation.build_context_for_prompt(
                session_id, f"分析{exercise_name}：{user_desc}"
            )

        yield ("stage", "[检索] 正在检索动作标准规范...")
        retrieved = self.retriever.retrieve({"subtasks": [exercise_name], "skill_config": {}})
        yield ("retriever_done", {"count": len(retrieved.get("exercises", []))})

        yield ("stage", "[分析] 正在分析动作问题...")
        full_text = ""
        for event, data in self.writer.write_analysis_stream(
            exercise_name, user_desc, retrieved, profile_dict,
            conv_context=conv_context,
        ):
            if event == "chunk":
                full_text += data
                yield ("writer_chunk", data)
            elif event == "done":
                result = data
        # 将检索到的参考动作名附加到结果中，供前端展示出处
        ref_exercises = [e.get("name", "") for e in retrieved.get("exercises", [])[:5]]
        result["reference_exercises"] = ref_exercises

        # === 多轮对话：保存助手回复 ===
        if session_id:
            self.conversation.add_turn(session_id, "assistant", full_text[:500])

        yield ("done", result)

    def answer_question_stream(self, question: str, profile: UserProfileInput, session_id: str = None):
        """【流式版】多源融合问答 —— 结合知识库 + 动作数据库 + 知识图谱回答用户问题。

        输入：
            question: str — 用户自由文本问题，如 "深蹲膝盖疼怎么办？"
            profile: UserProfileInput — 用户画像（提供身体数据和伤病背景）
            session_id: str | None — 会话 ID，非 None 时启用多轮对话上下文管理

        产出（Generator）：
            ("stage", str) / ("answer_chunk", str) / ("graph_done", dict) /
            ("knowledge_done", dict) / ("retriever_done", dict) / ("done", dict)

        检索流水线（5 步）：
            1. 伤病关键词检测 → 命中则启用 GraphRAG 知识图谱多跳推理
            2. 知识库搜索 → 向量 + 关键词检索 → RRF 融合 → LLM 重排序
            3. 动作数据库检索 → 查找与问题相关的训练动作
            4. 多轮对话上下文注入 → 滑动窗口 + 摘要压缩历史
            5. LLM 流式生成回答 + 构建引用来源列表 + 保存本轮对话

        设计要点：
            - GraphRAG 仅对伤病/疼痛类问题启用，因为知识图谱存储的是"动作-肌肉-伤病"关系，
              对通用健身问答（如"怎么增肌"）无帮助，启用反而浪费资源
            - 知识库搜索使用 search_with_fallback：
              向量检索 → 结果不够时降级到关键词检索 → RRF 融合 → 重排序，确保召回率
            - 多轮对话使用 sliding window + LLM 摘要，既保留近期细节又压缩远期历史
        """
        profile_dict = profile.model_dump()

        # === 检测是否为伤病/疼痛类问题 → 启用 GraphRAG ===
        # 知识图谱中存储了"动作→肌肉→伤病"的关系链，适合多跳推理。
        # 仅当问题包含疼痛相关关键词时才触发，避免无意义的图谱查询开销。
        pain_keywords = ["疼", "痛", "伤", "酸", "不舒服", "拉伤", "扭伤", "炎症", "恢复"]
        is_pain_q = any(kw in question for kw in pain_keywords)

        graph_data = None
        if is_pain_q:
            yield ("stage", "[图谱] 正在用知识图谱推理伤病关联...")
            exercise_name = self._extract_exercise_from_question(question)
            if exercise_name:
                # 调用 MCP 工具进行图谱推理：给定动作+症状，找出可能的伤病原因链
                pain_result = self.retriever.tools.call("graph_reason_pain", {
                    "exercise": exercise_name, "symptom": question
                })
                graph_data = {"exercise": exercise_name, "pain_data": pain_result}
                yield ("graph_done", graph_data)

        # === 知识库搜索 ===
        yield ("stage", "[知识库] 正在检索健身知识库...")
        knowledge_chunks = self.knowledge.search_with_fallback(question)
        yield ("knowledge_done", {"count": len(knowledge_chunks)})

        # === 用户文档搜索（如果 session 中有上传文件）===
        doc_chunks = []
        doc_list = []
        if session_id:
            doc_chunks = self.documents.search(question, session_id, top_k=5)
            doc_list = self.documents.get_documents_for_session(session_id)
            if doc_chunks:
                yield ("knowledge_done", {"count": len(knowledge_chunks),
                                          "doc_chunks": len(doc_chunks),
                                          "doc_files": len(doc_list)})

        # === 动作数据库检索 ===
        # 即使是一般性问题，也检索相关动作作为回答的参考素材
        retrieved = self.retriever.retrieve({"subtasks": [question], "skill_config": {}})
        exercises = retrieved.get("exercises", [])
        yield ("retriever_done", {"count": len(exercises),
                                   "names": [e.get("name", "?") for e in exercises[:6]]})

        # === 构建带引用来源和对话上下文的回答提示词 ===
        yield ("stage", "[解答] 正在为你解答...")
        sources_text = ""
        # 过滤低相关性 chunk：LLM 重排序分数 >= 6/10 的才纳入上下文
        RELEVANCE_THRESHOLD = 6
        relevant_sources = [
            c for c in knowledge_chunks
            if (c.get("rerank_score") or c.get("rrf_score") or 0) >= RELEVANCE_THRESHOLD
        ]
        if not relevant_sources:
            relevant_sources = knowledge_chunks[:2]  # 兜底：至少保留 2 条
            sources_text = "（以下知识库内容仅供参考，相关性可能不高）\n"
        for i, chunk in enumerate(relevant_sources, 1):
            snippet = chunk["content"][:400].replace("\n", " ")
            sources_text += f"\n[来源{i}] 《{chunk['title']}》：{snippet}\n"

        # === 用户上传文档的 chunk 注入 ===
        doc_sources_text = ""
        if doc_chunks:
            for i, chunk in enumerate(doc_chunks, 1):
                snippet = chunk["content"][:400].replace("\n", " ")
                doc_sources_text += f"\n[你的文件-{i}] {snippet}\n"

        # === 多轮对话：注入历史上下文 ===
        # 先记录本轮用户输入，再获取历史摘要。
        # build_context_for_prompt 内部：滑动窗口取最近 N 轮 + 超出窗口的轮次做 LLM 摘要压缩
        conv_context = ""
        if session_id:
            user_turn_preview = question[:200]
            self.conversation.add_turn(session_id, "user", user_turn_preview)
            conv_context = self.conversation.build_context_for_prompt(session_id, question)

        prompt = f"""你是资深健身教练和运动康复专家。请基于提供的知识库文档回答用户的问题。
如果知识库中没有足够信息，可以结合你的专业知识补充，但需要明确指出哪些来自文档、哪些是专业推断。

用户情况：{profile_dict.get('height')}cm, {profile_dict.get('weight')}kg, 训练{profile_dict.get('training_years')}年
伤病：{profile_dict.get('injuries', [])}

{conv_context}

知识库相关文档：
{sources_text if sources_text else '（未找到直接相关的知识库文档，请基于专业知识回答）'}

{"【用户上传的文件内容】" if doc_sources_text else ""}
{doc_sources_text if doc_sources_text else ""}

相关动作参考：
{exercises[:5] if exercises else '无特定动作关联'}

要求：
1. 先直接回答问题，给出明确结论
2. 解释原因（解剖/生理层面，但用大白话说）
3. 给出 2-3 条可执行的建议
4. 如果涉及危险信号，明确建议就医
5. 用自己的话自然回答，不要在正文里写 [来源N] 或类似标记（来源信息会单独展示给用户）
6. 200-350 字，口语化，像教练在聊天
7. 纯文字段落，不用 markdown
{"8. 如果用户使用了'改一下''换一个''刚才说的'等指代，请结合对话历史中的上下文理解用户的真正意图。" if conv_context else ""}"""

        full_text = ""
        for chunk in self.writer.llm.chat_stream([{"role": "user", "content": prompt}], temperature=0.5):
            full_text += chunk
            yield ("answer_chunk", chunk)

        # === 保存助手本轮回复到对话历史 ===
        # 截取前 500 字符存储，避免过长文本撑爆上下文窗口
        if session_id:
            self.conversation.add_turn(session_id, "assistant", full_text[:500])

        # === 构建引用来源列表 ===
        # 只返回实际在回答中可能被引用的高相关性来源
        source_citations = [
            {"title": c["title"], "source_file": c.get("source_file", ""),
             "score": c.get("rerank_score") or c.get("rrf_score", 0)}
            for c in relevant_sources
        ]

        yield ("done", {
            "answer": full_text,
            "sources": source_citations,
            "knowledge_count": len(relevant_sources),
            "exercise_count": len(exercises),
            "graph_data": graph_data,
            "session_id": session_id,
            "doc_chunks": len(doc_chunks),
            "doc_files": len(doc_list),
        })

    def _extract_exercise_from_question(self, question: str) -> str | None:
        """【私有方法】从用户问题文本中通过关键词匹配提取动作名称。

        输入：
            question: str — 用户原始问题文本
        输出：
            str | None — 匹配到的动作名称，无匹配则返回 None

        用途：为 GraphRAG 伤病推理提供动作锚点。
        局限性：依赖硬编码的动作列表，无法识别列表中不存在的动作或口语化表述。
                TODO: 可升级为 NER 模型或 LLM 实体提取以提高召回率。"""
        common_exercises = [
            "深蹲", "硬拉", "卧推", "推举", "划船", "弯举", "臂屈伸",
            "引体向上", "下拉", "飞鸟", "侧平举", "前平举", "面拉",
            "腿举", "弯举", "耸肩", "提踵", "臀推", "箭步蹲", "分腿蹲",
            "平板支撑", "举腿", "俄罗斯转体", "双杠臂屈伸", "直臂下压",
            "保加利亚分腿蹲", "高脚杯深蹲", "罗马尼亚硬拉", "史密斯机深蹲",
            "哑铃卧推", "杠铃卧推", "上斜卧推", "哑铃飞鸟", "绳索夹胸",
            "坐姿划船", "高位下拉", "哑铃推举", "杠铃推举", "哑铃弯举",
            "锤式弯举", "杠铃弯举", "绳索下压", "窄距卧推", "颈后臂屈伸",
        ]
        for ex in common_exercises:
            if ex in question:
                return ex
        return None

    def _summarize_plan_for_context(self, plan: dict) -> str:
        """【私有方法】从训练计划提取摘要，供多轮对话的 plan_state 存储。

        输入：
            plan: dict — 完整的训练计划结果（含 days 列表）
        输出：
            str — "第1天(胸+三头): 杠铃卧推/哑铃飞鸟... 第2天(背+二头): ..." 格式的摘要
        """
        days = plan.get("days", [])
        if not days:
            return ""
        parts = []
        for d in days:
            day_num = d.get("day", "?")
            focus = d.get("focus", "")
            exercises = d.get("exercises", [])
            ex_names = [e.get("name", "?") for e in exercises[:6]]
            ex_str = "/".join(ex_names)
            label = f"第{day_num}天" + (f"({focus})" if focus else "")
            parts.append(f"{label}: {ex_str}")
        return " | ".join(parts)
