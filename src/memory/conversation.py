"""
================================================================================
文件角色：多轮对话上下文管理器（ConversationManager）
================================================================================
- 被调用者：编排引擎或 API 层在每次用户请求时调用：
  1. add_turn() 记录本轮对话
  2. build_context_for_prompt() 获取带上下文的提示词前缀
- 调用者：本模块依赖 RedisClient（存储）和 LLMProvider（生成摘要）。
- 项目角色：实现"对话记忆"——让 AI 记住用户之前说了什么，支持指代消解
  （如用户说"把第二天改成哑铃动作"，AI 能从历史中找到"第二天"指什么）。
================================================================================
"""

"""多轮对话管理器，基于滑动窗口 + 摘要机制。

策略：
    - 保留最近 3 轮对话为完整文本
    - 将更早的对话压缩为滚动摘要（异步、非阻塞）
    - 存储在 Redis 中，基于 TTL 过期

Redis 存储结构：
    conv:{session_id}:turns   → JSON 数组，元素为 {role, content, timestamp}
    conv:{session_id}:summary → 纯文本摘要字符串
    TTL：24 小时（自动清理不活跃会话）
"""

import json
import logging
import threading
from datetime import datetime, timezone
from typing import List, Dict, Optional

from src.storage.redis_client import RedisClient
from src.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

# WINDOW_SIZE: 滑动窗口中保留的完整对话轮数。
# 为什么是 3：3 轮足够覆盖典型的指代消解场景（"换一个""上面那个"），
# 同时不会占用太多 token 预算。过大则浪费 token，过小则丢失上下文。
WINDOW_SIZE = 3

# SESSION_TTL: Redis 中的会话过期时间（秒）。
# 86400 秒 = 24 小时，之后自动清理以节省 Redis 内存。
SESSION_TTL = 86400


class ConversationManager:
    """
    管理问答功能的多轮对话上下文。

    职责：
    - 记录每轮对话（user/assistant 消息）到 Redis。
    - 构建注入 LLM prompt 的上下文文本（滑动窗口 + 滚动摘要）。
    - 异步生成早期对话的压缩摘要，平衡 token 消耗和记忆完整性。

    设计策略——为什么用"滑动窗口+摘要"而不是全量存储：
    - LLM 的 context window 有限，不能无限制塞入全量历史。
    - 最近的对话需要完整保留（因为用户最可能在指代这些内容）。
    - 更早的对话压缩为摘要，保留关键信息（话题、决策、用户数据）的同时
      大幅减少 token 占用。

    线程安全：
    - _summarize_lock 确保同一会话不会同时生成多份摘要（避免竞态写 Redis）。
    """

    def __init__(self):
        """
        初始化对话管理器。

        核心逻辑：
        1. 连接 Redis（用于持久化对话历史和摘要）。
        2. 持有 LLMProvider 实例（用于异步生成摘要）。
        3. 初始化 _summarize_lock（threading.Lock，防止并发摘要冲突）。
        """
        self.redis = RedisClient()
        self.llm = LLMProvider()
        self._summarize_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def add_turn(self, session_id: str, role: str, content: str):
        """
        记录一轮对话，并在需要时触发异步摘要生成。

        参数：
            session_id: str  - 会话唯一标识（通常是 UUID 或用户 ID + 时间戳）
            role: str        - 消息角色："user" 或 "assistant"
            content: str     - 消息正文（最大截断 2000 字符，防止单条消息过大）

        返回值：None

        核心逻辑：
        1. 构造 turn 字典（role + content[:2000] + 当前 UTC 时间戳）。
        2. 从 Redis 读取该 session 的历史 turns，追加新 turn，写回 Redis。
        3. 刷新 TTL 为 24 小时（每次有新对话都续期）。
        4. 如果 turns 数量超过 WINDOW_SIZE + 2（即窗口外还有至少 2 条旧对话），
           启动一个 daemon 线程异步生成摘要。
           阈值设为 +2 而非 +1，避免每条新消息都触发摘要（减少 LLM 调用频率）。

        为什么 content 要截断到 2000：
        单条消息过长会撑大 Redis 存储、拖慢摘要生成，且极长的单条消息对上下文
        理解帮助有限（超出摘要窗口的部分会被丢弃）。
        """
        turn = {
            "role": role,
            "content": content[:2000],          # 防止单条消息过大
            "timestamp": datetime.now(timezone.utc).isoformat(),  # UTC 时间，跨时区一致
        }
        key = f"conv:{session_id}:turns"
        raw = self.redis.get(key)
        turns = json.loads(raw) if raw else []
        turns.append(turn)
        # ex=SESSION_TTL：每次写入都续期 24h，活跃会话不会过期
        self.redis.set(key, json.dumps(turns, ensure_ascii=False), ex=SESSION_TTL)

        # 超过窗口大小 + 2 时触发异步摘要
        # 加 2 的缓冲：避免每条消息都触发摘要调用，节省 LLM API 开销
        if len(turns) > WINDOW_SIZE + 2:
            threading.Thread(
                target=self._summarize_old_turns,
                args=(session_id, turns),
                daemon=True,  # daemon 线程：主进程退出时自动终止，不阻塞关闭
            ).start()

    def get_context(self, session_id: str) -> str:
        """
        构建对话上下文文本，用于注入 LLM prompt。

        参数：
            session_id: str  - 会话唯一标识

        返回值：
            str              - 格式化的对话上下文字符串，如果没有历史则返回空串 ""

        返回格式示例：
            [对话历史摘要]
            用户之前讨论了增肌计划相关的内容...

            [最近对话]
            用户：给我一个增肌计划
            助手：好的，根据你的情况...
            用户：把第二天改成哑铃动作

        核心逻辑：
        1. 从 Redis 读取该 session 的完整 turns 和摘要。
        2. 摘要（如果有）放在最前面，帮助 LLM 快速了解历史话题。
        3. 最近的 WINDOW_SIZE 轮以完整文本形式展示，因为用户最可能在指代这些内容。
        4. 每条消息截断到 500 字符，防止上下文过长。
        """
        turns = self._get_turns(session_id)
        summary = self._get_summary(session_id)

        if not turns and not summary:
            return ""

        parts = []

        # 摘要放前面，让 LLM 先了解全局背景
        if summary:
            parts.append(f"[对话历史摘要]\n{summary}")

        # 最近 WINDOW_SIZE 轮作为完整文本
        recent = turns[-WINDOW_SIZE:] if len(turns) > WINDOW_SIZE else turns
        if recent:
            lines = ["[最近对话]"]
            for t in recent:
                role_label = "用户" if t["role"] == "user" else "助手"
                # 截断到 500 字符，平衡信息量和 token 消耗
                lines.append(f"{role_label}：{t['content'][:500]}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def build_context_for_prompt(self, session_id: str, question: str) -> str:
        """
        构建包含对话感知的完整提示词前缀。

        参数：
            session_id: str  - 会话唯一标识
            question: str    - 用户当前问题

        返回值：
            str              - 可直接拼接在 system prompt 前面的上下文字符串。
                              如果没有历史对话则返回空串 ""。

        为什么需要这个方法：
        get_context() 只返回对话历史本身，而此方法额外包装了指令性前缀
        （"以下是用户之前的对话记录..."）和当前问题，还包含了指代消解的提示
        （"如果用户说'改一下''换一个'，回顾历史..."），让 LLM 更好地利用上下文。

        核心逻辑：
        1. 调用 get_context() 获取历史上下文。
        2. 如果有历史，包装一段指令性前缀 + 历史 + 当前问题。
        3. 如果没有历史，返回空串（避免注入无意义的提示词前缀浪费 token）。
        """
        history = self.get_context(session_id)
        if not history:
            return ""

        return f"""以下是用户之前的对话记录。请基于此上下文理解当前问题：

{history}

---
当前问题：{question}
（注意：理解当前问题时，请结合对话历史中的上下文。如果用户说"改一下""换一个"，回顾历史找到上一轮讨论的是什么。）
"""

    def set_plan_state(self, session_id: str, plan_summary: str):
        """存储当前训练计划摘要，供多轮对话中的计划修改请求使用。

        当用户说"把第二天改成哑铃动作"时，系统需要知道"第二天"当前是什么内容。
        此方法将生成的计划摘要持久化到 Redis，后续请求可通过 get_plan_state() 获取。

        参数：
            session_id: str    - 会话唯一标识
            plan_summary: str  - 计划摘要文本（包含每天的关键动作列表，不超过 800 字符）
        """
        self.redis.set(f"conv:{session_id}:plan", plan_summary[:800], ex=SESSION_TTL)

    def get_plan_state(self, session_id: str) -> str | None:
        """获取当前会话中的训练计划摘要。

        参数：
            session_id: str - 会话唯一标识

        返回值：
            str | None - 计划摘要文本，没有则返回 None
        """
        return self.redis.get(f"conv:{session_id}:plan")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_turns(self, session_id: str) -> List[dict]:
        """
        从 Redis 读取指定会话的完整 turns 列表。

        参数：
            session_id: str  - 会话唯一标识

        返回值：
            List[dict]       - turns 列表，每项为 {role, content, timestamp}
                              Redis 中没有数据时返回空列表 []
        """
        raw = self.redis.get(f"conv:{session_id}:turns")
        return json.loads(raw) if raw else []

    def _get_summary(self, session_id: str) -> Optional[str]:
        """
        从 Redis 读取指定会话的滚动摘要。

        参数：
            session_id: str  - 会话唯一标识

        返回值：
            str | None        - 摘要文本，不存在时返回 None
        """
        return self.redis.get(f"conv:{session_id}:summary")

    def _summarize_old_turns(self, session_id: str, turns: List[dict]):
        """
        异步生成滚动摘要：将滑动窗口之前的旧对话压缩为一段摘要。

        参数：
            session_id: str   - 会话唯一标识
            turns: List[dict] - 当前完整的 turns 列表（调用时从 add_turn 传入，
                               避免竞态条件下重新从 Redis 读取）

        返回值：None（结果写入 Redis）

        核心逻辑：
        1. 加锁（self._summarize_lock）防止同一会话并发生成多份摘要。
        2. 取出滑动窗口之前的旧 turns（old_turns = turns[:-WINDOW_SIZE]）。
        3. 读取已有摘要（作为"历史摘要"合并进去，实现滚动更新）。
        4. 取 old_turns 的最后 10 条拼接成文本（限制 10 条防止 prompt 过长，
           每次也只取 300 字符/条）。
        5. 构造摘要 prompt，调用 LLM 生成摘要。
        6. 将新摘要写回 Redis（同样设置 TTL）。

        为什么是"滚动"摘要：
        每次生成摘要时，将旧摘要作为上下文传入，让 LLM 合并新旧信息。
        这样摘要始终是最新的全局概要，不会因为每次只摘要最新旧对话而丢失
        之前摘要中的信息。

        异常处理：
        如果 LLM 调用失败（网络超时、API 限流等），只 log warning 不抛异常。
        因为摘要是优化性功能而非必须功能，失败不应影响主流程。
        """
        with self._summarize_lock:
            # 取窗口之外的旧对话
            old_turns = turns[: -WINDOW_SIZE]
            if not old_turns:
                return

            # 读取已有的历史摘要，用于滚动合并
            existing_summary = self._get_summary(session_id) or ""

            # 构建摘要 prompt 的输入文本：最多取 10 条旧对话，每条截断 300 字符
            old_text = "\n".join(
                f"{'用户' if t['role'] == 'user' else '助手'}：{t['content'][:300]}"
                for t in old_turns[-10:]
            )

            # 摘要 prompt：要求 LLM 保留关键决策、用户数据和话题
            prompt = f"""你是对话摘要助手。请将以下旧的对话记录压缩为一段简洁的摘要（不超过 150 字）。

{"已有的历史摘要：" + existing_summary if existing_summary else ""}

新的对话内容：
{old_text}

请输出合并后的摘要，只输出摘要文本，不要其他内容。摘要应包含：用户讨论了什么话题、做了哪些决策、有什么关键信息（身高体重目标等如果提到过要保留）。"""

            try:
                # 低 temperature 让摘要更稳定一致
                resp = self.llm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                # 截断到 300 字符，数据库存储不宜过大
                new_summary = resp.content.strip()[:300]
                self.redis.set(
                    f"conv:{session_id}:summary",
                    new_summary,
                    ex=SESSION_TTL,
                )
                logger.info(f"Summarized {len(old_turns)} old turns for session {session_id[:8]}...")
            except Exception as e:
                # 摘要失败不阻塞主流程，仅记录警告
                logger.warning(f"Summarization failed for {session_id[:8]}: {e}")
