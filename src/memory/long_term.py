"""
================================================================================
文件角色：长期记忆管理器（LongTermMemory）
================================================================================
- 被调用者：编排引擎在构建 Prompt 时调用 build_context_for_prompt() 获取
  用户偏好和伤病史，注入 Planner 或 Writer 的 user message 中。
- 调用者：本模块仅依赖 RedisClient 做 key-value 持久化。
- 项目角色：实现跨 session 的"用户画像记忆"——用户的偏好（目标、器械、水平）
  和伤病史不会随会话过期而丢失，是持久化的用户数据层。

与 ConversationManager 的区别：
- ConversationManager 是短期记忆（24h TTL，会话级），记住"刚才聊了什么"。
- LongTermMemory 是长期记忆（无 TTL，用户级），记住"用户是什么样的人"。
================================================================================
"""

import json
from datetime import datetime
from src.storage.redis_client import RedisClient


class LongTermMemory:
    """
    长期记忆管理器，存储跨会话的用户画像信息。

    职责：
    - 保存/读取用户偏好（preferences）：目标、训练水平、可用器械等。
    - 记录用户对计划的反馈（feedback）：评分、评论。
    - 管理伤病史（injury history）：影响动作选择的硬约束。
    - 将以上信息格式化为 Prompt 上下文片段。

    Redis Key 设计说明：
    - memory:user:{user_id}:pref:{key}   → 单个偏好键值
    - memory:user:{user_id}:feedback:{plan_id} → 单条反馈记录
    前缀 memory:user: 将长期记忆与 ConversationManager 的 conv: 前缀区分开，
    便于运维时按前缀分类管理（如批量清理会话数据不影响长期记忆）。
    """

    def __init__(self):
        """
        初始化长期记忆管理器。

        核心逻辑：
        1. 连接 Redis。
        2. 设置 key 前缀为 "memory:user:"，所有 Redis 操作都基于此前缀拼接。
        """
        self.redis = RedisClient()
        self.prefix = "memory:user:"

    def save_preference(self, user_id: int, key: str, value):
        """
        保存用户的一个偏好项。

        参数：
            user_id: int  - 用户唯一 ID
            key: str      - 偏好键名，如 "goal"（目标）、"level"（训练水平）、
                           "equipment"（可用器械）、"injuries"（伤病史）
            value: any    - 偏好值，可以是 str / list / dict，内部自动 json.dumps

        返回值：None

        核心逻辑：
        构造 Redis key: memory:user:{user_id}:pref:{key}
        将 value JSON 序列化后存入 Redis（无 TTL，永久存储直到用户主动更新）。

        使用场景：
        - 用户首次注册时填写健身目标 → save_preference(uid, "goal", "增肌")
        - 用户更新可用器械 → save_preference(uid, "equipment", ["哑铃", "杠铃"])
        """
        self.redis.set(f"{self.prefix}{user_id}:pref:{key}", json.dumps(value))

    def get_preferences(self, user_id: int) -> dict:
        """
        读取用户的所有偏好项，组装为字典。

        参数：
            user_id: int  - 用户唯一 ID

        返回值：
            dict           - 所有偏好键值对，如 {"goal": "增肌", "level": "初级"}
                            没有偏好时返回空字典 {}

        核心逻辑：
        1. 用 Redis keys 命令扫描 memory:user:{user_id}:pref:* 所有 key。
           注意：keys 命令在数据量极大时可能阻塞 Redis，当前场景下用户数有限，
           可以接受。未来如果用户量大可改用 scan 迭代。
        2. 从 key 名中提取偏好 key 名（去掉前缀和 :pref: 后的部分）。
        3. 从 Redis 取值并 JSON 反序列化，组装为字典。
        """
        # 为什么要 decode：redis-py 返回的 key 是 bytes 类型，需要 decode 为 str
        # 注意：keys() 在大型生产环境应改为 scan_iter() 避免阻塞
        keys = self.redis.conn.keys(f"{self.prefix}{user_id}:pref:*")
        prefs = {}
        for k in keys:
            key_name = k.decode().split(":pref:")[-1]  # 从完整 key 中提取偏好名
            prefs[key_name] = json.loads(self.redis.get(k.decode()))
        return prefs

    def record_feedback(self, user_id: int, plan_id: str, rating: int, comment: str):
        """
        记录用户对某个训练计划的反馈。

        参数：
            user_id: int  - 用户唯一 ID
            plan_id: str  - 训练计划的唯一标识
            rating: int   - 评分（如 1-5 分）
            comment: str  - 文字反馈

        返回值：None

        核心逻辑：
        构造包含 plan_id / rating / comment / timestamp 的反馈字典，
        JSON 序列化后写入 Redis。
        Key 格式：memory:user:{user_id}:feedback:{plan_id}，
        每个计划只保留最新一条反馈（后写覆盖先写）。

        使用场景：
        用户对 AI 生成的计划打分/评论后调用，用于后续评估 AI 生成质量、
        迭代优化 Prompt。
        """
        feedback = {
            "plan_id": plan_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now().isoformat()  # 本地时间，用于反馈排序
        }
        key = f"{self.prefix}{user_id}:feedback:{plan_id}"
        self.redis.set(key, json.dumps(feedback))

    def get_injury_history(self, user_id: int) -> list:
        """
        读取用户的伤病史列表。

        参数：
            user_id: int  - 用户唯一 ID

        返回值：
            list           - 伤病史列表，如 ["肩袖损伤", "腰椎间盘突出"]
                            Redis 中没有数据时返回空列表 []

        核心逻辑：
        从固定 key memory:user:{user_id}:pref:injuries 读取，
        JSON 反序列化为列表。伤病史是硬约束，FactChecker 会据此标记危险动作。
        """
        data = self.redis.get(f"{self.prefix}{user_id}:pref:injuries")
        return json.loads(data) if data else []

    def build_context_for_prompt(self, user_id: int) -> str:
        """
        构建可注入 Prompt 的用户画像上下文文本。

        参数：
            user_id: int  - 用户唯一 ID

        返回值：
            str            - 格式化的用户画像文本，如：
                            "用户偏好：{'goal': '增肌', 'level': '初级'}
                            伤病史：['肩袖损伤']"
                            没有数据时返回空串 ""

        核心逻辑：
        1. 调用 get_preferences() 和 get_injury_history() 获取全部用户信息。
        2. 如果有偏好，添加 "用户偏好：..." 行。
        3. 如果有伤病史，添加 "伤病史：..." 行。
        4. 用换行符拼接。

        使用场景：
        编排引擎在构建 Planner / Writer 的 user message 时，
        调用此方法获取用户画像字符串，拼接到 prompt 中。
        """
        prefs = self.get_preferences(user_id)
        injuries = self.get_injury_history(user_id)
        parts = []
        if prefs:
            parts.append(f"用户偏好：{prefs}")
        if injuries:
            parts.append(f"伤病史：{injuries}")
        return "\n".join(parts)
