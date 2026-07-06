"""
redis_client.py - Redis 缓存操作客户端

角色：封装 Redis 连接和常用缓存操作，为 LLM 调用结果缓存和相似查询去重提供支持。
      在整个项目中作为"缓存加速层"——命中则直接返回，未命中则调用 LLM 后写入。
被调用者：core.orchestrator（编排器，用于缓存训练计划和动作分析结果）。
调用者：redis-py 库（Python Redis 客户端）。
"""
import redis
from src.config import REDIS_HOST, REDIS_PORT


class RedisClient:
    """
    Redis 客户端封装类

    职责：管理 Redis 连接，提供字符串和字节两种读写模式。
    使用场景：
        - 缓存 LLM 生成的训练计划（key=查询哈希, value=计划 JSON）
        - 缓存动作分析结果避免重复调用 LLM
        - 向量相似度缓存命中检测（配合 CACHE_SIMILARITY_THRESHOLD）
    设计要点：
        - decode_responses=False 表示不自动解码，由方法内部处理，
          这样既能存字符串也能存二进制（embedding 向量序列化数据）。
    """

    def __init__(self):
        """
        初始化 Redis 连接

        核心逻辑：
            使用配置中的 host 和 port 建立连接。
            decode_responses=False：保持原始字节存储，兼容字符串和二进制两种数据类型。
        """
        self.conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

    def get(self, key: str):
        """
        读取字符串类型的缓存值

        输入参数：
            key : str - 缓存键名
        返回值：
            str | bytes | None - 缓存值（自动将 bytes 解码为 UTF-8 字符串），键不存在时返回 None

        核心逻辑：
            1. 从 Redis 读取原始值（可能是 bytes）
            2. 如果是 bytes 类型，解码为 UTF-8 字符串返回
            3. 否则直接返回（可能是 None 或其他类型）
        """
        val = self.conn.get(key)
        if val and isinstance(val, bytes):
            return val.decode("utf-8")
        return val

    def set(self, key: str, value: str, ex: int = None):
        """
        写入字符串类型缓存，支持过期时间

        输入参数：
            key   : str  - 缓存键名
            value : str  - 要缓存的字符串值
            ex    : int  - 过期时间（秒），None 表示永不过期
        返回值：
            无

        核心逻辑：
            直接调用 Redis SET 命令写入键值对。
            ex 参数控制缓存自动过期，防止内存无限增长。
        """
        self.conn.set(key, value, ex=ex)

    def delete(self, key: str):
        """
        删除指定缓存键

        输入参数：
            key : str - 要删除的缓存键名
        返回值：
            无（Redis DEL 命令的结果被忽略）
        """
        self.conn.delete(key)

    def set_bytes(self, key: str, value: bytes):
        """
        写入二进制数据（用于存储序列化的 embedding 向量）

        输入参数：
            key   : str   - 缓存键名
            value : bytes - 二进制数据（通常是 pickle/numpy 序列化的向量）
        返回值：
            无

        使用场景：
            缓存 embedding 向量以加速向量相似度计算
        """
        self.conn.set(key, value)

    def get_bytes(self, key: str):
        """
        读取二进制缓存值

        输入参数：
            key : str - 缓存键名
        返回值：
            bytes | None - 原始字节数据，键不存在时返回 None

        使用场景：
            读取缓存的 embedding 向量进行反序列化
        """
        return self.conn.get(key)

    def flushdb(self):
        """
        清空当前 Redis 数据库的所有键

        返回值：
            无

        警告：此操作不可逆，用于开发/测试环境重置缓存，
              生产环境慎用！
        """
        self.conn.flushdb()

    def close(self):
        """
        关闭 Redis 连接

        说明：
            释放连接资源，应在应用关闭时调用。
        """
        self.conn.close()
