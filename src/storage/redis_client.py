import redis
from src.config import REDIS_HOST, REDIS_PORT

class RedisClient:
    def __init__(self):
        self.conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

    def get(self, key: str):
        val = self.conn.get(key)
        if val and isinstance(val, bytes):
            return val.decode("utf-8")
        return val

    def set(self, key: str, value: str, ex: int = None):
        self.conn.set(key, value, ex=ex)

    def delete(self, key: str):
        self.conn.delete(key)

    def set_bytes(self, key: str, value: bytes):
        self.conn.set(key, value)

    def get_bytes(self, key: str):
        return self.conn.get(key)

    def flushdb(self):
        self.conn.flushdb()

    def close(self):
        self.conn.close()
