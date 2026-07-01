import json
from datetime import datetime
from src.storage.redis_client import RedisClient

class LongTermMemory:
    def __init__(self):
        self.redis = RedisClient()
        self.prefix = "memory:user:"

    def save_preference(self, user_id: int, key: str, value):
        self.redis.set(f"{self.prefix}{user_id}:pref:{key}", json.dumps(value))

    def get_preferences(self, user_id: int) -> dict:
        keys = self.redis.conn.keys(f"{self.prefix}{user_id}:pref:*")
        prefs = {}
        for k in keys:
            key_name = k.decode().split(":pref:")[-1]
            prefs[key_name] = json.loads(self.redis.get(k.decode()))
        return prefs

    def record_feedback(self, user_id: int, plan_id: str, rating: int, comment: str):
        feedback = {
            "plan_id": plan_id, "rating": rating, "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
        key = f"{self.prefix}{user_id}:feedback:{plan_id}"
        self.redis.set(key, json.dumps(feedback))

    def get_injury_history(self, user_id: int) -> list:
        data = self.redis.get(f"{self.prefix}{user_id}:pref:injuries")
        return json.loads(data) if data else []

    def build_context_for_prompt(self, user_id: int) -> str:
        prefs = self.get_preferences(user_id)
        injuries = self.get_injury_history(user_id)
        parts = []
        if prefs:
            parts.append(f"用户偏好：{prefs}")
        if injuries:
            parts.append(f"伤病史：{injuries}")
        return "\n".join(parts)
