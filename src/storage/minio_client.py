from minio import Minio
from src.config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY

class MinioClient:
    def __init__(self):
        self.client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        self._ensure_bucket("fitness-plans")

    def _ensure_bucket(self, name: str):
        if not self.client.bucket_exists(name):
            self.client.make_bucket(name)

    def upload_json(self, key: str, data: dict):
        import json
        from io import BytesIO
        content = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.client.put_object(
            "fitness-plans", key, BytesIO(content), len(content),
            content_type="application/json"
        )

    def get_json(self, key: str) -> dict:
        import json
        response = self.client.get_object("fitness-plans", key)
        return json.loads(response.read().decode("utf-8"))
