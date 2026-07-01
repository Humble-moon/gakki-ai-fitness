import pytest
from src.storage.minio_client import MinioClient

@pytest.fixture
def minio():
    return MinioClient()

class TestMinioClient:
    def test_upload_and_get_json(self, minio):
        data = {"test": "hello", "nested": {"key": "value"}}
        minio.upload_json("test/plan_001.json", data)
        result = minio.get_json("test/plan_001.json")
        assert result["test"] == "hello"
        assert result["nested"]["key"] == "value"
