"""
minio_client.py - MinIO 对象存储操作客户端

角色：封装 MinIO SDK，提供 JSON 文件的上传和下载能力。
      MinIO 在图项目中作为"训练计划持久化存储"，将生成的训练计划以 JSON 格式存入对象存储。
被调用者：core.orchestrator（编排器，保存生成的训练计划）。
调用者：minio-py SDK（官方 Python MinIO 客户端库）。

说明：MinIO 是 S3 兼容的对象存储，这里可以无缝替换为 AWS S3 或阿里云 OSS。
"""
from minio import Minio
from src.config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY


class MinioClient:
    """
    MinIO 对象存储客户端封装类

    职责：管理 MinIO 连接，确保存储桶存在，提供 JSON 文件的上传和下载。
    使用场景：
        - 在生成训练计划后，将 plan_data 持久化到 MinIO
        - 后续可从 MinIO 读取历史训练计划做对比分析
    设计要点：
        - 构造时自动确保 "fitness-plans" 桶存在（_ensure_bucket）
        - secure=False 表示使用 HTTP 而非 HTTPS（本地开发环境）
    """

    def __init__(self):
        """
        初始化 MinIO 客户端并确保存储桶存在

        核心逻辑：
            1. 使用配置中的 endpoint、access_key、secret_key 创建 MinIO 客户端
            2. secure=False：本地开发使用 HTTP 协议
            3. 调用 _ensure_bucket 确保 "fitness-plans" 桶已创建
        """
        self.client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        # 自动创建存储桶（幂等操作，桶存在则跳过）
        self._ensure_bucket("fitness-plans")

    def _ensure_bucket(self, name: str):
        """
        确保指定名称的存储桶存在，不存在则创建

        输入参数：
            name : str - 存储桶名称
        返回值：
            无

        核心逻辑：
            检查桶是否存在，不存在则调用 make_bucket 创建。
            该方法是幂等的——重复调用不会出错也不会重复创建。

        说明：以 _ 开头表示这是内部方法，外部不应直接调用。
        """
        if not self.client.bucket_exists(name):
            self.client.make_bucket(name)

    def upload_json(self, key: str, data: dict):
        """
        将 Python 字典以 JSON 格式上传到 MinIO

        输入参数：
            key  : str  - 对象键名（相当于文件名路径，如 "plans/user_1_20240702.json"）
            data : dict - 要上传的 Python 字典数据
        返回值：
            无

        核心逻辑：
            1. 将 dict 序列化为 JSON 字符串（ensure_ascii=False 保留中文）
            2. 编码为 UTF-8 字节
            3. 包装为 BytesIO 内存流
            4. 调用 put_object 上传到 "fitness-plans" 桶

        说明：使用 BytesIO 而非写临时文件，避免磁盘 IO 开销。
        """
        import json
        from io import BytesIO
        # 序列化：ensure_ascii=False 确保中文不会被转义为 \uXXXX
        content = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.client.put_object(
            "fitness-plans", key, BytesIO(content), len(content),
            content_type="application/json"
        )

    def get_json(self, key: str) -> dict:
        """
        从 MinIO 下载 JSON 文件并解析为 Python 字典

        输入参数：
            key : str - 对象键名
        返回值：
            dict - 解析后的 JSON 数据

        核心逻辑：
            1. 调用 get_object 从 "fitness-plans" 桶获取对象
            2. 读取响应体的字节内容
            3. UTF-8 解码
            4. json.loads 解析为 Python dict

        说明：该操作是同步的，会阻塞直到下载完成。
        """
        import json
        response = self.client.get_object("fitness-plans", key)
        return json.loads(response.read().decode("utf-8"))
