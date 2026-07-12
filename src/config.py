"""
config.py - 全局配置中心

角色：集中管理项目中所有环境相关的配置项（API Key、数据库连接、模型参数等）。
      这是整个项目最底层的配置模块，所有其他模块通过 import 本文件获取配置。
被调用者：所有需要配置的模块（storage、models、rag、core 等）。
调用者：python-dotenv（.env 文件读取）+ os.getenv（环境变量读取）。

说明：每个配置项均优先从环境变量读取，未设置时使用开发环境的默认值。
"""
import os
from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件，将变量注入 os.environ
# 这样 os.getenv() 就能读取到 .env 中定义的值
load_dotenv()

# ============================================================
# LLM 多模型配置
# 支持多个模型提供商，通过 OpenAI 兼容 API 统一调用。
# 使用方式：
#   llm.chat(messages)                  → 使用默认模型
#   llm.chat(messages, model="fast")    → 使用 fast 模型
#   llm.chat(messages, model="deepseek-reasoner") → 使用指定模型名
#
# 模型配置在 .env 中定义，命名规则：LLM_<别名>_<属性>
#   例: LLM_FAST_MODEL=qwen-turbo → 别名 "fast" 的模型标识
#
# 默认模型（别名 "default"）必须有；其他别名可选。
# ============================================================
def _load_llm_configs() -> dict[str, dict]:
    """从环境变量加载所有 LLM 模型配置。

    .env 中的配置示例：
        # 默认模型（必配）
        LLM_DEFAULT_MODEL=deepseek-chat
        LLM_DEFAULT_BASE_URL=https://api.deepseek.com
        LLM_DEFAULT_API_KEY=sk-xxx

        # 推理模型（可选，同一 API 端点）
        LLM_REASONER_MODEL=deepseek-reasoner
        LLM_REASONER_BASE_URL=https://api.deepseek.com
        LLM_REASONER_API_KEY=sk-xxx

        # 快速模型（可选，不同提供商）
        LLM_FAST_MODEL=qwen-turbo
        LLM_FAST_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
        LLM_FAST_API_KEY=sk-yyy

    返回: {"default": {"model": "deepseek-chat", "base_url": "...", "api_key": "..."}, ...}
    """
    configs = {}
    for alias in _discover_llm_aliases():
        prefix = f"LLM_{alias.upper()}_"
        cfg = {
            "model": os.getenv(f"{prefix}MODEL", ""),
            "base_url": os.getenv(f"{prefix}BASE_URL", ""),
            "api_key": os.getenv(f"{prefix}API_KEY", ""),
        }
        if cfg["model"] and cfg["base_url"] and cfg["api_key"]:
            configs[alias] = cfg
    return configs


def _discover_llm_aliases() -> list[str]:
    """扫描环境变量，发现所有 LLM_<别名>_MODEL 配置的别名。"""
    aliases = set()
    for key in os.environ:
        if key.startswith("LLM_") and key.endswith("_MODEL"):
            # LLM_DEFAULT_MODEL → default, LLM_FAST_MODEL → fast
            alias = key[4:-6].lower()
            aliases.add(alias)
    return aliases


# 兼容旧配置：如果设置了 DEEPSEEK_API_KEY 但没有 LLM_DEFAULT_*，
# 自动构造 default 配置
_DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
_DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

LLM_CONFIGS = _load_llm_configs()

if not LLM_CONFIGS.get("default") and _DEEPSEEK_API_KEY:
    LLM_CONFIGS["default"] = {
        "model": "deepseek-chat",
        "base_url": _DEEPSEEK_BASE_URL,
        "api_key": _DEEPSEEK_API_KEY,
    }

# 向后兼容：保留旧变量名
DEEPSEEK_API_KEY = LLM_CONFIGS.get("default", {}).get("api_key",
                _DEEPSEEK_API_KEY)
DEEPSEEK_BASE_URL = LLM_CONFIGS.get("default", {}).get("base_url",
                _DEEPSEEK_BASE_URL)

# 默认模型名
LLM_DEFAULT_MODEL = LLM_CONFIGS.get("default", {}).get("model", "deepseek-chat")

# ============================================================
# PostgreSQL + pgvector 配置
# 关系型数据库：存储用户档案、训练计划、知识块等结构化数据
# pgvector 扩展：存储和检索向量嵌入（embedding）
# ============================================================
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")          # 数据库主机地址
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))          # 数据库端口
POSTGRES_USER = os.getenv("POSTGRES_USER", "ai_fitness")         # 数据库用户名
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fitness123") # 数据库密码
POSTGRES_DB = os.getenv("POSTGRES_DB", "fitness_assistant")      # 数据库名称

# 拼接 SQLAlchemy 连接字符串
# 格式：postgresql://用户名:密码@主机:端口/数据库名
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# ============================================================
# Neo4j 图数据库配置
# 图数据库：存储运动知识图谱（如动作-肌肉-器械之间的关系）
# ============================================================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")  # Neo4j Bolt 协议连接地址
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")                # Neo4j 用户名
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "fitness123")   # Neo4j 密码

# ============================================================
# Redis 缓存配置
# 内存缓存：缓存 LLM 调用结果、相似查询结果，加速重复请求
# ============================================================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")  # Redis 服务器地址
REDIS_PORT = int(os.getenv("REDIS_PORT", "6380"))  # Redis 端口（使用 6380 避免和默认 6379 冲突）

# ============================================================
# MinIO 对象存储配置
# 对象存储：存储生成的训练计划 JSON 文件，类似 AWS S3 的本地替代
# ============================================================
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")   # MinIO 服务地址
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")   # MinIO 访问密钥
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")   # MinIO 秘密密钥

# ============================================================
# Embedding（向量嵌入）配置
# 使用阿里云 DashScope Embedding API（OpenAI 兼容，国内直连稳定）
# 替换了原来的 BGE 本地模型（解决了 HuggingFace 下载超时问题）
# ============================================================
EMBEDDING_API_KEY = os.getenv("DASHSCOPE_API_KEY", DEEPSEEK_API_KEY)
EMBEDDING_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
# text-embedding-v4: 1024 维，中英文优化，DashScope 最新模型

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
# text-embedding-v4 的向量维度。注意：更换 embedding 模型后需重新摄入所有数据，
# 因为不同模型的向量空间不兼容。

# ============================================================
# 业务逻辑参数
# ============================================================
CACHE_SIMILARITY_THRESHOLD = 0.92
# 缓存命中阈值：当用户新查询和缓存查询的向量相似度 >= 0.92 时，
# 直接返回缓存结果，避免重复调用 LLM，节省 API 费用和响应时间

AGENTIC_RAG_MAX_RETRIES = 3
# Agentic RAG 最大重试次数：当检索结果质量不足时，
# 系统自动改写查询重新检索，最多重试 3 次

HITL_CONFIDENCE_THRESHOLD = 0.7
# 人工介入（Human-in-the-Loop）阈值：
# 当 AI 对分析结果的置信度 < 0.7 时，标记为需要人工审核

REWRITE_MODEL = os.getenv("REWRITE_MODEL", "deepseek-chat")
# 查询改写专用模型。Agentic RAG 的检索评估 + 查询改写环节使用此模型。
# 设计意图：改写是高频低成本操作，不需要大模型，用小的便宜模型即可。
# 默认 deepseek-chat（DeepSeek 目前最小可用模型），后续可换为更便宜的模型。
# 用法：配置独立的 LLM_REWRITE_* 或在 .env 设 REWRITE_MODEL=deepseek-chat
