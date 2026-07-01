# Gakki AI Fitness 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 AI 健身私教应用，Multi-Agent 协作生成个性化训练计划并分析动作质量

**Architecture:** Streamlit 前端 → Orchestrator 状态机 → 4 Agent 流水线（Planner/Retriever/Writer/FactChecker）→ RAG + GraphRAG + MCP 服务层 → PostgreSQL/Neo4j/Redis/MinIO 四存储

**Tech Stack:** Python 3.11, Streamlit, LangChain, pgvector, Neo4j, Redis Stack, MinIO, FAISS, BGE-small-zh, DeepSeek-V3

---

## 文件结构

```
gakki-ai-fitness/
├── docker-compose.yml
├── requirements.txt
├── .env
├── .env.example
├── app/
│   └── streamlit_app.py
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner.py
│   │   ├── retriever.py
│   │   ├── writer.py
│   │   └── fact_checker.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── harness.py
│   │   └── model_router.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   ├── vector_search.py
│   │   ├── keyword_search.py
│   │   ├── agentic_rag.py
│   │   └── semantic_cache.py
│   ├── graphrag/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   └── search.py
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── exercise_server.py
│   │   └── tool_registry.py
│   ├── a2a/
│   │   ├── __init__.py
│   │   └── messaging.py
│   ├── memory/
│   │   ├── __init__.py
│   │   └── long_term.py
│   ├── hitl/
│   │   ├── __init__.py
│   │   └── review.py
│   ├── skills/
│   │   ├── __init__.py
│   │   └── registry.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── provider.py
│   │   └── prompts/
│   │       ├── __init__.py
│   │       ├── planner.py
│   │       ├── retriever.py
│   │       ├── writer.py
│   │       └── fact_checker.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── pg.py
│   │   ├── redis_client.py
│   │   ├── neo4j_client.py
│   │   └── minio_client.py
│   └── models/
│       ├── __init__.py
│       ├── db_models.py
│       └── schemas.py
├── skills/
│   ├── muscle_building.md
│   ├── fat_loss.md
│   └── exercise_analysis.md
├── data/
│   └── seed_exercises.json
├── eval/
│   ├── __init__.py
│   ├── test_queries.json
│   └── eval_runner.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_agents/
    ├── test_core/
    ├── test_rag/
    ├── test_graphrag/
    └── test_storage/
```

---

## Phase 1: 项目脚手架

### Task 1: 项目初始化

**Files:** Create: `.env.example`, `.env`, `requirements.txt`, `docker-compose.yml`, `src/__init__.py`, `src/config.py`

- [ ] **Step 1: 创建 docker-compose.yml**

从研报助手复用 Docker 配置：
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    ports: ["5432:5432"]
    environment:
      - POSTGRES_USER=ai_fitness
      - POSTGRES_PASSWORD=fitness123
      - POSTGRES_DB=fitness_assistant
    volumes:
      - pg_data:/var/lib/postgresql/data

  neo4j:
    image: neo4j:5.20
    ports: ["7474:7474", "7687:7687"]
    environment:
      - NEO4J_AUTH=neo4j/fitness123
      - NEO4J_server_memory_heap_initial__size=512m
      - NEO4J_server_memory_heap_max__size=1g
    volumes:
      - neo4j_data:/data

  redis:
    image: redis/redis-stack:latest
    ports: ["6380:6379", "8002:8001"]
    volumes:
      - redis_data:/data

  minio:
    image: minio/minio:latest
    ports: ["9000:9000", "9001:9001"]
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_data:/data

volumes:
  pg_data:
  neo4j_data:
  redis_data:
  minio_data:
```

- [ ] **Step 2: 创建 requirements.txt**

```
streamlit>=1.28.0
langchain>=0.3.0
langchain-community>=0.3.0
openai>=1.50.0
psycopg2-binary>=2.9.9
pgvector>=0.3.0
neo4j>=5.20.0
redis>=5.0.0
minio>=7.2.0
faiss-cpu>=1.8.0
sentence-transformers>=2.7.0
pydantic>=2.0.0
sqlalchemy>=2.0.0
python-dotenv>=1.0.0
numpy>=1.26.0
tiktoken>=0.7.0
```

- [ ] **Step 3: 创建 .env.example**

```
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
CLAUDE_API_KEY=optional
HF_ENDPOINT=https://hf-mirror.com
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=ai_fitness
POSTGRES_PASSWORD=fitness123
POSTGRES_DB=fitness_assistant
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=fitness123
REDIS_HOST=localhost
REDIS_PORT=6380
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

- [ ] **Step 4: 创建 src/config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "ai_fitness")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fitness123")
POSTGRES_DB = os.getenv("POSTGRES_DB", "fitness_assistant")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "fitness123")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6380"))
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
EMBEDDING_MODEL = "BAAI/bge-small-zh"
EMBEDDING_DIM = 512
CACHE_SIMILARITY_THRESHOLD = 0.92
AGENTIC_RAG_MAX_RETRIES = 3
HITL_CONFIDENCE_THRESHOLD = 0.7
```

- [ ] **Step 5: Commit**

```bash
cd /e/gakki-ai-fitness
git init
git add -A
git commit -m "feat: project scaffold with docker-compose, config, requirements"
```

---

### Task 2: 数据模型

**Files:** Create: `src/models/__init__.py`, `src/models/db_models.py`, `src/models/schemas.py`

- [ ] **Step 1: 创建 SQLAlchemy 模型 — `src/models/db_models.py`**

```python
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from pgvector.sqlalchemy import Vector
from src.config import DATABASE_URL, EMBEDDING_DIM
from datetime import datetime

class Base(DeclarativeBase):
    pass

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    height = Column(Float, nullable=False)
    weight = Column(Float, nullable=False)
    training_years = Column(Float, nullable=False)
    goal = Column(String(20), nullable=False)  # "增肌" / "减脂"
    available_equipment = Column(JSON, nullable=False)  # ["哑铃", "杠铃", "绳索"]
    days_per_week = Column(Integer, nullable=False)
    injuries = Column(JSON, default=[])  # ["下背痛", "肩伤"]
    preferences = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Exercise(Base):
    __tablename__ = "exercises"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    name_en = Column(String(100))
    exercise_type = Column(String(20))  # "复合" / "孤立"
    difficulty = Column(String(10))  # "初级" / "中级" / "高级"
    equipment = Column(String(50))
    target_muscles = Column(JSON)  # ["胸大肌", "三角肌前束"]
    description = Column(Text)
    common_errors = Column(JSON)  # ["手肘打得太开", "..."],
    embedding = Column(Vector(EMBEDDING_DIM))

class TrainingPlan(Base):
    __tablename__ = "training_plans"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    goal = Column(String(20))
    plan_data = Column(JSON, nullable=False)
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)
```

- [ ] **Step 2: 创建 Pydantic schemas — `src/models/schemas.py`**

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class UserProfileInput(BaseModel):
    height: float = Field(..., ge=100, le=250, description="身高(cm)")
    weight: float = Field(..., ge=30, le=200, description="体重(kg)")
    training_years: float = Field(..., ge=0, le=30, description="训练年限")
    goal: str = Field(..., pattern="^(增肌|减脂)$")
    available_equipment: List[str] = Field(..., min_length=1)
    days_per_week: int = Field(..., ge=1, le=7)
    injuries: List[str] = Field(default=[])
    preferences: dict = Field(default={})

class ExerciseItem(BaseModel):
    name: str
    sets: int = Field(..., ge=1, le=10)
    reps: str  # "8-12"
    rest: str  # "90s"
    notes: str = ""

class TrainingDay(BaseModel):
    day: int
    focus: str  # "胸+三头"
    exercises: List[ExerciseItem]

class TrainingPlanOutput(BaseModel):
    plan_id: str
    user_id: int
    goal: str
    weeks: int
    sessions_per_week: int
    days: List[TrainingDay]
    warnings: List[str] = []

class ExerciseAnalysisInput(BaseModel):
    exercise_name: str
    user_description: str  # 用户描述的训练感受
    user_level: str = "中级"

class ExerciseAnalysisOutput(BaseModel):
    exercise_name: str
    issues_found: List[str]
    severity: str  # "安全" / "注意" / "警告"
    suggestions: List[str]
    confidence: float

class PlanRequest(BaseModel):
    user_profile: UserProfileInput
    query: str = ""

class AnalysisRequest(BaseModel):
    analysis: ExerciseAnalysisInput

class SearchResult(BaseModel):
    content: str
    score: float
    source: str  # "vector" / "keyword" / "graph"
    metadata: dict = {}
```

- [ ] **Step 3: 创建测试 — `tests/test_models.py`**

```python
import pytest
from src.models.schemas import UserProfileInput, ExerciseItem, TrainingDay, TrainingPlanOutput

class TestUserProfileInput:
    def test_valid_profile(self):
        p = UserProfileInput(
            height=180, weight=80, training_years=1.5,
            goal="增肌", available_equipment=["哑铃", "杠铃"],
            days_per_week=4, injuries=["下背痛"]
        )
        assert p.height == 180
        assert p.goal == "增肌"

    def test_invalid_goal_raises(self):
        with pytest.raises(Exception):
            UserProfileInput(
                height=180, weight=80, training_years=1,
                goal="塑形", available_equipment=["哑铃"], days_per_week=3
            )

    def test_height_out_of_range_raises(self):
        with pytest.raises(Exception):
            UserProfileInput(
                height=50, weight=80, training_years=1,
                goal="增肌", available_equipment=["哑铃"], days_per_week=3
            )

class TestTrainingPlanOutput:
    def test_valid_plan(self):
        plan = TrainingPlanOutput(
            plan_id="abc-123", user_id=1, goal="增肌",
            weeks=4, sessions_per_week=4,
            days=[
                TrainingDay(day=1, focus="胸+三头", exercises=[
                    ExerciseItem(name="哑铃卧推", sets=4, reps="8-12", rest="90s")
                ])
            ],
            warnings=[]
        )
        assert plan.weeks == 4
        assert len(plan.days[0].exercises) == 1
```

- [ ] **Step 4: 运行测试验证失败（数据库模型测试暂跳过，等 PG 启动）**

```bash
cd /e/gakki-ai-fitness
pip install pydantic pytest
python -m pytest tests/test_models.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/ tests/test_models.py
git commit -m "feat: add data models (SQLAlchemy + Pydantic schemas)"
```

---

## Phase 2: 存储客户端

### Task 3: PostgreSQL 客户端

**Files:** Create: `src/storage/__init__.py`, `src/storage/pg.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_storage/__init__.py`, `tests/test_storage/test_pg.py`

- [ ] **Step 1: 编写测试 — `tests/test_storage/test_pg.py`**

```python
import pytest
from src.storage.pg import PGClient

@pytest.fixture
def pg():
    client = PGClient()
    yield client
    client.close()

class TestPGClient:
    def test_connection(self, pg):
        assert pg.engine is not None

    def test_insert_and_search_exercise(self, pg):
        from src.models.db_models import Exercise, init_db
        init_db()
        # 插入测试动作（无 embedding）
        pg.execute("DELETE FROM exercises WHERE name = '测试动作'")
        result = pg.execute(
            "INSERT INTO exercises (name, exercise_type, difficulty, equipment, target_muscles) "
            "VALUES ('测试动作', '复合', '初级', '哑铃', '[\"胸大肌\"]')"
        )
        assert result is not None
        # 搜索
        rows = pg.fetch_all("SELECT * FROM exercises WHERE name = '测试动作'")
        assert len(rows) == 1
        assert rows[0][1] == '测试动作'
```

- [ ] **Step 2: 实现 — `src/storage/pg.py`**

```python
from sqlalchemy import text
from src.models.db_models import engine, SessionLocal

class PGClient:
    def __init__(self):
        self.engine = engine

    def execute(self, query: str, params: dict = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            conn.commit()
            return result

    def fetch_all(self, query: str, params: dict = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchall()

    def fetch_one(self, query: str, params: dict = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchone()

    def get_session(self):
        return SessionLocal()

    def close(self):
        pass
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_storage/test_pg.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/storage/ tests/test_storage/
git commit -m "feat: add PostgreSQL client"
```

---

### Task 4: Redis 客户端

**Files:** Create: `src/storage/redis_client.py`, `tests/test_storage/test_redis.py`

- [ ] **Step 1: 编写测试 — `tests/test_storage/test_redis.py`**

```python
import pytest
from src.storage.redis_client import RedisClient

@pytest.fixture
def redis():
    client = RedisClient()
    yield client
    client.flushdb()
    client.close()

class TestRedisClient:
    def test_set_and_get(self, redis):
        redis.set("test_key", "hello")
        assert redis.get("test_key") == "hello"

    def test_cache_json(self, redis):
        import json
        data = {"plan": "增肌计划", "exercises": ["卧推", "深蹲"]}
        redis.set("cache:plan:1", json.dumps(data))
        result = json.loads(redis.get("cache:plan:1"))
        assert result["plan"] == "增肌计划"
        assert len(result["exercises"]) == 2

    def test_delete(self, redis):
        redis.set("temp", "val")
        redis.delete("temp")
        assert redis.get("temp") is None

    def test_vector_search(self, redis):
        import numpy as np
        vec = np.random.rand(512).astype(np.float32).tobytes()
        redis.conn.set("vec:test", vec)
        stored = redis.conn.get("vec:test")
        assert stored is not None
```

- [ ] **Step 2: 实现 — `src/storage/redis_client.py`**

```python
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
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_storage/test_redis.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/storage/redis_client.py tests/test_storage/test_redis.py
git commit -m "feat: add Redis client"
```

---

### Task 5: Neo4j 客户端

**Files:** Create: `src/storage/neo4j_client.py`, `tests/test_storage/test_neo4j.py`

- [ ] **Step 1: 实现 — `src/storage/neo4j_client.py`**

```python
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def run(self, query: str, params: dict = None):
        with self.driver.session() as session:
            return session.run(query, params or {})

    def query(self, query: str, params: dict = None):
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def close(self):
        self.driver.close()
```

- [ ] **Step 2: 编写测试 — `tests/test_storage/test_neo4j.py`**

```python
import pytest
from src.storage.neo4j_client import Neo4jClient

@pytest.fixture
def neo4j():
    client = Neo4jClient()
    yield client
    client.run("MATCH (n:TestNode) DETACH DELETE n")
    client.close()

class TestNeo4jClient:
    def test_connection_and_query(self, neo4j):
        neo4j.run("CREATE (:TestNode {name: 'test_exercise'})")
        results = neo4j.query("MATCH (n:TestNode) RETURN n.name AS name")
        assert len(results) > 0
        assert results[0]["name"] == "test_exercise"
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_storage/test_neo4j.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/storage/neo4j_client.py tests/test_storage/test_neo4j.py
git commit -m "feat: add Neo4j client"
```

---

### Task 6: MinIO 客户端

**Files:** Create: `src/storage/minio_client.py`, `tests/test_storage/test_minio.py`

- [ ] **Step 1: 实现 — `src/storage/minio_client.py`**

```python
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
        content = json.dumps(data, ensure_ascii=False).encode("utf-8")
        from io import BytesIO
        self.client.put_object(
            "fitness-plans", key, BytesIO(content), len(content),
            content_type="application/json"
        )

    def get_json(self, key: str) -> dict:
        import json
        response = self.client.get_object("fitness-plans", key)
        return json.loads(response.read().decode("utf-8"))
```

- [ ] **Step 2: 编写测试 — `tests/test_storage/test_minio.py`**

```python
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
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_storage/test_minio.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/storage/minio_client.py tests/test_storage/test_minio.py
git commit -m "feat: add MinIO client"
```

---

## Phase 3: LLM 层

### Task 7: LLM Provider

**Files:** Create: `src/llm/__init__.py`, `src/llm/provider.py`

- [ ] **Step 1: 编写测试 — `tests/test_llm.py`**

```python
import pytest
from src.llm.provider import LLMProvider, LLMResponse

class TestLLMResponse:
    def test_response_model(self):
        resp = LLMResponse(content="测试回复", model="deepseek-v3", tokens=100)
        assert resp.content == "测试回复"
        assert resp.model == "deepseek-v3"

class TestLLMProvider:
    def test_chat_sync(self):
        provider = LLMProvider()
        resp = provider.chat(
            messages=[{"role": "user", "content": "说'你好'"}],
            temperature=0.1
        )
        assert "你好" in resp.content
        assert resp.model is not None
```

- [ ] **Step 2: 实现 — `src/llm/provider.py`**

```python
from dataclasses import dataclass
from openai import OpenAI
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

@dataclass
class LLMResponse:
    content: str
    model: str
    tokens: int

class LLMProvider:
    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        self.default_model = "deepseek-chat"

    def chat(self, messages: list, temperature: float = 0.3, model: str = None) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature
        )
        return LLMResponse(
            content=resp.choices[0].message.content,
            model=resp.model,
            tokens=resp.usage.total_tokens
        )

    def chat_with_json_mode(self, messages: list, model: str = None) -> dict:
        import json
        resp = self.chat(messages, temperature=0.1, model=model)
        try:
            # Try to extract JSON from markdown code block or raw
            content = resp.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {"raw": resp.content}
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_llm.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/llm/ tests/test_llm.py
git commit -m "feat: add LLM provider with DeepSeek"
```

---

### Task 8: Prompt 模板

**Files:** Create: `src/llm/prompts/__init__.py`, `src/llm/prompts/planner.py`, `src/llm/prompts/retriever.py`, `src/llm/prompts/writer.py`, `src/llm/prompts/fact_checker.py`

- [ ] **Step 1: 创建 Planner prompt — `src/llm/prompts/planner.py`**

```python
PLANNER_SYSTEM = """你是健身训练计划编排专家。根据用户的身体数据、训练目标和可用器械，分解任务并决定需要检索哪些信息。

输出 JSON 格式：
{
  "subtasks": ["检索推类动作", "检索拉类动作", "检索腿部动作"],
  "retrieval_strategy": "vector" | "keyword" | "graph" | "all",
  "output_format": "增肌计划" | "减脂计划" | "动作分析",
  "constraints": ["仅哑铃动作", "排除肩伤风险动作"]
}
"""

def build_planner_messages(user_input: str, profile: dict) -> list:
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"用户信息：{profile}\n用户请求：{user_input}"}
    ]
```

- [ ] **Step 2: 创建 Retriever prompt — `src/llm/prompts/retriever.py`**

```python
RETRIEVER_SYSTEM = """你是动作检索专家。根据 Planner 的指令，评估检索结果的质量。

判断标准：
- 检索到的动作是否匹配用户需求（目标肌肉、器械、难度）
- 结果是否足够全面（推/拉/腿各方向覆盖）
- 是否有明显的安全隐患（伤病冲突）

输出 JSON：
{
  "quality_score": 0.0-1.0,
  "missing_aspects": ["缺少肩部推举动作"],
  "rewritten_query": "优化后的查询词" | null
}
"""

def build_retriever_eval_messages(original_query: str, results: list) -> list:
    return [
        {"role": "system", "content": RETRIEVER_SYSTEM},
        {"role": "user", "content": f"查询：{original_query}\n检索结果：{results}"}
    ]
```

- [ ] **Step 3: 创建 Writer prompt — `src/llm/prompts/writer.py`**

```python
WRITER_SYSTEM = """你是训练计划编写专家。根据检索到的动作库和用户情况，生成结构化训练计划。

增肌计划参数：
- Rep Range: 6-12
- 组间休息: 60-90s
- 每部位每周 10-20 组

减脂计划参数：
- Rep Range: 12-15
- 组间休息: 30-60s
- 可加入超级组/HIIT

输出 JSON 必须符合 TrainingPlanOutput Schema。每个动作必须来自检索结果，不得编造。
"""

def build_writer_messages(retrieved_exercises: list, profile: dict, goal: str) -> list:
    return [
        {"role": "system", "content": WRITER_SYSTEM},
        {"role": "user", "content": f"目标：{goal}\n用户画像：{profile}\n可用动作：{retrieved_exercises}"}
    ]
```

- [ ] **Step 4: 创建 FactChecker prompt — `src/llm/prompts/fact_checker.py`**

```python
FACTCHECKER_SYSTEM = """你是训练安全审查专家。校验生成的训练计划是否安全合理。

检查项：
1. 动作难度是否匹配用户水平（初学者不推荐大重量自由重量动作）
2. 训练量是否合理（单次最多 20 组，每周每部位最多 25 组）
3. 是否存在已知伤病风险动作
4. 器械约束是否满足

输出 JSON：
{
  "is_safe": true | false,
  "issues": [{"exercise": "杠铃深蹲", "issue": "用户有下背伤史，建议改为高脚杯深蹲", "severity": "warning"}],
  "confidence": 0.0-1.0,
  "requires_human_review": true | false
}
"""

def build_fact_checker_messages(plan: dict, profile: dict) -> list:
    return [
        {"role": "system", "content": FACTCHECKER_SYSTEM},
        {"role": "user", "content": f"训练计划：{plan}\n用户画像：{profile}"}
    ]
```

- [ ] **Step 5: 运行 prompt 构建测试**

```python
# tests/test_prompts.py
import pytest
from src.llm.prompts.planner import build_planner_messages
from src.llm.prompts.fact_checker import build_fact_checker_messages

def test_planner_prompt_includes_profile():
    msgs = build_planner_messages("想增肌", {"height": 180, "goal": "增肌"})
    assert "增肌" in msgs[1]["content"]
    assert msgs[0]["role"] == "system"

def test_factchecker_prompt_includes_plan():
    plan = {"days": [{"day": 1, "exercises": []}]}
    msgs = build_fact_checker_messages(plan, {"injuries": ["下背痛"]})
    assert "下背痛" in msgs[1]["content"]
```

- [ ] **Step 6: Commit**

```bash
git add src/llm/prompts/ tests/test_prompts.py
git commit -m "feat: add prompt templates for all 4 agents"
```

---

## Phase 4: RAG 层

### Task 9: Embedding 服务

**Files:** Create: `src/rag/__init__.py`, `src/rag/embedding.py`, `tests/test_rag/__init__.py`, `tests/test_rag/test_embedding.py`

- [ ] **Step 1: 编写测试 — `tests/test_rag/test_embedding.py`**

```python
import pytest
import numpy as np
from src.rag.embedding import EmbeddingService
from src.config import EMBEDDING_DIM

@pytest.fixture
def emb():
    return EmbeddingService()

class TestEmbeddingService:
    def test_embed_query_returns_correct_dim(self, emb):
        vec = emb.embed("哑铃卧推")
        assert len(vec) == EMBEDDING_DIM
        assert isinstance(vec[0], float)

    def test_embed_batch(self, emb):
        texts = ["深蹲", "硬拉", "卧推"]
        vectors = emb.embed_batch(texts)
        assert len(vectors) == 3
        assert all(len(v) == EMBEDDING_DIM for v in vectors)

    def test_similarity_same_text(self, emb):
        v1 = emb.embed("增肌训练")
        v2 = emb.embed("增肌训练")
        sim = emb.similarity(v1, v2)
        assert sim > 0.95

    def test_similarity_different_text(self, emb):
        v1 = emb.embed("增肌训练")
        v2 = emb.embed("有氧减脂")
        sim = emb.similarity(v1, v2)
        assert sim < 0.95
```

- [ ] **Step 2: 实现 — `src/rag/embedding.py`**

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import EMBEDDING_MODEL

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, text: str) -> list:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list) -> list:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    def similarity(self, vec1: list, vec2: list) -> float:
        return float(np.dot(vec1, vec2))
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_rag/test_embedding.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/rag/ tests/test_rag/
git commit -m "feat: add embedding service with BGE-small-zh"
```

---

### Task 10: 向量检索

**Files:** Create: `src/rag/vector_search.py`, `tests/test_rag/test_vector_search.py`

- [ ] **Step 1: 实现 — `src/rag/vector_search.py`**

```python
from src.storage.pg import PGClient
from src.rag.embedding import EmbeddingService

class VectorSearch:
    def __init__(self):
        self.pg = PGClient()
        self.emb = EmbeddingService()

    def search(self, query: str, top_k: int = 10, filters: dict = None) -> list:
        vec = self.emb.embed(query)
        vec_str = f"[{','.join(str(v) for v in vec)}]"
        filter_clause = ""
        if filters:
            if "equipment" in filters:
                filter_clause = f"AND equipment = '{filters['equipment']}'"
        sql = f"""
            SELECT name, name_en, exercise_type, difficulty, equipment,
                   target_muscles, description, common_errors,
                   1 - (embedding <=> '{vec_str}'::vector) AS similarity
            FROM exercises
            WHERE embedding IS NOT NULL {filter_clause}
            ORDER BY embedding <=> '{vec_str}'::vector
            LIMIT {top_k}
        """
        rows = self.pg.fetch_all(sql)
        return [
            {"name": r[0], "name_en": r[1], "type": r[2], "difficulty": r[3],
             "equipment": r[4], "target_muscles": r[5], "description": r[6],
             "common_errors": r[7], "similarity": float(r[8]),
             "source": "vector"}
            for r in rows
        ]
```

- [ ] **Step 2: Commit**

```bash
git add src/rag/vector_search.py
git commit -m "feat: add pgvector vector search"
```

---

### Task 11: 关键词检索 + Agentic RAG + Semantic Cache

**Files:** Create: `src/rag/keyword_search.py`, `src/rag/agentic_rag.py`, `src/rag/semantic_cache.py`

- [ ] **Step 1: 关键词检索 — `src/rag/keyword_search.py`**

```python
from src.storage.pg import PGClient

class KeywordSearch:
    def __init__(self):
        self.pg = PGClient()

    def search(self, query: str, top_k: int = 10) -> list:
        self.pg.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        sql = """
            SELECT name, exercise_type, difficulty, equipment,
                   target_muscles, description, common_errors,
                   similarity(name, :query) AS sim
            FROM exercises
            WHERE name % :query OR name ILIKE '%' || :query || '%'
            ORDER BY sim DESC
            LIMIT :limit
        """
        rows = self.pg.fetch_all(sql, {"query": query, "limit": top_k})
        return [
            {"name": r[0], "type": r[1], "difficulty": r[2], "equipment": r[3],
             "target_muscles": r[4], "description": r[5], "common_errors": r[6],
             "similarity": float(r[7]) if r[7] else 0.0, "source": "keyword"}
            for r in rows
        ]
```

- [ ] **Step 2: Agentic RAG — `src/rag/agentic_rag.py`**

```python
from src.rag.vector_search import VectorSearch
from src.rag.keyword_search import KeywordSearch
from src.llm.provider import LLMProvider
from src.llm.prompts.retriever import build_retriever_eval_messages
from src.config import AGENTIC_RAG_MAX_RETRIES

class AgenticRAG:
    def __init__(self):
        self.vector = VectorSearch()
        self.keyword = KeywordSearch()
        self.llm = LLMProvider()

    def search(self, query: str, filters: dict = None, max_retries: int = None) -> list:
        max_retries = max_retries or AGENTIC_RAG_MAX_RETRIES
        current_query = query
        all_results = []

        for attempt in range(max_retries):
            vec_results = self.vector.search(current_query, top_k=5, filters=filters)
            kw_results = self.keyword.search(current_query, top_k=5)
            combined = self._deduplicate(vec_results + kw_results)
            all_results.extend(combined)

            if attempt < max_retries - 1:
                eval_msgs = build_retriever_eval_messages(query, combined)
                eval_result = self.llm.chat_with_json_mode(eval_msgs)
                score = eval_result.get("quality_score", 0)
                if score >= 0.7:
                    break
                current_query = eval_result.get("rewritten_query", current_query)

        return self._deduplicate(all_results)

    def _deduplicate(self, results: list) -> list:
        seen = set()
        unique = []
        for r in results:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique.append(r)
        return unique
```

- [ ] **Step 3: Semantic Cache — `src/rag/semantic_cache.py`**

```python
import json
import hashlib
from src.rag.embedding import EmbeddingService
from src.storage.redis_client import RedisClient
from src.config import CACHE_SIMILARITY_THRESHOLD

class SemanticCache:
    def __init__(self):
        self.redis = RedisClient()
        self.emb = EmbeddingService()

    def _make_key(self, profile: dict, query: str) -> str:
        raw = json.dumps(profile, sort_keys=True) + query
        return f"cache:fitness:{hashlib.md5(raw.encode()).hexdigest()}"

    def get(self, profile: dict, query: str) -> dict | None:
        cache_key = self._make_key(profile, query)
        data = self.redis.get(cache_key)
        if data:
            return json.loads(data)
        # 尝试找相似缓存
        query_vec = self.emb.embed(query)
        keys = self.redis.conn.keys("cache:fitness:*")
        for k in keys:
            cached = self.redis.get(k.decode())
            if cached:
                entry = json.loads(cached)
                stored_vec = entry.get("_embedding")
                if stored_vec and self.emb.similarity(query_vec, stored_vec) >= CACHE_SIMILARITY_THRESHOLD:
                    return entry.get("result")
        return None

    def set(self, profile: dict, query: str, result: dict):
        cache_key = self._make_key(profile, query)
        query_vec = self.emb.embed(query)
        entry = {"_embedding": query_vec, "result": result}
        self.redis.set(cache_key, json.dumps(entry, ensure_ascii=False), ex=3600)
```

- [ ] **Step 4: Commit**

```bash
git add src/rag/
git commit -m "feat: add keyword search, Agentic RAG, and semantic cache"
```

---

## Phase 5: GraphRAG

### Task 12: 知识图谱构建

**Files:** Create: `src/graphrag/__init__.py`, `src/graphrag/builder.py`, `tests/test_graphrag/__init__.py`

- [ ] **Step 1: 实现 — `src/graphrag/builder.py`**

```python
from src.storage.neo4j_client import Neo4jClient
from src.llm.provider import LLMProvider
import json

class GraphBuilder:
    def __init__(self):
        self.neo4j = Neo4jClient()
        self.llm = LLMProvider()

    def init_schema(self):
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Muscle) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Exercise) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (eq:Equipment) REQUIRE eq.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Injury) REQUIRE i.name IS UNIQUE",
        ]
        for c in constraints:
            try:
                self.neo4j.run(c)
            except:
                pass

    def build_from_seed(self, exercises: list):
        self.init_schema()
        for ex in exercises:
            self.neo4j.run("""
                MERGE (e:Exercise {name: $name})
                SET e.difficulty = $difficulty, e.type = $type
            """, {"name": ex["name"], "difficulty": ex.get("difficulty", "中级"),
                  "type": ex.get("exercise_type", "复合")})

            for muscle in ex.get("target_muscles", []):
                self.neo4j.run("""
                    MERGE (m:Muscle {name: $muscle})
                    MERGE (e:Exercise {name: $ex_name})
                    MERGE (e)-[:TARGETS]->(m)
                """, {"muscle": muscle, "ex_name": ex["name"]})

            if ex.get("equipment"):
                self.neo4j.run("""
                    MERGE (eq:Equipment {name: $equip})
                    MERGE (e:Exercise {name: $ex_name})
                    MERGE (e)-[:REQUIRES]->(eq)
                """, {"equip": ex["equipment"], "ex_name": ex["name"]})

    def extract_triples_with_llm(self, text: str) -> list:
        prompt = f"""从以下健身文本中提取（动作-关系-实体）三元组。
实体类型：Muscle, Exercise, Equipment, Injury
关系类型：TARGETS, REQUIRES, MAY_CAUSE, RECOVERED_BY

文本：{text}

输出 JSON 数组：[{{"subject":"","relation":"","object":"","obj_type":""}}]
"""
        result = self.llm.chat_with_json_mode([{"role": "user", "content": prompt}])
        return result if isinstance(result, list) else []
```

- [ ] **Step 2: Commit**

```bash
git add src/graphrag/
git commit -m "feat: add GraphRAG builder with LLM triple extraction"
```

---

### Task 13: 知识图谱搜索

**Files:** Create: `src/graphrag/search.py`, `tests/test_graphrag/test_search.py`

- [ ] **Step 1: 编写测试 — `tests/test_graphrag/test_search.py`**

```python
import pytest
from src.graphrag.search import GraphSearch

@pytest.fixture
def gs():
    return GraphSearch()

class TestGraphSearch:
    def test_find_exercises_by_muscle(self, gs):
        # 需要先有数据，用 seed 数据填充后测试
        results = gs.find_exercises_by_muscle("胸大肌")
        assert isinstance(results, list)

    def test_multi_hop(self, gs):
        results = gs.multi_hop_search("哑铃", "胸")
        assert isinstance(results, list)
```

- [ ] **Step 2: 实现 — `src/graphrag/search.py`**

```python
from src.storage.neo4j_client import Neo4jClient

class GraphSearch:
    def __init__(self):
        self.neo4j = Neo4jClient()

    def find_exercises_by_muscle(self, muscle: str, limit: int = 10) -> list:
        results = self.neo4j.query("""
            MATCH (e:Exercise)-[:TARGETS]->(m:Muscle)
            WHERE m.name CONTAINS $muscle
            RETURN e.name AS name, e.difficulty AS difficulty,
                   e.type AS type, m.name AS muscle
            LIMIT $limit
        """, {"muscle": muscle, "limit": limit})
        return results

    def find_equipment_for_exercise(self, exercise: str) -> list:
        results = self.neo4j.query("""
            MATCH (e:Exercise {name: $name})-[:REQUIRES]->(eq:Equipment)
            RETURN eq.name AS equipment
        """, {"name": exercise})
        return results

    def multi_hop_search(self, equipment: str, target: str) -> list:
        """找用指定器械练指定部位的动作用什么器械"""
        results = self.neo4j.query("""
            MATCH (e:Exercise)-[:REQUIRES]->(eq:Equipment)
            WHERE eq.name CONTAINS $equipment
            MATCH (e)-[:TARGETS]->(m:Muscle)
            WHERE m.name CONTAINS $target
            RETURN e.name AS name, e.difficulty AS difficulty,
                   collect(DISTINCT m.name) AS muscles,
                   collect(DISTINCT eq.name) AS equipment
        """, {"equipment": equipment, "target": target})
        return results

    def find_injury_risks(self, exercise: str) -> list:
        results = self.neo4j.query("""
            MATCH (e:Exercise {name: $name})-[:MAY_CAUSE]->(i:Injury)
            RETURN i.name AS injury
        """, {"name": exercise})
        return results

    def find_rehab_exercises(self, injury: str) -> list:
        results = self.neo4j.query("""
            MATCH (i:Injury {name: $injury})<-[:MAY_CAUSE]-(bad:Exercise)
            MATCH (i)-[:RECOVERED_BY]->(rehab:Exercise)
            RETURN rehab.name AS rehab_exercise,
                   collect(DISTINCT bad.name) AS avoid_exercises
        """, {"injury": injury})
        return results

    def reason_about_pain(self, exercise: str, symptom: str) -> dict:
        """多跳推理：动作 → 可能伤病 → 康复建议"""
        risks = self.find_injury_risks(exercise)
        causes = []
        solutions = []
        for r in risks:
            rehab = self.find_rehab_exercises(r["injury"])
            for item in rehab:
                causes.append({"injury": r["injury"], "avoid": item.get("avoid_exercises", [])})
                solutions.append(item.get("rehab_exercise"))
        return {
            "exercise": exercise,
            "symptom": symptom,
            "possible_causes": causes,
            "suggested_rehab": list(set(solutions)),
            "source": "graph"
        }
```

- [ ] **Step 3: Commit**

```bash
git add src/graphrag/search.py tests/test_graphrag/
git commit -m "feat: add GraphRAG multi-hop search with injury reasoning"
```

---

## Phase 6: 高级特性

### Task 14: MCP 协议

**Files:** Create: `src/mcp/__init__.py`, `src/mcp/exercise_server.py`, `src/mcp/tool_registry.py`

- [ ] **Step 1: MCP 练习服务器 — `src/mcp/exercise_server.py`**

```python
"""Exercise Standard Library MCP Server. 
提供标准动作库查询：按器械、肌肉、难度筛选。"""

EXERCISE_LIBRARY = [
    {"name": "哑铃卧推", "equipment": "哑铃", "muscles": ["胸大肌", "三角肌前束", "肱三头肌"],
     "difficulty": "初级", "type": "复合"},
    {"name": "杠铃深蹲", "equipment": "杠铃", "muscles": ["股四头肌", "臀大肌", "腘绳肌"],
     "difficulty": "中级", "type": "复合"},
    {"name": "引体向上", "equipment": "自重", "muscles": ["背阔肌", "肱二头肌"],
     "difficulty": "中级", "type": "复合"},
    {"name": "哑铃侧平举", "equipment": "哑铃", "muscles": ["三角肌中束"],
     "difficulty": "初级", "type": "孤立"},
    {"name": "杠铃硬拉", "equipment": "杠铃", "muscles": ["腘绳肌", "臀大肌", "竖脊肌"],
     "difficulty": "高级", "type": "复合"},
    {"name": "绳索下压", "equipment": "绳索", "muscles": ["肱三头肌"],
     "difficulty": "初级", "type": "孤立"},
    {"name": "哑铃弯举", "equipment": "哑铃", "muscles": ["肱二头肌"],
     "difficulty": "初级", "type": "孤立"},
    {"name": "腿举", "equipment": "腿举机", "muscles": ["股四头肌", "臀大肌"],
     "difficulty": "初级", "type": "复合"},
]


class ExerciseMCPServer:
    """模拟 MCP Server 接口：tools/list + tools/call"""

    def list_tools(self) -> list:
        return [
            {"name": "search_by_muscle", "description": "按目标肌肉搜索动作",
             "parameters": {"muscle": "string"}},
            {"name": "search_by_equipment", "description": "按器械搜索动作",
             "parameters": {"equipment": "string"}},
            {"name": "search_by_difficulty", "description": "按难度搜索动作",
             "parameters": {"difficulty": "string"}},
            {"name": "get_exercise_detail", "description": "获取动作详情",
             "parameters": {"name": "string"}},
        ]

    def call_tool(self, tool_name: str, params: dict) -> list:
        if tool_name == "search_by_muscle":
            muscle = params.get("muscle", "").lower()
            return [e for e in EXERCISE_LIBRARY
                    if any(muscle in m.lower() for m in e["muscles"])]
        elif tool_name == "search_by_equipment":
            equip = params.get("equipment", "").lower()
            return [e for e in EXERCISE_LIBRARY if equip in e["equipment"].lower()]
        elif tool_name == "search_by_difficulty":
            diff = params.get("difficulty", "")
            return [e for e in EXERCISE_LIBRARY if e["difficulty"] == diff]
        elif tool_name == "get_exercise_detail":
            name = params.get("name", "")
            for e in EXERCISE_LIBRARY:
                if e["name"] == name:
                    return [e]
            return []
        return []
```

- [ ] **Step 2: Tool Registry — `src/mcp/tool_registry.py`**

```python
from src.mcp.exercise_server import ExerciseMCPServer
from src.graphrag.search import GraphSearch
from src.rag.vector_search import VectorSearch

class ToolRegistry:
    def __init__(self):
        self.exercise_mcp = ExerciseMCPServer()
        self.graph_search = GraphSearch()
        self.vector_search = VectorSearch()
        self._register_tools()

    def _register_tools(self):
        self.tools = {
            "search_by_muscle": lambda p: self.exercise_mcp.call_tool("search_by_muscle", p),
            "search_by_equipment": lambda p: self.exercise_mcp.call_tool("search_by_equipment", p),
            "search_by_difficulty": lambda p: self.exercise_mcp.call_tool("search_by_difficulty", p),
            "get_exercise_detail": lambda p: self.exercise_mcp.call_tool("get_exercise_detail", p),
            "graph_multi_hop": lambda p: self.graph_search.multi_hop_search(
                p.get("equipment", ""), p.get("target", "")),
            "graph_injury_risk": lambda p: self.graph_search.find_injury_risks(p.get("exercise", "")),
            "graph_reason_pain": lambda p: self.graph_search.reason_about_pain(
                p.get("exercise", ""), p.get("symptom", "")),
        }

    def list_tools(self) -> list:
        return [{"name": k, "params": "dict"} for k in self.tools]

    def call(self, tool_name: str, params: dict):
        if tool_name in self.tools:
            return self.tools[tool_name](params)
        return None
```

- [ ] **Step 3: Commit**

```bash
git add src/mcp/
git commit -m "feat: add MCP exercise server and tool registry"
```

---

### Task 15: A2A + 长期记忆 + HITL + Skill 注册

**Files:** Create: `src/a2a/__init__.py`, `src/a2a/messaging.py`, `src/memory/__init__.py`, `src/memory/long_term.py`, `src/hitl/__init__.py`, `src/hitl/review.py`, `src/skills/__init__.py`, `src/skills/registry.py`, `skills/muscle_building.md`, `skills/fat_loss.md`, `skills/exercise_analysis.md`

- [ ] **Step 1: A2A 消息 — `src/a2a/messaging.py`**

```python
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"

@dataclass
class Artifact:
    artifact_id: str
    artifact_type: str  # "training_plan" | "analysis_report" | "safety_check"
    content: dict
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Task:
    task_id: str
    from_agent: str
    to_agent: str
    task_type: str
    payload: dict
    status: TaskStatus = TaskStatus.PENDING
    artifacts: list = field(default_factory=list)

    def add_artifact(self, artifact: Artifact):
        self.artifacts.append(artifact)

    def complete(self):
        self.status = TaskStatus.COMPLETED

    def fail(self):
        self.status = TaskStatus.FAILED


class MessageBus:
    def __init__(self):
        self.tasks: list[Task] = []

    def send(self, task: Task):
        self.tasks.append(task)
        return task

    def get_for_agent(self, agent_name: str) -> list:
        return [t for t in self.tasks if t.to_agent == agent_name and
                t.status == TaskStatus.PENDING]
```

- [ ] **Step 2: 长期记忆 — `src/memory/long_term.py`**

```python
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
```

- [ ] **Step 3: HITL 审批 — `src/hitl/review.py`**

```python
from dataclasses import dataclass
from src.config import HITL_CONFIDENCE_THRESHOLD

@dataclass
class ReviewDecision:
    needs_review: bool
    reason: str
    severity: str  # "safe" | "warning" | "danger"
    suggestions: list

class HITLReview:
    def check(self, fact_check_result: dict) -> ReviewDecision:
        confidence = fact_check_result.get("confidence", 0)
        issues = fact_check_result.get("issues", [])
        has_danger = any(i.get("severity") == "danger" for i in issues)
        has_warning = any(i.get("severity") == "warning" for i in issues)

        if confidence < HITL_CONFIDENCE_THRESHOLD or has_danger:
            return ReviewDecision(
                needs_review=True,
                reason=f"置信度 {confidence:.2f} 低于阈值或有危险建议",
                severity="danger" if has_danger else "warning",
                suggestions=[i["issue"] for i in issues]
            )
        if has_warning:
            return ReviewDecision(
                needs_review=True,
                reason="存在需要确认的警告项",
                severity="warning",
                suggestions=[i["issue"] for i in issues]
            )
        return ReviewDecision(
            needs_review=False, reason="", severity="safe", suggestions=[]
        )
```

- [ ] **Step 4: Skill 注册 — `src/skills/registry.py`**

```python
from dataclasses import dataclass, field

@dataclass
class Skill:
    name: str
    description: str
    triggers: list  # 触发关键词
    retrieval_filters: dict = field(default_factory=dict)
    plan_template: str = ""

class SkillRegistry:
    def __init__(self):
        self.skills: dict[str, Skill] = {}
        self._load_builtin()

    def _load_builtin(self):
        self.register(Skill(
            name="muscle_building",
            description="增肌训练计划生成",
            triggers=["增肌", "增重", "变大", "维度", "增肌塑形"],
            retrieval_filters={"rep_range": "6-12", "rest": "60-90s"},
            plan_template="四分化/五分化"
        ))
        self.register(Skill(
            name="fat_loss",
            description="减脂训练计划生成",
            triggers=["减脂", "减重", "瘦", "刷脂", "塑形"],
            retrieval_filters={"rep_range": "12-15", "rest": "30-60s"},
            plan_template="上下肢分化/全身"
        ))
        self.register(Skill(
            name="exercise_analysis",
            description="动作质量分析",
            triggers=["动作", "姿势", "感觉", "疼", "不舒服", "是不是"],
            retrieval_filters={},
            plan_template="分析报告"
        ))

    def register(self, skill: Skill):
        self.skills[skill.name] = skill

    def match(self, user_input: str) -> str | None:
        for name, skill in self.skills.items():
            for trigger in skill.triggers:
                if trigger in user_input:
                    return name
        return "muscle_building"  # 默认增肌

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)
```

- [ ] **Step 5: 创建 Skill 定义文件 — `skills/muscle_building.md`**

```markdown
# 增肌训练计划 Skill

## 触发条件
用户输入包含：增肌、增重、变大、维度

## 检索策略
- Rep Range: 6-12
- 组间休息: 60-90 秒
- 优先复合动作
- 每周每部位 10-20 组

## 计划模板
四分化：胸+三头 / 背+二头 / 肩 / 腿
或五分化：胸 / 背 / 肩 / 手臂 / 腿
```

- [ ] **Step 6: Commit**

```bash
git add src/a2a/ src/memory/ src/hitl/ src/skills/ skills/
git commit -m "feat: add A2A messaging, long-term memory, HITL review, and skill registry"
```

---

## Phase 7: Agent 层

### Task 16: Planner Agent

**Files:** Create: `src/agents/__init__.py`, `src/agents/planner.py`, `tests/test_agents/__init__.py`, `tests/test_agents/test_planner.py`

- [ ] **Step 1: 编写测试 — `tests/test_agents/test_planner.py`**

```python
import pytest
from src.agents.planner import PlannerAgent

@pytest.fixture
def planner():
    return PlannerAgent()

class TestPlannerAgent:
    def test_plan_muscle_building(self, planner):
        profile = {"height": 180, "weight": 80, "goal": "增肌",
                   "training_years": 1, "available_equipment": ["哑铃"]}
        result = planner.plan("我想增肌", profile)
        assert "subtasks" in result
        assert len(result["subtasks"]) > 0
        assert result["output_format"] in ["增肌计划", "减脂计划"]

    def test_plan_exercise_analysis(self, planner):
        profile = {"height": 175, "weight": 70, "goal": "增肌",
                   "training_years": 0.5, "available_equipment": ["哑铃"]}
        result = planner.plan("深蹲时膝盖不舒服", profile)
        assert result["output_format"] == "动作分析"
```

- [ ] **Step 2: 实现 — `src/agents/planner.py`**

```python
from src.llm.provider import LLMProvider
from src.llm.prompts.planner import build_planner_messages
from src.skills.registry import SkillRegistry

class PlannerAgent:
    def __init__(self):
        self.llm = LLMProvider()
        self.skills = SkillRegistry()

    def plan(self, user_input: str, profile: dict) -> dict:
        skill_name = self.skills.match(user_input)
        skill = self.skills.get(skill_name)
        messages = build_planner_messages(user_input, profile)
        plan = self.llm.chat_with_json_mode(messages)
        plan["skill"] = skill_name
        plan["skill_config"] = {
            "retrieval_filters": skill.retrieval_filters,
            "plan_template": skill.plan_template
        }
        return plan
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_agents/test_planner.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/agents/ tests/test_agents/
git commit -m "feat: add Planner Agent with skill-based routing"
```

---

### Task 17: Retriever Agent

**Files:** Create: `src/agents/retriever.py`, `tests/test_agents/test_retriever.py`

- [ ] **Step 1: 实现 — `src/agents/retriever.py`**

```python
from src.rag.agentic_rag import AgenticRAG
from src.mcp.tool_registry import ToolRegistry
from src.llm.provider import LLMProvider

class RetrieverAgent:
    def __init__(self):
        self.agentic_rag = AgenticRAG()
        self.tools = ToolRegistry()
        self.llm = LLMProvider()

    def retrieve(self, plan: dict) -> dict:
        results = {"exercises": [], "knowledge": []}
        for subtask in plan.get("subtasks", []):
            filters = plan.get("skill_config", {}).get("retrieval_filters", {})
            rag_results = self.agentic_rag.search(subtask, filters=filters)
            results["exercises"].extend(rag_results)
        # 调 MCP 补充动作
        body_parts = self._extract_body_parts(plan)
        for part in body_parts:
            mcp_results = self.tools.call("search_by_muscle", {"muscle": part})
            if mcp_results:
                results["exercises"].extend(
                    [{"name": r["name"], "source": "mcp",
                      "muscles": r["muscles"], "equipment": r["equipment"],
                      "difficulty": r["difficulty"], "type": r["type"]}
                     for r in mcp_results]
                )
        return results

    def _extract_body_parts(self, plan: dict) -> list:
        parts_map = {
            "推": ["胸", "肩", "三头"],
            "拉": ["背", "二头"],
            "腿": ["腿", "臀"],
        }
        subtasks_str = "".join(plan.get("subtasks", []))
        parts = []
        for key, vals in parts_map.items():
            if key in subtasks_str:
                parts.extend(vals)
        return parts if parts else ["胸", "背", "腿"]
```

- [ ] **Step 2: Commit**

```bash
git add src/agents/retriever.py tests/test_agents/test_retriever.py
git commit -m "feat: add Retriever Agent with Agentic RAG + MCP tool calling"
```

---

### Task 18: Writer Agent

**Files:** Create: `src/agents/writer.py`, `tests/test_agents/test_writer.py`

- [ ] **Step 1: 实现 — `src/agents/writer.py`**

```python
import uuid
from src.llm.provider import LLMProvider
from src.llm.prompts.writer import build_writer_messages
from src.models.schemas import TrainingPlanOutput, TrainingDay, ExerciseItem

class WriterAgent:
    def __init__(self):
        self.llm = LLMProvider()

    def write_plan(self, retrieved: dict, profile: dict, plan_config: dict) -> dict:
        goal = profile.get("goal", "增肌")
        messages = build_writer_messages(
            retrieved.get("exercises", []), profile, goal
        )
        plan_json = self.llm.chat_with_json_mode(messages)
        plan_json["plan_id"] = str(uuid.uuid4())[:8]
        plan_json["user_id"] = profile.get("id", 0)
        return plan_json

    def write_analysis(self, exercise_name: str, user_desc: str,
                       retrieved: dict, profile: dict) -> dict:
        prompt = f"""分析动作：{exercise_name}
用户描述：{user_desc}
用户水平：{profile.get('training_years', 1)}年经验
标准动作规范：{retrieved}

输出 JSON：
{{
  "exercise_name": "{exercise_name}",
  "issues_found": ["问题1", "问题2"],
  "severity": "安全" | "注意" | "警告",
  "suggestions": ["改进1", "改进2"],
  "confidence": 0.0-1.0
}}"""
        result = self.llm.chat_with_json_mode([{"role": "user", "content": prompt}])
        result["exercise_name"] = exercise_name
        result.setdefault("issues_found", [])
        result.setdefault("severity", "安全")
        result.setdefault("suggestions", [])
        result.setdefault("confidence", 0.5)
        return result
```

- [ ] **Step 2: Commit**

```bash
git add src/agents/writer.py tests/test_agents/test_writer.py
git commit -m "feat: add Writer Agent with structured output"
```

---

### Task 19: FactChecker Agent

**Files:** Create: `src/agents/fact_checker.py`, `tests/test_agents/test_fact_checker.py`

- [ ] **Step 1: 编写测试 — `tests/test_agents/test_fact_checker.py`**

```python
from src.agents.fact_checker import FactCheckerAgent

def test_fact_checker_returns_structured_result():
    checker = FactCheckerAgent()
    plan = {
        "days": [{"day": 1, "focus": "胸", "exercises": [
            {"name": "杠铃卧推", "sets": 5, "reps": "3-5", "rest": "120s"}
        ]}]
    }
    profile = {"training_years": 0.3, "injuries": [], "goal": "增肌",
               "available_equipment": ["哑铃"]}
    result = checker.check(plan, profile)
    assert "is_safe" in result
    assert "issues" in result
    assert "confidence" in result
```

- [ ] **Step 2: 实现 — `src/agents/fact_checker.py`**

```python
from src.llm.provider import LLMProvider
from src.llm.prompts.fact_checker import build_fact_checker_messages
from src.hitl.review import HITLReview

class FactCheckerAgent:
    def __init__(self):
        self.llm = LLMProvider()
        self.hitl = HITLReview()

    def check(self, plan: dict, profile: dict) -> dict:
        messages = build_fact_checker_messages(plan, profile)
        result = self.llm.chat_with_json_mode(messages)
        result.setdefault("is_safe", True)
        result.setdefault("issues", [])
        result.setdefault("confidence", 0.8)
        result.setdefault("requires_human_review", False)
        # HITL 检查
        review = self.hitl.check(result)
        result["requires_human_review"] = review.needs_review
        result["review_reason"] = review.reason
        result["review_severity"] = review.severity
        return result
```

- [ ] **Step 3: Commit**

```bash
git add src/agents/fact_checker.py tests/test_agents/test_fact_checker.py
git commit -m "feat: add FactChecker Agent with HITL integration"
```

---

## Phase 8: 核心编排

### Task 20: Harness + Model Router

**Files:** Create: `src/core/__init__.py`, `src/core/harness.py`, `src/core/model_router.py`

- [ ] **Step 1: Harness 工程化 — `src/core/harness.py`**

```python
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def with_retry(max_retries: int = 3, backoff: float = 2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    wait = backoff ** attempt
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}")
                    time.sleep(wait)
            raise last_error
        return wrapper
    return decorator

def with_timeout(seconds: int = 60):
    import signal
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            import threading
            t = threading.Thread(target=target)
            t.daemon = True
            t.start()
            t.join(seconds)
            if t.is_alive():
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")
            if exception[0]:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator
```

- [ ] **Step 2: Model Router — `src/core/model_router.py`**

```python
from src.llm.provider import LLMProvider

class ModelRouter:
    def __init__(self):
        self.llm = LLMProvider()

    def route(self, task_type: str) -> str:
        """简单任务的用 DeepSeek-V3，复杂的用 Claude（若配置）"""
        simple_tasks = ["retrieve", "keyword_search", "cache_lookup"]
        if task_type in simple_tasks:
            return "deepseek-chat"
        return "deepseek-chat"  # Claude 未配 key 时全用 DeepSeek
```

- [ ] **Step 3: Commit**

```bash
git add src/core/
git commit -m "feat: add harness (retry/timeout) and model router"
```

---

### Task 21: Orchestrator

**Files:** Create: `src/core/orchestrator.py`, `tests/test_core/__init__.py`, `tests/test_core/test_orchestrator.py`

- [ ] **Step 1: 编写测试 — `tests/test_core/test_orchestrator.py`**

```python
import pytest
from src.core.orchestrator import Orchestrator
from src.models.schemas import UserProfileInput

@pytest.fixture
def orch():
    return Orchestrator()

class TestOrchestrator:
    def test_generate_plan_complete_flow(self, orch):
        profile = UserProfileInput(
            height=180, weight=80, training_years=1,
            goal="增肌", available_equipment=["哑铃", "杠铃"],
            days_per_week=4
        )
        result = orch.generate_plan(profile, "帮我设计增肌计划")
        assert "plan_id" in result
        assert "days" in result
        assert len(result["days"]) > 0

    def test_analyze_exercise(self, orch):
        profile = UserProfileInput(
            height=175, weight=70, training_years=0.5,
            goal="增肌", available_equipment=["哑铃"],
            days_per_week=3
        )
        result = orch.analyze_exercise(
            "哑铃卧推", "推的时候肩膀前侧有点疼", profile
        )
        assert "exercise_name" in result
        assert "issues_found" in result
        assert "suggestions" in result

    def test_semantic_cache_hit(self, orch):
        profile = UserProfileInput(
            height=180, weight=80, training_years=1,
            goal="增肌", available_equipment=["哑铃", "杠铃"],
            days_per_week=4
        )
        # 第一次调用
        result1 = orch.generate_plan(profile, "增肌计划")
        # 第二次相同调用应命中缓存
        result2 = orch.generate_plan(profile, "增肌计划")
        assert result1["plan_id"] == result2["plan_id"]
```

- [ ] **Step 2: 实现 — `src/core/orchestrator.py`**

```python
import logging
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.writer import WriterAgent
from src.agents.fact_checker import FactCheckerAgent
from src.rag.semantic_cache import SemanticCache
from src.skills.registry import SkillRegistry
from src.a2a.messaging import MessageBus, Task, Artifact, TaskStatus
from src.models.schemas import UserProfileInput

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self):
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.writer = WriterAgent()
        self.fact_checker = FactCheckerAgent()
        self.cache = SemanticCache()
        self.skills = SkillRegistry()
        self.bus = MessageBus()

    def generate_plan(self, profile: UserProfileInput, query: str = "") -> dict:
        profile_dict = profile.model_dump()
        # 1. 检查 Semantic Cache
        cached = self.cache.get(profile_dict, query)
        if cached:
            logger.info("Cache hit for plan generation")
            return cached

        # 2. Planner
        plan = self.planner.plan(query or f"为{profile.goal}目标生成训练计划", profile_dict)

        # 3. Retriever
        retrieved = self.retriever.retrieve(plan)

        # 4. Writer → send task via A2A
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
        artifact = Artifact(
            artifact_id=task.task_id, artifact_type="training_plan", content=result
        )
        task.add_artifact(artifact)

        # 5. FactChecker
        fc_task = Task(
            task_id=f"check_{profile_dict.get('id', 0)}",
            from_agent="writer", to_agent="fact_checker",
            task_type="safety_check", payload={"plan": result, "profile": profile_dict}
        )
        self.bus.send(fc_task)
        check = self.fact_checker.check(result, profile_dict)
        result["warnings"] = [i["issue"] for i in check.get("issues", [])]
        result["requires_review"] = check.get("requires_human_review", False)
        result["confidence"] = check.get("confidence", 0)

        # 6. 存入缓存
        self.cache.set(profile_dict, query, result)
        task.complete()
        return result

    def analyze_exercise(self, exercise_name: str, user_desc: str,
                         profile: UserProfileInput) -> dict:
        profile_dict = profile.model_dump()
        # 检索该动作的标准规范
        retrieved = self.retriever.retrieve({"subtasks": [exercise_name], "skill_config": {}})
        return self.writer.write_analysis(exercise_name, user_desc, retrieved, profile_dict)
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_core/test_orchestrator.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/core/orchestrator.py tests/test_core/
git commit -m "feat: add Orchestrator with full agent pipeline and cache"
```

---

## Phase 9: 入口

### Task 22: 命令行入口 + 数据种子

**Files:** Create: `src/main.py`, `data/seed_exercises.json`

- [ ] **Step 1: 种子数据 — `data/seed_exercises.json`**

```json
[
  {"name": "哑铃卧推", "exercise_type": "复合", "difficulty": "初级", "equipment": "哑铃",
   "target_muscles": ["胸大肌", "三角肌前束", "肱三头肌"],
   "description": "平躺于凳上，双手持哑铃推起至手臂伸直，缓慢下放至胸部两侧。",
   "common_errors": ["手肘打得太开(超过45度)", "哑铃下放过低导致肩关节压力"]},
  {"name": "杠铃深蹲", "exercise_type": "复合", "difficulty": "中级", "equipment": "杠铃",
   "target_muscles": ["股四头肌", "臀大肌", "腘绳肌"],
   "description": "杠铃置于斜方肌上，双脚与肩同宽，下蹲至大腿与地面平行。",
   "common_errors": ["膝盖内扣", "脚跟抬起", "下背拱起"]},
  {"name": "引体向上", "exercise_type": "复合", "difficulty": "中级", "equipment": "自重",
   "target_muscles": ["背阔肌", "肱二头肌"],
   "description": "双手握杠，身体悬挂，用背部力量将身体拉起至下巴过杠。",
   "common_errors": ["借助身体摆动", "没有完全下放"]},
  {"name": "哑铃侧平举", "exercise_type": "孤立", "difficulty": "初级", "equipment": "哑铃",
   "target_muscles": ["三角肌中束"],
   "description": "站姿，双手持哑铃于体侧，向两侧平举至肩高。",
   "common_errors": ["耸肩借力斜方肌", "手臂完全伸直"]},
  {"name": "杠铃硬拉", "exercise_type": "复合", "difficulty": "高级", "equipment": "杠铃",
   "target_muscles": ["臀大肌", "腘绳肌", "竖脊肌"],
   "description": "杠铃置于脚前，屈膝俯身双手握杠，挺髋站起。",
   "common_errors": ["下背拱起", "杠铃离身体过远", "起始位置臀部过低"]},
  {"name": "绳索下压", "exercise_type": "孤立", "difficulty": "初级", "equipment": "绳索",
   "target_muscles": ["肱三头肌"],
   "description": "面对龙门架，双手握绳索，下压至手臂伸直。",
   "common_errors": ["肘部前移", "身体晃动借力"]},
  {"name": "哑铃弯举", "exercise_type": "孤立", "difficulty": "初级", "equipment": "哑铃",
   "target_muscles": ["肱二头肌"],
   "description": "站姿或坐姿，双手持哑铃，弯举至肩前。",
   "common_errors": ["身体后仰借力", "下放过快"]},
  {"name": "腿举", "exercise_type": "复合", "difficulty": "初级", "equipment": "腿举机",
   "target_muscles": ["股四头肌", "臀大肌"],
   "description": "坐于腿举机，双脚置于踏板上，蹬起至腿伸直。",
   "common_errors": ["膝盖完全锁死", "下放幅度不够"]},
  {"name": "哑铃推举", "exercise_type": "复合", "difficulty": "中级", "equipment": "哑铃",
   "target_muscles": ["三角肌前束", "三角肌中束", "肱三头肌"],
   "description": "坐姿，哑铃于肩部两侧，推举至头顶。",
   "common_errors": ["下背过度反弓", "哑铃在最低点低于肩部"]},
  {"name": "俯身哑铃划船", "exercise_type": "复合", "difficulty": "中级", "equipment": "哑铃",
   "target_muscles": ["背阔肌", "菱形肌", "肱二头肌"],
   "description": "俯身，一只手撑凳，另一只手持哑铃拉至体侧。",
   "common_errors": ["上身旋转借力", "肩膀耸起", "弧度不够完整"]},
  {"name": "高脚杯深蹲", "exercise_type": "复合", "difficulty": "初级", "equipment": "哑铃",
   "target_muscles": ["股四头肌", "臀大肌"],
   "description": "双手托哑铃于胸前，下蹲时保持躯干直立。替代杠铃深蹲的入门动作。",
   "common_errors": ["膝盖内扣", "上半身前倾过多"]},
  {"name": "哑铃罗马尼亚硬拉", "exercise_type": "复合", "difficulty": "中级", "equipment": "哑铃",
   "target_muscles": ["腘绳肌", "臀大肌", "竖脊肌"],
   "description": "双手持哑铃于体前，微屈膝，髋部后移俯身。对下背压力小于杠铃硬拉。",
   "common_errors": ["膝盖锁死", "下背拱起"]}
]
```

- [ ] **Step 2: CLI 入口 — `src/main.py`**

```python
import argparse
import json
import logging
from src.core.orchestrator import Orchestrator
from src.models.schemas import UserProfileInput
from src.models.db_models import init_db
from src.graphrag.builder import GraphBuilder
from src.rag.embedding import EmbeddingService
from src.storage.pg import PGClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def seed():
    """初始化数据库和知识图谱"""
    init_db()
    pg = PGClient()
    emb = EmbeddingService()
    with open("data/seed_exercises.json", "r", encoding="utf-8") as f:
        exercises = json.load(f)
    for ex in exercises:
        text = f"{ex['name']} {' '.join(ex['target_muscles'])} {ex.get('description', '')}"
        vec = emb.embed(text)
        vec_str = f"[{','.join(str(v) for v in vec)}]"
        pg.execute("""
            INSERT INTO exercises (name, exercise_type, difficulty, equipment,
                                   target_muscles, description, common_errors, embedding)
            VALUES (:name, :type, :diff, :equip, :muscles, :desc, :errors, :vec::vector)
            ON CONFLICT (name) DO NOTHING
        """, {"name": ex["name"], "type": ex["exercise_type"], "diff": ex["difficulty"],
              "equip": ex["equipment"], "muscles": json.dumps(ex["target_muscles"]),
              "desc": ex.get("description", ""), "errors": json.dumps(ex.get("common_errors", [])),
              "vec": vec_str})
    # 构建知识图谱
    builder = GraphBuilder()
    builder.build_from_seed(exercises)
    logger.info(f"Seeded {len(exercises)} exercises")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=str, help="自然语言查询")
    parser.add_argument("--height", type=float, default=180)
    parser.add_argument("--weight", type=float, default=80)
    parser.add_argument("--years", type=float, default=1)
    parser.add_argument("--goal", type=str, default="增肌")
    parser.add_argument("--equipment", type=str, default="哑铃,杠铃")
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument("--seed", action="store_true", help="初始化种子数据")
    parser.add_argument("--analyze", type=str, help="分析动作,格式: 动作名:感受描述")
    args = parser.parse_args()

    if args.seed:
        seed()
        logger.info("Seed complete")
        return

    orch = Orchestrator()
    profile = UserProfileInput(
        height=args.height, weight=args.weight, training_years=args.years,
        goal=args.goal, available_equipment=args.equipment.split(","),
        days_per_week=args.days
    )

    if args.analyze:
        parts = args.analyze.split(":", 1)
        name = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ""
        result = orch.analyze_exercise(name, desc, profile)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.search:
        result = orch.generate_plan(profile, args.search)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = orch.generate_plan(profile, f"为{args.goal}目标生成训练计划")
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add src/main.py data/
git commit -m "feat: add CLI entry point and seed exercise data"
```

---

## Phase 10: Streamlit UI

### Task 23: Streamlit Web 界面

**Files:** Create: `app/streamlit_app.py`

- [ ] **Step 1: 实现 — `app/streamlit_app.py`**

```python
import streamlit as st
import json
from src.core.orchestrator import Orchestrator
from src.models.schemas import UserProfileInput

st.set_page_config(page_title="AI 健身私教", page_icon="💪", layout="wide")
st.title("💪 AI 健身私教")

# 侧边栏：用户画像
with st.sidebar:
    st.header("身体数据")
    height = st.number_input("身高 (cm)", 150, 220, 180)
    weight = st.number_input("体重 (kg)", 35, 180, 80)
    years = st.slider("训练年限", 0.0, 15.0, 1.0, 0.5)
    goal = st.selectbox("目标", ["增肌", "减脂"])
    equipment = st.multiselect(
        "可用器械", ["哑铃", "杠铃", "绳索", "腿举机", "自重"],
        default=["哑铃", "杠铃"]
    )
    days = st.slider("每周训练天数", 1, 7, 4)
    injuries = st.multiselect(
        "伤病史", ["下背痛", "肩伤", "膝伤", "手腕伤", "无"], default=["无"]
    )
    if "无" in injuries:
        injuries = []

# 主区域
tab1, tab2, tab3 = st.tabs(["训练计划", "动作分析", "知识问答"])

with tab1:
    st.header("生成训练计划")
    query = st.text_area("补充说明", "帮我设计增肌计划，重点练胸和背")
    if st.button("生成计划", type="primary", key="gen_plan"):
        with st.spinner("AI 正在为你编排计划..."):
            profile = UserProfileInput(
                height=height, weight=weight, training_years=years,
                goal=goal, available_equipment=equipment, days_per_week=days,
                injuries=injuries
            )
            orch = Orchestrator()
            result = orch.generate_plan(profile, query)
            st.success(f"计划生成完成 | 置信度: {result.get('confidence', 0):.0%}")
            if result.get("warnings"):
                st.warning("⚠️ 安全提示：\n" + "\n".join(f"- {w}" for w in result["warnings"]))
            if result.get("requires_review"):
                st.info("此计划含低置信度建议，请谨慎执行")
            for day in result.get("days", []):
                with st.expander(f"第{day['day']}天 - {day['focus']}", expanded=True):
                    for ex in day["exercises"]:
                        st.markdown(f"**{ex['name']}** — {ex['sets']}组 × {ex['reps']}次 | 休息{ex['rest']}")
                        if ex.get("notes"):
                            st.caption(ex["notes"])

with tab2:
    st.header("动作分析")
    ex_name = st.text_input("动作名称", "哑铃卧推")
    ex_desc = st.text_area("描述你的训练感受", "推的时候右边肩膀前侧有点疼")
    if st.button("分析动作", type="primary", key="analyze"):
        with st.spinner("分析中..."):
            profile = UserProfileInput(
                height=height, weight=weight, training_years=years,
                goal=goal, available_equipment=equipment, days_per_week=days,
                injuries=injuries
            )
            orch = Orchestrator()
            result = orch.analyze_exercise(ex_name, ex_desc, profile)
            severity_color = {"安全": "green", "注意": "orange", "警告": "red"}
            color = severity_color.get(result.get("severity", "安全"), "green")
            st.markdown(f"严重程度: :{color}[{result.get('severity', '未知')}]")
            st.subheader("发现的问题")
            for issue in result.get("issues_found", []):
                st.markdown(f"- {issue}")
            st.subheader("改进建议")
            for sug in result.get("suggestions", []):
                st.markdown(f"- {sug}")

with tab3:
    st.header("知识问答")
    question = st.text_area("输入你的健身问题", "硬拉的时候下背酸正常吗？")
    if st.button("提问", type="primary", key="ask"):
        st.info("使用 GraphRAG 知识图谱推理中...（功能待完善）")
```

- [ ] **Step 2: Commit**

```bash
git add app/
git commit -m "feat: add Streamlit web UI with 3 tabs"
```

---

## Phase 11: 评估

### Task 24: 评估脚本

**Files:** Create: `eval/__init__.py`, `eval/test_queries.json`, `eval/eval_runner.py`

- [ ] **Step 1: 测试查询 — `eval/test_queries.json`**

```json
[
  {"id": "1", "query": "增肌计划", "goal": "增肌", "equipment": ["哑铃"], "days": 4,
   "checks": ["has_plan_id", "has_exercises", "exercises_count_gt:3"]},
  {"id": "2", "query": "减脂计划", "goal": "减脂", "equipment": ["自重"], "days": 3,
   "checks": ["has_plan_id", "has_exercises"]},
  {"id": "3", "query": "胸肌训练", "goal": "增肌", "equipment": ["哑铃", "杠铃"], "days": 4,
   "checks": ["has_exercises", "contains_chest_exercise"]},
  {"id": "4", "exercise": "深蹲", "desc": "膝盖疼", "goal": "增肌",
   "checks": ["has_issues", "has_suggestions"]},
  {"id": "5", "query": "增肌计划", "goal": "增肌", "equipment": ["哑铃"], "days": 4,
   "checks": ["cache_hit"]}
]
```

- [ ] **Step 2: 评估执行 — `eval/eval_runner.py`**

```python
import json
import time
import argparse
from src.core.orchestrator import Orchestrator
from src.models.schemas import UserProfileInput

def run_eval(limit: int = None):
    with open("eval/test_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    if limit:
        queries = queries[:limit]

    orch = Orchestrator()
    results = []
    total_latency = 0
    cache_hits = 0

    for q in queries:
        profile = UserProfileInput(
            height=180, weight=80, training_years=1,
            goal=q.get("goal", "增肌"),
            available_equipment=q.get("equipment", ["哑铃"]),
            days_per_week=q.get("days", 4)
        )
        start = time.time()
        if "exercise" in q:
            result = orch.analyze_exercise(q["exercise"], q.get("desc", ""), profile)
        else:
            result = orch.generate_plan(profile, q["query"])
        latency = time.time() - start
        total_latency += latency

        checks_passed = 0
        checks_total = len(q.get("checks", []))
        for check in q.get("checks", []):
            if check == "has_plan_id" and "plan_id" in result:
                checks_passed += 1
            elif check == "has_exercises":
                days = result.get("days", [])
                if any(len(d.get("exercises", [])) > 0 for d in days):
                    checks_passed += 1
            elif check == "has_issues" and result.get("issues_found"):
                checks_passed += 1
            elif check == "has_suggestions" and result.get("suggestions"):
                checks_passed += 1
            elif check.startswith("exercises_count_gt:"):
                threshold = int(check.split(":")[1])
                days = result.get("days", [])
                if any(len(d.get("exercises", [])) > threshold for d in days):
                    checks_passed += 1
            elif check == "cache_hit":
                if latency < 0.5:
                    cache_hits += 1
                    checks_passed += 1

        results.append({
            "id": q["id"], "latency_ms": round(latency * 1000),
            "checks": f"{checks_passed}/{checks_total}", "passed": checks_passed == checks_total
        })

    avg_latency = total_latency / len(queries) * 1000
    pass_count = sum(1 for r in results if r["passed"])
    print(f"总用例: {len(queries)} | 通过: {pass_count} | 平均延迟: {avg_latency:.0f}ms | 缓存命中: {cache_hits}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['id']}: {r['latency_ms']}ms ({r['checks']})")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_eval(args.limit)
```

- [ ] **Step 3: Commit**

```bash
git add eval/
git commit -m "feat: add eval dataset and batch runner"
```

---

## Appendix: 常用命令速查

```bash
# 环境准备
cd E:\gakki-ai-fitness
pip install -r requirements.txt

# 启动基础服务
docker compose up -d

# 初始化种子数据
python -m src.main --seed

# CLI 使用
python -m src.main --search "增肌计划" --height 180 --weight 80 --goal 增肌 --equipment 哑铃,杠铃
python -m src.main --analyze "哑铃卧推:肩膀前侧疼"

# Web 界面
streamlit run app/streamlit_app.py

# 测试
python -m pytest tests/ -v

# 评估
python -m eval.eval_runner --limit 5

# 停止服务
docker compose down
```
