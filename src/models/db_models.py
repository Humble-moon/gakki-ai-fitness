"""
db_models.py - 数据库表结构定义（ORM 模型层）

角色：使用 SQLAlchemy ORM 定义 PostgreSQL 数据库中所有表的 Schema。
      这是项目中所有关系型数据结构的"单点真相"（Single Source of Truth）。
被调用者：
    - core.orchestrator（写入/读取用户档案、训练计划）
    - rag 模块（读写知识块和向量数据）
    - graphrag.builder（从 exercises 表读取数据构建知识图谱）
    - main.py 中的 seed() 函数（初始化表结构和种子数据）
调用者：SQLAlchemy + pgvector 扩展。
"""
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from pgvector.sqlalchemy import Vector
from src.config import DATABASE_URL, EMBEDDING_DIM
from datetime import datetime


class Base(DeclarativeBase):
    """
    SQLAlchemy 声明式基类

    所有 ORM 模型类都继承它。SQLAlchemy 通过它发现和管理所有表定义。
    init_db() 函数调用 Base.metadata.create_all() 时，自动创建所有继承此类的子表。
    """
    pass


# ============================================================
# 用户档案表 (user_profiles)
# 存储用户的身体参数、训练目标和偏好设置
# ============================================================
class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 自增主键，唯一标识每个用户

    height = Column(Float, nullable=False)
    # 身高（厘米），如 180.0

    weight = Column(Float, nullable=False)
    # 体重（公斤），如 80.0

    training_years = Column(Float, nullable=False)
    # 训练年限（年），如 1.5 表示一年半健身经验，支持小数

    goal = Column(String(20), nullable=False)
    # 训练目标，取值限制为 "增肌" 或 "减脂"

    available_equipment = Column(JSON, nullable=False)
    # 可用器械列表，JSON 数组格式，如 ["哑铃", "杠铃", "龙门架"]

    days_per_week = Column(Integer, nullable=False)
    # 每周可训练天数，取值范围 1~7

    injuries = Column(JSON, default=[])
    # 伤病史，JSON 数组，如 ["腰椎间盘突出", "肩袖损伤"]
    # 生成训练计划时会根据伤病史排除危险动作

    preferences = Column(JSON, default={})
    # 用户偏好设置，JSON 对象，如 {"prefer_split": "推拉腿", "session_minutes": 60}
    # 灵活扩展，可存放任意自定义偏好键值对

    created_at = Column(DateTime, default=datetime.utcnow)
    # 记录创建时间（UTC），首次插入时自动赋值

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # 记录更新时间（UTC），每次 UPDATE 时自动刷新为当前时间


# ============================================================
# 运动动作表 (exercises)
# 存储所有健身动作的元数据和向量嵌入
# ============================================================
class Exercise(Base):
    __tablename__ = "exercises"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 自增主键

    name = Column(String(100), nullable=False, unique=True)
    # 动作中文名，如 "杠铃深蹲"，UNIQUE 约束防止重复插入

    name_en = Column(String(100))
    # 动作英文名，如 "Barbell Squat"，可选字段

    exercise_type = Column(String(20))
    # 动作类型：如 "复合动作"、"孤立动作"、"有氧"

    difficulty = Column(String(10))
    # 难度等级：如 "初级"、"中级"、"高级"

    equipment = Column(String(50))
    # 所需器械名：如 "杠铃"、"哑铃"、"自重"

    target_muscles = Column(JSON)
    # 目标肌肉群列表，JSON 数组：["股四头肌", "臀大肌", "竖脊肌"]

    description = Column(Text)
    # 动作详细描述和要领说明，长文本

    common_errors = Column(JSON)
    # 常见错误列表，JSON 数组：["膝盖内扣", "背部弯曲", "重心前移"]

    embedding = Column(Vector(EMBEDDING_DIM))
    # 向量嵌入，维度由 EMBEDDING_DIM 决定（DashScope text-embedding-v4: 1024）
    # 用于语义相似度搜索，pgvector 提供向量索引和余弦相似度计算


# ============================================================
# 知识块表 (knowledge_chunks)
# 存储知识库文档的分块内容和向量嵌入，用于 RAG 检索增强生成
# ============================================================
class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 自增主键

    chunk_id = Column(String(100), nullable=False, unique=True)
    # 分块唯一标识符，如 "nutrition_guide_chunk_005"
    # 由知识摄入流程（knowledge_ingestion）生成，用于去重和追踪

    title = Column(String(200), nullable=False)
    # 知识块标题，如 "蛋白质摄入量计算方法"

    content = Column(Text, nullable=False)
    # 知识块正文内容，用于 LLM 上下文拼接

    source_file = Column(String(200))
    # 来源文件名，如 "nutrition_basics.md"，用于溯源

    chunk_index = Column(Integer)
    # 在原文档中的分块序号（0-based），用于上下文拼接时恢复原始顺序

    metadata_json = Column(JSON, default={})
    # 元数据 JSON，如 {"category": "营养", "tags": ["蛋白质", "增肌"], "page": 3}
    # 灵活扩展，可存储任意结构化元信息

    embedding = Column(Vector(EMBEDDING_DIM))
    # 向量嵌入（512 维），与用户查询做相似度匹配实现语义检索


# ============================================================
# 训练计划表 (training_plans)
# 存储为每个用户生成的训练计划快照
# ============================================================
class TrainingPlan(Base):
    __tablename__ = "training_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 自增主键

    user_id = Column(Integer, nullable=False)
    # 关联的用户 ID，对应 user_profiles.id
    # 注意：当前未设置外键约束，仅做逻辑关联

    goal = Column(String(20))
    # 该计划对应的训练目标（"增肌" 或 "减脂"）

    plan_data = Column(JSON, nullable=False)
    # 训练计划完整数据，JSON 格式
    # 结构：{"weeks": 4, "days": [{"day": 1, "focus": "胸+三头", "exercises": [...]}]}

    confidence = Column(Float, default=0.0)
    # AI 生成该计划的置信度（0.0 ~ 1.0）
    # 低于 HITL_CONFIDENCE_THRESHOLD (0.7) 的计划建议人工审核

    created_at = Column(DateTime, default=datetime.utcnow)
    # 计划生成时间（UTC）


# ============================================================
# 用户上传文档表 (user_documents)
# 存储用户上传的 PDF/Word/MD 文档的解析结果
# ============================================================
class UserDocument(Base):
    __tablename__ = "user_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(128), nullable=False, index=True)
    filename = Column(String(256), nullable=False)
    file_type = Column(String(16), nullable=False)    # pdf / docx / md
    file_size = Column(Integer, default=0)            # 字节数
    raw_content = Column(Text, default="")            # 解析后的纯文本全文
    page_count = Column(Integer, default=1)
    title = Column(String(256), default="")           # 文档标题
    has_text = Column(Integer, default=1)             # 0=扫描件无文字, 1=正常
    parse_error = Column(String(512), default="")     # 解析错误信息
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================
# 文档切块表 (document_chunks)
# 独立于 knowledge_chunks，确保用户文档和公共知识库检索隔离
# ============================================================
class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String(128), nullable=False, index=True)
    chunk_index = Column(Integer, default=0)
    content = Column(Text, nullable=False)            # 已注入标题路径+页码上下文
    chunk_type = Column(String(16), default="text")   # "text" | "table"
    title_path = Column(String(512), default="")      # 章节路径
    page_number = Column(Integer, default=1)
    embedding = Column(Vector(EMBEDDING_DIM))
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================
# 数据库引擎和会话工厂
# ============================================================
# create_engine 创建全局唯一的数据库连接引擎，管理连接池
engine = create_engine(DATABASE_URL)

# sessionmaker 创建 Session 工厂函数，每次调用 SessionLocal() 获得一个新的数据库会话
# PGClient.get_session() 就是通过这个工厂获取 ORM session
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """
    初始化数据库：根据所有 ORM 模型定义创建表结构

    核心逻辑：
        调用 Base.metadata.create_all(engine)
        -> 检查所有继承 Base 的类
        -> 对比数据库现状
        -> 执行 CREATE TABLE IF NOT EXISTS 等 DDL

    调用时机：
        - 应用首次启动时
        - main.py 中 seed() 函数的第一步
        - 生产环境建议改用 Alembic 做数据库迁移管理

    注意：此函数只会"创建不存在的表"，不会修改已有表结构。
          如需变更表结构，需手动执行 ALTER TABLE 或使用 Alembic 迁移。
    """
    Base.metadata.create_all(engine)
