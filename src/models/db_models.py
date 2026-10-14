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
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, Date, Text, Index, Boolean, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
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
# 用户账号表 (users)
# 仅承载登录身份与凭据；健身档案仍存在长期记忆（Redis）里
# ============================================================
class User(Base):
    """登录账号。

    职责边界刻意收窄：只回答「你是谁、凭什么证明」，不回答「你的身高体重」。
    用户的健身档案本来就落在长期记忆（``memory:user:{key}:pref:profile``），
    若在这里再存一份，就会多出第三份档案真相——而 ``user_profiles`` 表
    至今仍是这种"定义了但从未写入"的空表，是这个坑的前车之鉴。

    身份如何接入业务数据：账号的 ``id`` 会被格式化为 ``u:{id}`` 作为
    ``athlete_key`` 使用，从而直接复用训练日志、长期记忆既有的字符串主体键，
    无需改动那些表的 schema。
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 自增主键，同时是业务侧的稳定身份（格式化为 u:{id} 作为 athlete_key）

    username = Column(String(64), unique=True, nullable=False, index=True)
    # 登录名，唯一。用 unique=True 而非独立索引，与 Exercise.name 一致。
    # 长度 64：足够容纳邮箱式用户名，也留出与 athlete_key 上限（64）对齐的余量。

    password_hash = Column(Text, nullable=False)
    # 自描述哈希串，形如 scrypt$n$r$p$<salt_hex>$<hash_hex>。
    # 把算法参数写进串里，是为了将来调强度或换算法时能按条目识别旧格式，
    # 而不必猜"这条是什么年代用什么参数算出来的"。

    is_active = Column(Boolean, default=True, nullable=False)
    # 停用开关。删除账号会连带个人数据，代价不可逆；停用是可逆的第一步。

    created_at = Column(DateTime, default=datetime.utcnow)
    # 记录创建时间（UTC）

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # 记录更新时间（UTC），每次 UPDATE 时自动刷新


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

    __table_args__ = (
        Index(
            "exercises_embedding_hnsw_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
    # HNSW 近似最近邻索引（余弦距离）。此前它只存在于
    # scripts/create_hnsw_indexes.sql 里、不在 ORM 模型中，带来两个后果：
    #   1. 新库靠 create_all 建表时不会创建它，向量检索退化为全表扫描——
    #      不报错，只是慢，小数据量下几乎看不出来；
    #   2. alembic autogenerate 看到「库里有、模型里没有」，会把它判定为
    #      多余索引并在迁移里 DROP 掉。
    # 写进模型后两处都根治：DDL 由 ORM 单点生成，迁移也不会误删。
    # m 与 ef_construction 沿用原脚本取值，当前数据量下够用。


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
    # 向量嵌入（维度由 EMBEDDING_DIM 配置，当前 1024 维），与用户查询做相似度匹配实现语义检索

    __table_args__ = (
        Index(
            "knowledge_chunks_embedding_hnsw_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
    # HNSW 近似最近邻索引，同 Exercise.__table_args__ 的说明


# ============================================================
# 训练计划表 (training_plans)
# 存储为每个用户生成的训练计划快照
# ============================================================
class TrainingPlan(Base):
    __tablename__ = "training_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 自增主键

    owner_key = Column(String(64), nullable=False, index=True)
    # 计划归属的业务身份键：游客为本机 UUID，登录用户为 u:{id}。
    # 与 training_logs.athlete_key 同一约定——身份是一个字符串，不是一个整数。
    # 建表时的 user_id 是整数且从未被写入过任何一行，因此改用这个字段承载归属；
    # 下面的 user_id 保留只为兼容旧库结构，不再参与任何逻辑。

    user_id = Column(Integer, nullable=True)
    # 【已弃用】原设计指向 user_profiles.id，但该表从未被写入，
    # 且整数 ID 与本项目基于字符串的身份体系不兼容。仅为旧库结构保留。

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
# 训练日志主表 (training_logs)
# 一次训练一条记录，动作明细见 training_log_entries
# 主从结构同 user_documents / document_chunks
# ============================================================
class TrainingLog(Base):
    """
    训练日志主表：记录一次完整训练的执行情况

    职责：记录"某天练了什么、练得怎么样"，作为自适应调整建议的数据来源。
    与 training_plans 的区别：计划是 AI 生成的处方，日志是用户执行的事实。
    """

    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    athlete_key = Column(String(64), nullable=False, index=True)
    # 稳定运动员标识：前端首次访问时生成并存入 localStorage，此后永不轮换。
    # 注意不要用 make_user_key(profile) —— 它由身高/体重/目标/伤病哈希而成，
    # 而增肌减脂期体重必然持续变化，等于每周换一个用户，训练史会断裂。
    # make_user_key 的语义是"同一份 profile 命中同一份缓存"，作为缓存键是对的，
    # 误用为持久身份才是错的，因此这里用独立的身份维度。
    log_date = Column(Date, nullable=False, index=True)
    # 训练日期（本地日历日，不含时刻）；与 created_at 的 UTC 写入时刻解耦，
    # 避免东八区晚间训练被记到次日
    day_index = Column(Integer, default=0)
    # 对应训练计划的第几个训练日（1-based）；0 表示自由训练、不关联计划
    focus = Column(String(64), default="")
    # 当日训练重点，如 "胸+三头"
    plan_id = Column(String(32), default="")
    # 该次训练依据的 plan_id，用于溯源到具体计划
    session_rpe = Column(Float)
    # 整节课主观疲劳度 1~10，可空
    duration_min = Column(Integer, default=0)
    # 训练时长（分钟）
    body_weight = Column(Float)
    # 当日体重（kg），可空。与 athlete_key 解耦后仍可单独追踪体重曲线
    notes = Column(String(512), default="")
    # 自由备注
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_training_logs_athlete_date", "athlete_key", "log_date"),
    )
    # 复合索引：历史列表与统计的固定查询形态是
    # WHERE athlete_key = :k AND log_date >= :since ORDER BY log_date


# ============================================================
# 训练日志明细表 (training_log_entries)
# 一次训练里的一个动作一行，是统计聚合与调整建议的最小数据单元
# ============================================================
class TrainingLogEntry(Base):
    """
    训练日志明细表：记录一个动作的实际执行参数

    职责：存储"某个动作做了多少组、每组多重、多少次"，供力量进步曲线与
    训练容量趋势聚合使用。reps 存整数而非计划里的 "8-12" 区间 —— 计划是
    处方（范围），日志是事实（点值），这样 1RM 估算与容量计算可直接算出，
    不需要解析字符串。
    """

    __tablename__ = "training_log_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    log_id = Column(Integer, nullable=False, index=True)
    # 关联 training_logs.id（逻辑关联，无外键约束，同 training_plans.user_id 的既有约定）
    athlete_key = Column(String(64), nullable=False, index=True)
    # 冗余主体标识，避免按动作跨会话聚合时回表 JOIN（同 document_chunks.session_id 的做法）
    exercise_name = Column(String(100), nullable=False, index=True)
    # 动作中文名，与 exercises.name 及计划中的 name 对齐
    sets = Column(Integer, default=0)
    # 实际完成组数
    reps = Column(Integer, default=0)
    # 每组实际次数（代表值）
    weight = Column(Float, default=0.0)
    # 使用重量（kg）；自重动作记 0
    rpe = Column(Float)
    # 该动作主观疲劳度 1~10，可空
    completed = Column(Integer, default=1)
    # 1=按计划完成, 0=跳过/未完成；完成率 = AVG(completed)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_training_log_entries_athlete_exercise", "athlete_key", "exercise_name"),
    )
    # 复合索引：单动作进步曲线的查询形态是
    # WHERE athlete_key = :k AND exercise_name = :n ORDER BY created_at


# ============================================================
# 体测记录表 (body_measurements)
# 不定期的身体围度与成分快照，是"力量涨了但人有没有变"的另一半答案
# ============================================================
class BodyMeasurement(Base):
    """体测记录表：某个时间点的身体围度与成分。

    与 training_logs.body_weight 的分工：
        那张表记的是"训练当天的体重"，依附于一次训练，字段只有体重一项；
        这张表是独立的体测快照——不要求当天训练，且包含体脂、肌肉量与
        各部位围度。两者不合并：一次深蹲训练顺手称的体重，和一次专门量的
        体测，可信度与用途都不同，混在一起会让趋势线出现假的尖峰。

    为什么围度字段是独立列而不是一个 JSON：
        查询形态固定是"某个部位的围度随时间怎么变"，独立列可以直接
        写 WHERE/SELECT 与建索引，JSON 里取字段则要每次解析。
        代价是新增部位要改表——但人体围度的部位集合是稳定的。
    """

    __tablename__ = "body_measurements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    athlete_key = Column(String(64), nullable=False, index=True)
    # 与 training_logs 同一套身份维度，见该表关于 make_user_key 的说明
    measure_date = Column(Date, nullable=False, index=True)
    # 测量日期（本地日历日），与 created_at 的解耦理由同 training_logs

    weight = Column(Float)
    # 体重（kg）
    body_fat_pct = Column(Float)
    # 体脂率（%），可空——家用秤与体脂钳的读数量级差异大，不强求
    muscle_mass = Column(Float)
    # 肌肉量（kg），可空

    chest = Column(Float)
    # 胸围（cm）
    waist = Column(Float)
    # 腰围（cm）
    hip = Column(Float)
    # 臀围（cm）
    arm = Column(Float)
    # 臂围（cm，惯用侧）
    thigh = Column(Float)
    # 大腿围（cm，惯用侧）

    notes = Column(String(512), default="")
    # 自由备注，如"晨起空腹""饭后"
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_body_measurements_athlete_date", "athlete_key", "measure_date"),
    )
    # 复合索引：趋势查询的固定形态是
    # WHERE athlete_key = :k AND measure_date >= :since ORDER BY measure_date


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
        1. 确保必需扩展存在（pgvector: embedding 列；pg_trgm: 关键词检索的 similarity/% 操作符）
        2. 调用 Base.metadata.create_all(engine)
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
    # 第 1 步：确保必需扩展已启用
    #   - pgvector: exercises / knowledge_chunks 的 embedding 列使用 vector 类型，
    #     缺失时报 type "vector" does not exist，create_all 会直接失败
    #   - pg_trgm:  knowledge_search.keyword_search 用 similarity() 与 % 操作符做关键词检索，
    #     缺失时报 function similarity(text, unknown) does not exist
    # 两者都必须先于建表/查询存在。全新数据库（例如刚起的 docker compose 数据卷）
    # 正是这种情况，而 scripts/create_hnsw_indexes.sql 要在建表"之后"才能跑，
    # 兜不住这个先后顺序，所以在这里幂等补齐。
    if engine.dialect.name == "postgresql":
        try:
            with engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        except SQLAlchemyError:
            # 权限不足等场景不阻塞建表；若确实缺扩展，create_all 会报出明确错误
            pass

    # 第 2 步：创建所有表结构
    Base.metadata.create_all(engine)
