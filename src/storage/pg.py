"""
pg.py - PostgreSQL 数据库操作客户端

角色：封装 SQLAlchemy 的原始 SQL 执行能力，为上层业务模块提供简洁的数据库操作接口。
      这是项目中所有关系型数据库读写的唯一入口。
被调用者：core.orchestrator（编排器）、rag（检索模块）、graphrag（图构建模块）。
调用者：SQLAlchemy engine（数据库连接引擎，在 db_models.py 中创建）。
"""
from sqlalchemy import text
from src.models.db_models import engine, SessionLocal


class PGClient:
    """
    PostgreSQL 客户端封装类

    职责：提供原始 SQL 执行、查询和 ORM session 管理能力。
    使用场景：在 Orchestrator 编排流程中执行数据读写，在 seed 流程中批量插入数据。
    设计思想：不走 ORM，直接用 text() 执行 SQL，保持灵活性和性能。
    """

    def __init__(self):
        """
        初始化客户端，复用 db_models.py 创建的全局数据库引擎。
        不创建新连接，避免连接池膨胀。
        """
        self.engine = engine

    def execute(self, query: str, params: dict = None):
        """
        执行写操作（INSERT / UPDATE / DELETE / DDL）

        输入参数：
            query  : str  - 要执行的 SQL 语句（支持 :param 命名参数占位符）
            params : dict - SQL 参数字典，键名对应 SQL 中的 :param 占位符，可选
        返回值：
            sqlalchemy CursorResult 对象（包含 rowcount 等信息）

        核心逻辑：
            1. 从引擎获取一个连接（自动从连接池复用）
            2. 用 text() 将字符串转为 SQLAlchemy 可执行对象
            3. 执行 SQL 并自动提交事务
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            conn.commit()
            return result

    def fetch_all(self, query: str, params: dict = None):
        """
        执行查询并返回所有行

        输入参数：
            query  : str  - SELECT 查询语句
            params : dict - SQL 参数字典，可选
        返回值：
            list[Row] - 查询结果行列表，每行是 sqlalchemy Row 对象（可用索引或键名访问）

        核心逻辑：
            执行 SQL -> 调用 fetchall() 一次性获取全部结果行
            注意：结果集大时需注意内存占用
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchall()

    def fetch_one(self, query: str, params: dict = None):
        """
        执行查询并返回第一行

        输入参数：
            query  : str  - SELECT 查询语句
            params : dict - SQL 参数字典，可选
        返回值：
            Row | None - 查询到的第一行，无结果时返回 None

        核心逻辑：
            执行 SQL -> 调用 fetchone() 获取单行结果
            适用于按主键查询或 LIMIT 1 的场景
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchone()

    def get_session(self):
        """
        获取一个新的 SQLAlchemy ORM Session

        返回值：
            Session 对象 - 用于 ORM 操作（add / commit / rollback / query）

        使用场景：
            当需要使用 ORM 方式（而非原始 SQL）操作数据库时调用
            调用方需自行管理 session 的生命周期（commit / close）
        """
        return SessionLocal()

    def close(self):
        """
        关闭客户端（当前为空实现）

        说明：由于复用了全局 engine，关闭操作应在应用退出时统一处理。
        此处预留接口供将来扩展（如需要关闭独立的连接池）。
        """
        pass
