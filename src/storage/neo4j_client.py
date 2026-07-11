"""
neo4j_client.py - Neo4j 图数据库操作客户端

角色：封装 Neo4j Python 驱动的查询能力，供知识图谱构建和查询模块使用。
      Neo4j 在图项目中存储"动作-肌肉-器械"之间的关联关系，构建健身知识图谱。
被调用者：graphrag.builder（知识图谱构建器）、core 层（图谱查询）。
调用者：Neo4j Python Driver（neo4j 官方驱动）。
"""
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class Neo4jClient:
    """
    Neo4j 图数据库客户端封装类

    职责：管理 Neo4j 驱动连接，提供 Cypher 查询的执行和结果解析能力。
    使用场景：
        - 在 seed 流程中构建健身知识图谱（GraphBuilder.build_from_seed）
        - 在训练计划生成时查询相关动作和肌肉群关系
    设计思想：封装底层驱动细节，对外暴露简洁的 run() 和 query() 接口。
    """

    def __init__(self):
        """
        初始化 Neo4j 驱动

        核心逻辑：
            使用配置中的 URI、用户名、密码创建 GraphDatabase.driver 实例。
            驱动内部维护连接池，session 在每次操作时创建和释放。
        """
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def run(self, query: str, params: dict = None):
        """执行 Cypher 写操作（CREATE/MERGE/DELETE），返回消耗计数。

        对于读操作请使用 query() 方法获取完整结果列表。
        """
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return result.consume()  # 返回 SummaryCounters

    def query(self, query: str, params: dict = None):
        """
        执行 Cypher 查询，返回 Python 字典列表

        输入参数：
            query  : str  - Cypher 查询语句
            params : dict - 查询参数字典，可选
        返回值：
            list[dict] - 每条记录转换为 dict，键为字段名，值为字段值

        核心逻辑：
            1. 创建 session 执行查询
            2. 遍历结果记录，调用 record.data() 将每行转为 dict
            3. 汇总返回列表

        使用场景：
            需要将结果直接序列化为 JSON 或传递给前端时使用
        """
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def close(self):
        """
        关闭 Neo4j 驱动连接

        说明：
            释放驱动持有的连接池资源。
            应在应用关闭时调用，避免连接泄漏。
        """
        self.driver.close()
