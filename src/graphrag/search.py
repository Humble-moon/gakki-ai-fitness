"""
=============================================================================
graphrag/search.py — 知识图谱检索（第 4 层，Neo4j + Cypher）
=============================================================================
【项目角色】
    这是 RAG 五层检索体系中的第 4 层 — 知识图谱推理层。
    基于 Neo4j 图数据库，利用 Cypher 查询语言对健身知识图谱做结构化推理：
    - 动作 → 训练 → 目标肌群（多跳关系查询）
    - 动作 → 可能造成 → 伤病（风险推理）
    - 伤病 → 康复动作（康复路径推理）
    图结构的优势在于能表示传统向量/关键词无法捕获的实体间复杂关系。

【五层检索关系】
    第 0 层:  EmbeddingService（embedding.py）—— 文本 → 向量
    第 1 层:  VectorSearch   —— 向量语义匹配
    第 1 层:  KeywordSearch  —— 关键词字面匹配
    第 2 层:  KnowledgeSearch —— RRF 融合 + 重排序
    第 3 层:  AgenticRAG     —— 迭代评估 + 查询改写
    第 4 层:  GraphSearch（本文件）—— 知识图谱推理                   ← 你在这里
    第 5 层:  SemanticCache  —— 语义缓存加速

【被谁调用】应用层 API（当用户询问伤病风险、康复建议、器材+肌肉组合查询时）
【调用谁】  Neo4jClient.query()（执行 Cypher 图查询）

【为什么需要 GraphRAG（知识图谱检索）】
    向量检索和关键词检索都是"扁平"的——它们只知道文档片段和查询的相关性，
    不知道"卧推"→ TARGETS → "胸大肌" → SYNERGIST → "三角肌前束" 这种层级结构关系。
    知识图谱把实体（动作、肌肉、器材、伤病）和关系（训练、需要、可能造成、康复）
    存储为图结构，可以做：
    - 多跳推理：哑铃 + 练胸 → 找到所有用哑铃且练胸的动作
    - 伤病关联：卧推 → 可能造成 → 肩袖损伤 → 康复动作 → 肩外旋
    - 因果分析：某个动作可能引起的伤病链

【知识图谱结构】
    节点类型: Exercise（训练动作）, Muscle（肌肉）, Equipment（器材）, Injury（伤病）
    关系类型:
        - TARGETS:     Exercise → Muscle      （该动作训练该肌群）
        - REQUIRES:    Exercise → Equipment   （该动作需要使用该器材）
        - MAY_CAUSE:   Exercise → Injury      （该动作可能导致该伤病）
        - RECOVERED_BY:Injury   → Exercise   （该伤病可通过该动作康复）
=============================================================================
"""

from src.storage.neo4j_client import Neo4jClient


class GraphSearch:
    """Neo4j 知识图谱检索类。

    【职责】
        对 Neo4j 中存储的健身知识图谱执行结构化 Cypher 查询，
        支持单跳/多跳关系推理、伤病风险分析、康复路径推荐。

    【使用场景】
        - 用户问"练哪些动作可以练到肱二头肌" → find_exercises_by_muscle("肱二头肌")
        - 用户问"卧推需要什么器材" → find_equipment_for_exercise("卧推")
        - 用户问"用哑铃练胸的动作有哪些" → multi_hop_search("哑铃", "胸")
        - 用户问"卧推会不会伤到肩膀" → reason_about_pain("卧推", "肩膀疼")
    """

    def __init__(self):
        self.neo4j = Neo4jClient()  # Neo4j 图数据库客户端

    # =====================================================================
    # 单跳查询：动作 → 肌肉
    # =====================================================================

    def find_exercises_by_muscle(self, muscle: str, limit: int = 10) -> list:
        """根据目标肌肉名查找训练动作（1 跳：动作 → TARGETS → 肌肉）。

        输入：
            muscle: str — 肌肉名称（支持模糊匹配，如"胸"匹配"胸大肌"、"胸小肌"）
            limit:  int — 返回数量上限

        输出：
            list[dict] — 每个字典包含 name（动作名）、difficulty（难度）、
                         type（动作类型）、muscle（匹配到的肌肉名）

        Cypher 逻辑：
            MATCH (e:Exercise)-[:TARGETS]->(m:Muscle)
            从 Exercise 节点出发，沿 TARGETS 关系找到 Muscle 节点
            CONTAINS 做子串模糊匹配（如搜"胸"匹配"胸大肌"、"胸小肌"）
        """
        results = self.neo4j.query(
            """
            MATCH (e:Exercise)-[:TARGETS]->(m:Muscle)
            WHERE m.name CONTAINS $muscle
            RETURN e.name AS name, e.difficulty AS difficulty,
                   e.type AS type, m.name AS muscle
            LIMIT $limit
            """,
            {"muscle": muscle, "limit": limit},
        )
        return results

    # =====================================================================
    # 单跳查询：动作 → 器材
    # =====================================================================

    def find_equipment_for_exercise(self, exercise: str) -> list:
        """查找指定动作所需的器材（1 跳：动作 → REQUIRES → 器材）。

        输入：
            exercise: str — 动作名称（精确匹配，如"卧推"）

        输出：
            list[dict] — 每项包含 equipment（器材名称）
        """
        results = self.neo4j.query(
            """
            MATCH (e:Exercise {name: $name})-[:REQUIRES]->(eq:Equipment)
            RETURN eq.name AS equipment
            """,
            {"name": exercise},
        )
        return results

    # =====================================================================
    # 多跳查询：器材 + 肌肉 → 动作（图结构的核心优势）
    # =====================================================================

    def multi_hop_search(self, equipment: str, target: str) -> list:
        """多跳组合查询：同时按器材和肌肉筛选动作（2 跳）。

        输入：
            equipment: str — 器材名称（模糊匹配）
            target:    str — 目标肌肉（模糊匹配）

        输出：
            list[dict] — 匹配的动作，包含名称、难度、聚合的肌肉列表和器材列表

        Cypher 逻辑：
            从 Exercise 节点出发，同时沿 REQUIRES 和 TARGETS 两条边分别匹配：
            Exercise → REQUIRES → Equipment（where eq.name CONTAINS equipment）
            Exercise → TARGETS → Muscle（where m.name CONTAINS target）
            collect(DISTINCT ...) 聚合多个匹配的肌肉/器材为数组

        为什么这是图检索的核心优势：
            关系型数据库需要 JOIN 两张表再子查询，SQL 会非常复杂。
            图数据库的路径遍历天然支持多跳查询，性能远优于关系表 JOIN。
        """
        results = self.neo4j.query(
            """
            MATCH (e:Exercise)-[:REQUIRES]->(eq:Equipment)
            WHERE eq.name CONTAINS $equipment
            MATCH (e)-[:TARGETS]->(m:Muscle)
            WHERE m.name CONTAINS $target
            RETURN e.name AS name, e.difficulty AS difficulty,
                   collect(DISTINCT m.name) AS muscles,
                   collect(DISTINCT eq.name) AS equipment
            """,
            {"equipment": equipment, "target": target},
        )
        return results

    # =====================================================================
    # 伤病风险查询
    # =====================================================================

    def find_injury_risks(self, exercise: str) -> list:
        """查找某动作可能导致的伤病（1 跳：动作 → MAY_CAUSE → 伤病）。

        输入：
            exercise: str — 动作名称（精确匹配）

        输出：
            list[dict] — 每项包含 injury（伤病名称），如"肩袖损伤"、"腰椎间盘突出"

        应用场景：
            用户问"深蹲会不会伤膝盖？"→ 调用此方法查询，返回深蹲相关的伤病列表
        """
        results = self.neo4j.query(
            """
            MATCH (e:Exercise)-[:MAY_CAUSE]->(i:Injury)
            WHERE e.name CONTAINS $name
            RETURN i.name AS injury
            """,
            {"name": exercise},
        )
        return results

    # =====================================================================
    # 康复路径查询（2 跳逆向推理）
    # =====================================================================

    def find_rehab_exercises(self, injury: str) -> list:
        """查找某伤病的康复动作及其需要避免的动作（2 跳逆向推理）。

        输入：
            injury: str — 伤病名称（精确匹配）

        输出：
            list[dict] — 每项包含：
                - rehab_exercise: str — 推荐的康复动作
                - avoid_exercises: list[str] — 应该避免的动作列表

        Cypher 逻辑（双向匹配）：
            1. (bad:Exercise)-[:MAY_CAUSE]->(i:Injury)
               → 找到可能造成该伤病的动作（应该避免）
            2. (i:Injury)-[:RECOVERED_BY]->(rehab:Exercise)
               → 找到该伤病的康复动作（应该执行）
            collect(DISTINCT bad.name) 聚合所有需要避免的动作

        应用场景：
            用户说"我有肩袖损伤，还能做什么动作？"
            → 推荐康复动作 + 警告避免的动作
        """
        results = self.neo4j.query(
            """
            MATCH (i:Injury {name: $injury})<-[:MAY_CAUSE]-(bad:Exercise)
            MATCH (i)-[:RECOVERED_BY]->(rehab:Exercise)
            RETURN rehab.name AS rehab_exercise,
                   collect(DISTINCT bad.name) AS avoid_exercises
            """,
            {"injury": injury},
        )
        return results

    # =====================================================================
    # 综合推理：疼痛原因分析 + 康复方案
    # =====================================================================

    def reason_about_pain(self, exercise: str, symptom: str) -> dict:
        """综合推理：分析某动作引起症状的可能原因并给出康复建议。

        输入：
            exercise: str — 用户正在做的动作（如"卧推"）
            symptom:  str — 用户出现的症状（如"肩膀疼"）

        输出：
            dict — 包含：
                - exercise:        str       — 原动作名
                - symptom:         str       — 症状描述
                - possible_causes: list[dict]— 可能原因（伤病名称 + 应避免的动作）
                - suggested_rehab: list[str] — 建议的康复动作（去重后）
                - source:          "graph"   — 标记来源为图检索

        推理链：
            1. 查找该动作可能导致的伤病（MAY_CAUSE 关系）
            2. 对每个可能的伤病，查找康复动作和应避免的动作
            3. 汇总并去重返回

        为什么这个方法是图检索在项目中的高价值应用：
            传统的向量检索只能找到"肩部疼痛"相关的文章片段，
            但无法推理出"卧推可能引起肩袖损伤，建议做肩外旋康复动作"这样的因果链。
            图结构天然支持这种 multi-hop 推理。
        """
        # 步骤 1：查找该动作可能导致的伤病
        risks = self.find_injury_risks(exercise)
        causes = []
        solutions = []

        # 步骤 2：对每个可能的伤病做康复路径推理
        for r in risks:
            rehab = self.find_rehab_exercises(r["injury"])
            for item in rehab:
                # 记录可能的伤病原因和需要避免的动作
                causes.append(
                    {
                        "injury": r["injury"],
                        "avoid": item.get("avoid_exercises", []),
                    }
                )
                # 收集康复动作
                rehab_ex = item.get("rehab_exercise")
                if rehab_ex:
                    solutions.append(rehab_ex)

        # 步骤 3：组装结果，去重康复动作（同一个康复动作可能被多种伤病推荐）
        return {
            "exercise": exercise,
            "symptom": symptom,
            "possible_causes": causes,           # 可能的原因列表
            "suggested_rehab": list(set(solutions)),  # 去重后的康复建议
            "source": "graph",                   # 标记来源，便于前端展示
        }
