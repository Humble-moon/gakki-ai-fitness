"""
=============================================================================
graphrag/builder.py — 知识图谱构建器（Neo4j 数据写入）
=============================================================================
【项目角色】
    这是 GraphRAG 的数据构建端，负责将健身数据（动作、肌肉、器材、伤病）
    写入 Neo4j 图数据库，建立实体节点和关系边。
    与 graphrag/search.py（查询端）配合使用：Builder 负责"造图"，Search 负责"查图"。

【五层检索关系】
    本模块不直接参与检索，而是为第 4 层 GraphSearch 提供数据基础。
    没有 Builder 构建图谱，GraphSearch 就无法执行图查询。

【被谁调用】数据初始化脚本、应用启动时的 bootstrap 流程
【调用谁】  Neo4jClient.run()（执行 Cypher 语句）、LLMProvider（文本三元组提取）

【两种构建方式】
    1. build_from_seed(): 从结构化数据（exercises 列表）构建图谱
       → 适用于已有结构化数据库导出的数据
    2. extract_triples_with_llm(): 从非结构化文本中用 LLM 提取三元组
       → 适用于科研论文、科普文章等文本的知识抽取

【知识图谱结构回顾】
    节点类型：
        - Exercise（训练动作）: name, difficulty, type
        - Muscle（肌肉）:      name
        - Equipment（器材）:   name
        - Injury（伤病）:      name

    关系类型：
        - TARGETS:      Exercise → Muscle    （该动作训练该肌群）
        - REQUIRES:     Exercise → Equipment （该动作需要使用该器材）
        - MAY_CAUSE:    Exercise → Injury    （该动作可能导致该伤病）
        - RECOVERED_BY: Injury   → Exercise  （该伤病可通过该动作康复）
=============================================================================
"""

from src.storage.neo4j_client import Neo4jClient
from src.llm.provider import LLMProvider


class GraphBuilder:
    """Neo4j 知识图谱构建器。

    【职责】
        1. 初始化图数据库的 Schema（唯一性约束 + 索引）
        2. 从结构化种子数据批量创建节点和关系
        3. 从非结构化文本中利用 LLM 提取知识三元组

    【使用流程】
        1. 应用启动 → GraphBuilder.init_schema() → 创建约束
        2. 数据导入 → GraphBuilder.build_from_seed(exercises) → 构建图谱
        3. 知识扩展 → GraphBuilder.extract_triples_with_llm(text) → LLM 提取三元组
    """

    def __init__(self):
        self.neo4j = Neo4jClient()  # Neo4j 图数据库客户端
        self.llm = LLMProvider()    # LLM 服务，用于非结构化文本的知识抽取

    # =====================================================================
    # Schema 初始化
    # =====================================================================

    def init_schema(self):
        """在 Neo4j 中创建唯一性约束（相当于关系型数据库的 UNIQUE + INDEX）。

        输入：无
        输出：无（副作用：在 Neo4j 中创建约束）

        逻辑：
            为每种实体类型创建 name 字段的唯一性约束，其作用：
            1. 保证每个实体名称在图中唯一（防止重复节点）
            2. 自动为 name 字段创建索引（加速 MATCH ... WHERE name = ... 查询）
            3. 与 MERGE 语句配合：MERGE 先检查约束再决定创建或复用

        为什么使用 CONSTRAINT 而非 INDEX：
            图数据库的约束同时提供唯一性保证和索引加速。
            如果没有唯一性约束，每次 build_from_seed 执行 MERGE 时
            都需要全表扫描检查是否存在同名节点，性能极差。
        """
        constraints = [
            # 每种实体类型都有一个 name 属性的唯一性约束
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Muscle) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Exercise) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (eq:Equipment) REQUIRE eq.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Injury) REQUIRE i.name IS UNIQUE",
        ]
        for c in constraints:
            try:
                self.neo4j.run(c)
            except Exception:
                # 不同 Neo4j 版本对约束语法可能有差异，失败时不中断
                # IF NOT EXISTS 已经做了幂等保证，重复执行也不会报错
                pass

    # =====================================================================
    # 从结构化数据构建图谱
    # =====================================================================

    def build_from_seed(self, exercises: list):
        """从结构化的 exercise 列表构建知识图谱。

        输入：
            exercises: list[dict] — 每个字典包含：
                - name:           str          — 动作名（如"卧推"）
                - difficulty:     str (可选)   — 难度（初级/中级/高级），默认"中级"
                - exercise_type:  str (可选)   — 动作类型（复合/孤立），默认"复合"
                - target_muscles: list[str](可选)— 目标肌群列表（如["胸大肌", "三角肌前束"]）
                - equipment:      str (可选)   — 所需器材（如"杠铃"）

        输出：
            无（副作用：在 Neo4j 中创建节点和关系）

        逻辑：
            对每个动作执行：
            1. MERGE Exercise 节点（如果已存在则复用，否则创建）
               → SET 设置难度和类型属性
            2. 为每个 target_muscle MERGE Muscle 节点 → MERGE TARGETS 关系
            3. 如果指定了 equipment，MERGE Equipment 节点 → MERGE REQUIRES 关系

        为什么使用 MERGE 而非 CREATE：
            CREATE 每次都会创建新节点，即使同名节点已存在，会导致重复。
            MERGE 是"匹配或创建"语义：
                - 如果节点已存在 → 复用已有节点
                - 如果节点不存在 → 创建新节点
            配合唯一性约束，MERGE 能保证幂等性（重复执行不会产生重复数据）。

        default 值使用 Unicode 转义的原因：
            "\u4e2d\u7ea7" = "中级", "\u590d\u5408" = "复合"
            这是为了兼容各种编码环境，防止中文在不同系统中出现乱码。
        """
        # 先初始化 Schema（确保约束存在）
        self.init_schema()

        for ex in exercises:
            # 步骤 1：创建/更新 Exercise 节点
            self.neo4j.run(
                """
                MERGE (e:Exercise {name: $name})
                SET e.difficulty = $difficulty, e.type = $type
                """,
                {
                    "name": ex["name"],
                    "difficulty": ex.get("difficulty", "\u4e2d\u7ea7"),  # 默认"中级"
                    "type": ex.get("exercise_type", "\u590d\u5408"),    # 默认"复合"
                },
            )

            # 步骤 2：创建 Muscle 节点和 TARGETS 关系
            # 一对多：一个动作可能训练多个肌群
            for muscle in ex.get("target_muscles", []):
                self.neo4j.run(
                    """
                    MERGE (m:Muscle {name: $muscle})
                    MERGE (e:Exercise {name: $ex_name})
                    MERGE (e)-[:TARGETS]->(m)
                    """,
                    {"muscle": muscle, "ex_name": ex["name"]},
                )

            # 步骤 3：创建 Equipment 节点和 REQUIRES 关系
            # 一对一：一个动作对应一套器材（若有）
            if ex.get("equipment"):
                self.neo4j.run(
                    """
                    MERGE (eq:Equipment {name: $equip})
                    MERGE (e:Exercise {name: $ex_name})
                    MERGE (e)-[:REQUIRES]->(eq)
                    """,
                    {"equip": ex["equipment"], "ex_name": ex["name"]},
                )

    # =====================================================================
    # 从非结构化文本提取知识
    # =====================================================================

    def extract_triples_with_llm(self, text: str) -> list:
        """利用 LLM 从非结构化健身文本中提取知识三元组。

        输入：
            text: str — 健身相关的非结构化文本（如科普文章、论文摘要）

        输出：
            list[dict] — 提取的三元组列表，每项包含：
                - subject:   str — 主体（动作名）
                - relation:  str — 关系类型（TARGETS/REQUIRES/MAY_CAUSE/RECOVERED_BY）
                - object:    str — 客体（肌肉/器材/伤病名）
                - obj_type:  str — 客体实体类型（Muscle/Exercise/Equipment/Injury）

        使用场景：
            知识库中有如下文本：
            "卧推主要训练胸大肌和三角肌前束，但不当操作可能引起肩袖损伤"
            LLM 提取的三元组：
            [
              {"subject": "卧推", "relation": "TARGETS", "object": "胸大肌", "obj_type": "Muscle"},
              {"subject": "卧推", "relation": "TARGETS", "object": "三角肌前束", "obj_type": "Muscle"},
              {"subject": "卧推", "relation": "MAY_CAUSE", "object": "肩袖损伤", "obj_type": "Injury"}
            ]

        为什么用 LLM 而非正则表达式：
            健身知识表达方式多样，例如"卧推能刺激胸肌"、"卧推可以练到胸部"、
            "做卧推对胸大肌有很好的刺激效果"意思相同但句式各异，
            正则需要穷举所有表达方式，LLM 能理解语义层面的等价性。

        安全兜底：
            如果 LLM 返回的不是 list（如 JSON 解析失败返回 dict），
            isinstance(result, list) 检查后返回空列表，防止后续代码崩溃。
        """
        prompt = (
            f"从以下健身文本中提取（动作-关系-实体）三元组。\n"
            f"实体类型：Muscle, Exercise, Equipment, Injury\n"
            f"关系类型：TARGETS, REQUIRES, MAY_CAUSE, RECOVERED_BY\n\n"
            f"文本：{text}\n\n"
            f'输出 JSON 数组：[{{"subject":"","relation":"","object":"","obj_type":""}}]'
        )
        result = self.llm.chat_with_json_mode(
            [{"role": "user", "content": prompt}]
        )
        # 安全校验：如果 LLM 返回的不是列表（异常情况），返回空列表
        return result if isinstance(result, list) else []
