"""
main.py - CLI（命令行）程序入口

角色：项目的命令行交互入口，支持以下操作模式：
      1. 生成训练计划（--search / 默认模式）
      2. 分析动作姿势（--analyze）
      3. 初始化种子数据（--seed）
      4. 导入知识库文档（--ingest-knowledge）
被调用者：终端用户通过 python src/main.py <参数> 启动。
调用者：
    - src.core.orchestrator（核心编排器）
    - src.models.db_models（数据库初始化）
    - src.models.schemas（用户输入校验）
    - src.graphrag.builder（知识图谱构建）
    - src.rag.embedding（向量嵌入服务）
    - src.rag.knowledge_ingestion（知识库导入）
    - src.storage.pg（数据库写入）
"""
import argparse
import json
import logging
from src.core.orchestrator import Orchestrator
from src.models.schemas import UserProfileInput
from src.models.db_models import init_db
from src.graphrag.builder import GraphBuilder
from src.rag.embedding import EmbeddingService
from src.storage.pg import PGClient

# 配置日志格式：时间戳 [级别] 消息内容
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def seed():
    """
    种子数据初始化函数

    功能：将 data/seed_exercises.json 中的预置动作数据写入 PostgreSQL 和 Neo4j，
         完成数据库初始化和知识图谱构建。

    核心逻辑步骤：
        1. init_db() - 创建所有数据库表（如果不存在）
        2. 读取种子 JSON 文件
        3. 遍历每个动作：
           a. 拼接动作名+肌肉群+描述为一段文本
           b. 调用 embedding 服务生成 512 维向量
           c. 将向量转为 pgvector 可接受的字符串格式 "[x1,x2,...,x512]"
           d. INSERT 到 exercises 表（ON CONFLICT DO NOTHING 防止重复导入）
        4. 调用 GraphBuilder 将动作数据导入 Neo4j 构建知识图谱
        5. 输出导入日志

    被调用者：main() 函数中解析到 --seed 参数时触发。
    调用者：PGClient（写入 Postgres）、EmbeddingService（生成向量）、GraphBuilder（写入 Neo4j）。
    """
    # 第 1 步：确保数据库表结构存在
    init_db()
    pg = PGClient()
    emb = EmbeddingService()

    # 第 2 步：读取种子动作数据文件
    with open("data/seed_exercises.json", "r", encoding="utf-8") as f:
        exercises = json.load(f)

    # 第 3 步：逐个导入动作到 PostgreSQL
    for ex in exercises:
        # 拼接文本用于生成向量嵌入
        text = f"{ex['name']} {' '.join(ex['target_muscles'])} {ex.get('description', '')}"
        vec = emb.embed(text)  # 调用 embedding 模型生成 512 维向量
        vec_str = f"[{','.join(str(v) for v in vec)}]"  # 转为 pgvector 格式

        try:
            # INSERT ... ON CONFLICT DO NOTHING: 如果动作名已存在则跳过
            # CAST(:vec AS vector): 将字符串转为 pgvector 的 vector 类型
            pg.execute("""
                INSERT INTO exercises (name, exercise_type, difficulty, equipment,
                                       target_muscles, description, common_errors, embedding)
                VALUES (:name, :type, :diff, :equip, :muscles, :desc, :errors, CAST(:vec AS vector))
                ON CONFLICT (name) DO NOTHING
            """, {"name": ex["name"], "type": ex["exercise_type"], "diff": ex["difficulty"],
                  "equip": ex["equipment"], "muscles": json.dumps(ex["target_muscles"]),
                  "desc": ex.get("description", ""), "errors": json.dumps(ex.get("common_errors", [])),
                  "vec": vec_str})
        except Exception as e:
            logger.warning(f"Insert {ex['name']} failed: {e}")

    # 第 4 步：构建 Neo4j 知识图谱（动作-肌肉-器械关系）
    builder = GraphBuilder()
    builder.build_from_seed(exercises)
    logger.info(f"Seeded {len(exercises)} exercises into PG and Neo4j")


def main():
    """
    CLI 主入口函数

    功能：解析命令行参数，根据参数分发到不同的操作模式。

    支持的命令行参数：
        --search          自然语言查询，用于生成训练计划
        --height          身高（cm），默认 180
        --weight          体重（kg），默认 80
        --years           训练年限，默认 1
        --goal            目标（增肌/减脂），默认 增肌
        --equipment       可用器械（逗号分隔），默认 哑铃,杠铃
        --days            每周训练天数，默认 4
        --seed            初始化数据库种子数据
        --ingest-knowledge 导入知识库文档
        --knowledge-dir   知识文档目录，默认 data/knowledge
        --analyze         分析动作格式：动作名:感受描述

    操作模式：
        1. --seed 模式：调用 seed() 初始化数据
        2. --ingest-knowledge 模式：调用 ingest() 导入知识文档
        3. --analyze 模式：调用 orchestrator 分析动作
        4. 默认模式：调用 orchestrator 生成训练计划
    """
    parser = argparse.ArgumentParser(description="AI 健身教练")
    parser.add_argument("--search", type=str, help="用于生成训练计划的自然语言查询")
    parser.add_argument("--height", type=float, default=180)
    parser.add_argument("--weight", type=float, default=80)
    parser.add_argument("--years", type=float, default=1, help="训练年限")
    parser.add_argument("--goal", type=str, default="增肌")
    parser.add_argument("--equipment", type=str, default="哑铃,杠铃", help="逗号分隔的器械列表")
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument("--seed", action="store_true", help="向数据库写入运动数据种子")
    parser.add_argument("--ingest-knowledge", action="store_true", help="导入知识库文档")
    parser.add_argument("--knowledge-dir", type=str, default="data/knowledge", help="知识文档目录")
    parser.add_argument("--analyze", type=str, help="分析动作，格式：动作名:感受描述")
    args = parser.parse_args()

    # ===== 模式 1：种子数据初始化 =====
    if args.seed:
        seed()
        logger.info("Seed complete!")
        return

    # ===== 模式 2：知识库文档导入 =====
    if args.ingest_knowledge:
        from src.rag.knowledge_ingestion import ingest
        ingest(args.knowledge_dir)
        logger.info("Knowledge ingestion complete!")
        return

    # 初始化编排器（加载所有子模块：LLM、RAG、缓存等）
    orch = Orchestrator()

    # 构建用户档案对象（Pydantic 会自动校验参数合法性）
    profile = UserProfileInput(
        height=args.height, weight=args.weight, training_years=args.years,
        goal=args.goal, available_equipment=args.equipment.split(","),
        days_per_week=args.days
    )

    # ===== 模式 3：动作分析 =====
    if args.analyze:
        # 解析 --analyze 参数：用第一个冒号分隔动作名和描述
        # 例如：--analyze "杠铃深蹲:膝盖内扣，腰部有酸痛感"
        parts = args.analyze.split(":", 1)
        name = parts[0].strip()            # "杠铃深蹲"
        desc = parts[1].strip() if len(parts) > 1 else ""  # "膝盖内扣..."
        result = orch.analyze_exercise(name, desc, profile)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # ===== 模式 4（默认）：生成训练计划 =====
        query = args.search or f"为{args.goal}目标生成训练计划"
        result = orch.generate_plan(profile, query)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
