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
    """Initialize database and knowledge graph with seed data."""
    init_db()
    pg = PGClient()
    emb = EmbeddingService()
    with open("data/seed_exercises.json", "r", encoding="utf-8") as f:
        exercises = json.load(f)
    for ex in exercises:
        text = f"{ex['name']} {' '.join(ex['target_muscles'])} {ex.get('description', '')}"
        vec = emb.embed(text)
        vec_str = f"[{','.join(str(v) for v in vec)}]"
        try:
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
    builder = GraphBuilder()
    builder.build_from_seed(exercises)
    logger.info(f"Seeded {len(exercises)} exercises into PG and Neo4j")

def main():
    parser = argparse.ArgumentParser(description="AI Fitness Coach")
    parser.add_argument("--search", type=str, help="Natural language query for plan generation")
    parser.add_argument("--height", type=float, default=180)
    parser.add_argument("--weight", type=float, default=80)
    parser.add_argument("--years", type=float, default=1, help="Training years")
    parser.add_argument("--goal", type=str, default="增肌")
    parser.add_argument("--equipment", type=str, default="哑铃,杠铃", help="Comma-separated equipment")
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument("--seed", action="store_true", help="Seed database with exercise data")
    parser.add_argument("--analyze", type=str, help="Analyze exercise, format: 动作名:感受描述")
    args = parser.parse_args()

    if args.seed:
        seed()
        logger.info("Seed complete!")
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
    else:
        query = args.search or f"为{args.goal}目标生成训练计划"
        result = orch.generate_plan(profile, query)
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
