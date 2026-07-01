import logging
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.writer import WriterAgent
from src.agents.fact_checker import FactCheckerAgent
from src.rag.semantic_cache import SemanticCache
from src.skills.registry import SkillRegistry
from src.a2a.messaging import MessageBus, Task, Artifact
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
        # 1. Check cache
        cached = self.cache.get(profile_dict, query)
        if cached:
            logger.info("Cache hit for plan generation")
            return cached

        # 2. Planner
        plan = self.planner.plan(query or f"为{profile.goal}目标生成训练计划", profile_dict)

        # 3. Retriever
        retrieved = self.retriever.retrieve(plan)

        # 4. Writer via A2A
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
        # Normalize LLM output: map variant keys to standard "days"
        days_data = None
        for key in ("weekly_plan", "weekly_schedule", "days", "schedule", "plan"):
            if key in result:
                days_data = result.pop(key)
                break
        if days_data:
            result["days"] = days_data
        for day in result.get("days", []):
            for ex in day.get("exercises", []):
                if "rest_seconds" in ex and "rest" not in ex:
                    ex["rest"] = f"{ex.pop('rest_seconds')}s"
                # Normalize exercise name key
                if "exercise" in ex and "name" not in ex:
                    ex["name"] = ex.pop("exercise")
                if "movement" in ex and "name" not in ex:
                    ex["name"] = ex.pop("movement")
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

        # 6. Cache and return
        self.cache.set(profile_dict, query, result)
        task.complete()
        return result

    def analyze_exercise(self, exercise_name: str, user_desc: str,
                         profile: UserProfileInput) -> dict:
        profile_dict = profile.model_dump()
        retrieved = self.retriever.retrieve({"subtasks": [exercise_name], "skill_config": {}})
        return self.writer.write_analysis(exercise_name, user_desc, retrieved, profile_dict)
