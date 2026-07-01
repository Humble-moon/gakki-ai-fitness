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
