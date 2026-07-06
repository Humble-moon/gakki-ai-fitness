"""
===========================================================================
文件角色：A2A (Agent-to-Agent) 消息传递系统 —— 定义任务、产物和消息总线的数据结构
===========================================================================
- 被谁调用：Orchestrator 在流水线各步骤间通过 MessageBus 记录任务流转
- 调用谁：无外部依赖，仅使用 Python 标准库
- 核心职责：
    1. Task: 表示一个 A2A 任务 —— 谁发起的、发给谁、做什么、当前状态
    2. Artifact: 表示任务执行过程中产生的产物（训练计划、分析报告、安全审查结果）
    3. TaskStatus: 枚举任务生命周期状态
    4. MessageBus: 简单的内存消息总线，记录所有任务及其产物
- 设计理念：
    - 这不是真正的消息队列（如 RabbitMQ/Kafka），而是内存中的任务日志
    - 所有 Agent 在当前进程中同步执行，MessageBus 仅用于追踪和调试
    - Task 和 Artifact 的结构参考了 Google A2A 协议的设计思路
- 在项目中的角色：可观测性（Observability）—— 通过 Task/Artifact 记录
  每次生成计划的完整执行链路，便于调试问题、统计在哪个环节出了错
===========================================================================
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class TaskStatus(Enum):
    """任务生命周期状态枚举。

    状态流转：PENDING → IN_PROGRESS → COMPLETED / FAILED / NEEDS_REVIEW
    - PENDING:      任务已创建，等待处理
    - IN_PROGRESS:  任务正在被目标 Agent 处理中
    - COMPLETED:    任务成功完成
    - FAILED:       任务执行失败
    - NEEDS_REVIEW: 任务完成但需要人工审核（由 FactChecker + HITL 触发）
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


@dataclass
class Artifact:
    """产物数据类：记录 Agent 执行任务时生成的工作成果。

    字段说明：
        artifact_id: str — 产物唯一标识，通常关联到对应 Task 的 task_id
        artifact_type: str — 产物类型，用于前端区分展示：
            "training_plan"    → 训练计划
            "analysis_report"  → 动作分析报告
            "safety_check"     → 安全审查结果
        content: dict — 产物的实际内容（JSON 可序列化的字典）
        created_at: str — ISO 8601 格式的创建时间戳，自动设置为当前时间

    在整个流程中的位置：
        Orchestrator 在 Writer 或 FactChecker 完成后，将其输出封装为 Artifact
        并挂载到对应的 Task 上，形成完整的任务-产物追溯链。
    """
    artifact_id: str
    artifact_type: str  # 产物类型："training_plan" | "analysis_report" | "safety_check"
    content: dict
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Task:
    """任务数据类：表示一个 Agent 间任务，记录完整的工作流上下文。

    字段说明：
        task_id: str — 任务唯一标识
        from_agent: str — 发起方 Agent 名称，如 "orchestrator"、"writer"
        to_agent: str — 接收方 Agent 名称，如 "writer"、"fact_checker"
        task_type: str — 任务类型，如 "generate_plan"、"safety_check"
        payload: dict — 任务携带的数据载荷
        status: TaskStatus — 当前状态，初始为 PENDING
        artifacts: list[Artifact] — 该任务执行过程中产生的所有产物

    方法说明：
        add_artifact(artifact): 将产物追加到 artifacts 列表，用于记录工作成果
        complete(): 标记任务为 COMPLETED，由 Orchestrator 在流水线成功完成后调用
        fail(): 标记任务为 FAILED（当前代码中预留，尚未集成到异常处理流程）
    """
    task_id: str
    from_agent: str
    to_agent: str
    task_type: str
    payload: dict
    status: TaskStatus = TaskStatus.PENDING
    artifacts: list = field(default_factory=list)

    def add_artifact(self, artifact: Artifact):
        """将产物挂载到任务上。一个 Task 可以挂多个 Artifact，
        例如：Writer 产出训练计划 → FactChecker 产出安全报告。"""
        self.artifacts.append(artifact)

    def complete(self):
        """标记任务完成。由 Orchestrator 在 generate_plan 成功返回前调用。"""
        self.status = TaskStatus.COMPLETED

    def fail(self):
        """标记任务失败。预留接口，当前未集成到异常处理流程中。"""
        self.status = TaskStatus.FAILED


class MessageBus:
    """内存消息总线：集中管理所有 Agent 间的任务流转。

    这不是真正的消息队列（无异步、无持久化、无消费者模式），
    而是一个中心化的任务日志存储。所有 Agent 在同步进程中执行，
    MessageBus 的作用是：
        1. 记录每个任务的完整生命周期
        2. 提供按目标 Agent 查询待处理任务的能力
        3. 为调试和监控提供任务追溯数据

    在 Orchestrator 中使用模式：
        bus.send(task)          → 记录"Writer 收到了生成计划的指令"
        task.add_artifact(...)  → 记录"Writer 产出了训练计划"
        task.complete()         → 记录"任务完成"
    """

    def __init__(self):
        """初始化空任务列表。MessageBus 实例由 Orchestrator 持有，生命周期同应用进程。"""
        self.tasks: list[Task] = []

    def send(self, task: Task):
        """发送（记录）一个任务到总线。

        输入：
            task: Task — 待记录的任务实例
        输出：
            Task — 返回同一个 task 实例，支持链式调用

        当前实现：简单 append 到列表。
        未来可扩展为：写入数据库、发送到外部消息队列、触发 Webhook 等。
        """
        self.tasks.append(task)
        return task

    def get_for_agent(self, agent_name: str) -> list:
        """查询指定 Agent 的待处理任务列表。

        输入：
            agent_name: str — 目标 Agent 的名称
        输出：
            list[Task] — 发给该 Agent 且状态为 PENDING 的所有任务

        用途：在异步/队列模式下，Agent 轮询自己的待处理任务。
              当前同步模式下，Orchestrator 直接调用 Agent 方法，此方法预留。
        """
        return [t for t in self.tasks if t.to_agent == agent_name and
                t.status == TaskStatus.PENDING]
