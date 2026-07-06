"""
===========================================================================
文件角色：模型路由器 —— 根据任务类型将请求分发到合适的 LLM 模型
===========================================================================
- 被谁调用：LLMProvider 内部，或在 Orchestrator 中按任务类型选择模型
- 调用谁：无（仅返回模型名称字符串，由 LLMProvider 实际创建客户端）
- 核心职责：根据 task_type 返回应使用的模型标识符
- 设计意图：为后续多模型分级调度预留扩展点。
    简单/高频任务 → 便宜快速的模型（如 DeepSeek-V3）
    复杂/核心任务 → 强力模型（如 DeepSeek-R1 / Claude Opus）
    当前阶段统一使用 deepseek-chat，等预算方案确定后再开启分级路由
===========================================================================
"""

from src.llm.provider import LLMProvider


class ModelRouter:
    """模型路由器：根据任务复杂度选择不同模型以平衡成本与质量。
    当前处于单模型阶段，所有任务均路由到 deepseek-chat。
    预留在 Orchestrator 流水线中使用 —— 写入/核查等关键任务可路由到更强模型。"""

    def __init__(self):
        """初始化 LLMProvider 实例。保留引用以便未来实现多模型切换。"""
        self.llm = LLMProvider()

    def route(self, task_type: str) -> str:
        """根据任务类型返回应使用的模型名称字符串。

        输入：
            task_type: str — 任务类型标识，如 "retrieve", "keyword_search", "cache_lookup"
        输出：
            str — LLM 模型标识符（如 "deepseek-chat"）

        当前行为：
            所有任务类型统一返回 "deepseek-chat"。
            注释中标明 simple_tasks 列表，为后续分级路由做准备：
            - simple_tasks（检索/搜索/缓存查询）→ 廉价模型
            - 其他任务（计划生成/写作/核查）→ 强力模型
        """
        # 预留：简单任务使用便宜模型，复杂任务使用强力模型
        simple_tasks = ["retrieve", "keyword_search", "cache_lookup"]
        if task_type in simple_tasks:
            return "deepseek-chat"
        return "deepseek-chat"
