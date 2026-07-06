"""
schemas.py - API 数据模型 / Pydantic Schema 定义

角色：使用 Pydantic 定义 API 层（CLI 和 FastAPI）的请求和响应数据结构。
      这些 Schema 负责数据校验、序列化/反序列化、类型提示和自动文档生成。
被调用者：
    - src/main.py（CLI 入口，用 UserProfileInput 校验命令行参数）
    - app/server.py（FastAPI 路由，用 Pydantic 模型校验 HTTP 请求体）
    - core.orchestrator（编排器，接收和返回这些模型的实例）
调用者：Pydantic v2 运行时。
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# ============================================================
# 用户输入模型
# ============================================================
class UserProfileInput(BaseModel):
    """
    用户健身档案的输入 Schema

    职责：校验用户提交的身体参数和训练偏好，确保数据在合法范围内。
    使用场景：CLI 命令行参数校验、FastAPI POST 请求体校验。
    """

    height: float = Field(..., ge=100, le=250, description="身高(cm)")
    # 身高（厘米），合法范围 100~250cm

    weight: float = Field(..., ge=30, le=200, description="体重(kg)")
    # 体重（公斤），合法范围 30~200kg

    training_years: float = Field(..., ge=0, le=30, description="训练年限")
    # 训练年限，合法范围 0~30 年

    goal: str = Field(..., pattern="^(增肌|减脂)$")
    # 训练目标，正则约束只能为 "增肌" 或 "减脂"

    available_equipment: List[str] = Field(..., min_length=1)
    # 可用器械列表，至少包含一项，如 ["哑铃", "杠铃"]

    days_per_week: int = Field(..., ge=1, le=7)
    # 每周训练天数，合法范围 1~7

    injuries: List[str] = Field(default=[])
    # 伤病史列表，默认空列表，如 ["腰椎间盘突出"]

    preferences: dict = Field(default={})
    # 用户偏好，灵活的键值对字典


# ============================================================
# 训练计划相关模型
# ============================================================
class ExerciseItem(BaseModel):
    """
    训练计划中的单个动作项

    职责：描述一个具体训练动作的执行参数（组数、次数、休息等）。
    使用场景：嵌套在 TrainingDay.exercises 列表中，组成一天的训练内容。
    """

    name: str
    # 动作名称，如 "杠铃卧推"

    sets: int = Field(..., ge=1, le=10)
    # 组数，合法范围 1~10 组

    reps: str
    # 次数范围（字符串格式），如 "8-12"、"12-15"

    rest: str
    # 组间休息时间，如 "60秒"、"90秒"

    notes: str = ""
    # 动作注意事项/提示，如 "注意顶峰收缩，离心阶段控制速度"


class TrainingDay(BaseModel):
    """
    训练计划中的某一天

    职责：描述一个训练日的内容（训练部位 + 动作列表）。
    使用场景：嵌套在 TrainingPlanOutput.days 列表中。
    """

    day: int
    # 训练日序号（1-based），如第 1 天、第 2 天

    focus: str
    # 该日训练重点/部位，如 "胸+三头"、"背+二头"、"腿"

    exercises: List[ExerciseItem]
    # 该日的动作列表


class TrainingPlanOutput(BaseModel):
    """
    训练计划的完整输出 Schema

    职责：定义 AI 生成的训练计划的完整结构，包含元信息、日程安排和警告。
    使用场景：LLM 生成 JSON 后由 Pydantic 校验结构完整性，再返回给前端/CLI。
    """

    plan_id: str
    # 计划唯一标识符，如 "plan_20240702_abc123"

    user_id: int
    # 关联的用户 ID

    goal: str
    # 训练目标（"增肌" 或 "减脂"）

    weeks: int
    # 计划总周数，如 4（周）

    sessions_per_week: int
    # 每周训练次数

    days: List[TrainingDay]
    # 训练日程列表，每个元素是一天的训练内容

    warnings: List[str] = []
    # 警告信息列表，如 ["注意：你的伤病史中包含'腰椎间盘突出'，已自动排除大重量硬拉"]


# ============================================================
# 动作分析相关模型
# ============================================================
class ExerciseAnalysisInput(BaseModel):
    """
    动作分析请求的输入 Schema

    职责：接收用户对自己做动作的感受描述，由 LLM 分析动作问题。
    使用场景：/api/analyze-exercise 端点的请求体。
    """

    exercise_name: str
    # 要分析的动作名称，如 "杠铃深蹲"

    user_description: str
    # 用户对自己做动作时的感受描述
    # 如 "深蹲时膝盖会内扣，腰部有酸痛感"

    user_level: str = "中级"
    # 用户自评水平，"初级"/"中级"/"高级"，默认 "中级"


class ExerciseAnalysisOutput(BaseModel):
    """
    动作分析结果的输出 Schema

    职责：封装 LLM 对用户动作的分析结果，包括问题、严重程度和建议。
    使用场景：/api/analyze-exercise 端点的响应体。
    """

    exercise_name: str
    # 被分析的动作名称（回显）

    issues_found: List[str]
    # 发现的问题列表，如 ["膝盖内扣", "核心未收紧"]

    severity: str
    # 严重程度等级，如 "低"、"中"、"高"

    suggestions: List[str]
    # 改进建议列表，如 ["做箱式深蹲练习膝外展意识", "降低负重至自重先矫正姿势"]

    confidence: float
    # LLM 对分析结果的置信度（0.0~1.0）


# ============================================================
# API 请求/响应包装模型
# ============================================================
class PlanRequest(BaseModel):
    """
    训练计划生成请求（schemas.py 中的版本）

    职责：包装用户档案和自然语言查询为一个完整请求。
    注意：app/server.py 中有一个独立定义的 PlanRequest，两者字段略有不同。
    """

    user_profile: UserProfileInput
    # 用户健身档案

    query: str = ""
    # 自然语言查询，如 "我想重点练胸部和肩部"
    # 为空时系统使用默认查询模板生成计划


class AnalysisRequest(BaseModel):
    """
    动作分析请求（schemas.py 中的版本）

    职责：包装动作分析输入。
    """

    analysis: ExerciseAnalysisInput
    # 动作分析输入数据


class SearchResult(BaseModel):
    """
    RAG 检索结果的数据模型

    职责：封装向量检索返回的单个结果（知识块），包含内容和相关性信息。
    使用场景：RAG 检索流程中，将 pgvector 查询结果转换为结构化对象。
    """

    content: str
    # 检索到的知识内容/文本片段

    score: float
    # 相似度分数（余弦相似度），范围通常是 0.0~1.0
    # 越高表示与查询越相关

    source: str
    # 来源标识，如文件名或知识库名称

    metadata: dict = {}
    # 额外的元数据，如 {"chunk_index": 3, "category": "营养"}
