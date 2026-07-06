"""
server.py - FastAPI Web 后端入口

角色：项目的 HTTP API 服务器，提供 RESTful API 接口和静态前端页面。
      与 CLI 入口（main.py）平行，面向 Web 用户而非终端用户。
被调用者：通过 uvicorn 启动（python server.py 或 uvicorn app.server:app）。
调用者：
    - src.core.orchestrator（核心编排器，处理所有业务逻辑）
    - src.models.schemas（用户输入校验）
    - FastAPI 框架（路由、SSE、静态文件服务）

API 端点总览：
    GET  /                  → 返回前端 SPA 页面（index.html）
    POST /api/generate-plan → 流式生成训练计划（SSE）
    POST /api/analyze-exercise → 流式分析动作（SSE）
    POST /api/ask-question  → 流式问答（SSE）

通信模式说明：
    所有业务 API 使用 Server-Sent Events (SSE) 流式返回，
    让前端能实时展示 AI 生成过程（类似 ChatGPT 的逐字输出）。
"""
import json
import sys
from pathlib import Path
# 将项目根目录加入 Python 模块搜索路径，确保 src.* 导入正常
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from src.core.orchestrator import Orchestrator
from src.models.schemas import UserProfileInput

# 创建 FastAPI 应用实例
app = FastAPI(title="AI Fitness Coach")

# CORS 中间件：允许前端跨域访问（部署到服务器时收紧 origins）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 开发阶段允许所有来源，生产环境改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 请求体数据模型（在 server.py 内部重新定义，方便独立部署）
# ============================================================
class PlanRequest(BaseModel):
    """
    训练计划生成请求体

    说明：与 schemas.py 中的 PlanRequest 不同，这里直接展开字段而非嵌套。
         这样设计是为了让前端直接提交扁平 JSON，无需嵌套结构。
    """
    height: float = 180
    # 身高（cm），默认 180
    weight: float = 80
    # 体重（kg），默认 80
    training_years: float = 1.0
    # 训练年限，默认 1.0
    goal: str = "增肌"
    # 训练目标，默认 "增肌"
    available_equipment: list[str] = ["哑铃", "杠铃"]
    # 可用器械列表，默认 ["哑铃", "杠铃"]
    days_per_week: int = 4
    # 每周训练天数，默认 4
    injuries: list[str] = []
    # 伤病史列表，默认空
    query: str = ""
    # 自然语言查询，为空时使用默认模板


class AnalysisRequest(BaseModel):
    """
    动作分析请求体

    说明：包含用户档案和动作分析所需的所有字段。
    """
    height: float = 180
    # 身高（cm）
    weight: float = 80
    # 体重（kg）
    training_years: float = 1.0
    # 训练年限
    goal: str = "增肌"
    # 训练目标
    available_equipment: list[str] = ["哑铃"]
    # 可用器械
    days_per_week: int = 4
    # 每周训练天数
    injuries: list[str] = []
    # 伤病史
    exercise_name: str = ""
    # 要分析的动作名称，如 "杠铃深蹲"
    user_description: str = ""
    # 用户对自己做动作时的感受描述


class QuestionRequest(BaseModel):
    """
    自由问答请求体

    说明：支持用户自由提问健身相关问题，session_id 用于多轮对话上下文管理。
    """
    height: float = 180
    weight: float = 80
    training_years: float = 1.0
    goal: str = "增肌"
    available_equipment: list[str] = ["哑铃"]
    days_per_week: int = 4
    injuries: list[str] = []
    question: str = ""
    # 用户提问内容，如 "减脂期应该怎么安排碳水摄入？"
    session_id: Optional[str] = None
    # 会话 ID，用于多轮对话上下文追踪
    # None 时系统自动创建新会话


# ============================================================
# 全局编排器实例（单例，应用启动时初始化，所有请求复用）
# ============================================================
orch = Orchestrator()


# ============================================================
# SSE 流式传输辅助函数
# ============================================================
def _stream_events(generator):
    """
    将 Python 生成器包装为 SSE（Server-Sent Events）格式

    输入参数：
        generator : Generator - 生成器，每次 yield 一个 (event_name, data) 元组
                   event_name 如 "token", "plan", "error"
                   data 如 {"content": "今天"} 或完整计划 dict

    输出：
        str 生成器 - 逐行产出 SSE 格式文本：
            data: {"event": "token", "data": {...}}

    核心逻辑：
        1. 遍历上层生成器的每次产出
        2. 将 (event, data) 包装为 JSON payload
        3. 加上 SSE 协议要求的 "data: " 前缀和 "\n\n" 结尾

    说明：必须是同步函数（非 async），因为 FastAPI StreamingResponse 需要同步迭代器。
    """
    for event, data in generator:
        payload = json.dumps({"event": event, "data": data}, ensure_ascii=False)
        yield f"data: {payload}\n\n"


# ============================================================
# API 端点定义
# ============================================================

@app.post("/api/generate-plan")
def generate_plan(req: PlanRequest):
    """
    POST /api/generate-plan - 流式生成训练计划（SSE）

    路由：/api/generate-plan
    方法：POST
    功能：接收用户档案和查询，流式返回 AI 生成的训练计划。

    请求体（JSON）：
        - height: float          身高（cm）
        - weight: float          体重（kg）
        - training_years: float  训练年限
        - goal: str              目标（"增肌"|"减脂"）
        - available_equipment: list[str]  可用器械列表
        - days_per_week: int     每周训练天数
        - injuries: list[str]    伤病史
        - query: str             自然语言查询

    响应（SSE 流）：
        事件类型包括：
        - "token":    LLM 逐 token 输出（流式打字效果）
        - "plan":     完整的训练计划 JSON
        - "complete": 生成完成信号
        - "error":    错误信息

    核心逻辑：
        1. 将扁平请求体转为 UserProfileInput Pydantic 对象
        2. 调用 orchestrator 的流式生成方法
        3. 将生成器包装为 SSE StreamingResponse 返回
    """
    profile = UserProfileInput(
        height=req.height, weight=req.weight, training_years=req.training_years,
        goal=req.goal, available_equipment=req.available_equipment,
        days_per_week=req.days_per_week, injuries=req.injuries
    )
    return StreamingResponse(
        _stream_events(orch.generate_plan_stream(profile, req.query)),
        media_type="text/event-stream"
    )


@app.post("/api/analyze-exercise")
def analyze_exercise(req: AnalysisRequest):
    """
    POST /api/analyze-exercise - 流式分析动作姿势（SSE）

    路由：/api/analyze-exercise
    方法：POST
    功能：接收用户的动作描述，AI 分析动作问题并给出改进建议。

    请求体（JSON）：
        - exercise_name: str     动作名称
        - user_description: str  用户感受描述
        - height: float, weight: float, ...  用户身体参数

    响应（SSE 流）：
        事件类型：
        - "token":    LLM 逐 token 输出分析文本
        - "result":   完整分析结果（issues_found, suggestions, severity, confidence）
        - "complete": 分析完成
        - "error":    错误信息

    核心逻辑：
        1. 构建 UserProfileInput 对象
        2. 调用 orchestrator 的流式分析方法
        3. 以 SSE 格式返回
    """
    profile = UserProfileInput(
        height=req.height, weight=req.weight, training_years=req.training_years,
        goal=req.goal, available_equipment=req.available_equipment,
        days_per_week=req.days_per_week, injuries=req.injuries
    )
    return StreamingResponse(
        _stream_events(orch.analyze_exercise_stream(req.exercise_name, req.user_description, profile)),
        media_type="text/event-stream"
    )


@app.post("/api/ask-question")
def ask_question(req: QuestionRequest):
    """
    POST /api/ask-question - 流式自由问答（SSE）

    路由：/api/ask-question
    方法：POST
    功能：用户自由提问健身相关问题，AI 基于知识库（RAG）实时回答。

    请求体（JSON）：
        - question: str          用户提问
        - session_id: str|None   会话 ID（多轮对话），None 时自动创建
        - height: float, weight: float, ...  用户身体参数

    响应（SSE 流）：
        事件类型：
        - "token":    LLM 逐 token 输出回答
        - "sources":  引用的知识来源列表
        - "complete": 回答完成
        - "error":    错误信息

    核心逻辑：
        1. 构建 UserProfileInput 对象
        2. 调用 orchestrator 的流式问答方法
        3. 以 SSE 格式返回

    说明：session_id 用于维持多轮对话上下文，
          前端在首次请求后获取并缓存 session_id，后续请求携带它。
    """
    profile = UserProfileInput(
        height=req.height, weight=req.weight, training_years=req.training_years,
        goal=req.goal, available_equipment=req.available_equipment,
        days_per_week=req.days_per_week, injuries=req.injuries
    )
    return StreamingResponse(
        _stream_events(orch.answer_question_stream(req.question, profile, req.session_id)),
        media_type="text/event-stream"
    )


@app.get("/")
async def index():
    """
    GET / - 前端 SPA 入口页

    路由：/
    方法：GET
    功能：返回静态前端页面（app/static/index.html），支持完整的 SPA 路由模式。

    说明：前端为单页应用（SPA），所有页面路由由前端 JavaScript 处理，
          后端只需返回 index.html 即可。
    """
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


# ============================================================
# 直接启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0": 监听所有网络接口，允许外部访问
    # port=8503: 应用端口
    uvicorn.run(app, host="0.0.0.0", port=8503)
