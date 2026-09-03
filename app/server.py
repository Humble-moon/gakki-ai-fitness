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
import logging
import sys
from pathlib import Path
# 将项目根目录加入 Python 模块搜索路径，确保 src.* 导入正常
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Literal, Optional
from src.core.orchestrator import Orchestrator
from src.core.goal_contract import GoalConsistencyError
from src.models.schemas import TrainingGoal, UserProfileInput
from src.health import readiness_checks
from src.llm.provider import LLMUnavailableError
from src.graph import build_inputs, build_runtime, graph_stream_events
from src.hitl.review_resolution import make_resolution
from langgraph.types import Command

logger = logging.getLogger(__name__)
_TERMINAL_EVENTS = {"done", "error", "cancelled"}
STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# 创建 FastAPI 应用实例
app = FastAPI(title="AI Fitness Coach")


@app.get("/health/live")
def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready():
    checks, ready = readiness_checks()
    return JSONResponse(
        {"status": "ok" if ready else "not_ready", "checks": checks},
        status_code=200 if ready else 503,
    )

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
    goal: TrainingGoal = "增肌"
    # 训练目标，默认 "增肌"
    available_equipment: list[str] = ["哑铃", "杠铃"]
    # 可用器械列表，默认 ["哑铃", "杠铃"]
    days_per_week: int = 4
    # 每周训练天数，默认 4
    injuries: list[str] = []
    # 伤病史列表，默认空
    query: str = ""
    # 自然语言查询，为空时使用默认模板
    session_id: Optional[str] = None
    # 会话 ID，用于多轮对话（修改计划时说"把第二天改成哑铃动作"）


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
    goal: TrainingGoal = "增肌"
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
    session_id: Optional[str] = None
    # 会话 ID，用于多轮对话上下文


class QuestionRequest(BaseModel):
    """
    自由问答请求体

    说明：支持用户自由提问健身相关问题，session_id 用于多轮对话上下文管理。
    """
    height: float = 180
    weight: float = 80
    training_years: float = 1.0
    goal: TrainingGoal = "增肌"
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

# LangGraph 版编排运行时（v2 API）。复用 orch 的同一批 Agent / 缓存 / 记忆 /
# 审核存储，并附带 checkpointer 以支持 interrupt 暂停与跨重启恢复。
graph_runtime = build_runtime(orch)


# ============================================================
# SSE 流式传输辅助函数
# ============================================================
def _stream_events(generator):
    """Convert an orchestrator iterator into one stable SSE stream."""
    terminal = False

    def emit(event, data):
        payload = json.dumps({"event": event, "data": data}, ensure_ascii=False)
        return f"data: {payload}\n\n"

    try:
        for event, data in generator:
            if terminal:
                break
            if event in _TERMINAL_EVENTS:
                terminal = True
            yield emit(event, data)
            if terminal:
                break
        if not terminal:
            yield emit("error", {"code": "STREAM_INCOMPLETE", "message": "流式请求未返回明确完成事件"})
    except GeneratorExit:
        raise
    except (ConnectionError, LLMUnavailableError):
        if not terminal:
            yield emit("error", {"code": "DEPENDENCY_UNAVAILABLE", "message": "演示依赖暂时不可用"})
    except Exception:
        logger.exception("SSE stream failed")
        if not terminal:
            yield emit("error", {"code": "STREAM_FAILED", "message": "流式请求暂时失败，请稍后重试"})
    finally:
        close = getattr(generator, "close", None)
        if callable(close):
            close()


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
        _stream_events(orch.generate_plan_stream(profile, req.query, req.session_id)),
        media_type="text/event-stream", headers=STREAM_HEADERS
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
        - session_id: str|None   会话 ID（多轮对话）

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
        _stream_events(orch.analyze_exercise_stream(
            req.exercise_name, req.user_description, profile, req.session_id)),
        media_type="text/event-stream", headers=STREAM_HEADERS
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
        media_type="text/event-stream", headers=STREAM_HEADERS
    )


@app.post("/api/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """上传 PDF/Word/MD 文档，解析并存入 session 级文档库。

    参数：
        file: 上传文件（PDF/Word/MD，最大 20MB）
        session_id: 当前会话 ID（必填）

    返回：
        JSON: {"document_id", "filename", "file_type", "total_chars",
               "page_count", "title", "has_text", "error"}
    """
    from src.parsers import parse_file
    from src.storage.document_store import MAX_FILE_SIZE

    # 读文件内容
    content = await file.read()

    # 大小检查
    if len(content) > MAX_FILE_SIZE:
        return {"error": f"文件过大（{len(content)/1024/1024:.1f}MB），限制 {MAX_FILE_SIZE/1024/1024:.0f}MB"}

    # 解析
    parsed = parse_file(content, file.filename or "unknown")

    # 存入文档库
    doc_id = orch.documents.save(
        session_id=session_id,
        filename=parsed.filename,
        file_type=parsed.file_type,
        file_size=len(content),
        full_text=parsed.full_text,
        page_count=parsed.page_count,
        title=parsed.title,
        has_text=parsed.has_text,
        parse_error=parsed.error or "",
    )

    return {
        "document_id": doc_id,
        "filename": parsed.filename,
        "file_type": parsed.file_type,
        "total_chars": parsed.total_chars,
        "page_count": parsed.page_count,
        "title": parsed.title,
        "has_text": parsed.has_text,
        "error": parsed.error,
    }


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
# 开发运维端点（不暴露给前端用户）
# ============================================================

@app.get("/admin/metrics")
async def admin_metrics():
    """
    GET /admin/metrics - 开发侧观测数据

    返回缓存命中率、LLM 成本等运维指标。仅开发人员使用，不在前端 UI 中展示。
    使用方式：浏览器打开 http://localhost:8503/admin/metrics

    返回格式：
    {
      "cache": {
        "hits_exact": 12,        // 精确匹配命中次数
        "hits_semantic": 3,      // 语义相似度命中次数
        "hits_total": 15,        // 总命中次数
        "misses": 8,             // 未命中次数
        "sets": 23,              // 写入缓存次数
        "total_lookups": 23,     // 总查询次数
        "hit_rate_exact": 0.52,  // 精确命中率
        "hit_rate_semantic": 0.13, // 语义命中率
        "hit_rate_total": 0.65   // 总命中率
      },
      "cost": { ... }            // LLM 成本统计
    }
    """
    from src.llm.cost_tracker import cost_tracker
    return {
        **orch.cache.get_stats(),
        "cost": cost_tracker.get_stats(),
    }


@app.post("/admin/reingest")
async def admin_reingest(strategy: str = "combo", incremental: bool = True):
    """POST /admin/reingest — 手动触发知识库摄入（增量模式）。

    使用方式: curl -X POST http://localhost:8503/admin/reingest?strategy=combo&incremental=true

    增量模式: 仅摄入变更/新增的文档（基于文件 hash 对比）。
    全量模式: incremental=false 时重摄全部文档。
    """
    from src.rag.knowledge_ingestion import ingest
    import threading, time as _time

    started_at = _time.time()

    # 后台线程执行摄入（避免阻塞响应）
    result = {"status": "started", "strategy": strategy, "incremental": incremental}

    def _run():
        try:
            ingest(
                knowledge_dir="data/knowledge",
                strategy=strategy,
                incremental=incremental,
                state_file="data/.ingestion_state.json",
            )
            result["status"] = "completed"
            result["elapsed_sec"] = round(_time.time() - started_at, 1)
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return result


@app.get("/admin/ingestion-status")
async def admin_ingestion_status():
    """GET /admin/ingestion-status — 查看摄入状态和文件变更列表。

    返回: 上次摄入时间、文件总数、变更文件列表。
    需要先跑过一次 --incremental 才会有状态文件。
    """
    from pathlib import Path as _Path
    import json as _json

    state_file = _Path("data/.ingestion_state.json")
    if not state_file.exists():
        return {"status": "no_state", "hint": "Run ingestion with --incremental first"}

    state = _json.loads(state_file.read_text(encoding="utf-8"))
    knowledge_dir = _Path("data/knowledge")
    current_files = set(f.name for f in knowledge_dir.glob("*.md"))

    changed = []
    added = []
    deleted = []
    for fname, old_hash in state.items():
        fpath = knowledge_dir / fname
        if not fpath.exists():
            deleted.append(fname)
        else:
            import hashlib
            current_hash = hashlib.md5(fpath.read_bytes()).hexdigest()
            if current_hash != old_hash:
                changed.append(fname)

    for fname in current_files:
        if fname not in state:
            added.append(fname)

    return {
        "status": "ok",
        "total_files": len(current_files),
        "tracked_files": len(state),
        "changed": changed,
        "added": added,
        "deleted": deleted,
        "needs_reingest": len(changed) + len(added) > 0,
    }


# ============================================================
# v2 API：基于 LangGraph 的计划生成与人工审核闭环
# ============================================================
class ReviewResolveRequest(BaseModel):
    """人工审核解除请求体。"""
    decision: Literal["approved", "rejected"]
    reviewer: str = ""
    comment: str = ""


def _profile_from_request(req) -> UserProfileInput:
    """Reuse the flat request fields shared by PlanRequest-shaped bodies."""
    return UserProfileInput(
        height=req.height, weight=req.weight, training_years=req.training_years,
        goal=req.goal, available_equipment=req.available_equipment,
        days_per_week=req.days_per_week, injuries=req.injuries,
    )


@app.post("/api/v2/plan")
def generate_plan_v2(req: PlanRequest):
    """同步版：走 LangGraph 状态图生成训练计划。

    返回三种情况之一：
    - 正常交付 → final_payload（delivery_status=safe_delivered）
    - 命中缓存 → final_payload（缓存计划）
    - 触发人工审核 → review_pending 载荷（含 thread_id，供后续 resolve）
    """
    profile = _profile_from_request(req)
    thread_id = graph_runtime.new_thread_id()
    config = graph_runtime.config_for(thread_id)
    inputs = build_inputs(profile, query=req.query, session_id=req.session_id,
                          thread_id=thread_id)
    try:
        state = graph_runtime.graph.invoke(inputs, config)
    except GoalConsistencyError as exc:
        return JSONResponse(status_code=422, content={
            "error": {"code": exc.code, "message": "训练计划目标校验失败，请重试"}})
    except LLMUnavailableError:
        return JSONResponse(status_code=503, content={
            "error": {"code": "DEPENDENCY_UNAVAILABLE", "message": "演示依赖暂时不可用"}})
    if state.get("__interrupt__"):
        payload = dict(state.get("review_payload") or {})
        payload["thread_id"] = thread_id
        return payload
    return state.get("final_payload")


@app.post("/api/v2/plan/stream")
def generate_plan_v2_stream(req: PlanRequest):
    """流式版（SSE）：复用 _stream_events 与既有事件协议，前端零改动。"""
    profile = _profile_from_request(req)
    thread_id = graph_runtime.new_thread_id()
    config = graph_runtime.config_for(thread_id)
    inputs = build_inputs(profile, query=req.query, session_id=req.session_id,
                          thread_id=thread_id)
    return StreamingResponse(
        _stream_events(graph_stream_events(graph_runtime, inputs, config)),
        media_type="text/event-stream", headers=STREAM_HEADERS,
    )


@app.get("/api/v2/reviews/{review_id}")
def get_review_v2(review_id: str):
    """查询审核工件及其解除结果（若已解除）。"""
    artifact = graph_runtime.deps.review_store.get(review_id)
    if artifact is None:
        return JSONResponse(status_code=404,
                            content={"error": {"code": "REVIEW_NOT_FOUND"}})
    resolution = graph_runtime.resolutions.get(review_id)
    return {
        "review": {
            "review_id": artifact.review_id,
            "status": artifact.status,
            "created_at": artifact.created_at,
            "issues": artifact.issues,
            "severity": artifact.severity,
            "prohibited_actions": artifact.prohibited_actions,
        },
        "resolution": ({
            "decision": resolution.decision, "reviewer": resolution.reviewer,
            "comment": resolution.comment, "resolved_at": resolution.resolved_at,
        } if resolution else None),
    }


@app.post("/api/v2/reviews/{review_id}/resolve")
def resolve_review_v2(review_id: str, req: ReviewResolveRequest,
                      thread_id: Optional[str] = None):
    """解除人工审核并恢复被暂停的 LangGraph 执行（HITL 闭环的关键一步）。

    旧系统只会创建 review_pending 工件、无法对其作出决定；该路由通过
    ``Command(resume=...)`` 恢复 checkpoint 中暂停的图，交付
    review_approved / review_rejected。thread_id 缺省时按 review_id 反查，
    服务重启后也可显式传入 thread_id 恢复。
    """
    resolved_thread = thread_id or graph_runtime.thread_index.thread_for(review_id)
    if not resolved_thread:
        return JSONResponse(status_code=404,
                            content={"error": {"code": "REVIEW_NOT_FOUND"}})
    config = graph_runtime.config_for(resolved_thread)
    snapshot = graph_runtime.graph.get_state(config)
    if "review_gate" not in (snapshot.next or ()):
        return JSONResponse(status_code=409,
                            content={"error": {"code": "REVIEW_ALREADY_RESOLVED"}})
    resolution = make_resolution(review_id, req.decision, req.reviewer, req.comment)
    graph_runtime.resolutions.record(resolution)
    state = graph_runtime.graph.invoke(
        Command(resume={"decision": req.decision, "review_id": review_id,
                        "reviewer": req.reviewer, "comment": req.comment}),
        config)
    return {
        "thread_id": resolved_thread,
        "resolution": {
            "decision": resolution.decision, "reviewer": resolution.reviewer,
            "comment": resolution.comment, "resolved_at": resolution.resolved_at,
        },
        "delivery": state.get("final_payload"),
    }


# ============================================================
# 直接启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0": 监听所有网络接口，允许外部访问
    # port=8503: 应用端口
    uvicorn.run(app, host="0.0.0.0", port=8503)
