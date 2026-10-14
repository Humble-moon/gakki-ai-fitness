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
from datetime import datetime, timezone
from pathlib import Path
# 将项目根目录加入 Python 模块搜索路径，确保 src.* 导入正常
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Literal, Optional
from src.core.orchestrator import Orchestrator
from src.core.goal_contract import GoalConsistencyError
from src.models.schemas import SexInput, TrainingGoal, UserProfileInput
from src.health import readiness_checks
from src.llm.provider import LLMUnavailableError
from src.graph import build_inputs, build_runtime, graph_stream_events
from src.hitl.review_resolution import make_resolution
from src import config
from src.security.api_guard import (AdminAuthMiddleware, ApiKeyMiddleware,
                                    CostLimitMiddleware, RateLimitMiddleware,
                                    cors_allow_origins)
from src.security.cost_guard import CostGuard
from src.security.session import (
    Principal,
    user_owner_key,
)
from app.auth_api import (
    get_current_principal,
    require_owner_key,
    router as auth_router,
)
from src.storage.redis_client import RedisClient
from src.llm.cost_tracker import cost_tracker
from langgraph.types import Command

logger = logging.getLogger(__name__)
_TERMINAL_EVENTS = {"done", "error", "cancelled"}
STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# 创建 FastAPI 应用实例
#
# 这里只补文档元信息，不关任何路由：`/docs`（Swagger UI）与 `/openapi.json`
# 一直是 FastAPI 默认开启的，此前缺的只是标题之外的说明，点进去是一份没有
# 描述、没有分组的端点清单。定位与运行契约以 docs/project-fact-baseline.md 为准，
# 不要在这里写"生产级""多租户"这类与事实不符的措辞。
app = FastAPI(
    title="IRON MIND · AI Fitness Coach",
    version="0.1.0",
    description=(
        "本地运行的 AI 健身私教。Multi-Agent 生成个性化训练计划，"
        "GraphRAG 做伤病推理，混合检索的知识问答带来源引用。\n\n"
        "**运行契约**：`localhost:8503`，游客可直接使用，也可注册本地账号"
        "把数据归到自己名下；不是公网多租户服务。\n\n"
        "**流式入口**：`/api/generate-plan`、`/api/analyze-exercise`、"
        "`/api/ask-question` 以 SSE 返回，其余为普通 JSON 接口。"
    ),
)


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

# CORS 中间件：白名单由 CORS_ALLOW_ORIGINS 配置（默认仅本机前端来源）。
#
# 中间件注册顺序说明：Starlette 中**后注册者在外层**，因此下面这段代码的
# 书写顺序与真实执行顺序**相反**。按注册顺序从下往上读，实际执行顺序为：
#     CORS → 限流 → 成本上限 → 管理端认证 → 业务 API Key → 路由
# CORS 放在最外层是必须的：预检 OPTIONS 不带凭据，若它先撞上认证中间件，
# 浏览器会收到 401 而不是跨源策略，前端表现为"跨域请求莫名失败"。
app.add_middleware(ApiKeyMiddleware, token=config.API_AUTH_TOKEN,
                   # 登录/注册/登出必须能在无凭据时访问，否则形成鸡生蛋
                   exempt_prefixes=("/api/auth/",))
app.add_middleware(AdminAuthMiddleware,
                   admin_token=config.ADMIN_TOKEN,
                   api_token=config.API_AUTH_TOKEN)

# 当日成本上限：防"请求不频繁但一天累积花超"。计数存 Redis，
# 因此重启不会把预算重置回零。接线方式是给 cost_tracker 挂一个 sink，
# 这样 LLMProvider 不必知道预算守卫的存在。
cost_guard = CostGuard(redis_client=RedisClient(),
                       daily_limit=config.COST_DAILY_LIMIT_YUAN)
cost_tracker.add_sink(cost_guard.record)
app.add_middleware(CostLimitMiddleware, guard=cost_guard)

app.add_middleware(RateLimitMiddleware,
                   requests_per_minute=config.RATE_LIMIT_PER_MINUTE)
# CORS 最后注册 → 最外层，先于一切认证处理预检
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 账号相关路由（注册/登录/登出/会话查询）
app.include_router(auth_router)

# 静态资源。此前项目没有这个挂载——首页是单独用 FileResponse 返回的，
# 所以只有 index.html 一个文件能被取到；动作图示这类资源需要真正的目录服务。
# 挂在整个 static 目录上（而不是只挂 exercises），后续新增静态资源不用再改这里。
class _CacheableStatic(StaticFiles):
    """给静态资源加 Cache-Control。

    默认只有 ETag/Last-Modified，浏览器每次仍要发一次条件请求；动作图示是
    内容不变的图片，让浏览器直接跳过请求更省事。前端页面本身不缓存
    （max-age 只加在图片上），避免改动后用户拿到旧界面。
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200 and path.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg")):
            response.headers["Cache-Control"] = "public, max-age=86400"
        return response


app.mount("/static",
          _CacheableStatic(directory=str(Path(__file__).parent / "static")),
          name="static")


# ============================================================
# 请求体数据模型（在 server.py 内部重新定义，方便独立部署）
# ============================================================
class PlanRequest(BaseModel):
    """
    训练计划生成请求体

    说明：与 schemas.py 中的 PlanRequest 不同，这里直接展开字段而非嵌套。
         这样设计是为了让前端直接提交扁平 JSON，无需嵌套结构。
    """
    height: float = Field(180, ge=100, le=250)
    # 身高（cm），默认 180，合法范围 100~250cm。范围与 src/models/schemas.py
    # 对齐——这里曾是复制出来的第二个模型，漏掉约束后让 height=0 /
    # days_per_week=0 这类值穿透到下游抛 500，而不是在入口被拒。
    weight: float = Field(80, ge=30, le=200)
    # 体重（kg），默认 80，合法范围 30~200kg
    age: Optional[int] = Field(None, ge=10, le=100)
    # 年龄，合法范围 10~100。可选：只影响热量估算，没填就退回按体重粗估
    sex: SexInput = "不填"
    # 性别，仅用于热量公式。可选且允许「不填」——不填时按两性中值估算
    training_years: float = Field(1.0, ge=0, le=30)
    # 训练年限，默认 1.0，合法范围 0~30 年
    goal: TrainingGoal = "增肌"
    # 训练目标，默认 "增肌"
    available_equipment: list[str] = ["哑铃", "杠铃"]
    # 可用器械列表，默认 ["哑铃", "杠铃"]。
    # 有意不加 min_length：徒手/零器械是合法场景，必须能生成计划。
    days_per_week: int = Field(4, ge=1, le=7)
    # 每周训练天数，默认 4，合法范围 1~7
    injuries: list[str] = []
    # 伤病史列表，默认空
    habits: list[str] = []
    # 运动习惯/爱好列表，默认空。与 injuries 同属用户属性，随画像落盘进长期记忆。
    query: str = Field("", max_length=2000)
    # 自然语言查询，为空时使用默认模板。上限 2000 字符：再长也撑不进
    # 提示词预算，只会白烧 token。
    session_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    # 会话 ID，用于多轮对话（修改计划时说"把第二天改成哑铃动作"）。
    # pattern 不是洁癖：session_id 直接拼进 Redis 键名，而会话清理用的是
    # SCAN 前缀匹配（glob），放进 "*" 就能匹配并删掉所有人的会话。
    athlete_key: Optional[str] = None
    # 稳定运动员标识（前端 localStorage 生成的 UUID）。提供时会读取该运动员的
    # 训练历史与已确认调整注入生成上下文，让新计划承接实际执行情况。
    # 注意不要塞进 UserProfileInput——那会改变 SemanticCache 的指纹、破坏缓存命中。


class AnalysisRequest(BaseModel):
    """
    动作分析请求体

    说明：包含用户档案和动作分析所需的所有字段。
    """
    height: float = Field(180, ge=100, le=250)
    # 身高（cm），合法范围 100~250cm
    weight: float = Field(80, ge=30, le=200)
    # 体重（kg），合法范围 30~200kg
    age: Optional[int] = Field(None, ge=10, le=100)
    # 年龄，合法范围 10~100。可选：只影响热量估算，没填就退回按体重粗估
    sex: SexInput = "不填"
    # 性别，仅用于热量公式。可选且允许「不填」——不填时按两性中值估算
    training_years: float = Field(1.0, ge=0, le=30)
    # 训练年限，合法范围 0~30 年
    goal: TrainingGoal = "增肌"
    # 训练目标
    available_equipment: list[str] = ["哑铃"]
    # 可用器械
    days_per_week: int = Field(4, ge=1, le=7)
    # 每周训练天数，合法范围 1~7
    injuries: list[str] = []
    # 伤病史
    exercise_name: str = Field(..., min_length=1, max_length=200)
    # 要分析的动作名称，如 "杠铃深蹲"。必填：没有动作名就无从分析，
    # 空值过去会一路跑到检索阶段白烧一次 LLM 调用。
    user_description: str = Field("", max_length=2000)
    # 用户对自己做动作时的感受描述
    session_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    # 会话 ID，用于多轮对话上下文（同样要挡住 glob 通配符）


class QuestionRequest(BaseModel):
    """
    自由问答请求体

    说明：支持用户自由提问健身相关问题，session_id 用于多轮对话上下文管理。
    """
    height: float = Field(180, ge=100, le=250)
    # 身高（cm），合法范围 100~250cm
    weight: float = Field(80, ge=30, le=200)
    # 体重（kg），合法范围 30~200kg
    age: Optional[int] = Field(None, ge=10, le=100)
    # 年龄，合法范围 10~100。可选：只影响热量估算，没填就退回按体重粗估
    sex: SexInput = "不填"
    # 性别，仅用于热量公式。可选且允许「不填」——不填时按两性中值估算
    training_years: float = Field(1.0, ge=0, le=30)
    # 训练年限，合法范围 0~30 年
    goal: TrainingGoal = "增肌"
    available_equipment: list[str] = ["哑铃"]
    days_per_week: int = Field(4, ge=1, le=7)
    # 每周训练天数，合法范围 1~7
    injuries: list[str] = []
    habits: list[str] = []
    # 运动习惯/爱好列表，默认空
    question: str = Field(..., min_length=1, max_length=2000)
    # 用户提问内容，如 "减脂期应该怎么安排碳水摄入？"。必填且限长：
    # 空问题无意义，超长问题（曾实测 10 万字符）会直接撑爆提示词。
    session_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    # 会话 ID，用于多轮对话上下文追踪
    # None 时系统自动创建新会话（同样要挡住 glob 通配符）
    athlete_key: Optional[str] = None
    # 稳定运动员标识（前端 localStorage）。用于读取跨会话长期记忆——
    # 与训练日志共用同一个身份，用户改体重后仍能读回自己的偏好。


class TrainingLogEntryInput(BaseModel):
    """
    训练日志里的单个动作明细。

    说明：reps 用整数而非计划里的 "8-12" 区间——计划是处方（范围），
         日志是事实（点值），这样才能直接算容量与估算 1RM。
    """
    exercise_name: str = Field("", max_length=200)
    # 动作名称，需与动作库/计划中的名称对齐
    sets: int = Field(0, ge=0, le=100)
    # 实际完成组数
    reps: int = Field(0, ge=0, le=1000)
    # 每组实际次数（代表值）
    weight: float = Field(0.0, ge=0, le=2000, allow_inf_nan=False)
    # 使用重量（kg）；自重动作填 0。
    # allow_inf_nan=False 是必须的：JSON 允许裸 NaN/Infinity 字面量，
    # pydantic 的 float 默认放行，而 NaN 会一路写进 Postgres 并在
    # JSONResponse 序列化时抛 ValueError——结果是这个运动员之后的
    # 统计类接口全部持续 500，且无法自愈。
    rpe: Optional[float] = Field(None, ge=0, le=10, allow_inf_nan=False)
    # 该动作主观疲劳度 1~10
    completed: bool = True
    # 是否按计划完成


class TrainingLogRequest(BaseModel):
    """
    训练日志提交请求体。

    说明：一次训练提交一条，entries 为该次训练包含的所有动作。
    """
    athlete_key: str = ""
    # 稳定运动员标识（前端 localStorage）
    log_date: Optional[str] = None
    # 训练日期 "YYYY-MM-DD"；缺省为今天（本地日历日）
    day_index: int = Field(0, ge=0, le=60)
    # 对应训练计划第几个训练日；0 表示自由训练
    focus: str = Field("", max_length=100)
    # 当日训练重点，如 "胸+三头"
    plan_id: str = Field("", max_length=64)
    # 该次训练依据的计划 ID
    session_rpe: Optional[float] = Field(None, ge=0, le=10, allow_inf_nan=False)
    # 整节课主观疲劳度 1~10
    duration_min: int = Field(0, ge=0, le=1440)
    # 训练时长（分钟）
    body_weight: Optional[float] = Field(None, ge=20, le=400, allow_inf_nan=False)
    # 当日体重（kg）
    notes: str = ""
    # 自由备注
    entries: list[TrainingLogEntryInput] = []
    # 动作明细列表


class BodyMeasurementRequest(BaseModel):
    """
    体测记录提交请求体。

    说明：所有测量项都可选——现实中很少有人每次把八项量齐，强制必填
         只会让人放弃记录。全空提交由接口层拒绝，避免留下无意义的空行。
    """
    athlete_key: str = ""
    # 稳定运动员标识（前端 localStorage）
    measure_date: Optional[str] = None
    # 测量日期 "YYYY-MM-DD"；缺省为今天（本地日历日）

    weight: Optional[float] = Field(None, ge=20, le=400, allow_inf_nan=False)
    # 体重（kg）。上下界按人体实际范围取，NaN 由 allow_inf_nan 拦下，
    # 理由同 TrainingLogEntryInput.weight。
    body_fat_pct: Optional[float] = Field(None, ge=1, le=70, allow_inf_nan=False)
    # 体脂率（%）
    muscle_mass: Optional[float] = Field(None, ge=0, le=200, allow_inf_nan=False)
    # 肌肉量（kg）

    chest: Optional[float] = Field(None, ge=20, le=250, allow_inf_nan=False)
    # 胸围（cm）
    waist: Optional[float] = Field(None, ge=30, le=250, allow_inf_nan=False)
    # 腰围（cm）
    hip: Optional[float] = Field(None, ge=30, le=250, allow_inf_nan=False)
    # 臀围（cm）
    arm: Optional[float] = Field(None, ge=10, le=100, allow_inf_nan=False)
    # 臂围（cm）
    thigh: Optional[float] = Field(None, ge=10, le=150, allow_inf_nan=False)
    # 大腿围（cm）

    notes: str = Field("", max_length=512)
    # 自由备注，如"晨起空腹"


class TrainingAdviceRequest(BaseModel):
    """调整建议生成请求体。"""
    athlete_key: str = ""
    # 稳定运动员标识
    weeks: int = 8
    # 参与分析的回溯周数


class TrainingAdviceApplyRequest(BaseModel):
    """调整建议确认请求体——用户勾选采纳哪些条目。"""
    advice_id: str = ""
    # 待确认的建议工件 ID
    accepted_item_ids: list[str] = []
    # 用户勾选采纳的条目 ID；空列表表示全部不采纳


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
def _llm_failure_payload(exc) -> dict:
    """把 LLM 不可用翻译成用户能据此行动的提示。

    原来一律回「演示依赖暂时不可用」——把配额耗尽和网络抖动说成了同一回事。
    前者要充值、不会自愈，后者等一会儿就好；混在一起用户只能反复重试。
    文案由 LLMUnavailableError.payload 统一给出，这里只处理裸 ConnectionError。
    """
    if isinstance(exc, LLMUnavailableError):
        return exc.payload
    return {"code": "DEPENDENCY_UNAVAILABLE", "message": "AI 服务暂时不可用，请稍后重试。"}


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
            yield emit("error", {"code": "STREAM_INCOMPLETE", "message": "请求没有正常结束，请重试"})
    except GeneratorExit:
        raise
    except (ConnectionError, LLMUnavailableError) as exc:
        if not terminal:
            yield emit("error", _llm_failure_payload(exc))
    except Exception:
        logger.exception("SSE stream failed")
        if not terminal:
            yield emit("error", {"code": "STREAM_FAILED", "message": "请求暂时失败，请稍后重试"})
    finally:
        close = getattr(generator, "close", None)
        if callable(close):
            close()


# ============================================================
# API 端点定义
# ============================================================

@app.post("/api/generate-plan")
def generate_plan(req: PlanRequest,
                  principal: Principal = Depends(get_current_principal)):
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
    profile = _profile_from_request(req)
    owner_key = require_owner_key(principal, req.athlete_key)
    return StreamingResponse(
        _stream_events(orch.generate_plan_stream(profile, req.query, req.session_id,
                                                 athlete_key=owner_key)),
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
    profile = _profile_from_request(req)
    return StreamingResponse(
        _stream_events(orch.analyze_exercise_stream(
            req.exercise_name, req.user_description, profile, req.session_id)),
        media_type="text/event-stream", headers=STREAM_HEADERS
    )


@app.post("/api/ask-question")
def ask_question(req: QuestionRequest,
                 principal: Principal = Depends(get_current_principal)):
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
    profile = _profile_from_request(req)
    owner_key = require_owner_key(principal, req.athlete_key)
    return StreamingResponse(
        _stream_events(orch.answer_question_stream(req.question, profile, req.session_id,
                                                   athlete_key=owner_key)),
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


# ============================================================
# 训练执行闭环（训练日志 / 统计 / 调整建议）
# ============================================================
# 说明：本组端点全部返回普通 JSON，不使用 SSE。前端有契约测试约束
#      consumeSse( 的出现次数必须与流式动作数严格一致，新增流式端点
#      会破坏该不变量。且这几个端点的交互形态（提交表单、看图表、
#      勾选确认）本就不需要逐字输出。

_training_log_store = None
_plan_store = None
_body_measure_store = None


def _get_training_log_store():
    """惰性构造训练日志存储——避免模块导入时就建立数据库连接。"""
    global _training_log_store
    if _training_log_store is None:
        from src.storage.training_log_store import TrainingLogStore
        _training_log_store = TrainingLogStore()
    return _training_log_store


def _get_plan_store():
    """惰性构造计划存储——同 _get_training_log_store，避免导入时连库。"""
    global _plan_store
    if _plan_store is None:
        from src.storage.plan_store import PlanStore
        _plan_store = PlanStore()
    return _plan_store


def _get_body_measure_store():
    """惰性构造体测记录存储——同 _get_training_log_store，避免导入时连库。"""
    global _body_measure_store
    if _body_measure_store is None:
        from src.storage.body_measure_store import BodyMeasureStore
        _body_measure_store = BodyMeasureStore()
    return _body_measure_store


def _storage_unavailable():
    """训练日志存储不可用时的统一响应。"""
    return JSONResponse(
        status_code=503,
        content={"error": {"code": "STORAGE_UNAVAILABLE",
                           "message": "训练日志暂时不可用，请确认数据服务已启动"}},
    )


@app.post("/api/training-logs")
async def create_training_log(req: TrainingLogRequest,
                              principal: Principal = Depends(get_current_principal)):
    """
    POST /api/training-logs - 记录一次训练

    请求体：TrainingLogRequest（含动作明细 entries）
    返回：JSON {"log_id", "entries_saved", "log_date"}

    说明：主表与明细在同一事务写入——明细写一半失败会留下一条没有动作的
          空日志，统计时会把完成率的分母撑大、稀释真实完成度。
    """
    if not req.athlete_key and not principal.is_authenticated:
        return JSONResponse(status_code=400, content={
            "error": {"code": "MISSING_ATHLETE_KEY", "message": "缺少身份信息"}})

    store = _get_training_log_store()
    store.ensure_table()
    if not store.enabled:
        return _storage_unavailable()

    entries = [e.model_dump() for e in req.entries]
    log_id = store.save_log(
        athlete_key=require_owner_key(principal, req.athlete_key),
        entries=entries, log_date=req.log_date,
        day_index=req.day_index, focus=req.focus, plan_id=req.plan_id,
        session_rpe=req.session_rpe, duration_min=req.duration_min,
        body_weight=req.body_weight, notes=req.notes,
    )
    if log_id is None:
        return _storage_unavailable()

    return {
        "log_id": log_id,
        "entries_saved": len([e for e in entries if (e.get("exercise_name") or "").strip()]),
        "log_date": req.log_date or "",
    }


@app.get("/api/training-logs")
async def list_training_logs(athlete_key: str = "", weeks: int = 12,
                             limit: int = 50, exercise: Optional[str] = None,
                             principal: Principal = Depends(get_current_principal)):
    """
    GET /api/training-logs - 查询历史训练日志（含动作明细）

    查询参数：
        athlete_key: 运动员标识（必填）
        weeks: 回溯周数，默认 12
        limit: 返回条数上限，默认 50、最大 100
        exercise: 只返回包含该动作的日志

    返回：JSON {"logs": [...], "count"}
    """
    if not athlete_key and not principal.is_authenticated:
        return JSONResponse(status_code=400, content={
            "error": {"code": "MISSING_ATHLETE_KEY", "message": "缺少身份信息"}})

    store = _get_training_log_store()
    store.ensure_table()
    if not store.enabled:
        return _storage_unavailable()

    logs = store.get_logs(require_owner_key(principal, athlete_key),
                          weeks=max(1, min(weeks, 52)),
                          limit=max(1, min(limit, 100)), exercise=exercise)
    return {"logs": logs, "count": len(logs)}


@app.post("/api/body-measurements")
async def create_body_measurement(req: BodyMeasurementRequest,
                                  principal: Principal = Depends(get_current_principal)):
    """
    POST /api/body-measurements - 记录一次体测

    请求体：BodyMeasurementRequest
    返回：JSON {"measurement_id"}

    说明：至少要有一项有效测量才落库。全空提交不是"记录了一次体测"，
    而是误触——存下来只会在趋势图上多一个空洞。
    """
    if not req.athlete_key and not principal.is_authenticated:
        return JSONResponse(status_code=400, content={
            "error": {"code": "MISSING_ATHLETE_KEY", "message": "缺少身份信息"}})

    measured = [req.weight, req.body_fat_pct, req.muscle_mass,
                req.chest, req.waist, req.hip, req.arm, req.thigh]
    if all(v is None for v in measured):
        return JSONResponse(status_code=400, content={
            "error": {"code": "NO_MEASUREMENT", "message": "至少填写一项测量数据"}})

    store = _get_body_measure_store()
    store.ensure_table()
    if not store.enabled:
        return _storage_unavailable()

    measurement_id = store.save(
        require_owner_key(principal, req.athlete_key),
        measure_date=req.measure_date,
        weight=req.weight, body_fat_pct=req.body_fat_pct,
        muscle_mass=req.muscle_mass, chest=req.chest, waist=req.waist,
        hip=req.hip, arm=req.arm, thigh=req.thigh, notes=req.notes,
    )
    if measurement_id is None:
        return _storage_unavailable()
    return {"measurement_id": measurement_id}


@app.get("/api/body-measurements")
async def list_body_measurements(athlete_key: str = "", weeks: int = 0,
                                 principal: Principal = Depends(get_current_principal)):
    """
    GET /api/body-measurements - 查询历史体测记录

    查询参数：
        athlete_key: 运动员标识（必填）
        weeks: 回溯周数；0 表示不限时间（默认）

    返回：JSON {"measurements": [...], "count"}

    默认不限时间：体测是低频记录，按月甚至按季度才量一次，
    默认回溯 12 周会让多数用户看到空列表。
    """
    if not athlete_key and not principal.is_authenticated:
        return JSONResponse(status_code=400, content={
            "error": {"code": "MISSING_ATHLETE_KEY", "message": "缺少身份信息"}})

    store = _get_body_measure_store()
    store.ensure_table()
    if not store.enabled:
        return _storage_unavailable()

    rows = store.get_measurements(
        require_owner_key(principal, athlete_key),
        weeks=max(1, min(weeks, 520)) if weeks else None,
    )
    return {"measurements": rows, "count": len(rows)}


@app.delete("/api/body-measurements/{measurement_id}")
async def delete_body_measurement(measurement_id: int, athlete_key: str = "",
                                  principal: Principal = Depends(get_current_principal)):
    """
    DELETE /api/body-measurements/{measurement_id} - 删除一条体测记录

    返回：JSON {"deleted": bool}

    说明：删除范围限定在该身份名下。只按自增 id 删会让任何人凭一个可枚举的
    整数抹掉别人的记录，因此 athlete_key 一并参与过滤；不匹配时返回
    deleted=false 而非 404，不暴露"这个 id 存在但不属于你"。
    """
    if not athlete_key and not principal.is_authenticated:
        return JSONResponse(status_code=400, content={
            "error": {"code": "MISSING_ATHLETE_KEY", "message": "缺少身份信息"}})

    store = _get_body_measure_store()
    store.ensure_table()
    if not store.enabled:
        return _storage_unavailable()

    deleted = store.delete_one(require_owner_key(principal, athlete_key), measurement_id)
    return {"deleted": deleted}


@app.get("/api/user-data/export")
def export_user_data(athlete_key: str = "", session_id: Optional[str] = None,
                     principal: Principal = Depends(get_current_principal)):
    """导出该身份名下的全部个人数据（JSON）。

    健康数据属于个人信息，用户有权取回自己产生的全部内容。导出覆盖三处
    存储：训练日志（PostgreSQL）、长期记忆（Redis）、当前会话（Redis）。
    缓存不在导出范围——它是"同一份身体数据 + 同一目标命中同一计划"的
    去重产物，不绑定个人身份。

    登录后 athlete_key 参数被忽略，一律导出登录账号自己的数据。这一点在
    账号场景下是必须的：否则任何登录用户传别人的 key 就能把他人健康数据
    整份拉走。
    """
    if not athlete_key and not principal.is_authenticated:
        return JSONResponse(status_code=400, content={
            "error": "athlete_key_required",
            "message": "必须提供身份信息才能定位你的数据。",
        })

    athlete_key = require_owner_key(principal, athlete_key)
    payload: dict = {
        "athlete_key": athlete_key,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "training_logs": [],
        "body_measurements": [],
        "saved_plans": [],
        "long_term_memory": {},
        "conversation": None,
        "note": (
            "本文件包含你的全部个人数据。训练计划由 AI 生成，仅供参考，"
            "不构成医疗建议。"
        ),
    }

    try:
        store = _get_training_log_store()
        # 取足够长的窗口以覆盖全部历史（10 年）——导出不做时间裁剪
        payload["training_logs"] = store.get_logs(athlete_key, weeks=520, limit=10_000)
    except Exception as exc:  # noqa: BLE001 - 单源失败不该让整份导出失败
        logger.warning("[data-rights] 导出训练日志失败：%s", exc)
        payload["training_logs_error"] = str(exc)

    try:
        measure_store = _get_body_measure_store()
        measure_store.ensure_table()
        if measure_store.enabled:
            # 体测记录与训练日志同属个人健康数据，必须一并可导出——
            # 少导一类数据，这份「导出全部」就是假的。
            payload["body_measurements"] = measure_store.get_measurements(
                athlete_key, limit=10_000)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[data-rights] 导出体测记录失败：%s", exc)
        payload["body_measurements_error"] = str(exc)

    try:
        plan_store = _get_plan_store()
        plan_store.ensure_table()
        if plan_store.enabled:
            # 导出整份计划正文而不是列表摘要——用户要的是能带走的内容，
            # 只有标题和日期的话，这份导出对他没用。
            exported = []
            for summary in plan_store.list_plans(athlete_key, limit=10_000):
                full = plan_store.get_plan(athlete_key, summary.get("plan_id"))
                exported.append(full or summary)
            payload["saved_plans"] = exported
    except Exception as exc:  # noqa: BLE001 - 单源失败不该让整份导出失败
        logger.warning("[data-rights] 导出已保存计划失败：%s", exc)
        payload["saved_plans_error"] = str(exc)

    try:
        payload["long_term_memory"] = orch.long_term.get_preferences(athlete_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[data-rights] 导出长期记忆失败：%s", exc)
        payload["long_term_memory_error"] = str(exc)

    if session_id:
        try:
            payload["conversation"] = {
                "session_id": session_id,
                "context": orch.conversation.get_context(session_id),
                "plan_state": orch.conversation.get_plan_state(session_id),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("[data-rights] 导出会话失败：%s", exc)
            payload["conversation_error"] = str(exc)

    return payload


@app.delete("/api/user-data")
def delete_user_data(athlete_key: str = "", confirm: str = "",
                     session_id: Optional[str] = None,
                     principal: Principal = Depends(get_current_principal)):
    """删除该身份名下的全部个人数据。

    **不可逆操作**，因此要求显式传 ``confirm=DELETE``——一个拼错的请求
    不该把用户几个月的训练记录抹掉。

    账号场景下这是破坏性最强的端点：不校验归属的话，任何登录用户传一个
    受害者 key 就能删光他人数据。因此登录后一律以登录身份为准，客户端
    传来的 athlete_key 被忽略。游客模式沿用自证式 key（本机数据本机可见），
    这条审计日志记录实际删除的主体，便于事后追溯。

    删除范围与导出对齐（训练日志 / 体测记录 / 已保存计划 / 长期记忆 / 会话）。
    **不做部分成功**：任一步抛错都如实返回失败与已完成的部分，不谎报"已全部删除"。
    """
    if not athlete_key and not principal.is_authenticated:
        return JSONResponse(status_code=400, content={
            "error": "athlete_key_required",
            "message": "必须提供身份信息才能定位你的数据。",
        })
    if confirm != "DELETE":
        return JSONResponse(status_code=400, content={
            "error": "confirmation_required",
            "message": "删除不可恢复。请带上 confirm=DELETE 明确确认。",
        })

    athlete_key = require_owner_key(principal, athlete_key)
    logger.warning("[data-rights] 收到删除请求 subject=%s session_id=%s",
                   athlete_key, session_id or "-")

    result: dict = {"athlete_key": athlete_key, "deleted": {}, "errors": {}}

    try:
        result["deleted"]["training_logs"] = _get_training_log_store().delete_logs(athlete_key)
    except Exception as exc:  # noqa: BLE001
        logger.error("[data-rights] 删除训练日志失败：%s", exc)
        result["errors"]["training_logs"] = str(exc)

    try:
        measure_store = _get_body_measure_store()
        measure_store.ensure_table()
        result["deleted"]["body_measurements"] = measure_store.delete_all(athlete_key)
    except Exception as exc:  # noqa: BLE001
        logger.error("[data-rights] 删除体测记录失败：%s", exc)
        result["errors"]["body_measurements"] = str(exc)

    try:
        result["deleted"]["long_term_keys"] = orch.long_term.purge(athlete_key)
    except Exception as exc:  # noqa: BLE001
        logger.error("[data-rights] 删除长期记忆失败：%s", exc)
        result["errors"]["long_term"] = str(exc)

    try:
        plan_store = _get_plan_store()
        plan_store.ensure_table()
        result["deleted"]["plans"] = plan_store.delete_all(athlete_key)
    except Exception as exc:  # noqa: BLE001
        logger.error("[data-rights] 删除已保存计划失败：%s", exc)
        result["errors"]["plans"] = str(exc)

    if session_id:
        try:
            result["deleted"]["conversation_keys"] = orch.conversation.purge(session_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("[data-rights] 删除会话失败：%s", exc)
            result["errors"]["conversation"] = str(exc)

    result["ok"] = not result["errors"]
    if not result["ok"]:
        result["message"] = "部分数据删除失败，详见 errors。请重试或联系支持。"
    return JSONResponse(status_code=200 if result["ok"] else 500, content=result)


@app.get("/api/plans")
def list_plans(limit: int = 20,
               principal: Principal = Depends(get_current_principal),
               athlete_key: str = ""):
    """列出当前身份下已保存的训练计划。

    游客用自己的本机标识、登录用户用账号身份——与训练日志、长期记忆同一套
    归属规则，客户端伪造的 key 在登录态下会被忽略。
    """
    store = _get_plan_store()
    store.ensure_table()
    if not store.enabled:
        return JSONResponse(status_code=503, content={
            "error": {"code": "STORAGE_UNAVAILABLE",
                      "message": "计划暂时不可用，请确认数据服务已启动"}})
    owner = require_owner_key(principal, athlete_key or None)
    plans = store.list_plans(owner, limit=limit)
    return {"plans": plans, "count": len(plans)}


@app.get("/api/plans/{plan_id}")
def get_plan(plan_id: int,
             principal: Principal = Depends(get_current_principal),
             athlete_key: str = ""):
    """取一份计划的完整正文。不属于当前身份时返回 404。"""
    store = _get_plan_store()
    store.ensure_table()
    if not store.enabled:
        return JSONResponse(status_code=503, content={
            "error": {"code": "STORAGE_UNAVAILABLE", "message": "计划暂时不可用"}})
    owner = require_owner_key(principal, athlete_key or None)
    plan = store.get_plan(owner, plan_id)
    if plan is None:
        # 用 404 而不是 403：403 等于确认"这个 id 存在但不是你的"
        return JSONResponse(status_code=404, content={
            "error": {"code": "PLAN_NOT_FOUND", "message": "计划不存在。"}})
    return plan


class PlanSaveRequest(BaseModel):
    """保存计划请求体。plan 是模型产出的计划对象原文。"""
    plan: dict = {}
    athlete_key: Optional[str] = None


@app.post("/api/plans")
def save_plan(req: PlanSaveRequest,
              principal: Principal = Depends(get_current_principal)):
    """保存一份训练计划。

    只接受**可交付**的计划：被安全门扣下的、或在生成中途的草稿不该进历史，
    否则用户翻历史时会看到一份从未被批准执行的计划，而历史列表并不显示
    "这份被拦下过"。
    """
    store = _get_plan_store()
    store.ensure_table()
    if not store.enabled:
        return JSONResponse(status_code=503, content={
            "error": {"code": "STORAGE_UNAVAILABLE", "message": "计划暂时不可用"}})

    plan = req.plan or {}
    status = plan.get("delivery_status")
    if status not in ("safe_delivered", "review_approved"):
        return JSONResponse(status_code=400, content={
            "error": {"code": "PLAN_NOT_DELIVERED",
                      "message": "只能保存已交付的计划。"}})
    # 没有训练日的计划没有保存价值。这里先拦是因为 store.save 对这种情况也返回
    # None，而路由把 None 一律当成"存储不可用"报 503——那会让调用方以为该稍后
    # 重试，实际重试多少次都一样：问题在请求，不在存储。
    if not plan.get("days"):
        return JSONResponse(status_code=400, content={
            "error": {"code": "PLAN_EMPTY",
                      "message": "该计划没有训练日内容，无需保存。"}})

    owner = require_owner_key(principal, req.athlete_key)
    plan_id = store.save(owner, plan)
    if plan_id is None:
        return JSONResponse(status_code=503, content={
            "error": {"code": "STORAGE_UNAVAILABLE", "message": "计划保存失败"}})
    return {"plan_id": plan_id, "saved": True}


#: 计划名长度上限。够放"周四下肢·深蹲强化"这类描述性名字，又不至于
#: 长到在「我的计划」列表里把日期挤成省略号。
MAX_PLAN_TITLE_LEN = 40


class PlanRenameRequest(BaseModel):
    """计划重命名请求体。"""
    title: str = ""
    athlete_key: Optional[str] = None


@app.patch("/api/plans/{plan_id}")
def rename_plan(plan_id: int,
                req: PlanRenameRequest,
                principal: Principal = Depends(get_current_principal)):
    """给已保存的计划改名字。不属于当前身份时返回 404。

    校验放在路由而不是存储层：存储层的 False 只有一个含义——"没这行"，
    路由要把它翻成 404。若把"名字非法"也塞进同一个 False，用户会收到
    "计划不存在"这种驴唇不对马嘴的提示。
    """
    store = _get_plan_store()
    store.ensure_table()
    if not store.enabled:
        return JSONResponse(status_code=503, content={
            "error": {"code": "STORAGE_UNAVAILABLE", "message": "计划暂时不可用"}})

    title = (req.title or "").strip()
    if not title:
        return JSONResponse(status_code=400, content={
            "error": {"code": "PLAN_TITLE_EMPTY", "message": "计划名称不能为空。"}})
    if len(title) > MAX_PLAN_TITLE_LEN:
        return JSONResponse(status_code=400, content={
            "error": {"code": "PLAN_TITLE_TOO_LONG",
                      "message": f"计划名称不能超过 {MAX_PLAN_TITLE_LEN} 个字符。"}})

    owner = require_owner_key(principal, req.athlete_key)
    if not store.rename_plan(owner, plan_id, title):
        return JSONResponse(status_code=404, content={
            "error": {"code": "PLAN_NOT_FOUND", "message": "计划不存在。"}})
    return {"renamed": True, "plan_id": plan_id, "title": title}


@app.delete("/api/plans/{plan_id}")
def delete_plan(plan_id: int,
                principal: Principal = Depends(get_current_principal),
                athlete_key: str = ""):
    """删除一份已保存的计划。"""
    store = _get_plan_store()
    store.ensure_table()
    if not store.enabled:
        return JSONResponse(status_code=503, content={
            "error": {"code": "STORAGE_UNAVAILABLE", "message": "计划暂时不可用"}})
    owner = require_owner_key(principal, athlete_key or None)
    if not store.delete_plan(owner, plan_id):
        return JSONResponse(status_code=404, content={
            "error": {"code": "PLAN_NOT_FOUND", "message": "计划不存在。"}})
    return {"deleted": True, "plan_id": plan_id}


@app.get("/api/exercise-images")
def exercise_images():
    """动作图示清单：中文动作名 → 起止位两张图的静态路径。

    只有经过逐张目视核对、确认与动作是同一个的条目才在清单里——健身场景里
    照着错的图练会受伤，宁可没有。未收录的动作前端不显示任何东西（不占位、
    不加提示），这是刻意的：缺图是常态，为它加一行说明只会让卡片更满。
    清单来自 data/exercise_images.json，由 data/ 下的构建脚本产出。

    返回空对象而不是报错：图示是增强，取不到不该让计划渲染不出来。
    """
    path = Path(__file__).resolve().parent.parent / "data" / "exercise_images.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[exercise-images] 清单读取失败：%s", exc)
        return {}


@app.get("/api/training-stats")
async def training_stats(athlete_key: str = "", weeks: int = 12,
                         principal: Principal = Depends(get_current_principal)):
    """
    GET /api/training-stats - 训练统计（图表数据源）

    查询参数：
        athlete_key: 运动员标识（必填）
        weeks: 回溯周数，默认 12

    返回：JSON {totals, weekly, exercises, muscles, records}，结构直接对应前端
        图表的 series。无数据时返回结构完整的空壳，前端不必判空。
    """
    if not athlete_key and not principal.is_authenticated:
        return JSONResponse(status_code=400, content={
            "error": {"code": "MISSING_ATHLETE_KEY", "message": "缺少身份信息"}})

    from src.core.training_analytics import (
        compute_muscle_volume,
        compute_records,
        compute_stats,
    )

    store = _get_training_log_store()
    store.ensure_table()
    if not store.enabled:
        return _storage_unavailable()

    athlete_key = require_owner_key(principal, athlete_key)
    logs = store.get_logs(athlete_key, weeks=max(1, min(weeks, 52)))
    stats = compute_stats(logs)
    # 肌群分布要先知道每个动作练的是哪儿。动作库取不到时退化成空映射：
    # 动作会全部落进 unmatched_exercises，前端据此提示"这些没算进去"，
    # 而不是把"查不到"直接画成"这个肌群没练过"。
    try:
        from src.rag.exercise_catalog import load_exercise_muscles
        exercise_muscles = load_exercise_muscles()
    except Exception as exc:
        logger.warning("动作肌群标注读取失败，肌群分布将标记为未匹配: %s", exc)
        exercise_muscles = {}
    stats["muscles"] = compute_muscle_volume(logs, exercise_muscles)
    stats["records"] = compute_records(logs)
    stats["athlete_key"] = athlete_key
    stats["weeks"] = weeks
    return stats


@app.post("/api/training-advice")
async def create_training_advice(req: TrainingAdviceRequest,
                                 principal: Principal = Depends(get_current_principal)):
    """
    POST /api/training-advice - 生成调整建议（**只生成，不生效**）

    请求体：TrainingAdviceRequest
    返回：JSON {advice_id, status, summary, summary_source, summary_detail,
                items, stats}

    说明：**调整条目**由确定性规则引擎推导（见 training_analytics.rule_engine），
          不依赖 LLM，模型不可用时依然可用；**summary_detail** 是 LLM 把统计
          与规则结论重述成的一份周报，属于增强层，失败时回退规则摘要并把
          summary_source 标成 "rule"。用户必须在 apply 接口确认勾选后才会
          写入长期记忆。
    """
    if not req.athlete_key and not principal.is_authenticated:
        return JSONResponse(status_code=400, content={
            "error": {"code": "MISSING_ATHLETE_KEY", "message": "缺少身份信息"}})

    from src.core.training_advice import build_advice

    store = _get_training_log_store()
    store.ensure_table()
    if not store.enabled:
        return _storage_unavailable()

    # 丢进线程池：周报会调一次 LLM（秒级）。build_advice 是同步函数，
    # 直接在 async 端点里执行会阻塞事件循环——生成周报的那几秒里，同一进程
    # 上正在跑的 SSE 流会一起卡住。build_advice 内部已做降级，LLM 不可用时
    # 回退规则摘要，异常不会冒到这里。
    return await run_in_threadpool(
        build_advice,
        require_owner_key(principal, req.athlete_key),
        store,
        weeks=max(1, min(req.weeks, 52)),
        long_term=orch.long_term,
    )


@app.post("/api/training-advice/apply")
async def apply_training_advice(req: TrainingAdviceApplyRequest,
                                principal: Principal = Depends(get_current_principal)):
    """
    POST /api/training-advice/apply - 确认采纳所选调整

    请求体：TrainingAdviceApplyRequest（advice_id + 勾选的 item_id 列表）
    返回：JSON {"advice_id", "status", "applied", "persisted", "note"}

    说明：只接受该工件中确实存在的 item_id，防止伪造或过期 id 写入。
          确认后写入长期记忆，在下一次生成计划时注入上下文。
          **用户确认不等于安全审核通过**——若调整与伤病冲突，
          FactChecker 仍会拦下并触发人工审核。

    归属：工件记录了自己属于谁，登录用户只能结算自己的工件。不匹配返回
          404 而非 403——403 等于告诉对方"这个 id 存在但不属于你"，
          可被用来枚举有效 advice_id。
    """
    from src.core.training_advice import advice_store, resolve_advice

    artifact = advice_store.get(req.advice_id)
    if (artifact is not None and principal.is_authenticated
            and artifact.athlete_key != user_owner_key(principal.user_id)):
        return JSONResponse(status_code=404, content={
            "error": {"code": "ADVICE_NOT_FOUND",
                      "message": "该调整建议不存在或已失效，请重新生成"}})

    result = resolve_advice(req.advice_id, req.accepted_item_ids)
    if result is None:
        return JSONResponse(status_code=409, content={
            "error": {"code": "ADVICE_EXPIRED",
                      "message": "该调整建议已失效，请重新生成"}})
    return result


# ----------------------------------------------------------------------
# AI 记忆面板
# ----------------------------------------------------------------------
# 让用户看到 AI 记住了自己什么、从哪来、能不能删。长期记忆一直在被写入
# （训练画像、已确认的调整、计划反馈），但此前只有写入没有出口——用户既
# 看不到内容，也无法单独拿掉某一条。

# 偏好键的来源归类。面板要如实区分「你填的」和「AI 推的」：后者更该被
# 质疑。用户看到 training_adjustments 里有一条不合适，应该能单独删掉它，
# 而不是只能在「全留」和「全清」之间二选一。
_USER_FILLED_PREF_KEYS = {"profile", "goal", "equipment", "habits"}
_ADJUSTMENT_PREF_KEY = "training_adjustments"

# 偏好新鲜度阈值，与 long_term._get_preference_ages 的口径保持一致
_PREF_STALE_DAYS = 90


def _pref_source_label(key: str) -> str:
    """把偏好键翻译成用户能看懂的记忆来源。"""
    if key == _ADJUSTMENT_PREF_KEY:
        return "AI 建议，你确认采纳"
    if key in _USER_FILLED_PREF_KEYS:
        return "你在训练档案里填写的"
    return "系统记录"


def _pref_age_days(updated_at) -> int | None:
    """把写入时间换算成天数差；缺失或无法解析时返回 None。"""
    if not updated_at:
        return None
    try:
        ts = datetime.fromisoformat(str(updated_at))
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        # 旧格式可能不带时区（record_feedback 写的是本地时间），按 UTC 兜底
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - ts).days)


@app.get("/api/memory")
async def get_memory(athlete_key: str = "",
                     principal: Principal = Depends(get_current_principal)):
    """
    GET /api/memory - 读取该身份名下的全部 AI 记忆

    返回：JSON {athlete_key, preferences, feedback, injuries, counts}
        preferences 每条含 key / value / updated_at / age_days / stale /
        source_label，前端据此分组并标出过旧条目。

    说明：只读。来源标签如实区分用户填写与 AI 推断，让用户知道哪些记忆
          值得质疑。
    """
    if not athlete_key and not principal.is_authenticated:
        return JSONResponse(status_code=400, content={
            "error": {"code": "MISSING_ATHLETE_KEY", "message": "缺少身份信息"}})

    athlete_key = require_owner_key(principal, athlete_key)
    memory = orch.long_term

    try:
        raw_prefs = memory.get_preferences_detailed(athlete_key)
    except Exception as exc:  # noqa: BLE001 - 读不到要如实上报，不能假装没有记忆
        logger.error("[memory] 读取长期记忆失败：%s", exc)
        return JSONResponse(status_code=503, content={
            "error": {"code": "MEMORY_UNAVAILABLE",
                      "message": "长期记忆暂不可用，请确认数据服务已启动"}})

    preferences = []
    for key, entry in sorted(raw_prefs.items()):
        age = _pref_age_days(entry.get("updated_at"))
        preferences.append({
            "key": key,
            "value": entry.get("value"),
            "updated_at": entry.get("updated_at"),
            "age_days": age,
            "stale": bool(age is not None and age > _PREF_STALE_DAYS),
            "source_label": _pref_source_label(key),
        })

    # 反馈与伤病是补充信息，读失败不该让整个面板打不开
    try:
        feedback = memory.get_feedback(athlete_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[memory] 读取计划反馈失败：%s", exc)
        feedback = []

    try:
        injuries = memory.get_injury_history(athlete_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[memory] 读取伤病记录失败：%s", exc)
        injuries = []

    return {
        "athlete_key": athlete_key,
        "preferences": preferences,
        "feedback": feedback,
        "injuries": injuries,
        "counts": {
            "preferences": len(preferences),
            "feedback": len(feedback),
            "injuries": len(injuries),
        },
    }


@app.delete("/api/memory")
async def clear_memory(athlete_key: str = "", confirm: str = "",
                       principal: Principal = Depends(get_current_principal)):
    """
    DELETE /api/memory - 清空该身份名下的全部 AI 记忆（**不可逆**）

    只清长期记忆，不动训练日志与训练计划——与 DELETE /api/user-data 的
    全量删除明确区分开。要求显式传 confirm=DELETE，理由同该端点：
    一个拼错的请求不该把用户攒了几个月的画像抹掉。
    """
    if not athlete_key and not principal.is_authenticated:
        return JSONResponse(status_code=400, content={
            "error": {"code": "MISSING_ATHLETE_KEY", "message": "缺少身份信息"}})
    if confirm != "DELETE":
        return JSONResponse(status_code=400, content={
            "error": {"code": "CONFIRMATION_REQUIRED",
                      "message": "清空记忆不可恢复。请带上 confirm=DELETE 明确确认。"}})

    athlete_key = require_owner_key(principal, athlete_key)
    try:
        deleted = orch.long_term.purge(athlete_key)
    except Exception as exc:  # noqa: BLE001 - 删除失败要如实上报
        logger.error("[memory] 清空长期记忆失败：%s", exc)
        return JSONResponse(status_code=503, content={
            "error": {"code": "MEMORY_UNAVAILABLE",
                      "message": "长期记忆暂不可用，未能清空"}})

    logger.warning("[memory] 清空长期记忆 subject=%s 删除 %d 个键", athlete_key, deleted)
    return {"deleted_keys": deleted,
            "note": "已清空。下次生成计划时不再参考任何历史记忆。"}


@app.delete("/api/memory/{pref_key}")
async def delete_memory_preference(pref_key: str, athlete_key: str = "",
                                   principal: Principal = Depends(get_current_principal)):
    """
    DELETE /api/memory/{pref_key} - 删除单条 AI 记忆

    删除立即生效：下次生成计划时该条不再进入上下文。用于「某条记忆不对，
    我要单独拿掉它」——比清空全部更常用。
    """
    if not athlete_key and not principal.is_authenticated:
        return JSONResponse(status_code=400, content={
            "error": {"code": "MISSING_ATHLETE_KEY", "message": "缺少身份信息"}})

    athlete_key = require_owner_key(principal, athlete_key)
    try:
        deleted = orch.long_term.delete_preference(athlete_key, pref_key)
    except Exception as exc:  # noqa: BLE001
        logger.error("[memory] 删除单条记忆失败：%s", exc)
        return JSONResponse(status_code=503, content={
            "error": {"code": "MEMORY_UNAVAILABLE",
                      "message": "长期记忆暂不可用，未能删除"}})

    if not deleted:
        return JSONResponse(status_code=404, content={
            "error": {"code": "MEMORY_NOT_FOUND",
                      "message": "这条记忆不存在或已被删除"}})

    return {"deleted": True, "key": pref_key,
            "note": "已删除。下次生成计划时不再包含这条记忆。"}


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
    # no-cache 不是"不缓存"，而是"每次都要先校验"：内容没变时仍走 304，
    # 变了就拿到新版。不加它时浏览器会对 HTML 做启发式缓存，于是**改了前端
    # 刷新却看不到新版本**——实测中被绊过一次，演示前改一版然后刷新风险很大。
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"),
                        headers={"Cache-Control": "no-cache"})


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
    import threading
    import time as _time

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
    """由请求体构造用户档案，字段清单以 ``UserProfileInput`` 为准。

    这里曾经逐字段手写映射，四个调用点各写一份。这种写法在新增字段时会
    **静默丢数据**：age/sex 加进请求模型和档案模型之后，映射没跟上，于是
    用户填了、前端传了、模型收下了，到流水线里却是 None——热量目标悄悄退回
    粗估，界面还显示"缺身高或年龄"，而没有任何环节报错。

    改成按模型字段反射取值：请求模型没有这个字段就跳过，档案模型有默认值的
    自然落到默认值。以后再加字段，映射不必改，也就不会再漏。
    """
    payload = {name: getattr(req, name)
               for name in UserProfileInput.model_fields if hasattr(req, name)}
    return UserProfileInput(**payload)


@app.post("/api/v2/plan")
def generate_plan_v2(req: PlanRequest,
                     principal: Principal = Depends(get_current_principal)):
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
                          thread_id=thread_id,
                          athlete_key=require_owner_key(principal, req.athlete_key))
    try:
        state = graph_runtime.graph.invoke(inputs, config)
    except GoalConsistencyError as exc:
        return JSONResponse(status_code=422, content={
            "error": {"code": exc.code, "message": "生成结果与目标不符，请重试"}})
    except LLMUnavailableError as exc:
        return JSONResponse(status_code=503, content={"error": _llm_failure_payload(exc)})
    if state.get("__interrupt__"):
        payload = dict(state.get("review_payload") or {})
        payload["thread_id"] = thread_id
        return payload
    return state.get("final_payload")


@app.post("/api/v2/plan/stream")
def generate_plan_v2_stream(req: PlanRequest,
                            principal: Principal = Depends(get_current_principal)):
    """流式版（SSE）：复用 _stream_events 与既有事件协议，前端零改动。"""
    profile = _profile_from_request(req)
    thread_id = graph_runtime.new_thread_id()
    config = graph_runtime.config_for(thread_id)
    inputs = build_inputs(profile, query=req.query, session_id=req.session_id,
                          thread_id=thread_id,
                          athlete_key=require_owner_key(principal, req.athlete_key))
    return StreamingResponse(
        _stream_events(graph_stream_events(graph_runtime, inputs, config)),
        media_type="text/event-stream", headers=STREAM_HEADERS,
    )


def _review_owned_by(artifact, owner_key: str | None) -> bool:
    """审核工件是否属于当前身份。

    旧工件（本字段引入前落库）的 owner_key 为空，此时无法判定归属，退回
    "凭 review_id 访问"——review_id 是 uuid4，靠不可猜性兜底。新工件一律
    带归属，登录用户只能访问自己的。这是一次有意的取舍：宁可对历史数据
    保守兼容，也不为了整齐而让旧工件变得完全无法审核。
    """
    if not getattr(artifact, "owner_key", ""):
        return True
    return artifact.owner_key == owner_key


def _review_denied_response():
    return JSONResponse(status_code=404,
                        content={"error": {"code": "REVIEW_NOT_FOUND"}})


@app.get("/api/v2/reviews/{review_id}")
def get_review_v2(review_id: str,
                  principal: Principal = Depends(get_current_principal)):
    """查询审核工件及其解除结果（若已解除）。"""
    artifact = graph_runtime.deps.review_store.get(review_id)
    if artifact is None:
        return JSONResponse(status_code=404,
                            content={"error": {"code": "REVIEW_NOT_FOUND"}})
    # 不属于当前身份时返回 404 而非 403：403 等于确认"这个 id 存在但不是你的"，
    # 可被用来枚举有效 review_id。
    if principal.is_authenticated and not _review_owned_by(
            artifact, user_owner_key(principal.user_id)):
        return _review_denied_response()
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
                      thread_id: Optional[str] = None,
                      principal: Principal = Depends(get_current_principal)):
    """解除人工审核并恢复被暂停的 LangGraph 执行（HITL 闭环的关键一步）。

    旧系统只会创建 review_pending 工件、无法对其作出决定；该路由通过
    ``Command(resume=...)`` 恢复 checkpoint 中暂停的图，交付
    review_approved / review_rejected。thread_id 缺省时按 review_id 反查，
    服务重启后也可显式传入 thread_id 恢复。

    归属：解除是改状态的操作，登录用户只能解除自己的工件。不校验的话，
    任何登录用户拿到 review_id 就能替别人放行一个被安全门拦下的计划。
    """
    artifact = graph_runtime.deps.review_store.get(review_id)
    if artifact is not None and principal.is_authenticated and not _review_owned_by(
            artifact, user_owner_key(principal.user_id)):
        return _review_denied_response()

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
