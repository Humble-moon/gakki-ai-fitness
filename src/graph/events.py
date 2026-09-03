"""Adapt LangGraph stream output to the project's ``(event, data)`` SSE protocol.

``app/server.py``'s ``_stream_events`` consumes an iterator of ``(event, data)``
tuples and stops on a terminal event (``done``/``error``/``cancelled``). This module
turns ``graph.stream(..., stream_mode=["updates","custom"])`` into exactly that
shape, reusing the legacy event names so the frontend and terminal-event logic need
no changes.
"""

from src.core.goal_contract import GoalConsistencyError
from src.llm.provider import LLMUnavailableError


def build_inputs(profile, query: str = "", session_id=None, thread_id: str = "") -> dict:
    """Build the initial state dict handed to ``graph.invoke``/``graph.stream``."""
    profile_dict = profile.model_dump() if hasattr(profile, "model_dump") else dict(profile)
    return {
        "profile": profile_dict,
        "query": query or "",
        "session_id": session_id,
        "thread_id": thread_id,
    }


def _node_events(node_name: str, update: dict):
    """Map one node's state update to the legacy event names."""
    if node_name == "plan" and "plan" in update:
        p = update["plan"]
        yield "planner_done", {"skill": p.get("skill", "unknown"),
                               "subtasks": p.get("subtasks", [])}
    elif node_name == "retrieve" and "retrieved" in update:
        exercises = update["retrieved"].get("exercises", [])
        yield "retriever_done", {"count": len(exercises),
                                 "names": [e.get("name", "?") for e in exercises[:8]]}
    elif node_name == "check" and "latest_check" in update:
        c = update["latest_check"]
        yield "factcheck_done", {"safe": c.get("is_safe", True),
                                 "issues": len(c.get("issues", [])),
                                 "confidence": c.get("confidence", 0)}
    elif node_name == "deliver_cached" and "final_payload" in update:
        yield "cache_hit", update["final_payload"]
        yield "done", update["final_payload"]
    elif node_name == "deliver" and "final_payload" in update:
        yield "done", update["final_payload"]


def graph_stream_events(runtime, inputs: dict, config: dict):
    """Yield ``(event, data)`` tuples for the SSE wrapper.

    Terminal outcomes:
    * normal / review / cached delivery → ``("done", payload)``
    * goal-contract failure → ``("error", {"code": "GOAL_CONSISTENCY_FAILED", ...})``
    * all LLM providers down → ``("error", {"code": "DEPENDENCY_UNAVAILABLE", ...})``
    """
    try:
        for mode, chunk in runtime.graph.stream(inputs, config,
                                                stream_mode=["updates", "custom"]):
            if mode == "custom":
                # Forwarded from nodes via get_stream_writer(): stage / writer_chunk.
                yield chunk["event"], chunk["data"]
                continue
            if "__interrupt__" in chunk:
                # Held for human review: surface the review payload as the terminal.
                yield "done", chunk["__interrupt__"][0].value
                return
            for node_name, update in chunk.items():
                for event in _node_events(node_name, update):
                    yield event
                    if event[0] == "done":
                        return
    except GoalConsistencyError as exc:
        yield "error", {"code": exc.code, "message": "训练计划目标校验失败，请重试"}
    except LLMUnavailableError:
        yield "error", {"code": "DEPENDENCY_UNAVAILABLE", "message": "演示依赖暂时不可用"}
