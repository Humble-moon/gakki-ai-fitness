"""Dependency container for the coach graph.

Nodes never pull live objects out of the state (the checkpointer must serialize
state to JSON). Instead every collaborator — the four agents, the cache, the
memory stores, the review stores — is gathered here and bound to each node with
``functools.partial`` when the graph is built. This keeps the state schema pure
and lets tests substitute any dependency with a ``SimpleNamespace`` fake, exactly
like the existing agent tests do.
"""

from dataclasses import dataclass

from src.hitl.review_resolution import InMemoryReviewResolutionStore, ReviewThreadIndex


@dataclass
class CoachGraphDeps:
    """All collaborators the coach graph needs, injected at build time."""
    planner: object
    retriever: object
    writer: object
    fact_checker: object
    cache: object
    conversation: object
    long_term: object
    review_store: object
    resolutions: InMemoryReviewResolutionStore
    thread_index: ReviewThreadIndex
    #: 可调用对象，返回动作库全部动作名（懒求值——首次调用时才查库）。
    #: 传 None 表示该能力不可用，此时 finalize_node 跳过动作库校验，
    #: 而不是把图跑挂。测试用 SimpleNamespace 伪造 deps 时通常不带该字段。
    library_exercise_names_fn: object = None


def deps_from_orchestrator(orch, resolutions: InMemoryReviewResolutionStore,
                           thread_index: ReviewThreadIndex) -> CoachGraphDeps:
    """Build graph dependencies by reusing an Orchestrator's already-wired modules.

    This is the reuse seam: the graph pipeline drives the *same* agent instances,
    cache, memory and review store that the legacy orchestrator uses, so the two
    backends are directly comparable.
    """
    return CoachGraphDeps(
        planner=orch.planner,
        retriever=orch.retriever,
        writer=orch.writer,
        fact_checker=orch.fact_checker,
        cache=orch.cache,
        conversation=orch.conversation,
        long_term=orch.long_term,
        review_store=orch.review_store,
        resolutions=resolutions,
        thread_index=thread_index,
        # 绑定方法而非预求值：动作库查询推迟到首次真正用到时，
        # 避免启动期数据库未就绪就把空集合缓存住。
        library_exercise_names_fn=getattr(orch, "_library_exercise_names", None),
    )
