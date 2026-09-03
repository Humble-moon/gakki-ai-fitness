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
    )
