from src.hitl.review_store import InMemoryReviewArtifactStore, ReviewArtifact


def test_create_assigns_pending_artifact_with_review_context():
    store = InMemoryReviewArtifactStore()

    artifact = store.create(
        profile_summary={"goal": "增肌", "injuries": ["膝盖疼"]},
        query="膝盖疼还能深蹲吗",
        issues=[{"issue": "膝盖风险", "severity": "danger"}],
        severity="danger",
        prohibited_actions=["深蹲", "加大强度"],
    )

    assert artifact.review_id
    assert artifact.status == "review_pending"
    assert artifact.created_at
    assert artifact.profile_summary == {"goal": "增肌", "injuries": ["膝盖疼"]}
    assert artifact.query == "膝盖疼还能深蹲吗"
    assert artifact.issues == [{"issue": "膝盖风险", "severity": "danger"}]
    assert artifact.severity == "danger"
    assert artifact.prohibited_actions == ["深蹲", "加大强度"]
    assert store.get(artifact.review_id) == artifact


def test_artifacts_are_kept_only_in_process_memory():
    store = InMemoryReviewArtifactStore()
    artifact = store.create({}, "q", [], "warning", [])

    assert isinstance(artifact, ReviewArtifact)
    assert store.get("unknown") is None
