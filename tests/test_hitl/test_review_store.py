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


class TestInjuryProfileLabel:
    """伤病档案 ID 必须能翻成中文名再给用户看。

    档案里只有英文 id（knee-osteoarthritis），而这条文案会出现在
    「计划已暂缓交付」面板的风险明细里。中文界面冒英文内部标识，
    用户看不懂也无法据此行动。
    """

    def test_known_ids_map_to_chinese_names(self):
        from src.hitl.review import injury_profile_label

        assert injury_profile_label("knee-osteoarthritis") == "膝关节骨关节炎"
        assert injury_profile_label("lumbar-disc") == "腰椎间盘突出"
        assert injury_profile_label("shoulder-impingement") == "肩峰撞击综合征"

    def test_every_shipped_profile_has_a_chinese_label(self):
        """所有内置档案都要能翻译——新增档案时不能漏掉。"""
        from src.hitl.review import INJURY_PROFILES, injury_profile_label

        for profile in INJURY_PROFILES:
            label = injury_profile_label(profile["id"])
            assert label != profile["id"], f"{profile['id']} 没有中文名"
            assert not label.isascii(), f"{profile['id']} 的中文名仍是英文：{label}"

    def test_unknown_id_falls_back_to_itself(self):
        """未知 ID 原样返回而不是抛异常——文案生成不该成为新的故障点。"""
        from src.hitl.review import injury_profile_label

        assert injury_profile_label("no-such-profile") == "no-such-profile"
