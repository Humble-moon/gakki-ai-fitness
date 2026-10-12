"""肌群粗分组映射的测试。

最重要的一条是 ``test_every_seed_muscle_is_mapped``：动作库里出现的每个
细分肌群名都必须已被映射覆盖。映射漏一个，那部分训练量就会在肌群分布图上
凭空消失，而且**不会报错**——图照样画出来，只是少了一块，看的人无从察觉。
所以这条断言是 fail-closed 的：宁可加动作时测试失败，也不要静默少算。
"""

import json
from pathlib import Path

import pytest

from src.core.muscle_groups import (
    GROUP_ORDER,
    OTHER_GROUP,
    group_of,
    is_known,
    known_muscles,
)

SEED_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "seed_exercises.json"


def _seed_muscle_names() -> set[str]:
    """动作库里出现过的全部细分肌群名（去重）。"""
    data = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    return {
        m.strip()
        for item in data
        for m in (item.get("target_muscles") or [])
        if m and m.strip()
    }


def test_every_seed_muscle_is_mapped():
    """动作库里的每个肌群名都要有归属，不允许静默漏掉。"""
    missing = sorted(m for m in _seed_muscle_names() if not is_known(m))
    assert not missing, (
        f"以下肌群名未登记到 muscle_groups._MUSCLE_TO_GROUP，"
        f"它们的训练量会在肌群分布图上消失：{missing}"
    )


def test_seed_muscle_names_are_not_trivially_empty():
    """先确认真的读到了动作库，否则上面的全覆盖断言是空转的。"""
    names = _seed_muscle_names()
    assert len(names) > 50, f"只读到 {len(names)} 个肌群名，动作库可能没读进来"


def test_group_of_returns_declared_group_for_known_muscle():
    assert group_of("股四头肌") == "腿"
    assert group_of("胸大肌") == "胸"
    assert group_of("背阔肌") == "背"
    assert group_of("三角肌前束") == "肩"
    assert group_of("臀大肌") == "臀"
    assert group_of("肱三头肌") == "手臂"
    assert group_of("腹直肌") == "核心"


def test_variants_of_same_muscle_land_in_same_group():
    """同一块肉的不同写法必须归到同一组，否则组数会被同义词拆散。"""
    for variants in (
        ["胸大肌", "胸肌", "胸大肌上束", "上胸", "胸肌中缝"],
        ["核心", "核心肌群", "腹横肌", "腹斜肌"],
        ["小腿", "小腿肌群", "腓肠肌", "比目鱼肌"],
    ):
        groups = {group_of(v) for v in variants}
        assert len(groups) == 1, f"{variants} 被拆到了多组：{groups}"


def test_unknown_muscle_falls_back_to_other_group_without_raising():
    """未登记的名字归"其他"而不是抛异常：动作库是可扩充的数据。"""
    assert group_of("某块未来才会有的肌肉") == OTHER_GROUP
    assert group_of("") == OTHER_GROUP
    assert group_of(None) == OTHER_GROUP


@pytest.mark.parametrize("muscle", sorted(known_muscles()))
def test_every_mapped_group_is_in_display_order(muscle):
    """映射出的组名必须都在展示顺序里，否则图表会漏画这一格。"""
    assert group_of(muscle) in GROUP_ORDER


def test_non_muscle_training_tags_are_not_forced_into_a_muscle_group():
    """"全身/心肺"是训练属性，硬塞进某个肌群会让那一格虚高。"""
    assert group_of("全身/心肺") == OTHER_GROUP
    assert group_of("全身") == OTHER_GROUP
