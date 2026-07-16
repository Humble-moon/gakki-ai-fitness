"""
===========================================================================
expand_exercises.py — LLM 批量扩展健身动作库 (39 → 100)
===========================================================================
用法:
    python scripts/expand_exercises.py                    # 生成新动作，追加到 seed_exercises.json
    python scripts/expand_exercises.py --validate-only    # 仅校验现有动作库
    python scripts/expand_exercises.py --dry-run          # 生成但不写入文件

输出:
    data/seed_exercises_expanded.json — 合并后的完整动作库 (~100 个)

设计:
    1. 按类别分批生成，每批 10-15 个，确保各肌群均匀覆盖
    2. LLM 生成后用 JSON Schema + 去重校验
    3. 校验失败自动要求 LLM 修正（最多 2 次）
    4. 合并到现有 seed_exercises.json 并写入新文件
===========================================================================
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.provider import LLMProvider

# 现有39个动作中已覆盖的，不要重复
EXISTING_NAMES = {
    "哑铃卧推", "杠铃深蹲", "引体向上", "哑铃侧平举", "杠铃硬拉",
    "绳索下压", "哑铃弯举", "腿举", "哑铃推举", "俯身哑铃划船",
    "高脚杯深蹲", "哑铃罗马尼亚硬拉", "杠铃卧推", "哑铃飞鸟", "双杠臂屈伸",
    "上斜哑铃卧推", "杠铃划船", "坐姿绳索划船", "高位下拉", "直臂下压",
    "杠铃推举", "哑铃前平举", "哑铃俯身飞鸟", "绳索面拉", "杠铃弯举",
    "锤式弯举", "窄距杠铃卧推", "哑铃颈后臂屈伸", "保加利亚分腿蹲",
    "杠铃臀推", "腿弯举", "站姿提踵", "哑铃耸肩", "平板支撑",
    "悬垂举腿", "俄罗斯转体", "绳索夹胸", "哑铃负重箭步蹲", "史密斯机深蹲",
}

# 分批生成提示词 — 每批指定目标肌群/类型，避免 LLM 自由发挥导致覆盖不均
BATCHES = [
    # === 腿臀补充 ===
    {
        "batch_name": "腿部补充",
        "prompt": """请生成 10 个腿部训练动作，补充以下空缺：
- 股四头肌专项孤立动作（如腿屈伸）
- 后链/腘绳肌补充（如北欧降、臀桥变式）
- 小腿补充（坐姿提踵、农夫行走提踵）
- 髋内收/外展
- 不同器械变式（史密斯机、哈克机）

严格排除以下已有动作：杠铃深蹲, 腿举, 高脚杯深蹲, 保加利亚分腿蹲, 哑铃负重箭步蹲, 史密斯机深蹲, 杠铃硬拉, 哑铃罗马尼亚硬拉, 杠铃臀推, 腿弯举, 站姿提踵""",
        "count": 10,
    },
    # === 背部补充 ===
    {
        "batch_name": "背部补充",
        "prompt": """请生成 6 个背部训练动作，补充以下空缺：
- 下背专项（如山羊挺身、反向山羊挺身）
- 单侧划船变式（如哑铃单臂划船、海豹划船）
- 下拉变式（如窄距下拉、反握下拉、直臂下拉）
- T杆划船、潘德勒划船等高强度变式

严格排除以下已有动作：引体向上, 俯身哑铃划船, 杠铃划船, 坐姿绳索划船, 高位下拉, 直臂下压""",
        "count": 6,
    },
    # === 胸部补充 ===
    {
        "batch_name": "胸部补充",
        "prompt": """请生成 4 个胸部训练动作，补充以下空缺：
- 下胸专项（如下斜卧推、臂屈伸变式）
- 哑铃/器械夹胸变式
- 俯卧撑变式（如负重俯卧撑、吊环俯卧撑、窄距俯卧撑）

严格排除以下已有动作：哑铃卧推, 杠铃卧推, 上斜哑铃卧推, 哑铃飞鸟, 双杠臂屈伸, 绳索夹胸""",
        "count": 4,
    },
    # === 肩部补充 ===
    {
        "batch_name": "肩部补充",
        "prompt": """请生成 4 个肩部训练动作，补充以下空缺：
- 后束专项（如反向蝴蝶机、绳索后束拉）
- 中束补充（如器械侧平举、单臂绳索侧平举）
- 古巴旋转、阿诺德推举等综合肩部动作

严格排除以下已有动作：哑铃侧平举, 哑铃推举, 杠铃推举, 哑铃前平举, 哑铃俯身飞鸟, 绳索面拉, 哑铃耸肩""",
        "count": 4,
    },
    # === 手臂补充 ===
    {
        "batch_name": "手臂补充",
        "prompt": """请生成 8 个手臂训练动作，补充以下空缺：
- 二头专项（如牧师凳弯举、集中弯举、蜘蛛弯举、上斜凳弯举）
- 三头专项（如仰卧臂屈伸、单臂哑铃臂屈伸、器械下压）
- 前臂专项（如腕弯举、反向腕弯举、农夫行走）
- 肱肌/肱桡肌补充

严格排除以下已有动作：绳索下压, 哑铃弯举, 杠铃弯举, 锤式弯举, 窄距杠铃卧推, 哑铃颈后臂屈伸""",
        "count": 8,
    },
    # === 核心补充 ===
    {
        "batch_name": "核心补充",
        "prompt": """请生成 4 个核心训练动作，补充以下空缺：
- 下腹专项（如反向卷腹、剪刀腿）
- 侧腹/腹斜肌（如侧平板变式、伐木式）
- 动态核心（如药球砸地、死虫式）
- 抗旋转/抗伸展（如帕罗夫推）

严格排除以下已有动作：平板支撑, 悬垂举腿, 俄罗斯转体""",
        "count": 4,
    },
    # === 有氧/心肺（全新类别）===
    {
        "batch_name": "有氧心肺",
        "prompt": """请生成 6 个有氧/心肺训练动作，这是目前完全缺失的类别。覆盖：
- 传统有氧（跑步机、椭圆机、划船机、风阻单车、登山机）
- HIIT（波比跳、开合跳、高抬腿、登山者）
- 低冲击有氧（游泳、快走、骑行）

格式与其他动作一致，但 exercise_type 填"有氧"，target_muscles 填主要发力肌群或"全身/心肺"。

排除：任何已有力量训练动作""",
        "count": 6,
    },
    # === 拉伸/柔韧（全新类别）===
    {
        "batch_name": "拉伸柔韧",
        "prompt": """请生成 6 个拉伸/柔韧性动作，这是目前完全缺失的类别。覆盖：
- 动态拉伸（如猫式伸展、最伟大拉伸、腿摆）
- 静态拉伸（如腘绳肌拉伸、髋屈肌拉伸、胸椎旋转）
- 泡沫轴放松（如泡沫轴滚腿、泡沫轴滚背）

exercise_type 填"拉伸"，difficulty 填"初级"，equipment 填"自重"或"泡沫轴"。
target_muscles 填被拉伸的肌群名称。description 写清"怎么做+拉伸哪里+保持多久"。
common_errors 写过度拉伸/憋气/姿势错误等。

排除：任何已有力量训练动作""",
        "count": 6,
    },
    # === 爆发力/增强式（全新类别）===
    {
        "batch_name": "爆发力增强式",
        "prompt": """请生成 6 个爆发力/增强式训练动作，这是目前完全缺失的类别。覆盖：
- 跳类（跳箱、深蹲跳、分腿跳、连续跳）
- 抛类（药球胸前推、药球过顶砸、药球旋转抛）
- 奥举简化（高翻、抓举简化版、借力推举）

exercise_type 填"爆发力"，difficulty 填"中级"或"高级"。
description 中强调"落地缓冲/核心收紧/避免关节锁死"等安全要点。

严格排除已有力量训练动作""",
        "count": 6,
    },
    # === 固定器械补充 ===
    {
        "batch_name": "固定器械",
        "prompt": """请生成 5 个固定器械训练动作，覆盖：
- 器械推胸、器械飞鸟（蝴蝶机夹胸）
- 器械推举、器械划船
- 哈克深蹲、倒蹬机变式

这类动作的特点是轨迹固定、安全系数高、适合新手。请选择已有动作库中缺失的固定器械变式。

严格排除已有动作：腿举, 坐姿绳索划船, 高位下拉, 史密斯机深蹲""",
        "count": 5,
    },
    # === 纯自重补充 ===
    {
        "batch_name": "纯自重动作",
        "prompt": """请生成 4 个纯自重训练动作（equipment="自重"），覆盖：
- 上身自重（如反握引体、宽握引体、澳式引体、片式俯卧撑）
- 下身自重（如单腿深蹲、手枪蹲、臀桥、驴踢）
- 全身自重（如熊爬、螃蟹走、虫爬）

确保不与已有动作重复。difficulty 按动作实际难度填写，不要都是"初级"。

严格排除已有动作：引体向上, 平板支撑, 悬垂举腿, 站姿提踵, 双杠臂屈伸""",
        "count": 4,
    },
]

SCHEMA_REQUIREMENT = """
每个动作为一个 JSON 对象，包含以下字段：
{
  "name": "动作中文名（唯一，不与已有动作重复）",
  "exercise_type": "复合|孤立|有氧|拉伸|爆发力",
  "difficulty": "初级|中级|高级",
  "equipment": "所需器械（如哑铃、杠铃、绳索、自重、固定器械、药球、泡沫轴）",
  "target_muscles": ["目标肌群1", "目标肌群2", ...],
  "description": "动作标准做法，30-80字",
  "common_errors": ["常见错误1", "常见错误2", "常见错误3"]
}

返回格式：纯 JSON 数组，不要 markdown 代码块。
[
  {"name": "...", ...},
  {"name": "...", ...}
]
"""


def validate_exercise(ex: dict) -> list[str]:
    """校验单个动作的字段完整性，返回错误列表。"""
    errors = []
    if not ex.get("name") or not isinstance(ex["name"], str):
        errors.append("name 缺失或非字符串")
    if ex.get("exercise_type") not in ("复合", "孤立", "有氧", "拉伸", "爆发力"):
        errors.append(f"exercise_type 非法: {ex.get('exercise_type')}")
    if ex.get("difficulty") not in ("初级", "中级", "高级"):
        errors.append(f"difficulty 非法: {ex.get('difficulty')}")
    if not ex.get("equipment") or not isinstance(ex["equipment"], str):
        errors.append("equipment 缺失或非字符串")
    if not ex.get("target_muscles") or not isinstance(ex["target_muscles"], list):
        errors.append("target_muscles 缺失或非数组")
    if len(ex.get("target_muscles", [])) == 0:
        errors.append("target_muscles 为空")
    if not ex.get("description") or len(ex.get("description", "")) < 15:
        errors.append("description 太短（<15字）")
    if not ex.get("common_errors") or len(ex.get("common_errors", [])) < 1:
        errors.append("common_errors 缺失或为空")
    return errors


def deduplicate(new_exercises: list[dict], existing_names: set[str]) -> list[dict]:
    """去重：按 name 排除已有动作和批次内重复。"""
    seen = set(existing_names)
    result = []
    for ex in new_exercises:
        name = ex.get("name", "")
        if name in seen:
            print(f"  [跳过] 重复动作: {name}")
            continue
        seen.add(name)
        result.append(ex)
    return result


def generate_batch(llm: LLMProvider, batch: dict, retries: int = 2) -> list[dict]:
    """生成一批动作，校验失败时自动重试。"""
    full_prompt = batch["prompt"] + "\n\n" + SCHEMA_REQUIREMENT
    print(f"\n{'='*60}")
    print(f"批次: {batch['batch_name']} (目标 {batch['count']} 个)")
    print(f"{'='*60}")

    for attempt in range(retries + 1):
        if attempt > 0:
            print(f"  [重试] 第 {attempt} 次修正...")

        try:
            resp = llm.chat(
                [{"role": "user", "content": full_prompt}],
                temperature=0.5,
                model="default",
            )
            content = resp.content
            # 剥离可能的 markdown 代码块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            exercises = json.loads(content.strip())

            if not isinstance(exercises, list):
                print(f"  [错误] LLM 返回的不是数组")
                full_prompt += "\n\n上次输出格式错误，请返回纯 JSON 数组。"
                continue
        except json.JSONDecodeError as e:
            print(f"  [错误] JSON 解析失败: {e}")
            full_prompt += "\n\n上次输出的 JSON 解析失败，请确保是合法 JSON 数组。"
            continue

        # 校验 + 去重
        all_errors = []
        for ex in exercises:
            errs = validate_exercise(ex)
            if errs:
                all_errors.append(f"  {ex.get('name', '?')}: {'; '.join(errs)}")

        exercises = deduplicate(exercises, EXISTING_NAMES)

        if all_errors:
            print(f"  [校验] {len(all_errors)} 个动作有字段问题:")
            for e in all_errors[:5]:
                print(e)
            if len(all_errors) > 5:
                print(f"  ... 还有 {len(all_errors) - 5} 个")
            if attempt < retries:
                full_prompt += (
                    f"\n\n上次有 {len(all_errors)} 个动作校验失败。"
                    f"请修正：{'; '.join(all_errors[:3])}"
                )
                continue

        print(f"  [完成] 有效新动作: {len(exercises)}")
        for ex in exercises[:3]:
            print(f"    - {ex['name']} ({ex['exercise_type']}, {ex['difficulty']})")
        if len(exercises) > 3:
            print(f"    ... 还有 {len(exercises) - 3} 个")
        return exercises

    print(f"  [警告] 重试耗尽，返回空列表")
    return []


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM 批量扩展健身动作库")
    parser.add_argument("--validate-only", action="store_true", help="仅校验现有动作库")
    parser.add_argument("--dry-run", action="store_true", help="生成但不写入文件")
    parser.add_argument("--output", default="data/seed_exercises_expanded.json",
                        help="输出文件路径")
    args = parser.parse_args()

    # 加载现有动作库
    seed_path = Path("data/seed_exercises.json")
    if not seed_path.exists():
        seed_path = Path("../data/seed_exercises.json")
    with open(seed_path, encoding="utf-8") as f:
        existing = json.load(f)

    if args.validate_only:
        print(f"现有动作: {len(existing)}")
        errors = []
        for ex in existing:
            errs = validate_exercise(ex)
            if errs:
                errors.append(f"{ex.get('name','?')}: {'; '.join(errs)}")
        if errors:
            print(f"校验失败 {len(errors)} 个:")
            for e in errors:
                print(f"  {e}")
        else:
            print("全部校验通过。")
        return

    print(f"现有动作: {len(existing)}")
    print(f"将生成 {sum(b['count'] for b in BATCHES)} 个新动作 ({len(BATCHES)} 批次)\n")

    llm = LLMProvider()
    all_new = []

    for batch in BATCHES:
        new_exercises = generate_batch(llm, batch)
        all_new.extend(new_exercises)
        print(f"  累计: {len(all_new)} 新动作")

    if not all_new:
        print("\n[失败] 没有生成任何有效新动作。")
        sys.exit(1)

    # 二次去重（跨批次）
    all_new = deduplicate(all_new, EXISTING_NAMES)
    merged = existing + all_new

    print(f"\n{'='*60}")
    print(f"汇总: 原有 {len(existing)} → 新增 {len(all_new)} → 合并 {len(merged)}")
    print(f"{'='*60}")

    # 统计覆盖
    muscles = set()
    equips = set()
    for ex in merged:
        for m in ex.get("target_muscles", []):
            muscles.add(m)
        equips.add(ex.get("equipment", ""))
    print(f"覆盖肌群: {len(muscles)}")
    print(f"覆盖器械: {len(equips)} | {sorted(equips)}")

    if args.dry_run:
        print("\n[Dry Run] 不写入文件。")
        return

    # 同时更新原文件
    import shutil
    shutil.copy2(seed_path, str(seed_path) + ".bak")
    with open(seed_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n已写入: {seed_path} (备份: {seed_path}.bak)")
    print("下一步: python -m src.main --seed  # 重新灌入动作库")


if __name__ == "__main__":
    main()
