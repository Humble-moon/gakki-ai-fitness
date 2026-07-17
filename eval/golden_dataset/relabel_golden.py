"""
=============================================================================
Golden Set 补标脚本 — 动作库扩容（39→106）后的相关性标签补全
=============================================================================
背景：
  2026-07-16 动作库从 39 个扩到 106 个，但 queries.json 的 relevant_doc_ids
  仍基于旧的 39 动作库标注。新增 67 个动作中与查询相关的动作未被标注，
  会系统性低估所有消融组的 Recall（检索到了新相关动作却被判为不相关）。

方法论（面试可讲）：
  1. 穷举判定而非 pooling：动作库仅 106 个，直接把全量动作列表给裁判 LLM
     逐查询判定相关性——避免"用自己的检索器生成候选"的循环偏差
     （pooling 会让标签偏向检索器已能找到的结果，虚高 Precision/Recall）。
  2. 异构裁判：用 qwen-plus（LLM_JUDGE_* 配置）做标注，与生成/检索链路的
     DeepSeek 不同厂商，标注独立性更强。
  3. 只增不删：原有标签视为人工确认过的 ground truth，脚本只补充新相关
     动作，不删除任何已有标签；与 hard_negative_ids 冲突的候选直接丢弃。
  4. 可审计：输出 relabel_diff.md 逐查询列出新增标签及裁判理由，供人工抽查。

用法：
    python eval/golden_dataset/relabel_golden.py            # 全量 80 条
    python eval/golden_dataset/relabel_golden.py --limit 3  # 冒烟测试
    python eval/golden_dataset/relabel_golden.py --dry-run  # 只出 diff 不写回
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.provider import LLMProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QUERIES_FILE = Path(__file__).parent / "queries.json"
EXERCISES_FILE = PROJECT_ROOT / "data" / "seed_exercises.json"
DIFF_FILE = Path(__file__).parent / "relabel_diff.md"

JUDGE_SYSTEM = """你是健身领域的检索相关性标注专家。给定一条用户查询和一份动作库清单，
你需要判断哪些动作与该查询【相关】。

相关的定义：如果用户带着这个查询使用健身助手，这个动作出现在检索结果的前几位是
合理且有帮助的。判断时考虑：
- 目标肌群是否匹配查询意图（如"练胸"→ 卧推类、飞鸟类）
- 器械是否匹配查询约束（如"只有哑铃"→ 排除杠铃/器械类动作）
- 训练目标是否匹配（增肌/减脂/力量）
- 伤病安全性（如查询提到膝伤，深蹲类动作即使肌群匹配也不相关）

严格要求：
1. 只从提供的动作清单中选择，动作名必须与清单完全一致，不要自己发明
2. 宁缺毋滥：只选高度相关的，边缘相关的不要选（标注过松会毁掉评测区分度）
3. 输出 JSON，不要输出其他内容：
{"relevant": ["动作名1", "动作名2"], "reason": "一句话说明选择逻辑"}"""


def build_messages(query: dict, exercises: list) -> list:
    ex_lines = []
    for e in exercises:
        muscles = "/".join(e.get("target_muscles", []))
        ex_lines.append(f"- {e['name']}（{e.get('exercise_type','')}｜{e.get('equipment','')}｜{muscles}）")
    ex_text = "\n".join(ex_lines)
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"""【用户查询】
{query['query']}

【查询类别】{query.get('category', '')}

【动作库清单（共 {len(exercises)} 个）】
{ex_text}

请判定哪些动作与该查询相关，输出 JSON。"""},
    ]


def name_covered(candidate: str, existing: list) -> bool:
    """与 rag_metrics._name_matches 一致的判重逻辑：
    精确匹配或互为子串即视为已覆盖，避免加入冗余标签（如已有"深蹲"再加"杠铃深蹲"）。
    """
    for ex in existing:
        if candidate == ex or candidate in ex or ex in candidate:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Relabel golden set after exercise library expansion")
    parser.add_argument("--limit", type=int, default=None, help="Limit queries for smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Generate diff only, don't write queries.json")
    args = parser.parse_args()

    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    exercises = json.loads(EXERCISES_FILE.read_text(encoding="utf-8"))
    valid_names = {e["name"] for e in exercises}

    targets = queries[: args.limit] if args.limit else queries
    logger.info(f"Relabeling {len(targets)}/{len(queries)} queries against {len(exercises)} exercises...")

    llm = LLMProvider()
    if "judge" not in llm.available_models:
        logger.error("未配置 LLM_JUDGE_*（异构裁判），补标必须用独立裁判，退出")
        sys.exit(1)
    logger.info(f"Judge model: {llm.available_models['judge']}")

    diff_lines = [
        "# Golden Set 补标 Diff（动作库 39→106 扩容）",
        "",
        f"**补标时间**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**裁判模型**: {llm.available_models['judge']}（异构裁判，穷举判定，只增不删）",
        "",
    ]

    total_added = 0
    changed_queries = 0

    for i, q in enumerate(targets, 1):
        qid = q["id"]
        existing = q["relevant_doc_ids"]
        hard_neg = q.get("hard_negative_ids", [])

        try:
            result = llm.chat_with_json_mode(build_messages(q, exercises), model="judge")
            judged = result.get("relevant", [])
            reason = result.get("reason", "-")
        except Exception as e:
            logger.error(f"[{i}/{len(targets)}] {qid} judge failed: {e}")
            continue

        added, rejected = [], []
        for name in judged:
            if name not in valid_names:
                rejected.append(f"{name}(不在库中)")
                continue
            if name_covered(name, existing):
                continue  # 已有标签覆盖，跳过
            if name_covered(name, hard_neg):
                rejected.append(f"{name}(与hard_negative冲突,尊重原标注)")
                continue
            added.append(name)

        if added:
            q["relevant_doc_ids"] = existing + added
            changed_queries += 1
            total_added += len(added)

        logger.info(f"[{i}/{len(targets)}] {qid}: +{len(added)} labels"
                    + (f" | rejected: {len(rejected)}" if rejected else ""))

        diff_lines.append(f"## {qid}: {q['query'][:60]}")
        diff_lines.append(f"- 原有标签({len(existing)}): {', '.join(existing)}")
        diff_lines.append(f"- **新增({len(added)})**: {', '.join(added) if added else '（无）'}")
        if rejected:
            diff_lines.append(f"- 已拒绝: {', '.join(rejected)}")
        diff_lines.append(f"- 裁判理由: {reason}")
        diff_lines.append("")

    diff_lines.insert(4, f"**汇总**: {changed_queries}/{len(targets)} 条查询有新增，共 +{total_added} 个标签\n")
    DIFF_FILE.write_text("\n".join(diff_lines), encoding="utf-8")
    logger.info(f"Diff report saved to {DIFF_FILE}")

    if args.dry_run:
        logger.info("Dry-run mode: queries.json NOT modified")
        return

    if args.limit:
        logger.warning("--limit 模式下不写回 queries.json（避免部分补标造成不一致），仅生成 diff")
        return

    backup = QUERIES_FILE.with_suffix(f".json.bak-{time.strftime('%Y%m%d')}")
    shutil.copy(QUERIES_FILE, backup)
    logger.info(f"Backup saved to {backup}")

    QUERIES_FILE.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"queries.json updated: {changed_queries} queries changed, +{total_added} labels")


if __name__ == "__main__":
    main()
