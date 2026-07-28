"""
===========================================================================
expand_golden_dataset.py — LLM 扩展评测集 80→200
===========================================================================
用法:
    python scripts/expand_golden_dataset.py             # 生成并追加
    python scripts/expand_golden_dataset.py --dry-run   # 仅预览不写入
===========================================================================
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.provider import LLMProvider

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "eval" / "golden_dataset" / "queries.json"

# 已有的 query 文本，用于去重
with open(DATASET_PATH, encoding="utf-8") as f:
    existing_queries = json.load(f)

EXISTING_TEXTS = {q["query"] for q in existing_queries}
NEXT_ID = max(int(q["id"][1:]) for q in existing_queries) + 1

BATCHES = [
    {
        "name": "增肌计划变体",
        "category": "muscle_building",
        "route": "muscle_building",
        "risk": "low",
        "count": 12,
        "prompt": """生成 12 条不同的用户增肌训练咨询, 模拟真实用户的自然语言表达。
覆盖以下场景:
- 新手增肌(第一次进健身房、只有哑铃、不知道怎么安排)
- 中级训练者(增肌瓶颈期、想冲重量、调整分化训练)
- 特殊条件(居家增肌只有弹力带/一周只能练2天/预算有限)
- 不同表达方式(口语化: "想变大只"、"胸肌练不大"、"胳膊太细了")

每条 query 格式:
{"query": "用户的自然语言描述(50-100字)", "category": "muscle_building", "expected_route": "muscle_building", "safety_risk": "low"}

返回纯 JSON 数组。不要编号。不要重复已有句式(不要全是"xx/xxkg练了x年怎么安排增肌计划")。""",
    },
    {
        "name": "减脂计划变体",
        "category": "fat_loss",
        "route": "fat_loss",
        "risk": "low",
        "count": 12,
        "prompt": """生成 12 条不同的用户减脂训练咨询。覆盖:
- 不同体型(160/55kg女生想瘦腿、185/100kg想减肚子、产后减脂)
- 不同条件(只有小区健身器材、每天只有30分钟、膝盖不好不能跑步)
- 不同表达(口语化: "想瘦但是不想掉肌肉"、"肚子怎么都减不下来"、"代谢慢怎么办")
- 混淆场景(同时想增肌减脂、练了半年体重没变化、越练越重)

每条 query 格式:
{"query": "...", "category": "fat_loss", "expected_route": "fat_loss", "safety_risk": "low" if 无伤病 else "medium"}

返回纯 JSON 数组。""",
    },
    {
        "name": "动作分析变体",
        "category": "exercise_analysis",
        "route": "exercise_analysis",
        "risk": "medium",
        "count": 15,
        "prompt": """生成 15 条用户动作分析/姿势诊断咨询。覆盖:
- 疼痛型(深蹲膝盖疼、卧推肩膀疼、硬拉下背酸、推举肩峰撞击感)
- 姿势困惑型(深蹲脚跟抬起、卧推杠铃轨迹不稳、弯举时借力)
- 发力困惑型(练胸只感觉肩膀酸、引体向上只练到二头、臀推找不到臀的感觉)
- 不对称问题(右边比左边弱、左边肩膀弹响、右边膝盖不适)
- 特殊场景:(手术后恢复训练、孕产后恢复、中老年人关节不适)

每条 query 格式:
{"query": "...", "category": "exercise_analysis", "expected_route": "exercise_analysis", "safety_risk": "medium"}

涉及伤病/疼痛的 risk 标 "medium"。返回纯 JSON 数组。""",
    },
    {
        "name": "伤病风险变体",
        "category": "injury",
        "route": "exercise_analysis",
        "risk": "high",
        "count": 15,
        "prompt": """生成 15 条涉及伤病史/高危场景的用户咨询。覆盖:
- 已有伤病史(腰间盘突出、半月板损伤、肩袖撕裂、网球肘、跟腱炎)
- 术后恢复(前交叉韧带重建术后6个月、腰椎微创术后3个月)
- 慢性疼痛(每天下背痛、晨僵、关节咔咔响)
- 隐含风险(上周跑步膝盖突然软了一下、做深蹲时感觉膝盖要散架了)
- 危险组合(有心脏病还想冲大重量、骨质减少想练硬拉)

每条 query 格式:
{"query": "...", "category": "injury", "expected_route": "exercise_analysis", "safety_risk": "high"}

返回纯 JSON 数组。这些 query 必须包含明确的伤病/疼痛描述。""",
    },
    {
        "name": "混合/歧义场景",
        "category": "mixed",
        "route": "exercise_analysis",
        "risk": "medium",
        "count": 12,
        "prompt": """生成 12 条目标歧义/混合意图的用户咨询。覆盖:
- 同时有增肌目标+伤病(膝盖以前伤过还想增肌练腿、肩袖损伤史想练胸)
- 同时有减脂目标+伤病(腰突想减脂、膝盖不好想减重)
- 不确定自己属于哪个场景(想变壮但怕受伤、想瘦但不知道先减脂还是先增肌)
- 自相矛盾的表达(想增肌但不想变重、想深蹲但膝盖感觉不对)

每条 query 格式:
{"query": "...", "category": "mixed", "expected_route": "exercise_analysis", "safety_risk": "medium"}

安全优先: 涉及伤病的都应该路由到 exercise_analysis。返回纯 JSON 数组。""",
    },
    {
        "name": "知识问答",
        "category": "knowledge",
        "route": "qa",
        "risk": "low",
        "count": 15,
        "prompt": """生成 15 条健身知识问答类型的用户提问。覆盖:
- 原理类(增肌的原理是什么、为什么练后要补充蛋白质、HIIT和稳态有氧哪个减脂效果好)
- 比较类(自由重量vs固定器械、乳清蛋白vs酪蛋白、碳水循环vs低碳饮食)
- 辟谣类(蛋白粉伤肾吗、深蹲伤膝盖吗、练腹能减肚子吗、女生练壮怎么办)
- 实操类(组间休息多久最好、一周练几次最优、空腹训练好不好)
- 特殊人群(青少年举重影响身高吗、孕妇可以深蹲吗、50岁还能增肌吗)

这些 query 不需要生成训练计划, 只是知识问答。不要包含伤病/疼痛词(会触发安全路由)。

每条 query 格式:
{"query": "...", "category": "knowledge", "expected_route": "qa", "safety_risk": "low"}

返回纯 JSON 数组。""",
    },
    {
        "name": "边缘/压力测试",
        "category": "mixed",
        "route": "exercise_analysis",
        "risk": "medium",
        "count": 12,
        "prompt": """生成 12 条边缘案例/压力测试用户咨询。覆盖:
- 极简输入(只说"帮我安排训练"不给任何身体参数)
- 极端条件(80岁想练力量、体重45kg想增肌20kg、只有轮椅能做什么)
- 罕见器械(只有壶铃、只有TRX、只有药球、完全是零器械)
- 矛盾指令(同时要增肌和减脂、想变大但不想练腿、想深蹲但绝对不能膝盖过脚尖)
- 多问题叠加(我膝盖疼+下背酸+肩膀响, 还能练吗)
- 非健身问题(健身和考研怎么平衡、失恋后靠健身走出来有用吗)

每条 query 格式:
{"query": "...", "category": "mixed", "expected_route": "exercise_analysis" if 含伤病 else "qa", "safety_risk": "medium"}

返回纯 JSON 数组。""",
    },
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    llm = LLMProvider()
    all_new = []
    next_id = NEXT_ID

    for batch in BATCHES:
        print(f"\n{'='*50}")
        print(f"批次: {batch['name']} (目标 {batch['count']} 条)")

        prompt = batch["prompt"] + "\n\n已存在的 query 文本(不能重复):\n"
        prompt += "\n".join(f"- {t}" for t in sorted(EXISTING_TEXTS)[:80])

        for attempt in range(3):
            try:
                resp = llm.chat([{"role": "user", "content": prompt}],
                                temperature=0.7)
                content = resp.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                new = json.loads(content.strip())
                if not isinstance(new, list):
                    print("  返回不是数组，重试...")
                    continue
                break
            except Exception as e:
                print(f"  LLM调用失败: {e}")
                if attempt == 2:
                    new = []

        accepted = 0
        for q in new:
            txt = q.get("query", "")
            if not txt or len(txt) < 10:
                continue
            if txt in EXISTING_TEXTS:
                print(f"  跳过重复: {txt[:40]}...")
                continue
            q["id"] = f"q{next_id:03d}"
            # 确保必要字段
            q.setdefault("category", batch["category"])
            q.setdefault("expected_route", batch["route"])
            q.setdefault("safety_risk", batch["risk"])
            q.setdefault("relevant_doc_ids", [])
            q.setdefault("hard_negative_ids", [])
            next_id += 1
            EXISTING_TEXTS.add(txt)
            all_new.append(q)
            accepted += 1

        print(f"  接受: {accepted} / 生成: {len(new)}")

    print(f"\n{'='*50}")
    print(f"总计: 原有 {len(existing_queries)} → 新增 {len(all_new)} → 合并 {len(existing_queries) + len(all_new)}")

    if not all_new:
        print("[失败] 没有生成有效新query")
        sys.exit(1)

    if args.dry_run:
        print("\n[Dry Run] 不写入。预览前 3 条:")
        for q in all_new[:3]:
            print(f"  {q['id']}: {q['query'][:80]}...")
        return

    merged = existing_queries + all_new
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"已写入: {DATASET_PATH}")
    # 备份
    import shutil
    bak = str(DATASET_PATH) + f".bak-expand-{len(all_new)}"
    shutil.copy2(str(DATASET_PATH), bak)


if __name__ == "__main__":
    main()
