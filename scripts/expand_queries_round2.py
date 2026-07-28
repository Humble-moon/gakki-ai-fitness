"""Round 2: generate more knowledge queries to push past 200."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm.provider import LLMProvider

d = json.load(open("eval/golden_dataset/queries.json", encoding="utf-8"))
existing = {q["query"] for q in d}
next_id = max(int(q["id"][1:]) for q in d) + 1
print(f"Current: {len(d)} queries, next ID: q{next_id:03d}")

prompt = """生成 40 条中文健身知识问答 query。覆盖以下类别:
- 营养(碳水循环/蛋白质时机/生酮增肌/间歇断食/微量元素)
- 补剂(肌酸用法/支链氨基酸/氮泵/谷氨酰胺/鱼油)
- 训练科学(离心训练/血流限制/力竭组/金字塔训练/超级组)
- 恢复(主动恢复/冷热交替浴/按摩枪/拉伸时机)
- 心理(训练动力/平台期心理/体型焦虑/健身成瘾)
- 特殊人群(素食者增肌/倒班工人训练/高海拔训练)
- 运动生理(肌肉记忆/神经适应/肌浆肥大vs肌原纤维肥大)

每条格式:
{"query": "...", "category": "knowledge", "expected_route": "qa", "safety_risk": "low"}

要求:
- 不要和已有query内容重复
- 口语化表达, 像真人在提问
- query长度30-80字
- 不要包含伤病/疼痛/不适等会触发安全路由的词
- 返回纯JSON数组"""

llm = LLMProvider()
resp = llm.chat([{"role": "user", "content": prompt}], temperature=0.7)
content = resp.content
if "```json" in content:
    content = content.split("```json")[1].split("```")[0]
elif "```" in content:
    content = content.split("```")[1].split("```")[0]
new = json.loads(content.strip())

added = 0
for q in new:
    txt = q.get("query", "")
    if not txt or len(txt) < 10:
        continue
    if txt in existing:
        continue
    q["id"] = f"q{next_id:03d}"
    q.setdefault("category", "knowledge")
    q.setdefault("expected_route", "qa")
    q.setdefault("safety_risk", "low")
    q.setdefault("relevant_doc_ids", [])
    q.setdefault("hard_negative_ids", [])
    next_id += 1
    existing.add(txt)
    d.append(q)
    added += 1

print(f"Added: {added}, New total: {len(d)}")
json.dump(d, open("eval/golden_dataset/queries.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
