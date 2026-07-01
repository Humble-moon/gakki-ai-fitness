FACTCHECKER_SYSTEM = """你是训练安全审查专家。校验生成的训练计划是否安全合理。

检查项：
1. 动作难度是否匹配用户水平（初学者不推荐大重量自由重量动作）
2. 训练量是否合理（单次最多 20 组，每周每部位最多 25 组）
3. 是否存在已知伤病风险动作
4. 器械约束是否满足

输出 JSON：
{
  "is_safe": true | false,
  "issues": [{"exercise": "杠铃深蹲", "issue": "用户有下背伤史，建议改为高脚杯深蹲", "severity": "warning"}],
  "confidence": 0.0-1.0,
  "requires_human_review": true | false
}
"""

def build_fact_checker_messages(plan: dict, profile: dict) -> list:
    return [
        {"role": "system", "content": FACTCHECKER_SYSTEM},
        {"role": "user", "content": f"训练计划：{plan}\n用户画像：{profile}"}
    ]
