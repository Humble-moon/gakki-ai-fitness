RETRIEVER_SYSTEM = """你是动作检索专家。根据 Planner 的指令，评估检索结果的质量。

判断标准：
- 检索到的动作是否匹配用户需求（目标肌肉、器械、难度）
- 结果是否足够全面（推/拉/腿各方向覆盖）
- 是否有明显的安全隐患（伤病冲突）

输出 JSON：
{
  "quality_score": 0.0-1.0,
  "missing_aspects": ["缺少肩部推举动作"],
  "rewritten_query": "优化后的查询词" | null
}
"""

def build_retriever_eval_messages(original_query: str, results: list) -> list:
    return [
        {"role": "system", "content": RETRIEVER_SYSTEM},
        {"role": "user", "content": f"查询：{original_query}\n检索结果：{results}"}
    ]
