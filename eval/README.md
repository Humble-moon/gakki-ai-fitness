# 评测索引与使用边界

本目录保存历史评测结果及其可解释的元数据。`evaluation_manifest.json` 是结果索引，不覆盖或改写原始 JSON。

## 数据集与类型

- 主检索集：206 条 Golden Query（`retrieval`）。
- 知识检索子集：54 条自动生成并人工复核的问题（`retrieval`）。该子集的多标签 Recall 可能按相关文档数累计，必须以 manifest 的 `metric_definition` 解读，不能与普通单标签 Recall 直接比较。
- 条件上下文生成集：10 条（`generation_conditioned`），给定正确上下文后评估生成质量，不是完整检索 E2E。
- RAGAS 子集：68 条（`ragas_subset`），不是 206 条全量。
- 本地负载结果：`load_local`；不等同于生产 SLA。

## 指标约定

普通 `precision`、`recall`、`ndcg`、`mrr` 以及其 `@K` 变体必须位于 `[0, 1]`。若指标采用多标签、累计覆盖或其他特殊定义，必须在对应条目写明 `metric_definition`，并保留 `known_limits`。

运行校验：

```bash
python eval/scripts/validate_metrics.py --manifest eval/evaluation_manifest.json
```

历史结果的 `status`、`comparability` 和限制说明用于防止把不同数据集、模型或实验协议的数字直接横比。该目录不包含任何密钥，也不读取 `.env`。
