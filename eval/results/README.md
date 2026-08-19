# 条件路由对比结果

本目录的新结果是 **deterministic fixture（确定性夹具）**，用于复核同一批 query 在以下两种调用策略下的路由差异：

- `baseline`：保持旧调用方式，所有 query 都传 `route=None`。
- `conditional`：纯函数分类器命中明确类别时传显式 route；分类为 fallback 时仍传 `route=None`。

这是一项**非检索质量实验**。脚本不启动 Retriever、GraphSearch、向量库、LLM 或其他 provider，因此不能用于声称 precision、recall、排序质量或回答质量提升。输出中的 fixture metrics 只描述路由覆盖率和改变显式 route 参数的比例。

`latency_ms` 和 `provider_calls` 没有真实测量，必须保持 `null`，不得用脚本进程耗时或静态推测值代替。每份输出同时记录数据集大小、SHA-256、route distribution、逐条路由和 limitations，便于本地复核。

生成示例：

```bash
python eval/scripts/compare_retrieval_routes.py \
  --dataset eval/golden_dataset/queries.json \
  --output eval/results/conditional-route-fixture.json
```

脚本拒绝覆盖已存在的结果文件；如需新一轮记录，请使用包含日期或代码版本的新文件名。
