# 可观测性

## 三个 SLI

| SLI | 指标来源 | 当前定义 |
|---|---|---|
| 请求成功率 | `http_requests_total` | `(total - 4xx - 5xx) / total` |
| P95 延迟 | `http_request_duration_seconds` | 按路由汇总后取 P95，流式接口按**响应完成**计时 |
| LLM 调用失败率 | `llm_calls_total` | `outcome != success`，熔断跳过计入失败侧 |

直接读数：`GET /admin/sli`。Prometheus 原始指标：`GET /admin/metrics/prometheus`。

两个入口都在 `/admin/*` 下，受 `ADMIN_TOKEN` 保护。Prometheus 用标准 Bearer 头：

```yaml
scrape_configs:
  - job_name: ironmind
    metrics_path: /admin/metrics/prometheus
    authorization:
      type: Bearer
      credentials_file: /etc/prometheus/admin_token
    static_configs:
      - targets: ["app:8503"]
```

完整示例与告警规则见 `deploy/prometheus/`。

## 为什么路径按模板记

`http_requests_total` 的 `path` 标签取自路由模板（`/api/plans/{plan_id}`），不是原始 URL。用原始路径时，每个计划 ID 都会生成一组新时间序列，几个用户就能把内存撑爆。未匹配到路由的请求（404）退化为两级前缀。

## 告警

| 告警 | 条件 | 级别 | 含义与处置 |
|---|---|---|---|
| `IronMindDown` | 抓取失败 2 分钟 | critical | 进程或网络不通；先看容器状态与 `/health/ready` |
| `ApiErrorRateHigh` | 5xx 占比 > 5%，持续 5 分钟 | critical | 看结构化日志里的 `request_id`，沿该 ID 找同一条链路的错误 |
| `ApiLatencyP95High` | P95 > 5s，持续 10 分钟 | warning | 先区分是 LLM 上游慢还是本地处理慢：看 `llm_call_duration_seconds` |
| `LLMFailureRateHigh` | 失败率 > 10%，持续 10 分钟 | critical | 上游故障或密钥失效；降级链可能已耗尽 |
| `LLMCircuitOpen` | 10 分钟内出现熔断跳过 | warning | 某模型持续失败；确认主模型与备用模型状态 |

阈值是起始值，不是结论。上线后应按真实分布调整——尤其 P95，流式问答的长回答会天然抬高它。

## 日志与请求关联

日志是单行 JSON，带 `request_id`；该 ID 同时回写到响应头 `X-Request-ID`。用户报错时拿响应头里的 ID 就能检索整条链路。

没有日志聚合：容器重启后本地日志会丢。要做集中检索，需要接入 Loki 或 ELK，这一步尚未完成。

## 审计

`data/audit.db` 中只追加的 `audit_events` 表记录：登录/注册/登出/游客签发/数据认领、审核创建与解除、数据导出与删除。

查询：`GET /admin/audit-events?limit=100&action=&actor=`。

审计写失败只记日志、不阻塞业务；`detail` 不存密码、令牌、健康描述原文。没有归档与保留期策略。
