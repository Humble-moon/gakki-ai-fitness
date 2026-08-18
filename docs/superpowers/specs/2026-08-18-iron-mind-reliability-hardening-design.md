# IRON MIND 可靠性与安全终态加固设计

- 日期：2026-08-18
- 范围：localhost 单用户 Demo 的 P0/P1 可靠性整改
- 目标：默认离线测试通过；模型、缓存、文档存储失败语义明确；未通过安全检查的结果不得持久化

## 1. 背景与完成标准

IRON MIND 当前已经完成视觉改造和第一轮前端/SSE 可靠性修复，但后端仍存在几类风险：所有模型失败时可能返回可被上层误当成正常内容的降级文本；同步与流式 Orchestrator 的最终安全判断不统一；不安全结果可能写入 SemanticCache、Conversation 或 LongTermMemory；SemanticCache 缓存条目缺少显式版本和画像隔离；DocumentStore 在 embedding 或 chunk 写入中途失败时可能留下半成品。

本轮完成标准：

1. 默认 `pytest` 不访问 DeepSeek、DashScope 或其他外部 API，并通过所有默认测试。
2. Provider 全部失败时抛出明确异常，不返回可继续加工的伪成功内容。
3. 同步和流式结果共享同一最终安全闸门。
4. 只有 schema 有效、FactChecker 明确安全、无未解决问题、无需人工复核且没有 provider degraded 的结果，才能写入 cache、conversation 和 long-term memory。
5. SemanticCache 故障按 miss/best-effort 处理，不阻断主流程；缓存条目带版本和 profile fingerprint。
6. DocumentStore 的保存操作原子化，任何失败都不留下文档或 chunks 半成品。
7. 真实 API 测试只在显式设置 `RUN_LIVE_LLM_TESTS=1` 后运行，并只发送脱敏固定问题。

## 2. 方案选择

采用“保留现有模块边界、增加统一失败语义和持久化闸门”的方案。相比直接散落异常捕获的最小修补，该方案能让同步/流式入口共享规则；相比完整状态机重构，改动范围更适合当前 localhost Demo，回归风险较低。

## 3. Provider 失败语义

新增 `LLMUnavailableError`，携带已尝试模型和错误摘要。`LLMProvider.chat()` 保留 retry/fallback；全部模型失败后抛出异常，不再返回“服务暂时不可用”的正常 `LLMResponse`。fallback 成功仍返回真实内容，并以 `degraded=True` 表示实际使用了备用模型。

`chat_stream()` 在连接尚未产生内容时允许继续 fallback；所有模型不可用时抛出 `LLMUnavailableError`。已开始输出后发生中断则向上抛出流式错误，不插入伪造文本。日志只记录别名、错误类型和摘要，不记录密钥、完整 prompt 或业务数据。

## 4. Orchestrator 最终安全闸门

在 Orchestrator 中增加共享的最终化与持久化辅助逻辑，供同步和流式链路使用：

- `_finalize_result(result, checks, rewrite_count)`：归一化 warnings、requires_review、confidence 和 rewrite_count，并按 fail-closed 规则生成最终状态。
- `_persist_if_safe(profile, query, result, session_id=None)`：只有安全且完整的结果才写入持久化层。

安全结果必须同时满足：

- 输出 schema 校验通过；
- FactChecker 的 `is_safe` 明确为 `True`；
- `issues` 为空；
- `requires_human_review` 为 `False`；
- provider 未 degraded；
- 没有异常或缺失状态。

不安全或需要复核的结果仍可返回给前端并携带警告，但不得写入 cache、conversation 或 long-term memory。Provider 失败和 schema 无效应作为错误事件/异常结束，不进入持久化。

## 5. SemanticCache

缓存条目升级为带版本的数据结构：

```json
{
  "schema_version": 2,
  "profile_fingerprint": "...",
  "query": "...",
  "_embedding": [...],
  "result": {...}
}
```

缓存键采用 `cache:fitness:v2:{profile_fingerprint}:{query_hash}`。fingerprint 来自稳定规范化后的 profile 摘要；日志不输出原始画像。

读取策略：Redis 异常按 miss 处理；embedding 异常跳过语义匹配；JSON 损坏、版本不匹配、画像 fingerprint 不匹配或结果结构无效的条目忽略。写入失败只记录 warning，不影响主流程。只有经过 Orchestrator 最终安全闸门的结果才能写入。

## 6. DocumentStore 原子性

`save()` 使用事务边界覆盖旧文档清理、user_documents 插入、切块、embedding 生成和 document_chunks 插入。全部成功才提交。任何异常都回滚文档和 chunks，并向上层抛出明确存储错误，不返回半成品 document_id。

若现有 `PGClient` 没有事务上下文，则增加最小事务支持，不进行无关数据库重构。删除旧文档也必须在同一事务中完成，避免 quota 清理和新文档保存之间出现中间状态。

## 7. 测试策略

默认测试使用 fake provider、fake Redis 和 fake DB，覆盖：

- 全模型失败抛出 `LLMUnavailableError`；
- fallback 成功和 degraded 标记；
- 同步/流式安全闸门阻止三类持久化；
- SemanticCache profile 隔离、版本兼容和 best-effort；
- DocumentStore embedding/chunk 失败回滚；
- 默认测试不触网。

Live 测试使用 `@pytest.mark.live`，没有 `RUN_LIVE_LLM_TESTS=1` 时自动 skip。显式运行时仅使用固定脱敏问题，不上传训练档案或文档，不写入项目持久化层，也不打印密钥。

## 8. 实施顺序

1. 先写 Provider 失败语义测试并修改 provider。
2. 写 Orchestrator 安全闸门测试，接入同步和流式路径。
3. 修改 SemanticCache 的 schema/version、fingerprint 和 best-effort 行为。
4. 修改 DocumentStore 事务与回滚。
5. 增加 live marker 和显式 opt-in 测试。
6. 完善 health/ASGI 测试。
7. 运行定向测试、完整默认测试、静态检查和本地运行验证。
8. 最后再由用户决定是否使用本地新 key 进行 live 验证。

## 9. 非目标

- 不在本轮改造成公网多用户生产系统。
- 不在本轮重构为完整事件状态机。
- 不在本轮把真实 API key 写入仓库、测试、日志或文档。
- 不在本轮把公司数据或用户上传文档发送到外部服务。
