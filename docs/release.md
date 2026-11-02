# 发布与回滚

## 版本约定

- 版本号来自 git tag，形如 `v1.2.3`。
- 镜像地址：`ghcr.io/humble-moon/gakki-ai-fitness:<tag>`，同时打 `<sha>` 与 `latest`。
- 镜像 label 写入 `org.opencontainers.image.version` 与 `revision`，回滚时先看 label 再动手。
- `/health/live` 返回 `version` 字段，可直接确认运行中的是哪个版本。

## 发布

```bash
git tag v1.2.3
git push origin v1.2.3
```

推送 tag 会触发 `.github/workflows/release.yml`：

1. 构建镜像并推送到 GHCR；
2. 用 Trivy 扫描，存在**有修复版本的 HIGH / CRITICAL** 漏洞时失败；
3. 生成 CycloneDX SBOM，作为构建产物保留 90 天；
4. 创建 GitHub Release。

任何一步失败，都不会产生可用的 Release。扫描失败时的处理顺序：

1. 升级对应依赖并重新生成 `requirements.lock`；
2. 跑全量测试确认无回归；
3. 重新打一个更高的 patch tag（不要复用已推送的 tag）。

## 部署

```bash
export APP_IMAGE=ghcr.io/humble-moon/gakki-ai-fitness:v1.2.3
docker compose pull app
docker compose up -d --no-build app
curl -fsS http://127.0.0.1:8503/health/ready
curl -fsS http://127.0.0.1:8503/health/live   # 确认 version
```

`--no-build` 是有意的：compose 里保留了 `build: .`，不加这个参数时
`up` 可能用本地源码重新构建，部署的就不是扫描过的那份镜像了。

启动期会自动执行 `alembic upgrade head`；迁移失败会拒绝启动，而不是带着旧表提供服务。

## 回滚

```bash
# 1. 切回上一个 tag
export APP_IMAGE=ghcr.io/humble-moon/gakki-ai-fitness:v1.2.2
docker compose pull app
docker compose up -d --no-build app

# 2. 确认版本与就绪
curl -fsS http://127.0.0.1:8503/health/live
curl -fsS http://127.0.0.1:8503/health/ready
```

**数据库迁移不回滚。** 回滚前必须确认上一版本能读当前 schema。当前所有迁移都是向后兼容的加表/加列；如果将来出现破坏性迁移（删列、改类型），需要单独写 `downgrade` 并在发布说明里明确标注。

## 已知边界

- 没有蓝绿或金丝雀：`docker compose up -d` 是滚动替换，存在秒级中断。
- 没有自动回滚：健康检查失败只会让容器重启，不会自动切回上一版本。
- 应用仍是单副本：审核工件、计划与检查点写本地 SQLite，扩副本前需要先迁移存储。
