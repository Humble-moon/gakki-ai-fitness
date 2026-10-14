# 贡献指南

这是一个本地运行的 AI 健身私教项目。本文只写这个仓库真实的约定，不写通用模板——如果某条和你在别处见过的规范不一样，以本文为准。

## 环境准备

需要 Python 3.11+。外部依赖（PostgreSQL / Redis / Neo4j / MinIO）由 Docker 提供：

```bash
git clone <repo> && cd gakki-ai-fitness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # 按需填 DASHSCOPE_API_KEY 等
docker compose up -d        # 起四个服务
python scripts/run_demo.py --mode demo --host 127.0.0.1 --port 8503
# → 浏览器打开 http://localhost:8503
```

接口文档：服务起来后访问 <http://localhost:8503/docs>（FastAPI 自带，无需额外生成）。

## 提交前必须做的三件事

```bash
python -m ruff check .          # 静态检查，规则与理由见 ruff.toml
python -m pytest -q             # 离线测试，默认排除 integration / live
python scripts/verify_project_facts.py   # 确认事实基线仍然成立
```

CI 会跑同样的三件事，外加一条署名检查（见下）。一个或多个失败就不要提 PR。

### 测试标记

默认的 `pytest -q` **不会**跑 `integration` 和 `live` 标记的用例——它们需要外部服务或真实模型。改动涉及真实调用链时，显式开跑：

```bash
pytest -m integration
```

不要把需要网络的用例改写成不需要，那只会让它在 CI 里假装通过。

## 文档与事实的约定

这个仓库对「文档说了什么」有明确要求：**可复核的才写成事实**。

- 数量、路径、行为这类能由仓库文件独立核实的，写进 `docs/project-fact-baseline.md` 并标注证据路径与统计日期。
- 无法由仓库独立复核的（开发时间线、外部服务的历史状态等），要么标注「无法独立复核」，要么不写。
- 改了数据文件或代码量，顺手复核基线文档里的对应条目——`verify_project_facts.py` 会把差异打出来。

**不要为了让文档好看而修改原始资料去凑证据。** 面向非技术读者的说明（README、界面文案）也不能出现"生产级""多租户""医疗级"这类与项目边界不符的措辞——本项目就是跑在 localhost 上的本地应用。

## 提交信息

采用 `<type>(<scope>): <中文描述>` 的形式，描述写清楚**为什么改**，而不只是改了什么。

```
fix(ui): 打开已保存计划时空态引导区未清除，并更新 README 与截图
feat(plan): 接入确定性热量目标，修通运动习惯与跨会话画像链路
docs: 更新界面截图，对齐最新 UI
```

常见 type：`feat` / `fix` / `refactor` / `docs` / `chore` / `test`。

**不要在提交信息里添加 AI 署名。** CI 的 `attribution-guard` 会扫描提交范围，发现 `Co-Authored-By:` 里出现 AI 工具名就直接失败。这不只是流程偏好：仓库首页的贡献者统计会把署名计入贡献者列表，而这类记录无法通过后续删除提交抹掉。

## 分支与 PR

- 从 `master` 切分支，不要直接往 `master` 推。
- 一个 PR 做一件事。把格式化、重命名和功能改动混在同一个 PR 里，review 时没人分得清哪行是逻辑变更。
- PR 描述里说清楚：**改了什么、为什么、怎么验证的**。涉及界面改动请附截图；涉及安全门、检索或生成链路的改动，请说明覆盖了哪些失败路径。

## 安全相关的改动

涉及以下任一处，请格外说明测试方式，并在 PR 里指出你验证过的失败路径：

- `src/agents/planner.py` 的安全闸门（伤病词 / 急症词覆盖 LLM 决策的逻辑）
- `src/agents/fact_checker.py` 与 `src/core/qa_safety.py`
- `src/security/` 下的认证、限流、成本熔断
- HITL 审核工件的创建与解除

这些路径的共同点是：**出错时用户看不出自己走的是哪条路**。改它们时，宁可多写一个失败用例。

## 数据与隐私

仓库里不放任何真实的个人健康数据。测试夹具用构造数据；`data/knowledge/` 下的文档可以提交，但不要加入含个人身份信息的材料。

## 许可

提交即表示你同意以本仓库的 [MIT 许可](LICENSE) 发布你的贡献。
