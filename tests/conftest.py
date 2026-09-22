"""测试会话的全局准备。

## 为什么需要这个文件

部分测试会在构造 Agent 时创建 OpenAI 客户端（`src/llm/agent` → `LLMProvider`
→ `OpenAI(...)`），而 `openai>=2.x` 在 `api_key` 为空时会**直接抛**
`OpenAIError: Missing credentials`。同理 `app/server.py` 在**模块级**就
`orch = Orchestrator()`，任何 `from app import server` 的测试都会踩到。

开发机上因为有 `.env`，本地跑得通——但刚 clone 仓库、还没配 key 的人
（以及 CI）会看到一片失败。**那不是测试想验证的东西**：离线套件全部走
mock/fake，不发真实请求；真正联网的评测由 `integration` / `live` 标记
隔离，默认不跑（见 `pytest.ini`）。

## 设计：不改变开发机行为

先 `load_dotenv()` 把开发机上的 `.env` 读进来，再用 `setdefault` **只补
缺失项**。因此：

* 有 `.env`（开发机）：真实 key 生效，行为与以前完全一致；
* 无 `.env`（CI / 新克隆）：补占位值，测试得以运行。

占位值刻意写成一眼可辨的字符串，且指向 `.invalid` 域名——万一某条测试
真的发起了网络请求，它会立刻失败而不是悄悄打到真实服务。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# 先把开发机的 .env 灌进环境（若存在）。src/config.py 里也会调用一次，
# 重复调用无副作用。
load_dotenv()

#: 一眼可辨的占位值；不是真实凭据。
_PLACEHOLDER_KEY = "test-placeholder-key-not-a-real-credential"
#: RFC 2606 保留域名，保证任何真实请求都会立即失败。
_PLACEHOLDER_URL = "https://placeholder.invalid/v1"

# 缺失时才补，不覆盖 .env 里的真实值。
for _var in (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_BASE_URL",
    "LLM_JUDGE_API_KEY",
    "LLM_JUDGE_BASE_URL",
    "EVAL_LLM_API_KEY",
    "EVAL_LLM_BASE_URL",
    # openai SDK 自身的兜底变量：即使代码里没显式传 api_key，
    # 只要它存在于环境里，客户端就能构造成功。
    "OPENAI_API_KEY",
):
    os.environ.setdefault(
        _var, _PLACEHOLDER_URL if _var.endswith("BASE_URL") else _PLACEHOLDER_KEY
    )

del _var
