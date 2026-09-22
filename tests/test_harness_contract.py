"""harness 层契约测试。

这些断言存在的意义是**防止漂移**：``src/harness/config.py` 声称自己是执行
预算的唯一事实源，但如果消费方（provider / graph）偷偷改了各自的值，
这份声明就成了谎话。测试把两者钉在一起，任何一处改动都会立刻失败。

同样地，``CAPABILITIES`` 是一张"脚手架由什么构成"的登记表；一旦有人
重构掉某个模块而没更新登记表，这里会报警，而不是等到线上发现少了兜底。
"""

from __future__ import annotations

import importlib.util

import pytest

from src.harness import CAPABILITIES
from src.harness import config as harness_config


def _find_spec(dotted: str):
    return importlib.util.find_spec(dotted)


class TestConfigMatchesConsumers:
    """harness.config 的取值必须与各消费方实际使用的常量一致。"""

    def test_retry_policy_matches_llm_provider(self):
        from src.llm import provider

        assert harness_config.RETRY_MAX_ATTEMPTS == provider._MAX_RETRIES
        assert harness_config.RETRY_BACKOFF_BASE == provider._BACKOFF_BASE

    def test_max_rewrites_matches_graph_state(self):
        from src.graph import state

        assert harness_config.MAX_REWRITES == state.MAX_REWRITES

    def test_recursion_limit_matches_graph_runtime(self):
        from src.graph import runtime

        assert harness_config.RECURSION_LIMIT == runtime.RECURSION_LIMIT


class TestCapabilityRegistry:
    """登记表里的每个能力都必须真实存在。"""

    @pytest.mark.parametrize("capability,module_path", sorted(CAPABILITIES.items()))
    def test_capability_module_exists(self, capability, module_path):
        assert _find_spec(module_path) is not None, (
            f"CAPABILITIES['{capability}'] 指向 {module_path}，但该模块不存在。"
            "要么模块被重构了，要么登记表过期了。"
        )

    def test_registry_covers_every_capability_with_a_nonempty_path(self):
        assert CAPABILITIES, "能力登记表不应为空"
        for capability, module_path in CAPABILITIES.items():
            assert capability.strip(), "能力名不能为空"
            assert module_path.strip(), f"{capability} 的模块路径不能为空"


class TestHarnessConfigDataclass:
    def test_defaults_track_module_constants(self):
        cfg = harness_config.HarnessConfig()
        assert cfg.retry_max_attempts == harness_config.RETRY_MAX_ATTEMPTS
        assert cfg.max_rewrites == harness_config.MAX_REWRITES
        assert cfg.agent_max_steps == harness_config.AGENT_MAX_STEPS
        assert cfg.agent_token_budget == harness_config.AGENT_TOKEN_BUDGET

    def test_config_is_frozen(self):
        """配置在一次运行内不应被改写，否则续跑行为会与首次运行不一致。"""
        cfg = harness_config.HarnessConfig()
        with pytest.raises(Exception):
            cfg.max_rewrites = 99  # type: ignore[misc]

    def test_budgets_are_positive(self):
        """预算必须是正数——0 或负数会让循环立刻放弃，等同于功能失效。"""
        cfg = harness_config.HarnessConfig()
        assert cfg.agent_max_steps > 0
        assert cfg.agent_token_budget > 0
        assert cfg.retry_max_attempts > 0
        assert cfg.default_timeout_seconds > 0
