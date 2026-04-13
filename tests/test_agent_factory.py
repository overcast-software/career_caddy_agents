"""Tests for agents.agent_factory — model selection, registry, and agent creation."""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

# Mock the optional pydanticai_ollama dependency before importing agent_factory.
# We need real classes (not MagicMock) because agent_factory subclasses them with @dataclass.
import types

_ollama_pkg = types.ModuleType("pydanticai_ollama")
_ollama_models = types.ModuleType("pydanticai_ollama.models")
_ollama_models_ollama = types.ModuleType("pydanticai_ollama.models.ollama")
_ollama_providers = types.ModuleType("pydanticai_ollama.providers")
_ollama_providers_ollama = types.ModuleType("pydanticai_ollama.providers.ollama")
_ollama_settings = types.ModuleType("pydanticai_ollama.settings")
_ollama_settings_ollama = types.ModuleType("pydanticai_ollama.settings.ollama")


class _StubOllamaModel:
    def __init__(self, *args, **kwargs):
        pass


class _StubOllamaStreamedResponse:
    def __init__(self, *args, **kwargs):
        pass


class _StubOllamaProvider:
    def __init__(self, *args, **kwargs):
        self.base_url = kwargs.get("base_url", "")


class _StubOllamaModelSettings:
    def __init__(self, *args, **kwargs):
        pass


_ollama_models_ollama.OllamaModel = _StubOllamaModel
_ollama_models_ollama.OllamaStreamedResponse = _StubOllamaStreamedResponse
_ollama_providers_ollama.OllamaProvider = _StubOllamaProvider
_ollama_settings_ollama.OllamaModelSettings = _StubOllamaModelSettings

for name, mod in [
    ("pydanticai_ollama", _ollama_pkg),
    ("pydanticai_ollama.models", _ollama_models),
    ("pydanticai_ollama.models.ollama", _ollama_models_ollama),
    ("pydanticai_ollama.providers", _ollama_providers),
    ("pydanticai_ollama.providers.ollama", _ollama_providers_ollama),
    ("pydanticai_ollama.settings", _ollama_settings),
    ("pydanticai_ollama.settings.ollama", _ollama_settings_ollama),
]:
    sys.modules.setdefault(name, mod)

from agents.agent_factory import (
    get_model,
    get_model_name,
    AgentConfig,
    get_agent,
    register_agent,
    get_agent_config,
    register_defaults,
    _AGENT_REGISTRY,
)


# ---------------------------------------------------------------------------
# get_model — env-var-driven model selection
# ---------------------------------------------------------------------------


class TestGetModel:
    def test_default_model(self):
        with patch.dict(os.environ, {}, clear=True):
            result = get_model()
            assert result == "openai:gpt-4o-mini"

    def test_caddy_default_model_override(self):
        with patch.dict(os.environ, {"CADDY_DEFAULT_MODEL": "openai:gpt-4o"}, clear=True):
            result = get_model()
            assert result == "openai:gpt-4o"

    def test_role_specific_override(self):
        with patch.dict(os.environ, {"CADDY_MODEL": "anthropic:claude-3-5-sonnet"}, clear=True):
            result = get_model("caddy")
            assert result == "anthropic:claude-3-5-sonnet"

    def test_role_falls_back_to_caddy_default(self):
        with patch.dict(os.environ, {"CADDY_DEFAULT_MODEL": "openai:gpt-4o"}, clear=True):
            result = get_model("caddy")
            assert result == "openai:gpt-4o"

    def test_all_roles_have_env_mapping(self):
        roles = ["caddy", "chat", "email_classifier", "job_extractor", "pipeline", "browser_scraper"]
        for role in roles:
            result = get_model(role)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_unknown_role_returns_default(self):
        result = get_model("nonexistent_role")
        assert result == "openai:gpt-4o-mini"

    def test_pipeline_and_browser_share_env_var(self):
        with patch.dict(os.environ, {"PIPELINE_MODEL": "openai:gpt-4o"}, clear=True):
            assert get_model("pipeline") == "openai:gpt-4o"
            assert get_model("browser_scraper") == "openai:gpt-4o"


# ---------------------------------------------------------------------------
# get_model_name — extract string names from model objects
# ---------------------------------------------------------------------------


class TestGetModelName:
    def test_string_passthrough(self):
        assert get_model_name("openai:gpt-4o-mini") == "openai:gpt-4o-mini"

    def test_object_with_model_name_attr(self):
        model = MagicMock()
        model._model_name = "phi3:14b"
        model.provider_name = None
        # Has _model_name → uses ollama provider prefix
        assert get_model_name(model) == "ollama:phi3:14b"

    def test_object_with_provider_name(self):
        model = MagicMock()
        model._model_name = "gpt-4o"
        model.provider_name = "openai"
        assert get_model_name(model) == "openai:gpt-4o"

    def test_object_with_callable_model_name(self):
        model = MagicMock(spec=[])  # no _model_name
        model.model_name = MagicMock(return_value="test-model")
        assert get_model_name(model) == "test-model"

    def test_fallback_to_str(self):
        class PlainModel:
            def __str__(self):
                return "some-model"
        assert get_model_name(PlainModel()) == "some-model"


# ---------------------------------------------------------------------------
# AgentConfig + Registry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def setup_method(self):
        # Save and clear the registry for each test
        self._saved = dict(_AGENT_REGISTRY)
        _AGENT_REGISTRY.clear()

    def teardown_method(self):
        _AGENT_REGISTRY.clear()
        _AGENT_REGISTRY.update(self._saved)

    def test_register_and_retrieve(self):
        config = AgentConfig(role="test_role", system_prompt="Hello")
        register_agent("test_role", config)
        assert get_agent_config("test_role") is config

    def test_unregistered_returns_none(self):
        assert get_agent_config("nonexistent") is None

    def test_register_overwrites(self):
        config1 = AgentConfig(role="test", system_prompt="v1")
        config2 = AgentConfig(role="test", system_prompt="v2")
        register_agent("test", config1)
        register_agent("test", config2)
        assert get_agent_config("test").system_prompt == "v2"


# ---------------------------------------------------------------------------
# get_agent — creates Agent instances from registry
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_openai_key(monkeypatch):
    """Agent() validates the OpenAI key at construction; provide a dummy."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")


class TestGetAgent:
    def setup_method(self):
        self._saved = dict(_AGENT_REGISTRY)
        _AGENT_REGISTRY.clear()

    def teardown_method(self):
        _AGENT_REGISTRY.clear()
        _AGENT_REGISTRY.update(self._saved)

    def test_unregistered_role_returns_minimal_agent(self):
        agent = get_agent("unknown_role")
        assert agent.name == "unknown_role"

    def test_registered_role_uses_config(self):
        config = AgentConfig(
            role="test_agent",
            system_prompt="You are a test agent.",
            name="my_test_agent",
        )
        register_agent("test_agent", config)
        agent = get_agent("test_agent")
        assert agent.name == "my_test_agent"

    def test_override_system_prompt(self):
        config = AgentConfig(
            role="test_agent",
            system_prompt="Default prompt",
        )
        register_agent("test_agent", config)
        agent = get_agent("test_agent", system_prompt="Custom prompt")
        # The agent should use the override
        assert agent.name == "test_agent"

    def test_override_name(self):
        config = AgentConfig(role="test_agent", system_prompt="test")
        register_agent("test_agent", config)
        agent = get_agent("test_agent", name="custom_name")
        assert agent.name == "custom_name"

    def test_toolset_factories_called(self):
        mock_toolset = MagicMock()
        factory = MagicMock(return_value=mock_toolset)

        config = AgentConfig(
            role="tool_agent",
            system_prompt="test",
            toolset_factories=[factory],
        )
        register_agent("tool_agent", config)
        get_agent("tool_agent")
        factory.assert_called_once()

    def test_override_model(self):
        config = AgentConfig(role="test_agent", system_prompt="test")
        register_agent("test_agent", config)
        # Should not raise — model override is accepted
        agent = get_agent("test_agent", model="openai:gpt-4o")
        assert agent is not None


# ---------------------------------------------------------------------------
# register_defaults — all built-in roles
# ---------------------------------------------------------------------------


class TestRegisterDefaults:
    def test_registers_all_roles(self):
        # register_defaults is idempotent, call it
        register_defaults()
        expected_roles = {"caddy", "chat", "email_classifier", "job_extractor", "pipeline", "browser_scraper"}
        for role in expected_roles:
            assert get_agent_config(role) is not None, f"Missing registration for role: {role}"

    def test_caddy_has_deps_type(self):
        register_defaults()
        config = get_agent_config("caddy")
        assert config.deps_type is not None

    def test_email_classifier_has_toolset_factory(self):
        register_defaults()
        config = get_agent_config("email_classifier")
        assert len(config.toolset_factories) > 0

    def test_job_extractor_has_no_toolsets(self):
        register_defaults()
        config = get_agent_config("job_extractor")
        assert len(config.toolset_factories) == 0

    def test_browser_scraper_has_toolset_factory(self):
        register_defaults()
        config = get_agent_config("browser_scraper")
        assert len(config.toolset_factories) > 0
