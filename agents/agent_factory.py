import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Any, Callable
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage
from pydantic_ai._utils import PeekableAsyncStream

# Optional: Ollama support — only available when pydanticai_ollama is installed
try:
    from pydanticai_ollama.models.ollama import OllamaModel, OllamaStreamedResponse
    from pydanticai_ollama.providers.ollama import OllamaProvider
    from pydanticai_ollama.settings.ollama import OllamaModelSettings
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False
    OllamaModel = None
    OllamaStreamedResponse = None
    OllamaProvider = None
    OllamaModelSettings = None


# Define a Pydantic model for your tool's arguments
class GetCurrentWeatherArgs(BaseModel):
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(
        default="celsius",
        description="The unit of temperature (celsius or fahrenheit) to return",
    )


# ---------------------------------------------------------------------------
# Ollama model infrastructure — only available when pydanticai_ollama is installed.
# Skipped entirely in Docker / environments without local Ollama.
# ---------------------------------------------------------------------------
if _HAS_OLLAMA:
    class ConcreteOllamaProvider(OllamaProvider):
        """Concrete implementation of OllamaProvider with provider_url method."""

        def provider_url(self) -> str:
            return self.base_url

    @dataclass
    class ConcreteOllamaStreamedResponse(OllamaStreamedResponse):
        """Concrete implementation of OllamaStreamedResponse with provider_url method."""

        _model_name: str
        _model_profile: ModelProfile
        _response: PeekableAsyncStream[Any]
        _timestamp: datetime

        @property
        def model_name(self) -> str:
            return self._model_name

        @property
        def provider_name(self) -> str | None:
            return "ollama"

        @property
        def timestamp(self) -> datetime:
            return self._timestamp

        def provider_url(self) -> str:
            return "http://localhost:11434"

        async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
            """Asynchronously iterates over Ollama stream chunks and yields ModelResponseStreamEvent."""
            async for chunk in self._response:
                self._usage += RequestUsage(input_tokens=0, output_tokens=1)
                if hasattr(chunk, "message") and chunk.message.content:
                    text_event = self._parts_manager.handle_text_delta(
                        vendor_part_id="content",
                        content=chunk.message.content,
                        thinking_tags=self._model_profile.thinking_tags,
                        ignore_leading_whitespace=self._model_profile.ignore_streamed_leading_whitespace,
                    )
                    if text_event:
                        yield text_event

    class ConcreteOllamaModel(OllamaModel):
        """Concrete implementation of OllamaModel that returns ConcreteOllamaStreamedResponse."""

        async def _process_streamed_response(
            self,
            response: AsyncIterator[Any],
            model_request_parameters: ModelRequestParameters,
        ) -> ConcreteOllamaStreamedResponse:
            """Process a streamed response, and prepare a streaming response to return."""
            peekable_response = PeekableAsyncStream(response)
            await peekable_response.peek()

            return ConcreteOllamaStreamedResponse(
                model_request_parameters=model_request_parameters,
                _response=peekable_response,
                _model_name=self._model_name,
                _model_profile=self.profile,
                _timestamp=datetime.now(timezone.utc),
            )

    # Configure OllamaModelSettings with desired parameters
    ollama_settings = OllamaModelSettings(
        temperature=0.1,
        num_predict=1024,
        num_ctx=16384,
        repeat_penalty=1.1,
        num_gpu=1,
        top_k=40,
        top_p=0.9,
    )

    # Initialize OllamaProvider with your Ollama server's base URL
    ollama_provider = ConcreteOllamaProvider(base_url="http://localhost:11434")

    voytas26_openclaw_oss = "voytas26/openclaw-oss-20b-deterministic"
    voytas26_openclaw_oss_model = ConcreteOllamaModel(
        model_name=voytas26_openclaw_oss,
        provider=ollama_provider,
    )
    phi3 = "phi3:14b"
    phi3_model = ConcreteOllamaModel(
        model_name=phi3,
        provider=ollama_provider,
    )
    voytas26_openclaw_qwen3vl = "voytas26/openclaw-qwen3vl-8b-opt"
    voytas26_openclaw_qwen3vl_model = ConcreteOllamaModel(
        model_name=voytas26_openclaw_qwen3vl,
        provider=ollama_provider,
    )

    astrail3_tools = "60MPH/astral3-tools:12b"
    astrail3_model = ConcreteOllamaModel(
        model_name=astrail3_tools,
        provider=ollama_provider,
    )

    gpt_4o_mini_tools = "chevalblanc/gpt-4o-mini:latest"
    gpt_40_mini_model = ConcreteOllamaModel(
        model_name=gpt_4o_mini_tools,
        provider=ollama_provider,
    )

    # Tool-capable models via OpenAI-compatible endpoint (/v1)
    _ollama_openai_provider = OpenAIProvider(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    qwen3_4b_model = OpenAIChatModel(
        "qwen3:4b-instruct",
        provider=_ollama_openai_provider,
    )

    qwen25_coder_7b_model = OpenAIChatModel(
        "qwen2.5-coder:7b",
        provider=_ollama_openai_provider,
    )

    llama3 = ConcreteOllamaModel(
        model_name="llama3.3",
        provider=ollama_provider,
    )

    # Browser-optimised model: larger context window + low temperature for
    # reliable multi-step tool sequencing (create_tab → navigate → snapshot).
    browser_ollama_model = OpenAIChatModel(
        "qwen3:4b-instruct",
        provider=_ollama_openai_provider,
        settings={
            "temperature": 0.1,
            "extra_body": {
                "options": {
                    "num_ctx": 16384,
                    "num_predict": 1024,
                    "repeat_penalty": 1.1,
                }
            },
        },
    )

_DEFAULT_MODEL = "openai:gpt-4o-mini"

# ---------------------------------------------------------------------------
# Per-agent model overrides via environment variables.
# Set any of these to switch an individual agent's model:
#   CADDY_MODEL            — career_caddy_agent (main agent + add_job_post)
#   CHAT_MODEL             — chat_server (already read there, re-exported here)
#   EMAIL_CLASSIFIER_MODEL — email_classifier_agent
#   JOB_EXTRACTOR_MODEL    — job_extractor_agent
#   PIPELINE_MODEL         — pipeline agents (email search)
#   BROWSER_SCRAPER_MODEL  — browser scraper agent
#   CADDY_DEFAULT_MODEL    — fallback for all agents not individually overridden
# ---------------------------------------------------------------------------

def get_model(role: str | None = None) -> str:
    """Return the model string for a given agent role.

    Checks role-specific env var first, then CADDY_DEFAULT_MODEL, then _DEFAULT_MODEL.
    """
    role_env_map = {
        "caddy": "CADDY_MODEL",
        "chat": "CHAT_MODEL",
        "email_classifier": "EMAIL_CLASSIFIER_MODEL",
        "job_extractor": "JOB_EXTRACTOR_MODEL",
        "pipeline": "PIPELINE_MODEL",
        "browser_scraper": "BROWSER_SCRAPER_MODEL",
    }
    if role and role in role_env_map:
        val = os.environ.get(role_env_map[role])
        if val:
            return val
    return os.environ.get("CADDY_DEFAULT_MODEL", _DEFAULT_MODEL)


def get_model_name(model) -> str:
    """Extract a string model name from a model object or string.

    Works with pydantic-ai model strings, OpenAIChatModel, ConcreteOllamaModel, etc.
    """
    if isinstance(model, str):
        return model
    if hasattr(model, "_model_name"):
        provider = getattr(model, "provider_name", None) or "ollama"
        return f"{provider}:{model._model_name}"
    if hasattr(model, "model_name"):
        name = model.model_name
        if callable(name):
            name = name()
        return str(name)
    return str(model)


# Backwards-compatible alias
global_model = get_model("caddy")


# ---------------------------------------------------------------------------
# Agent Factory — create configured Agent instances with sensible defaults
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration blueprint for creating a pydantic-ai Agent.

    Toolsets and deps that require fresh instances per agent (e.g. MCPServerStdio)
    should be passed as factory callables that return the object.
    """

    role: str
    system_prompt: str = ""
    output_type: type | None = None
    deps_type: type | None = None
    # Callables that return toolset instances — called at agent creation time.
    # Use callables for stateful resources (MCPServerStdio) that need fresh
    # instances. Plain objects (CareerCaddyToolset) can be wrapped in a lambda.
    toolset_factories: list[Callable] = field(default_factory=list)
    history_processors: list[Callable] | None = None
    name: str | None = None  # Agent name override; defaults to role


# Global registry: role → AgentConfig
_AGENT_REGISTRY: dict[str, AgentConfig] = {}


def register_agent(role: str, config: AgentConfig) -> None:
    """Register (or replace) an agent configuration for a given role."""
    _AGENT_REGISTRY[role] = config


def get_agent_config(role: str) -> AgentConfig | None:
    """Return the registered AgentConfig for a role, or None."""
    return _AGENT_REGISTRY.get(role)


def get_agent(role: str, **overrides) -> Agent:
    """Create a configured Agent instance for the given role.

    Uses the registered AgentConfig defaults, with any keyword overrides
    applied on top. Supports overriding: model (str), system_prompt,
    output_type, deps_type, toolset_factories, history_processors, name.

    Unregistered roles get a bare agent with just the model.

    Args:
        role: Agent role name (must match a key in the registry or get_model).
        **overrides: Any AgentConfig field or 'model' to override the default.

    Returns:
        A ready-to-use pydantic-ai Agent instance.
    """
    config = _AGENT_REGISTRY.get(role)
    model = overrides.pop("model", None) or get_model(role)

    if config is None:
        # Unregistered role — return a minimal agent
        return Agent(
            model,
            name=overrides.get("name", role),
            system_prompt=overrides.get("system_prompt", ""),
        )

    # Merge overrides into config values
    system_prompt = overrides.get("system_prompt", config.system_prompt)
    output_type = overrides.get("output_type", config.output_type)
    deps_type = overrides.get("deps_type", config.deps_type)
    name = overrides.get("name", config.name or config.role)
    history_processors = overrides.get("history_processors", config.history_processors)

    # Build toolsets — call factories for fresh instances
    toolset_factories = overrides.get("toolset_factories", config.toolset_factories)
    toolsets = [factory() for factory in toolset_factories]

    kwargs: dict[str, Any] = {
        "name": name,
        "system_prompt": system_prompt,
    }
    if toolsets:
        kwargs["toolsets"] = toolsets
    if output_type is not None:
        kwargs["output_type"] = output_type
    if deps_type is not None:
        kwargs["deps_type"] = deps_type
    if history_processors:
        kwargs["history_processors"] = history_processors

    return Agent(model, **kwargs)


# ---------------------------------------------------------------------------
# Default agent registrations
# ---------------------------------------------------------------------------
# These are lazy-imported to avoid circular imports — the register_defaults()
# function is called once, typically at application startup or on first use.

_defaults_registered = False


def register_defaults() -> None:
    """Register all built-in agent configs. Safe to call multiple times."""
    global _defaults_registered
    if _defaults_registered:
        return
    _defaults_registered = True

    from lib.history import sanitize_orphaned_tool_calls, truncate_message_history
    from lib.toolsets import CareerCaddyToolset, CareerCaddyDeps

    _common_history = [truncate_message_history, sanitize_orphaned_tool_calls]

    # -- career_caddy (add_job_post, general queries) --
    register_agent("caddy", AgentConfig(
        role="caddy",
        system_prompt=(
            "You are a helpful agent to facilitate adding job posts and job "
            "applications to the career caddy API. Follow the standard workflow: "
            "check duplicates → create with company check → report result."
        ),
        output_type=None,  # set by caller (CareerCaddyResponse)
        deps_type=CareerCaddyDeps,
        toolset_factories=[lambda: CareerCaddyToolset(scope="career_caddy")],
        history_processors=_common_history,
    ))

    # -- chat (streaming web UI) --
    register_agent("chat", AgentConfig(
        role="chat",
        system_prompt="",  # chat_server injects user-profile-aware prompt at runtime
        deps_type=CareerCaddyDeps,
        toolset_factories=[lambda: CareerCaddyToolset(scope="all")],
        history_processors=_common_history,
    ))

    # -- email_classifier --
    from pydantic_ai.mcp import MCPServerStdio

    register_agent("email_classifier", AgentConfig(
        role="email_classifier",
        system_prompt=(
            "You are an email classifier. Read the email, determine if it "
            "contains a job posting, and tag accordingly."
        ),
        toolset_factories=[
            lambda: MCPServerStdio("python", args=["mcp_servers/email_server.py"]),
        ],
        history_processors=_common_history,
    ))

    # -- job_extractor (no tools, structured output) --
    register_agent("job_extractor", AgentConfig(
        role="job_extractor",
        system_prompt=(
            "You are a precise job posting data extractor. Given raw job posting "
            "text or markdown, extract and return structured data."
        ),
        output_type=None,  # set by caller (JobPostData)
    ))

    # -- pipeline (email job search) --
    register_agent("pipeline", AgentConfig(
        role="pipeline",
        system_prompt=(
            "Search for emails tagged 'job_post'. For each email found, read it "
            "and extract the job title and one primary job posting URL."
        ),
        toolset_factories=[
            lambda: MCPServerStdio(
                "python", args=["mcp_servers/email_server.py"], env=os.environ.copy()
            ),
        ],
    ))

    # -- browser_scraper --
    register_agent("browser_scraper", AgentConfig(
        role="browser_scraper",
        system_prompt=(
            "Use the scrape_page tool to retrieve all visible text from the "
            "given URL. Return the raw text."
        ),
        toolset_factories=[
            lambda: MCPServerStdio(
                "python", args=["mcp_servers/browser_server.py"], env=os.environ.copy()
            ),
        ],
    ))
