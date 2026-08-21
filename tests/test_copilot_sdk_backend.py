"""Mocked tests for the optional GitHub Copilot SDK backend."""
from __future__ import annotations

import asyncio
import base64
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from graphify import copilot_sdk_backend as backend
from graphify import llm

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


_GRAPH_JSON = '{"nodes":[{"id":"alpha","label":"Alpha"}],"edges":[],"hyperedges":[]}'


def test_copilot_extra_requires_session_lockdown_capabilities():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    expected = "github-copilot-sdk>=1.0.11,<2; python_version >= '3.11'"
    assert extras["copilot"] == [expected]
    assert expected in extras["all"]


def test_resolve_settings_precedence_and_sentinel(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_COPILOT_MODEL", "env-model")
    monkeypatch.setenv("GRAPHIFY_COPILOT_REASONING_EFFORT", "high")
    monkeypatch.setenv("GRAPHIFY_COPILOT_CONTEXT_TIER", "long_context")
    assert backend.resolve_settings() == ("env-model", "high", "long_context")
    assert backend.resolve_settings(
        model="cli-model", reasoning_effort="low", context_tier="default"
    ) == ("cli-model", "low", "default")
    monkeypatch.setenv("GRAPHIFY_COPILOT_MODEL", backend.COPILOT_DEFAULT_MODEL)
    assert backend.resolve_settings()[0] is None


@pytest.mark.parametrize(
    ("variable", "value", "needle"),
    [
        ("GRAPHIFY_COPILOT_REASONING_EFFORT", "invalid", "reasoning effort"),
        ("GRAPHIFY_COPILOT_CONTEXT_TIER", "huge", "context tier"),
    ],
)
def test_invalid_settings_fail_before_sdk(monkeypatch, variable, value, needle):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=needle):
        backend.call_copilot_sdk(
            "source", system_prompt="system", model=None, reasoning_effort=None,
            context_tier=None, timeout_seconds=2,
        )


def test_missing_optional_dependency_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "copilot", None)
    with pytest.raises(ImportError, match=r'graphifyy\[copilot\]'):
        backend.call_copilot_sdk(
            "source", system_prompt="system", model=None, reasoning_effort=None,
            context_tier=None, timeout_seconds=2,
        )


def test_unsupported_python_is_checked_before_sdk(monkeypatch):
    monkeypatch.setattr(backend.sys, "version_info", (3, 10, 0))
    with pytest.raises(RuntimeError, match="Python 3.11 or later"):
        backend.call_copilot_sdk(
            "source", system_prompt="system", model=None, reasoning_effort=None,
            context_tier=None, timeout_seconds=2,
        )


def test_blob_attachments_are_inline_and_never_absolute_paths():
    attachments = backend.blob_attachments(
        [backend.CopilotImage(b"pixels", "image/png", "/private/source/diagram.png")]
    )
    assert attachments == [{
        "type": "blob",
        "data": base64.b64encode(b"pixels").decode("ascii"),
        "mimeType": "image/png",
        "displayName": "diagram.png",
    }]


def test_helper_event_and_attachment_error_paths():
    with pytest.raises(TypeError, match="must be bytes"):
        backend.blob_attachments(
            [backend.CopilotImage("not bytes", "image/png", "x")]  # pyright: ignore[reportArgumentType]
        )
    assert backend._system_message("") is None
    assert backend._event_type(SimpleNamespace(type=SimpleNamespace(value="custom"))) == "custom"
    assert backend._event_type({"raw_type": "fallback"}) == "fallback"
    assert backend._event_type({}) == ""
    assert backend._value({"answer": 42}, "answer") == 42
    assert backend._value(SimpleNamespace(answer=42), "answer") == 42
    assert backend._number(True) == 0
    assert backend._content_from_event({"data": {"content": 123}}) is None
    assert backend._content_from_event({"data": None}) is None
    assert backend._content_from_event(SimpleNamespace(content=123)) is None


def test_usage_collector_ignores_child_and_covers_message_metadata():
    collector = backend._UsageCollector()
    collector(SimpleNamespace(agent_id="child", type="assistant.usage", data=SimpleNamespace(input_tokens=99)))
    assert collector.values["input_tokens"] == 0
    collector(SimpleNamespace(
        type="assistant.usage",
        data=SimpleNamespace(
            input_tokens=0, output_tokens=0, cache_read_tokens=0,
            cache_write_tokens=0, reasoning_tokens=0, cost=None,
            copilot_usage=SimpleNamespace(total_nano_aiu=3),
        ),
    ))
    collector(SimpleNamespace(
        type="assistant.message",
        data=SimpleNamespace(model="message-model", output_tokens=8),
    ))
    assert collector.values["copilot_usage_cost"] == 3
    assert collector.values["output_tokens"] == 8
    assert collector.values["model"] == "message-model"


def test_friendly_error_categories():
    timeout = backend.CopilotSdkTimeoutError("timeout")
    assert backend._friendly_error(timeout, model=None) is timeout
    assert "download-runtime" in str(backend._friendly_error(FileNotFoundError("x"), model=None))
    assert "authentication" in str(backend._friendly_error(RuntimeError("forbidden"), model=None))
    assert "requested" not in str(backend._friendly_error(RuntimeError("model bad"), model=None))
    assert "m-1" in str(backend._friendly_error(RuntimeError("model bad"), model="m-1"))


def _install_fake_copilot(monkeypatch, captured, *, response=None, fail=None):
    class FakeSession:
        async def send(self, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["send"] = kwargs
            if fail is not None:
                raise fail
            handler = captured["session"]["on_event"]
            if response is None:
                handler(SimpleNamespace(type="session.idle", data=None))
            else:
                handler(response)
            return "message-1"

        async def get_events(self):
            return []

        async def disconnect(self):
            captured["disconnected"] = True

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        async def start(self):
            captured["started"] = True

        async def create_session(self, **kwargs):
            captured["session"] = kwargs
            kwargs["on_event"](SimpleNamespace(type="session.tools_updated", data=None))
            return FakeSession()

        async def stop(self):
            captured["stopped"] = True

    module = types.ModuleType("copilot")
    setattr(module, "CopilotClient", FakeClient)
    monkeypatch.setitem(sys.modules, "copilot", module)


def _usage_event():
    return SimpleNamespace(
        type="assistant.usage",
        data=SimpleNamespace(
            model="gpt-test",
            input_tokens=12,
            output_tokens=7,
            cache_read_tokens=3,
            cache_write_tokens=2,
            reasoning_tokens=4,
            cost=0.25,
            finish_reason="stop",
        ),
    )


def test_call_uses_locked_down_session_and_cleans_temp_workspace(monkeypatch):
    captured = {}
    response = SimpleNamespace(
        type="assistant.message",
        data=SimpleNamespace(content=_GRAPH_JSON, model="gpt-test", output_tokens=7),
    )
    _install_fake_copilot(monkeypatch, captured, response=response)

    result = backend.call_copilot_sdk(
        "UNTRUSTED_SOURCE", system_prompt="GRAPHIFY_SYSTEM", model=None,
        reasoning_effort="low", context_tier="default", timeout_seconds=3,
        images=[backend.CopilotImage(b"x", "image/png", "diagram.png")],
    )
    assert result["content"] == _GRAPH_JSON
    assert captured["client"]["use_logged_in_user"] is True
    assert captured["client"]["mode"] == "copilot-cli"
    workdir = Path(captured["client"]["working_directory"])
    assert not workdir.exists()
    session = captured["session"]
    assert session["tools"] == []
    assert session["available_tools"] == []
    assert session["mcp_servers"] == {}
    assert session["config_directory"] == session["working_directory"]
    for key in (
        "enable_session_telemetry", "enable_file_change_tracking", "enable_session_store",
        "enable_skills", "enable_config_discovery", "enable_on_demand_instruction_discovery",
        "enable_file_hooks", "enable_host_git_operations",
    ):
        assert session[key] is False
    assert session["skip_custom_instructions"] is True
    assert session["skip_embedding_retrieval"] is True
    assert session["memory"] == {"enabled": False}
    assert session["system_message"]["mode"] == "customize"
    assert captured["prompt"].startswith("Extract the knowledge graph")
    assert captured["send"]["attachments"][0]["type"] == "blob"
    assert captured["disconnected"] and captured["stopped"]


def test_usage_events_aggregate(monkeypatch):
    captured = {}

    class FakeSession:
        async def send(self, prompt, **kwargs):
            handler = captured["session"]["on_event"]
            handler(_usage_event())
            handler(SimpleNamespace(
                type="assistant.usage",
                data=SimpleNamespace(
                    model="gpt-test", input_tokens=5, output_tokens=2,
                    cache_read_tokens=1, cache_write_tokens=0,
                    reasoning_tokens=0, cost=0.5, finish_reason="stop",
                ),
            ))
            handler(SimpleNamespace(
                type="session.usage_info",
                data=SimpleNamespace(current_tokens=55, token_limit=100),
            ))
            handler(
                SimpleNamespace(
                    type="assistant.message",
                    data=SimpleNamespace(content=_GRAPH_JSON),
                )
            )
            return "message-1"

        async def get_events(self):
            return []

        async def disconnect(self):
            pass

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
        async def start(self):
            pass
        async def create_session(self, **kwargs):
            captured["session"] = kwargs
            kwargs["on_event"](SimpleNamespace(type="session.tools_updated", data=None))
            return FakeSession()
        async def stop(self):
            pass

    module = types.ModuleType("copilot")
    setattr(module, "CopilotClient", FakeClient)
    monkeypatch.setitem(sys.modules, "copilot", module)
    result = backend.call_copilot_sdk(
        "source", system_prompt="system", model="m", reasoning_effort=None,
        context_tier=None, timeout_seconds=2,
    )
    assert result["input_tokens"] == 17
    assert result["output_tokens"] == 9
    assert result["cache_read_tokens"] == 4
    assert result["cache_write_tokens"] == 2
    assert result["reasoning_tokens"] == 4
    assert result["copilot_usage_cost"] == pytest.approx(0.75)
    assert result["context_current_tokens"] == 55
    assert result["context_limit"] == 100
    assert result["model"] == "gpt-test"


def test_permission_handler_returns_reject_decision(monkeypatch):
    captured = {}

    class Reject:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    generated = types.ModuleType("copilot.generated.rpc")
    setattr(generated, "PermissionDecisionReject", Reject)
    package = types.ModuleType("copilot.generated")
    setattr(package, "rpc", generated)
    monkeypatch.setitem(sys.modules, "copilot.generated", package)
    monkeypatch.setitem(sys.modules, "copilot.generated.rpc", generated)
    decision = backend._deny_permission(object(), object())
    assert isinstance(decision, Reject)
    assert "does not permit tools" in captured["feedback"]


def test_extract_dispatches_copilot_without_api_key(tmp_path, monkeypatch):
    source = tmp_path / "note.md"
    source.write_text("# Alpha\n")
    calls = {}

    def fake_call(prompt, **kwargs):
        calls.update(prompt=prompt, kwargs=kwargs)
        return {"content": _GRAPH_JSON, "input_tokens": 2, "output_tokens": 3, "model": "m"}

    monkeypatch.setattr("graphify.copilot_sdk_backend.call_copilot_sdk", fake_call)
    result = llm.extract_files_direct([source], backend="copilot-sdk", root=tmp_path)
    assert result["nodes"][0]["id"] == "alpha"
    assert calls["kwargs"]["model"] is None
    assert "untrusted_source" in calls["prompt"]


def test_copilot_is_not_auto_detected(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "MOONSHOT_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert llm.detect_backend() != "copilot-sdk"


def test_adapter_works_inside_running_event_loop(monkeypatch):
    captured = {}
    response = SimpleNamespace(type="assistant.message", data=SimpleNamespace(content=_GRAPH_JSON))
    _install_fake_copilot(monkeypatch, captured, response=response)

    async def run():
        return backend.call_copilot_sdk(
            "source", system_prompt="system", model=None, reasoning_effort=None,
            context_tier=None, timeout_seconds=2,
        )

    result = asyncio.run(run())
    assert result["content"] == _GRAPH_JSON


def test_unstarted_session_fails_without_resending(monkeypatch):
    monkeypatch.setattr(backend, "_SESSION_SETTLE_SECONDS", 0)
    monkeypatch.setattr(backend, "_SESSION_START_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(backend, "_SESSION_POLL_SECONDS", 0.001)
    captured = {"clients": 0, "sends": 0, "disconnects": 0, "stops": 0}

    class FakeSession:
        async def send(self, _prompt, **_kwargs):
            captured["sends"] += 1
            return "message-1"

        async def get_events(self):
            return []

        async def disconnect(self):
            captured["disconnects"] += 1

    class FakeClient:
        def __init__(self, **_kwargs):
            captured["clients"] += 1

        async def start(self):
            pass

        async def create_session(self, **_kwargs):
            return FakeSession()

        async def stop(self):
            captured["stops"] += 1

    module = types.ModuleType("copilot")
    setattr(module, "CopilotClient", FakeClient)
    monkeypatch.setitem(sys.modules, "copilot", module)

    with pytest.raises(RuntimeError, match="did not start processing it"):
        backend.call_copilot_sdk(
            "source",
            system_prompt="system",
            model=None,
            reasoning_effort=None,
            context_tier=None,
            timeout_seconds=2,
        )

    assert captured == {
        "clients": 1,
        "sends": 1,
        "disconnects": 1,
        "stops": 1,
    }


def test_history_fallback_ignores_child_agent_messages():
    child = SimpleNamespace(
        agent_id="child",
        type="assistant.message",
        data=SimpleNamespace(content="child output"),
    )
    assert backend._history_state([child]) == (False, False, False, None)

    root = SimpleNamespace(
        type="assistant.message",
        data=SimpleNamespace(content=_GRAPH_JSON),
    )
    assert backend._history_state([child, root]) == (True, False, True, root)


def test_history_polling_recovers_final_message_without_callback(monkeypatch):
    monkeypatch.setattr(backend, "_SESSION_SETTLE_SECONDS", 0)
    response = SimpleNamespace(
        type="assistant.message",
        data=SimpleNamespace(content=_GRAPH_JSON),
    )

    class FakeSession:
        async def send(self, _prompt, **_kwargs):
            return "message-1"

        async def get_events(self):
            return [_usage_event(), response]

        async def disconnect(self):
            pass

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            pass

        async def create_session(self, **_kwargs):
            return FakeSession()

        async def stop(self):
            pass

    module = types.ModuleType("copilot")
    setattr(module, "CopilotClient", FakeClient)
    monkeypatch.setitem(sys.modules, "copilot", module)

    result = backend.call_copilot_sdk(
        "source",
        system_prompt="system",
        model=None,
        reasoning_effort=None,
        context_tier=None,
        timeout_seconds=2,
    )
    assert result["content"] == _GRAPH_JSON
    assert result["input_tokens"] == 12
    assert result["output_tokens"] == 7


def test_sdk_error_does_not_echo_corpus_and_still_cleans_up(monkeypatch):
    captured = {}
    _install_fake_copilot(
        monkeypatch,
        captured,
        fail=RuntimeError("source secret and authorization header must stay private"),
    )
    with pytest.raises(RuntimeError) as exc_info:
        backend.call_copilot_sdk(
            "source secret", system_prompt="system", model="m", reasoning_effort=None,
            context_tier=None, timeout_seconds=2,
        )
    assert "source secret" not in str(exc_info.value)
    assert "authorization header" not in str(exc_info.value)
    assert captured["disconnected"] and captured["stopped"]


def test_sdk_timeout_preserves_timeout_type_and_cleanup(monkeypatch):
    captured = {}
    _install_fake_copilot(monkeypatch, captured, fail=asyncio.TimeoutError())
    with pytest.raises(backend.CopilotSdkTimeoutError, match="timed out after 2 seconds"):
        backend.call_copilot_sdk(
            "source", system_prompt="system", model=None, reasoning_effort=None,
            context_tier=None, timeout_seconds=2,
        )
    assert captured["disconnected"] and captured["stopped"]


def test_missing_final_assistant_message_is_actionable(monkeypatch):
    captured = {}
    _install_fake_copilot(monkeypatch, captured, response=None)
    with pytest.raises(RuntimeError, match="request failed"):
        backend.call_copilot_sdk(
            "source", system_prompt="system", model=None, reasoning_effort=None,
            context_tier=None, timeout_seconds=2,
        )
    assert captured["disconnected"] and captured["stopped"]


@pytest.mark.parametrize("primary_failure", [False, True])
def test_cleanup_failure_does_not_mask_primary_error(monkeypatch, primary_failure):
    captured = {}

    class FakeSession:
        def __init__(self, handler):
            self.handler = handler

        async def send(self, _prompt, **_kwargs):
            if primary_failure:
                raise RuntimeError("primary extraction failure")
            self.handler(
                SimpleNamespace(
                    type="assistant.message",
                    data=SimpleNamespace(content=_GRAPH_JSON),
                )
            )
            return "message-1"

        async def get_events(self):
            return []

        async def disconnect(self):
            raise RuntimeError("cleanup failure")

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            pass

        async def create_session(self, **kwargs):
            kwargs["on_event"](SimpleNamespace(type="session.tools_updated", data=None))
            return FakeSession(kwargs["on_event"])

        async def stop(self):
            captured["stopped"] = True

    module = types.ModuleType("copilot")
    setattr(module, "CopilotClient", FakeClient)
    monkeypatch.setitem(sys.modules, "copilot", module)
    if primary_failure:
        with pytest.raises(RuntimeError, match="request failed") as exc_info:
            backend.call_copilot_sdk(
                "source", system_prompt="system", model=None, reasoning_effort=None,
                context_tier=None, timeout_seconds=2,
            )
        assert "cleanup failure" not in str(exc_info.value)
    else:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            backend.call_copilot_sdk(
                "source", system_prompt="system", model=None, reasoning_effort=None,
                context_tier=None, timeout_seconds=2,
            )
    assert captured["stopped"] is True
