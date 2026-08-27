"""Mocked tests for the optional GitHub Copilot SDK backend."""
from __future__ import annotations

import asyncio
import base64
import gc
import json
import math
import sys
import time
import traceback
import types
import warnings
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
    with pytest.raises(ImportError, match=r'graphify\[copilot\]'):
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
    assert backend._number(False) == 0
    assert backend._number(-1) == 0
    assert backend._number(float("nan")) == 0
    assert backend._number(float("inf")) == 0
    assert backend._number(float("-inf")) == 0
    assert backend._number(0.25) == pytest.approx(0.25)
    assert backend._number(10**1000) == 10**1000
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


def test_usage_collector_normalizes_invalid_numeric_metadata():
    collector = backend._UsageCollector()
    collector(SimpleNamespace(
        type="assistant.usage",
        data=SimpleNamespace(
            input_tokens=float("nan"),
            output_tokens=-2,
            cache_read_tokens=float("inf"),
            cache_write_tokens=float("-inf"),
            reasoning_tokens=True,
            cost=float("inf"),
            copilot_usage=SimpleNamespace(total_nano_aiu=float("nan")),
        ),
    ))
    collector(SimpleNamespace(
        type="session.usage_info",
        data=SimpleNamespace(current_tokens=-1, token_limit=float("inf")),
    ))

    assert collector.values["input_tokens"] == 0
    assert collector.values["output_tokens"] == 0
    assert collector.values["cache_read_tokens"] == 0
    assert collector.values["cache_write_tokens"] == 0
    assert collector.values["reasoning_tokens"] == 0
    assert collector.values["copilot_usage_cost"] == 0
    assert collector.values["context_current_tokens"] == 0
    assert collector.values["context_limit"] == 0
    json.dumps(collector.values, allow_nan=False)


def test_usage_aggregation_clamps_float_overflow_and_keeps_large_integers():
    collector = backend._UsageCollector()
    for _ in range(2):
        collector(SimpleNamespace(
            type="assistant.usage",
            data=SimpleNamespace(
                input_tokens=1e308,
                output_tokens=1e308,
                cache_read_tokens=1e308,
                cache_write_tokens=1e308,
                reasoning_tokens=1e308,
                cost=1e308,
            ),
        ))
    assert all(
        math.isfinite(collector.values[key])
        for key in (
            "input_tokens", "output_tokens", "cache_read_tokens",
            "cache_write_tokens", "reasoning_tokens", "copilot_usage_cost",
        )
    )
    assert backend._add_numbers(10**1000, 10**1000) == 2 * 10**1000
    json.dumps(collector.values, allow_nan=False)


def test_retry_and_corpus_usage_aggregation_clamp_float_overflow():
    merged = {
        "nodes": [], "edges": [], "hyperedges": [],
        "input_tokens": 0, "output_tokens": 0,
    }
    result = {
        "input_tokens": 1e308,
        "output_tokens": 1e308,
        "cache_read_tokens": 1e308,
        "cache_write_tokens": 1e308,
        "reasoning_tokens": 1e308,
        "copilot_usage_cost": 1e308,
    }
    llm._merge_into(merged, result)
    llm._merge_into(merged, result)
    retry_usage = llm._merged_provider_usage(result, result)
    assert llm._usage_add(10**1000, 10**1000) == 2 * 10**1000
    for values in (merged, retry_usage):
        assert all(
            math.isfinite(values[key])
            for key in (
                "cache_read_tokens", "cache_write_tokens",
                "reasoning_tokens", "copilot_usage_cost",
            )
        )
        json.dumps(values, allow_nan=False)


def test_friendly_error_categories():
    timeout = backend.CopilotSdkTimeoutError("timeout")
    assert backend._friendly_error(timeout, model=None) is timeout
    assert "download-runtime" in str(backend._friendly_error(FileNotFoundError("x"), model=None))
    assert "authentication" in str(backend._friendly_error(RuntimeError("forbidden"), model=None))
    assert "requested" not in str(backend._friendly_error(RuntimeError("model bad"), model=None))
    assert "m-1" in str(backend._friendly_error(RuntimeError("model bad"), model="m-1"))


def test_session_error_surfaces_only_safe_type_and_code():
    observer = backend._SessionObserver()
    observer(SimpleNamespace(
        type="session.error",
        data=SimpleNamespace(
            error_type="rate_limit",
            error_code="429",
            message="secret prompt and authorization header",
            stack="private stack",
        ),
    ))

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(backend._wait_for_response(
            SimpleNamespace(get_events=lambda: None),
            observer,
            timeout_seconds=1,
        ))

    text = str(exc_info.value)
    assert "rate_limit" in text
    assert "429" in text
    assert "secret prompt" not in text
    assert "private stack" not in text


def test_session_error_rejects_unrecognized_secret_metadata():
    observer = backend._SessionObserver()
    observer(SimpleNamespace(
        type="session.error",
        data=SimpleNamespace(
            error_type="SENTINEL_PRIVATE_PROMPT_AUTH",
            error_code="SENTINEL_PRIVATE_PROMPT_AUTH",
        ),
    ))

    with pytest.raises(backend._CopilotSessionError) as exc_info:
        asyncio.run(backend._wait_for_response(
            SimpleNamespace(get_events=lambda: None),
            observer,
            timeout_seconds=1,
        ))

    rendered = "".join(traceback.format_exception(exc_info.value))
    assert "SENTINEL_PRIVATE_PROMPT_AUTH" not in str(exc_info.value)
    assert "SENTINEL_PRIVATE_PROMPT_AUTH" not in rendered
    assert "type=" not in str(exc_info.value)
    assert "code=" not in str(exc_info.value)


def _install_fake_copilot(monkeypatch, captured, *, response=None, fail=None):
    class FakeSession:
        session_id = "session-1"

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

        async def delete_session(self, session_id):
            captured["deleted_session"] = session_id

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
    assert captured["client"]["mode"] == "empty"
    assert captured["client"]["enable_remote_sessions"] is False
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


def test_cleanup_finishes_before_temporary_directory_is_removed(monkeypatch):
    captured = {}

    class FakeSession:
        async def send(self, _prompt, **_kwargs):
            captured["handler"](
                SimpleNamespace(
                    type="assistant.message",
                    data=SimpleNamespace(content=_GRAPH_JSON),
                )
            )

        async def get_events(self):
            return []

        async def disconnect(self):
            captured["directory_during_disconnect"] = Path(captured["workdir"]).exists()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["workdir"] = kwargs["working_directory"]

        async def start(self):
            pass

        async def create_session(self, **kwargs):
            captured["handler"] = kwargs["on_event"]
            kwargs["on_event"](SimpleNamespace(type="session.tools_updated", data=None))
            return FakeSession()

        async def stop(self):
            captured["directory_during_stop"] = Path(captured["workdir"]).exists()

        async def force_stop(self):
            captured["force_stopped"] = True

    module = types.ModuleType("copilot")
    setattr(module, "CopilotClient", FakeClient)
    monkeypatch.setitem(sys.modules, "copilot", module)

    backend.call_copilot_sdk(
        "source", system_prompt="system", model=None,
        reasoning_effort=None, context_tier=None, timeout_seconds=2,
    )

    assert captured["directory_during_disconnect"] is True
    assert captured["directory_during_stop"] is True
    assert not Path(captured["workdir"]).exists()


def test_stop_timeout_succeeds_when_forced_shutdown_completes(monkeypatch):
    monkeypatch.setattr(backend, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    captured: dict = {"force_stops": 0}

    class FakeSession:
        async def send(self, _prompt, **_kwargs):
            captured["handler"](
                SimpleNamespace(
                    type="assistant.message",
                    data=SimpleNamespace(content=_GRAPH_JSON),
                )
            )

        async def get_events(self):
            return []

        async def disconnect(self):
            pass

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            pass

        async def create_session(self, **kwargs):
            captured["handler"] = kwargs["on_event"]
            kwargs["on_event"](SimpleNamespace(type="session.tools_updated", data=None))
            return FakeSession()

        async def stop(self):
            await asyncio.Event().wait()

        async def force_stop(self):
            captured["force_stops"] += 1

    module = types.ModuleType("copilot")
    setattr(module, "CopilotClient", FakeClient)
    monkeypatch.setitem(sys.modules, "copilot", module)

    result = backend.call_copilot_sdk(
        "source", system_prompt="system", model=None,
        reasoning_effort=None, context_tier=None, timeout_seconds=2,
    )

    assert result["content"] == _GRAPH_JSON
    assert captured["force_stops"] == 1


def test_call_uses_login_home_but_never_persists_or_deletes_session(
    monkeypatch, tmp_path,
):
    captured = {}
    response = SimpleNamespace(
        type="assistant.message",
        data=SimpleNamespace(content=_GRAPH_JSON),
    )
    _install_fake_copilot(monkeypatch, captured, response=response)
    copilot_home = tmp_path / "copilot-home"
    monkeypatch.setenv("COPILOT_HOME", str(copilot_home))

    backend.call_copilot_sdk(
        "source", system_prompt="system", model=None,
        reasoning_effort=None, context_tier=None, timeout_seconds=3,
    )

    assert captured["client"]["mode"] == "empty"
    assert captured["client"]["base_directory"] == str(copilot_home)
    assert captured["session"]["enable_session_store"] is False
    assert "deleted_session" not in captured
    assert captured["disconnected"] is True
    assert captured["stopped"] is True


def test_usage_events_aggregate(monkeypatch):
    captured = {}

    class FakeSession:
        session_id = "usage-session"

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
        async def delete_session(self, session_id):
            captured["deleted_session"] = session_id
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
        session_id = "history-session"

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

        async def delete_session(self, _session_id):
            pass

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


def test_history_message_does_not_replace_callback_usage():
    response = SimpleNamespace(
        type="assistant.message",
        data=SimpleNamespace(content=_GRAPH_JSON, output_tokens=8056),
    )
    observer = backend._SessionObserver()
    observer(SimpleNamespace(
        type="assistant.usage",
        data=SimpleNamespace(
            model="gpt-5.6-luna", input_tokens=1234, output_tokens=8056,
            cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=0,
            cost=2.0, finish_reason="stop",
        ),
    ))
    observer(response)

    class HistoryOnlyMessage:
        async def get_events(self):
            return [response]

    final, usage = asyncio.run(backend._wait_for_response(
        HistoryOnlyMessage(), observer, timeout_seconds=1,
    ))

    assert final is response
    assert usage["input_tokens"] == 1234
    assert usage["output_tokens"] == 8056
    assert usage["copilot_usage_cost"] == pytest.approx(2.0)


def test_waits_for_turn_end_when_usage_arrives_after_message(monkeypatch):
    monkeypatch.setattr(backend, "_SESSION_POLL_SECONDS", 0.001)
    response = SimpleNamespace(
        type="assistant.message",
        data=SimpleNamespace(content=_GRAPH_JSON, output_tokens=7),
    )
    turn_end = SimpleNamespace(type="assistant.turn_end", data=None)

    class UsageAfterMessage:
        def __init__(self):
            self.polls = 0

        async def get_events(self):
            self.polls += 1
            if self.polls == 1:
                return [response]
            return [response, _usage_event(), turn_end]

    final, usage = asyncio.run(backend._wait_for_response(
        UsageAfterMessage(), backend._SessionObserver(), timeout_seconds=1,
    ))

    assert final is response
    assert usage["input_tokens"] == 12
    assert usage["output_tokens"] == 7
    assert usage["copilot_usage_cost"] == pytest.approx(0.25)


def test_history_poll_is_bounded_by_request_timeout(monkeypatch):
    monkeypatch.setattr(backend, "_SESSION_START_TIMEOUT_SECONDS", 1)

    class HungHistory:
        async def get_events(self):
            await asyncio.Event().wait()

    async def run():
        return await asyncio.wait_for(
            backend._wait_for_response(
                HungHistory(), backend._SessionObserver(), timeout_seconds=0.02,
            ),
            timeout=0.1,
        )

    with pytest.raises(backend.CopilotSdkTimeoutError):
        asyncio.run(run())


def test_hung_history_obeys_short_session_start_deadline(monkeypatch):
    monkeypatch.setattr(backend, "_SESSION_START_TIMEOUT_SECONDS", 0.02)

    class HungHistory:
        async def get_events(self):
            await asyncio.Event().wait()

    started = time.monotonic()
    with pytest.raises(backend._CopilotSessionNotReadyError):
        asyncio.run(
            backend._wait_for_response(
                HungHistory(), backend._SessionObserver(), timeout_seconds=0.2,
            )
        )
    assert time.monotonic() - started < 0.1


def test_hung_history_obeys_short_usage_settle_deadline(monkeypatch):
    monkeypatch.setattr(backend, "_USAGE_SETTLE_SECONDS", 0.02)
    response = SimpleNamespace(
        type="assistant.message",
        data=SimpleNamespace(content=_GRAPH_JSON, output_tokens=7),
    )
    observer = backend._SessionObserver()
    observer(response)

    class HungHistory:
        async def get_events(self):
            await asyncio.Event().wait()

    started = time.monotonic()
    final, usage = asyncio.run(
        backend._wait_for_response(HungHistory(), observer, timeout_seconds=0.2)
    )
    assert final is response
    assert usage["output_tokens"] == 7
    assert time.monotonic() - started < 0.1


def test_session_failure_wins_simultaneous_stale_history_race():
    observer = backend._SessionObserver()
    stale = SimpleNamespace(
        type="assistant.message",
        data=SimpleNamespace(content="stale success", output_tokens=7),
    )
    turn_end = SimpleNamespace(type="assistant.turn_end", data=None)

    class RacingHistory:
        async def get_events(self):
            observer(
                SimpleNamespace(
                    type="session.error",
                    data=SimpleNamespace(error_type="authentication", error_code="401"),
                )
            )
            return [stale, turn_end]

    with pytest.raises(backend._CopilotSessionError, match="authentication"):
        asyncio.run(
            backend._wait_for_response(RacingHistory(), observer, timeout_seconds=1)
        )


def test_empty_history_snapshot_preserves_last_populated_usage(monkeypatch):
    monkeypatch.setattr(backend, "_SESSION_POLL_SECONDS", 0)
    monkeypatch.setattr(backend, "_USAGE_SETTLE_SECONDS", 0.01)
    response = SimpleNamespace(
        type="assistant.message",
        data=SimpleNamespace(content=_GRAPH_JSON, output_tokens=7),
    )

    class EmptyAfterPopulated:
        def __init__(self):
            self.calls = 0

        async def get_events(self):
            self.calls += 1
            return [_usage_event(), response] if self.calls == 1 else []

    final, usage = asyncio.run(
        backend._wait_for_response(
            EmptyAfterPopulated(), backend._SessionObserver(), timeout_seconds=1,
        )
    )
    assert final is response
    assert usage["input_tokens"] == 12
    assert usage["output_tokens"] == 7
    assert usage["copilot_usage_cost"] == pytest.approx(0.25)


def test_cancellation_resistant_history_invokes_bounded_abort(monkeypatch):
    monkeypatch.setattr(backend, "_USAGE_SETTLE_SECONDS", 0.01)
    monkeypatch.setattr(backend, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    response = SimpleNamespace(
        type="assistant.message",
        data=SimpleNamespace(content=_GRAPH_JSON, output_tokens=7),
    )
    observer = backend._SessionObserver()
    observer(response)
    released = asyncio.Event()
    aborts = {"count": 0}

    class CancellationResistantHistory:
        async def get_events(self):
            while not released.is_set():
                try:
                    await released.wait()
                except asyncio.CancelledError:
                    continue
            return []

    async def abort():
        aborts["count"] += 1
        released.set()

    started = time.monotonic()
    final, usage = asyncio.run(
        backend._wait_for_response(
            CancellationResistantHistory(),
            observer,
            timeout_seconds=0.2,
            abort=abort,
        )
    )
    assert final is response
    assert usage["output_tokens"] == 7
    assert aborts["count"] == 1
    assert time.monotonic() - started < 0.1


def test_expired_budget_does_not_leak_unawaited_sdk_coroutine(monkeypatch):
    captured = {}
    _install_fake_copilot(monkeypatch, captured, response=None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(backend.CopilotSdkTimeoutError):
            backend.call_copilot_sdk(
                "source", system_prompt="system", model=None,
                reasoning_effort=None, context_tier=None, timeout_seconds=0,
            )
        gc.collect()

    assert not any("was never awaited" in str(item.message) for item in caught)


def test_final_message_at_deadline_returns_best_effort_usage(monkeypatch):
    monkeypatch.setattr(backend, "_SESSION_POLL_SECONDS", 0.001)
    monkeypatch.setattr(backend, "_USAGE_SETTLE_SECONDS", 0.25)
    response = SimpleNamespace(
        type="assistant.message",
        data=SimpleNamespace(content=_GRAPH_JSON, output_tokens=7),
    )

    class MessageNearDeadline:
        async def get_events(self):
            await asyncio.sleep(0.04)
            return [response]

    final, usage = asyncio.run(backend._wait_for_response(
        MessageNearDeadline(), backend._SessionObserver(), timeout_seconds=0.05,
    ))

    assert final is response
    assert usage["output_tokens"] == 7


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
    rendered = "".join(traceback.format_exception(exc_info.value))
    assert "source secret" not in rendered
    assert "authorization header" not in rendered
    assert captured["disconnected"] and captured["stopped"]


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("SENTINEL_PRIVATE_PROMPT_AUTH"),
        ImportError("SENTINEL_PRIVATE_PROMPT_AUTH"),
    ],
)
def test_sdk_value_and_import_errors_are_fully_sanitized(monkeypatch, failure):
    captured = {}
    _install_fake_copilot(monkeypatch, captured, fail=failure)

    with pytest.raises(RuntimeError) as exc_info:
        backend.call_copilot_sdk(
            "source", system_prompt="system", model=None,
            reasoning_effort=None, context_tier=None, timeout_seconds=2,
        )

    rendered = "".join(traceback.format_exception(exc_info.value))
    assert "SENTINEL_PRIVATE_PROMPT_AUTH" not in str(exc_info.value)
    assert "SENTINEL_PRIVATE_PROMPT_AUTH" not in rendered
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_force_stopped_history_does_not_turn_completed_response_into_cleanup_error(
    monkeypatch,
):
    monkeypatch.setattr(backend, "_SESSION_SETTLE_SECONDS", 0)
    monkeypatch.setattr(backend, "_USAGE_SETTLE_SECONDS", 0.01)
    monkeypatch.setattr(backend, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    captured: dict = {"force_stops": 0, "disconnects": 0, "stops": 0}
    released = asyncio.Event()

    class FakeSession:
        async def send(self, _prompt, **_kwargs):
            captured["handler"](
                SimpleNamespace(
                    type="assistant.message",
                    data=SimpleNamespace(content=_GRAPH_JSON, output_tokens=7),
                )
            )

        async def get_events(self):
            while not released.is_set():
                try:
                    await released.wait()
                except asyncio.CancelledError:
                    continue
            return []

        async def disconnect(self):
            captured["disconnects"] += 1
            if captured["force_stops"]:
                raise RuntimeError("transport already closed")

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            pass

        async def create_session(self, **kwargs):
            captured["handler"] = kwargs["on_event"]
            kwargs["on_event"](SimpleNamespace(type="session.tools_updated", data=None))
            return FakeSession()

        async def stop(self):
            captured["stops"] += 1

        async def force_stop(self):
            captured["force_stops"] += 1
            released.set()

    module = types.ModuleType("copilot")
    setattr(module, "CopilotClient", FakeClient)
    monkeypatch.setitem(sys.modules, "copilot", module)

    result = backend.call_copilot_sdk(
        "source", system_prompt="system", model=None,
        reasoning_effort=None, context_tier=None, timeout_seconds=1,
    )
    assert result["content"] == _GRAPH_JSON
    assert captured["force_stops"] == 1
    assert captured["disconnects"] == 0
    assert captured["stops"] == 0


@pytest.mark.parametrize("stage", ["start", "create", "send"])
def test_cancellation_resistant_request_stage_respects_deadline(monkeypatch, stage):
    monkeypatch.setattr(backend, "_SESSION_SETTLE_SECONDS", 0)
    monkeypatch.setattr(backend, "_CLEANUP_TIMEOUT_SECONDS", 0.01)

    async def resist_cancellation():
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                continue

    class FakeSession:
        async def send(self, _prompt, **_kwargs):
            if stage == "send":
                await resist_cancellation()

        async def get_events(self):
            return []

        async def disconnect(self):
            pass

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            if stage == "start":
                await resist_cancellation()

        async def create_session(self, **_kwargs):
            if stage == "create":
                await resist_cancellation()
            return FakeSession()

        async def stop(self):
            pass

    module = types.ModuleType("copilot")
    setattr(module, "CopilotClient", FakeClient)
    monkeypatch.setitem(sys.modules, "copilot", module)

    started = time.monotonic()
    with pytest.raises(backend.CopilotSdkTimeoutError):
        backend.call_copilot_sdk(
            "source", system_prompt="system", model=None,
            reasoning_effort=None, context_tier=None, timeout_seconds=0.02,
        )
    assert time.monotonic() - started < 0.3


def test_cancellation_resistant_stop_and_force_stop_are_bounded(monkeypatch):
    monkeypatch.setattr(backend, "_SESSION_SETTLE_SECONDS", 0)
    monkeypatch.setattr(backend, "_USAGE_SETTLE_SECONDS", 0)
    monkeypatch.setattr(backend, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    captured: dict = {}

    async def resist_cancellation():
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                continue

    class FakeSession:
        async def send(self, _prompt, **_kwargs):
            captured["handler"](
                SimpleNamespace(
                    type="assistant.message",
                    data=SimpleNamespace(content=_GRAPH_JSON),
                )
            )

        async def get_events(self):
            return []

        async def disconnect(self):
            pass

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            pass

        async def create_session(self, **kwargs):
            captured["handler"] = kwargs["on_event"]
            kwargs["on_event"](SimpleNamespace(type="session.tools_updated", data=None))
            return FakeSession()

        async def stop(self):
            await resist_cancellation()

        async def force_stop(self):
            await resist_cancellation()

    module = types.ModuleType("copilot")
    setattr(module, "CopilotClient", FakeClient)
    monkeypatch.setitem(sys.modules, "copilot", module)

    started = time.monotonic()
    with pytest.raises(backend._CopilotCleanupError):
        backend.call_copilot_sdk(
            "source", system_prompt="system", model=None,
            reasoning_effort=None, context_tier=None, timeout_seconds=1,
        )
    assert time.monotonic() - started < 0.3


def test_production_cleanup_constant_does_not_extend_request_timeout():
    async def resist_cancellation():
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                continue

    started = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        backend._run_async(
            lambda: backend._run_bounded(
                resist_cancellation,
                timeout=0.02,
                abort=resist_cancellation,
            )
        )
    assert time.monotonic() - started < 0.3


def test_run_async_restores_dormant_caller_event_loop():
    original = asyncio.new_event_loop()
    asyncio.set_event_loop(original)
    try:
        assert backend._run_async(lambda: asyncio.sleep(0, result="ok")) == "ok"
        assert asyncio.get_event_loop() is original
    finally:
        asyncio.set_event_loop(None)
        original.close()


def test_production_history_shutdown_uses_one_short_window():
    response = SimpleNamespace(
        type="assistant.message",
        data=SimpleNamespace(content=_GRAPH_JSON),
    )
    observer = backend._SessionObserver()
    observer(response)

    class CancellationResistantHistory:
        async def get_events(self):
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    continue

    async def quick_abort():
        return None

    started = time.monotonic()
    final, _usage = backend._run_async(
        lambda: backend._wait_for_response(
            CancellationResistantHistory(),
            observer,
            timeout_seconds=0.02,
            abort=quick_abort,
        )
    )
    assert final is response
    assert time.monotonic() - started < 0.3


def test_run_async_restores_custom_policy_dormant_loop():
    original_policy = asyncio.get_event_loop_policy()
    dormant = original_policy.new_event_loop()

    class CustomPolicy(type(original_policy)):
        def __init__(self):
            self.loop = dormant

        def get_event_loop(self):
            if self.loop is None:
                raise RuntimeError("no loop")
            return self.loop

        def set_event_loop(self, loop):
            self.loop = loop

        def new_event_loop(self):
            return original_policy.new_event_loop()

        def get_child_watcher(self):
            getter = getattr(original_policy, "get_child_watcher", None)
            return getter() if getter is not None else None

        def set_child_watcher(self, watcher):
            setter = getattr(original_policy, "set_child_watcher", None)
            if setter is not None:
                setter(watcher)

    custom = CustomPolicy()
    asyncio.set_event_loop_policy(custom)
    try:
        assert backend._run_async(lambda: asyncio.sleep(0, result="ok")) == "ok"
        assert custom.get_event_loop() is dormant
    finally:
        asyncio.set_event_loop_policy(original_policy)
        dormant.close()


def test_run_async_ignores_custom_policy_decoy_private_state():
    original_policy = asyncio.get_event_loop_policy()
    dormant = original_policy.new_event_loop()

    class CustomPolicy(type(original_policy)):
        def __init__(self):
            self.loop = dormant
            self._local = SimpleNamespace(_loop="decoy", _set_called=True)

        def get_event_loop(self):
            if self.loop is None:
                raise RuntimeError("no loop")
            return self.loop

        def set_event_loop(self, loop):
            self.loop = loop

        def new_event_loop(self):
            return original_policy.new_event_loop()

        def get_child_watcher(self):
            getter = getattr(original_policy, "get_child_watcher", None)
            return getter() if getter is not None else None

        def set_child_watcher(self, watcher):
            setter = getattr(original_policy, "set_child_watcher", None)
            if setter is not None:
                setter(watcher)

    async def fail():
        raise RuntimeError("expected")

    custom = CustomPolicy()
    asyncio.set_event_loop_policy(custom)
    try:
        assert backend._run_async(lambda: asyncio.sleep(0, result="ok")) == "ok"
        assert custom.get_event_loop() is dormant
        with pytest.raises(RuntimeError, match="expected"):
            backend._run_async(fail)
        assert custom.get_event_loop() is dormant
    finally:
        asyncio.set_event_loop_policy(original_policy)
        dormant.close()


def test_sdk_exception_class_name_is_not_reflected():
    secret_type = type("SENTINEL_PRIVATE_PROMPT_AUTH", (RuntimeError,), {})
    safe = backend._friendly_error(secret_type("hidden"), model=None)
    rendered = "".join(traceback.format_exception(safe))
    assert "SENTINEL_PRIVATE_PROMPT_AUTH" not in str(safe)
    assert "SENTINEL_PRIVATE_PROMPT_AUTH" not in rendered


def test_copilot_plain_llm_preserves_fractional_usage(monkeypatch):
    usage: dict = {}

    def fake_call(*_args, **_kwargs):
        return {
            "content": "label",
            "input_tokens": 10,
            "output_tokens": 2,
            "copilot_usage_cost": 0.25,
        }

    monkeypatch.setattr("graphify.copilot_sdk_backend.call_copilot_sdk", fake_call)
    assert llm._call_llm("prompt", backend="copilot-sdk", usage_out=usage) == "label"
    assert usage["input"] == 10
    assert usage["output"] == 2
    assert usage["copilot_usage_cost"] == pytest.approx(0.25)


def test_sdk_timeout_preserves_timeout_type_and_cleanup(monkeypatch):
    captured = {}
    _install_fake_copilot(
        monkeypatch,
        captured,
        fail=asyncio.TimeoutError("SECRET_AUTH_HEADER"),
    )
    with pytest.raises(
        backend.CopilotSdkTimeoutError,
        match="timed out after 2 seconds",
    ) as exc_info:
        backend.call_copilot_sdk(
            "source", system_prompt="system", model=None, reasoning_effort=None,
            context_tier=None, timeout_seconds=2,
        )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "SECRET_AUTH_HEADER" not in "".join(
        traceback.format_exception(exc_info.value)
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
            raise RuntimeError("cleanup credential secret")

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
        assert "cleanup credential secret" not in str(exc_info.value)
        assert "cleanup credential secret" not in "".join(
            traceback.format_exception(exc_info.value)
        )
    else:
        with pytest.raises(RuntimeError, match="cleanup failed") as exc_info:
            backend.call_copilot_sdk(
                "source", system_prompt="system", model=None, reasoning_effort=None,
                context_tier=None, timeout_seconds=2,
            )
        assert "cleanup credential secret" not in "".join(
            traceback.format_exception(exc_info.value)
        )
    assert captured["stopped"] is True


def test_sdk_wrapper_falls_back_without_reflecting_sdk_error(monkeypatch, capsys):
    llm._COPILOT_SDK_FALLBACK_WARNED.clear()
    monkeypatch.delenv("GRAPHIFY_COPILOT_SDK_FALLBACK", raising=False)

    def fail_sdk(*_args, **_kwargs):
        raise RuntimeError("SECRET_AUTH_HEADER prompt corpus")

    monkeypatch.setattr(backend, "call_copilot_sdk", fail_sdk)
    monkeypatch.setattr(llm, "_run_copilot_cli", lambda *_a, **_k: "fallback")

    response = llm._run_copilot_sdk(
        "source",
        system_prompt="system",
        model=backend.COPILOT_DEFAULT_MODEL,
    )

    assert response["content"] == "fallback"
    assert response["_transport"] == "copilot-cli"
    assert response["model"] == "auto"
    stderr = capsys.readouterr().err
    assert "failure category: runtime" in stderr
    assert "SECRET_AUTH_HEADER" not in stderr
    assert "prompt corpus" not in stderr


def test_sdk_wrapper_disabled_fallback_suppresses_exception_chain(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_COPILOT_SDK_FALLBACK", "0")

    def fail_sdk(*_args, **_kwargs):
        raise RuntimeError("SECRET_AUTH_HEADER prompt corpus")

    monkeypatch.setattr(backend, "call_copilot_sdk", fail_sdk)

    with pytest.raises(RuntimeError, match="fallback is disabled") as exc_info:
        llm._run_copilot_sdk(
            "source",
            system_prompt="system",
            model=None,
        )

    rendered = "".join(traceback.format_exception(exc_info.value))
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "SECRET_AUTH_HEADER" not in rendered
    assert "prompt corpus" not in rendered
