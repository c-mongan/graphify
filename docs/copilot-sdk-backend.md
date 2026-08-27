# GitHub Copilot SDK backend

Graphify can use the official Python GitHub Copilot SDK for headless semantic extraction. The backend is explicit and optional: it is never auto-selected merely because Copilot is installed. The separate `copilot-cli` backend remains directly selectable and is the compatibility fallback when an SDK request fails.

## Install

The SDK requires Python 3.11 or later. Graphify core continues to support Python 3.10.

```bash
python -m pip install "graphifyy[copilot]"
python -m copilot download-runtime  # optional prefetch
```

On Python 3.10, the optional SDK package is not installed. An explicit `--backend copilot-sdk` request can still use the installed `copilot` CLI fallback. Select `--backend copilot-cli` directly when the CLI transport is required.

## Use

```bash
graphify extract ./docs --backend copilot-sdk
graphify extract ./docs --backend copilot-sdk --model <available-model-id>
```

For GitHub Enterprise Cloud, authenticate the official CLI and select the host before running Graphify:

```bash
copilot login --host https://example.ghe.com
export COPILOT_GH_HOST=example.ghe.com
graphify extract ./docs --backend copilot-sdk
```

The backend reuses the login held by the Copilot runtime. It does not require an OpenAI, Anthropic, or other provider API key.

Model precedence is:

1. `--model`
2. `GRAPHIFY_COPILOT_MODEL`
3. the account/runtime default

Optional settings:

```text
GRAPHIFY_COPILOT_REASONING_EFFORT=low|medium|high|xhigh|max
GRAPHIFY_COPILOT_CONTEXT_TIER=default|long_context
GRAPHIFY_COPILOT_SDK_PARALLEL=1
GRAPHIFY_COPILOT_SDK_FALLBACK=0
GRAPHIFY_COPILOT_CLI_MODEL=<available-model-id>
GRAPHIFY_COPILOT_CLI_PARALLEL=1
GRAPHIFY_API_TIMEOUT=<seconds>
COPILOT_HOME=<copilot state directory>
```

Calls are serial by default. Parallel execution is an expert opt-in because each call starts a Copilot runtime and consumes account quota or AI credits.

## Security and privacy boundary

The backend uses SDK `empty` mode and creates a fresh session for every request.

It explicitly disables:

- tools and shell access;
- file, Git, and host operations;
- MCP servers and apps;
- skills and custom instructions;
- memory and embedding retrieval;
- config and instruction discovery;
- file hooks and change tracking;
- session telemetry and session persistence;
- remote sessions.

The runtime reads the existing Copilot login from `COPILOT_HOME`, which defaults to `~/.copilot`. The session's working and configuration paths use one isolated temporary directory that is separate from `COPILOT_HOME`. It remains available until session/runtime cleanup completes and is then removed. Session persistence is disabled; the session is disconnected and the runtime is stopped under bounded cleanup, with `force_stop()` used if graceful shutdown exceeds its deadline.

Graphify sends the prepared semantic source blocks and inline image bytes to the Copilot service associated with the signed-in account. Code-only AST extraction remains local and does not use this backend.

Graphify surfaces only sanitized SDK error categories and suppresses the original exception chain, including timeout causes and contexts, so normal rendered tracebacks do not disclose prompt text, authorization headers, credentials, private SDK stack details, or arbitrary session-error metadata. Session error type/code diagnostics appear only when they match fixed local allowlists.

## Usage accounting

Graphify records:

- input and output tokens;
- cache read/write tokens when supplied;
- reasoning tokens when supplied;
- the model reported by the runtime;
- Copilot's usage/credit signal.

Numeric usage metadata is accepted only when it is finite, non-negative, and not boolean. Invalid values normalize to zero; valid fractional Copilot usage and arbitrary-size non-negative integers are preserved. Float totals that would overflow are capped at the largest finite float, keeping retry/corpus totals valid under strict JSON serialization. Usage is aggregated across every hollow/truncated retry and child call.

A Copilot usage signal is not treated as a zero-dollar API-cost claim.

Request processing uses one overall request deadline. Session readiness, start/create/send operations, history polling, and post-message usage settling share that deadline. When an operation reaches it, task supervision uses one short shared post-timeout drain/abort window rather than adding full cleanup windows. After success or failure, disconnect/stop/force-stop teardown uses one separate shared cleanup deadline, so lifecycle shutdown may add only that bounded teardown window; a successful `force_stop()` is the valid fallback when graceful `stop()` times out. Terminal callbacks race history reads so a hung read cannot hide a completion or failure. The private adapter loop closes after a final short drain. The last populated history snapshot is retained for usage accounting, terminal failure takes precedence over stale success, and the adapter checks the remaining request budget before constructing SDK awaitables.

## CLI fallback

Fallback is enabled by default. It covers an unavailable SDK, Python 3.10, startup and transport failures, timeouts, session failures, response failures, and cleanup failures. The fallback warning contains only a fixed failure category and does not reflect arbitrary SDK error text.

Set `GRAPHIFY_COPILOT_SDK_FALLBACK=0` when the run must prove that the SDK path succeeded. A timeout can be ambiguous because the remote service might have received the request before the local deadline. Disabling fallback avoids a possible replay in workloads where duplicate model calls are unacceptable.

The CLI fallback cannot attach image pixels through the SDK blob channel. It receives an accurate text-only image reference instead of a false claim that the pixels were attached. See [GitHub Copilot CLI backend](copilot-cli-backend.md) for direct CLI usage.

## Troubleshooting

### Authentication failure

Confirm the Copilot CLI works:

```bash
copilot -p "Reply exactly OK" --silent --no-custom-instructions --disable-builtin-mcps
```

If `COPILOT_HOME` is customized, export the same path before running Graphify.

### Runtime unavailable

Prefetch the SDK runtime:

```bash
python -m copilot download-runtime
```

### Model unavailable

Omit `--model` to use the account default, or choose a model available to the signed-in Copilot account.

### Timeout

Raise the bounded request deadline:

```bash
GRAPHIFY_API_TIMEOUT=900 graphify extract ./docs --backend copilot-sdk
```

## Verification

Unit tests use protocol fakes and require no account or network:

```bash
uv sync --extra openai --extra copilot
uv run pytest -q tests/test_copilot_sdk_backend.py tests/test_cli_extract_copilot_sdk.py tests/test_evidence_binding.py
```

The opt-in authenticated E2E uses only a synthetic one-line fixture:

```bash
GRAPHIFY_COPILOT_E2E=1 \
GRAPHIFY_COPILOT_E2E_HOME="$HOME" \
GRAPHIFY_COPILOT_E2E_MODEL=<available-model-id> \
uv run pytest -q tests/e2e/test_copilot_sdk_live.py
```

`GRAPHIFY_COPILOT_E2E_HOME` is required because Graphify's test harness deliberately replaces `HOME` to protect developer configuration during normal tests.
