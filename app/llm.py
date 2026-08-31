"""
llm.py — Optional, provider-agnostic LLM client.

Purpose
-------
The template Q&A in ``explainer`` handles seven question shapes well and
everything else not at all. This module lets an LLM field the rest, while the
templates stay in place as a guaranteed fallback.

Three rules govern everything here
----------------------------------
1.  **It is optional.** With no key configured, ``is_configured()`` returns
    ``False`` and the app behaves exactly as before. The project's "runs locally,
    no API keys" property is preserved for anyone who does not opt in.

2.  **It never raises.** ``complete()`` returns ``(text, error)`` rather than
    propagating exceptions. A dead network, a bad key, a rate limit or a wrong
    model name must degrade to the templates, not break a live demo.

3.  **The key is never hardcoded and never logged.** It is read from Streamlit
    secrets or the environment. ``describe()`` redacts it. Nothing in this module
    prints or returns it.

Configuration
-------------
Either put it in ``.streamlit/secrets.toml`` (gitignored):

    LPR_LLM_API_KEY = "sk-..."
    LPR_LLM_PROVIDER = "openai"        # optional, inferred when omitted
    LPR_LLM_MODEL    = "gpt-4o-mini"   # optional
    LPR_LLM_BASE_URL = "..."           # optional, for OpenAI-compatible hosts

or set environment variables of the same names. Common provider-standard names
(``OPENAI_API_KEY``, ``GROQ_API_KEY``, ``GEMINI_API_KEY`` and so on) are also
picked up, so an existing key in your shell works without extra setup.

Supported providers
-------------------
``openai``  Any OpenAI-compatible ``/chat/completions`` endpoint. That covers
            OpenAI, Groq, OpenRouter, Together, DeepSeek, and local servers such
            as Ollama or LM Studio — pointing ``LPR_LLM_BASE_URL`` at them is
            enough. One adapter, most of the ecosystem.
``gemini``  Google's ``generativeContent`` endpoint, which is shaped differently
            enough to need its own adapter.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

#: Hard ceiling on how long to wait for an answer before falling back.
#:
#: Deliberately short. Measured against Google's endpoint from a home connection,
#: successful calls return in 1-2 seconds while roughly one request in four hangs
#: indefinitely — and the hangs are not correlated with prompt size, so they are a
#: transient network fault rather than model latency. Waiting 20 seconds to
#: discover that is far worse than falling back in 8, because the templates answer
#: instantly and a stalled UI in front of an audience is the real failure.
DEFAULT_TIMEOUT = 8.0

#: Extra attempts after the first, for transient failures only.
#:
#: Since the hangs are intermittent, an immediate retry usually succeeds. Retries
#: apply to timeouts, connection errors, 429s and 5xx. They never apply to other
#: 4xx responses: a bad key or a retired model name will fail identically every
#: time, so retrying only doubles the delay before the inevitable fallback.
MAX_RETRIES = 1

#: Sensible defaults per provider. Models get renamed and retired constantly, so
#: these are a starting point rather than a guarantee — set ``LPR_LLM_MODEL`` if a
#: default is unavailable to your account. A wrong model name surfaces as the
#: provider's own error message via ``complete()``, which makes it diagnosable.
_PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_keys": ("OPENAI_API_KEY",),
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "env_keys": ("GROQ_API_KEY",),
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "env_keys": ("OPENROUTER_API_KEY",),
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.0-flash",
        "env_keys": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    },
}

#: Providers that speak the OpenAI chat-completions shape.
_OPENAI_COMPATIBLE = {"openai", "groq", "openrouter"}


@dataclass(frozen=True)
class LLMConfig:
    """Resolved provider settings. Holds the key, so never log this directly."""

    provider: str
    api_key: str
    model: str
    base_url: str
    timeout: float = DEFAULT_TIMEOUT

    def describe(self) -> str:
        """A safe, human-readable summary with the key redacted."""
        return f"{self.provider}:{self.model}"

    @property
    def redacted_key(self) -> str:
        """The key with all but its last four characters masked."""
        if len(self.api_key) <= 8:
            return "*" * len(self.api_key)
        return f"{'*' * 8}{self.api_key[-4:]}"


#: Values that mean "not filled in yet". Copying secrets.toml.example without
#: editing it leaves the literal placeholder in place, which otherwise produces a
#: confusing state: the app reports an LLM as configured, then every call fails
#: with 401. Treating these as absent keeps "configured" honest.
_PLACEHOLDERS = {
    "paste-your-key-here",
    "your-key-here",
    "your-api-key",
    "your_api_key_here",
    "changeme",
    "change-me",
    "todo",
    "xxx",
    "...",
    "sk-...",
}


def _is_placeholder(value: str) -> bool:
    """Whether a configured value is an unedited template placeholder."""
    cleaned = value.strip().strip("\"'").lower()
    if cleaned in _PLACEHOLDERS:
        return True
    # Catch variants like "paste-your-openai-key-here" without listing them all.
    return "paste" in cleaned or "your-key" in cleaned or "your_key" in cleaned


#: The environment as it stood at import, before anything could touch
#: ``st.secrets``.
#:
#: Streamlit copies every entry in ``secrets.toml`` into ``os.environ`` the first
#: time secrets are read, **overwriting** values already present, and the change
#: persists for the life of the process. So a real key exported in the shell was
#: being destroyed by the unedited ``paste-your-key-here`` placeholder from the
#: file — and because the overwrite happens on first access, ``is_configured()``
#: and ``load_config()`` could disagree within a single process depending on which
#: ran first. Keeping a pristine copy makes the lookup deterministic.
_PRISTINE_ENV = dict(os.environ)


def _env(name: str) -> str:
    """
    Return the environment value for ``name``, immune to Streamlit's injection.

    A value set deliberately after import wins, which is what lets the self-test
    and any runtime override work. Otherwise the pristine copy is used, so a shell
    variable clobbered by ``secrets.toml`` is still recoverable.
    """
    current = os.environ.get(name, "").strip()
    if current and not _is_placeholder(current):
        return current
    pristine = _PRISTINE_ENV.get(name, "").strip()
    return pristine if pristine and not _is_placeholder(pristine) else ""


def _secret(name: str) -> str | None:
    """
    Read a value from Streamlit secrets, falling back to the environment.

    Precedence: a real value in ``secrets.toml`` wins, since that is the documented
    setup route. A shell variable wins when the file is absent or still holds an
    unedited placeholder.

    The broad ``except`` is required because ``st.secrets`` raises when no secrets
    file exists, and because this module must import cleanly outside Streamlit —
    the self-test and any command-line use depend on that.
    """
    try:
        import streamlit as st

        if name in st.secrets:
            value = str(st.secrets[name]).strip()
            if value and not _is_placeholder(value):
                return value
    except Exception:
        pass

    return _env(name) or None


def load_config() -> LLMConfig | None:
    """
    Resolve provider settings, or ``None`` when no key is available.

    Deliberately not cached, so editing secrets and re-running takes effect
    without restarting the process.
    """
    explicit_key = _secret("LPR_LLM_API_KEY")
    provider = (_secret("LPR_LLM_PROVIDER") or "").strip().lower()

    api_key = explicit_key

    # With no explicit provider, infer one from whichever standard key exists.
    # This makes an already-exported OPENAI_API_KEY or GEMINI_API_KEY just work.
    if not provider:
        for candidate, defaults in _PROVIDER_DEFAULTS.items():
            found = next((_secret(k) for k in defaults["env_keys"] if _secret(k)), None)
            if found:
                provider, api_key = candidate, api_key or found
                break
        if not provider and api_key:
            provider = "openai"  # a bare key with no hint is most often OpenAI

    if provider and not api_key:
        defaults = _PROVIDER_DEFAULTS.get(provider, {})
        api_key = next(
            (_secret(k) for k in defaults.get("env_keys", ()) if _secret(k)), None
        )

    if not api_key or not provider:
        return None
    if provider not in _PROVIDER_DEFAULTS:
        return None

    defaults = _PROVIDER_DEFAULTS[provider]
    return LLMConfig(
        provider=provider,
        api_key=api_key,
        model=_secret("LPR_LLM_MODEL") or str(defaults["model"]),
        base_url=(_secret("LPR_LLM_BASE_URL") or str(defaults["base_url"])).rstrip("/"),
        timeout=float(_secret("LPR_LLM_TIMEOUT") or DEFAULT_TIMEOUT),
    )


def is_configured() -> bool:
    """Whether an LLM is available. Cheap enough to call on every render."""
    return load_config() is not None


def provider_label() -> str:
    """A short label for the UI, or ``"templates"`` when no LLM is configured."""
    config = load_config()
    return config.describe() if config else "templates"


def complete(
    system: str,
    user: str,
    *,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> tuple[str | None, str | None]:
    """
    Send one prompt and return ``(text, error)``.

    Exactly one of the two is populated. Errors are returned rather than raised so
    a caller can silently fall back; the message is kept because a wrong model
    name or an expired key is otherwise very hard to diagnose from the UI.

    ``temperature`` defaults low: these are factual answers about a specific
    course, so variation is a defect rather than a feature.
    """
    config = load_config()
    if config is None:
        return None, "no LLM configured"

    try:
        import requests
    except ImportError:
        return None, "the 'requests' package is not installed"

    if config.provider in _OPENAI_COMPATIBLE:
        call = _complete_openai
    elif config.provider == "gemini":
        call = _complete_gemini
    else:
        return None, f"unsupported provider {config.provider!r}"

    last_error = "unknown error"
    for attempt in range(MAX_RETRIES + 1):
        try:
            text = (call(requests, config, system, user, max_tokens, temperature) or "").strip()
            if text:
                return text, None
            last_error = "the model returned an empty response"
        except _Permanent as exc:
            # A bad key or a retired model fails the same way every time.
            return None, _scrub(str(exc), config.api_key)
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
            # str(exc) can contain the request URL, and Gemini puts the key in the
            # query string, so scrub before this reaches a screen or a log.
            last_error = _scrub(str(exc) or exc.__class__.__name__, config.api_key)

        if attempt < MAX_RETRIES:
            continue
    return None, last_error


class _Permanent(RuntimeError):
    """A failure that retrying cannot fix, such as a bad key or unknown model."""


def _complete_openai(requests, config: LLMConfig, system: str, user: str,
                     max_tokens: int, temperature: float) -> str:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint."""
    response = requests.post(
        f"{config.base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=config.timeout,
    )
    if response.status_code != 200:
        raise _classify(response, config.model)
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def _complete_gemini(requests, config: LLMConfig, system: str, user: str,
                     max_tokens: int, temperature: float) -> str:
    """
    Call Google's ``generateContent`` endpoint.

    Gemini has no "system" role; the instruction goes in ``systemInstruction``.
    The key travels as a query parameter, which is why ``_scrub`` exists.
    """
    response = requests.post(
        f"{config.base_url}/models/{config.model}:generateContent",
        params={"key": config.api_key},
        headers={"Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        },
        timeout=config.timeout,
    )
    if response.status_code != 200:
        raise _classify(response, config.model)
    payload = response.json()
    candidates = payload.get("candidates") or []
    if not candidates:
        # Usually a safety block; surface the reason rather than a blank answer.
        reason = (payload.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise RuntimeError(f"Gemini returned no answer ({reason})")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts)


def _classify(response, model: str) -> Exception:
    """
    Turn a failed HTTP response into a retryable or permanent error.

    429 and 5xx are worth another attempt. Other 4xx codes describe a
    configuration problem — a rejected key, an unknown model — which will fail
    identically on every retry, so they short-circuit to the fallback instead of
    doubling the delay first.
    """
    message = _http_error(response, model)
    if response.status_code == 429 or response.status_code >= 500:
        return RuntimeError(message)
    return _Permanent(message)


def _http_error(response, model: str) -> str:
    """Turn a failed HTTP response into a short, actionable message."""
    detail = ""
    try:
        body = response.json()
        detail = (
            body.get("error", {}).get("message")
            if isinstance(body.get("error"), dict)
            else str(body.get("error") or "")
        ) or ""
    except Exception:
        detail = (response.text or "")[:200]

    hint = ""
    if response.status_code == 401:
        hint = " (check the API key)"
    elif response.status_code == 404:
        hint = f" (is model {model!r} available to this account?)"
    elif response.status_code == 429:
        hint = " (rate limited or out of quota)"
    return f"HTTP {response.status_code}{hint}: {detail[:200]}"


def _scrub(text: str, secret: str) -> str:
    """Remove an API key from a string before it can be displayed or logged."""
    if secret and secret in text:
        text = text.replace(secret, "***")
    return text


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # ── redaction must never expose a key ────────────────────────────────────
    cfg = LLMConfig(provider="openai", api_key="sk-abcdefghijklmnop1234",
                    model="gpt-4o-mini", base_url="https://example.test/v1")
    assert cfg.redacted_key == "********1234", cfg.redacted_key
    assert "abcdefghijklmnop" not in cfg.redacted_key
    assert "abcdefghijklmnop" not in cfg.describe(), "describe() must not leak the key"
    assert cfg.describe() == "openai:gpt-4o-mini"

    short = LLMConfig(provider="openai", api_key="abc", model="m", base_url="u")
    assert short.redacted_key == "***", short.redacted_key

    # ── scrubbing catches the key wherever it appears ────────────────────────
    assert _scrub("failed for key=sk-secret123 at url", "sk-secret123") == \
        "failed for key=*** at url"
    assert _scrub("nothing sensitive", "sk-secret123") == "nothing sensitive"
    assert _scrub("text", "") == "text"

    # ── unedited template placeholders must count as "not configured" ────────
    # Copying secrets.toml.example without editing it otherwise reports an LLM as
    # available and then fails every call with a 401.
    for placeholder in ("paste-your-key-here", "PASTE-YOUR-KEY-HERE", "changeme",
                        "your-key-here", "sk-...", "paste-your-openai-key-here",
                        '"paste-your-key-here"'):
        assert _is_placeholder(placeholder), placeholder
    for real in ("sk-proj-abc123def456", "gsk_realkeymaterial", "AIzaSyRealKey",
                 "ollama"):
        assert not _is_placeholder(real), real

    # ── REGRESSION: a shell variable must survive Streamlit's env injection ───
    # st.secrets copies secrets.toml into os.environ, overwriting what is already
    # there. Reading the environment after touching st.secrets returned the file's
    # placeholder instead of the exported key, so the client reported itself
    # unconfigured for no visible reason. _secret() snapshots the environment
    # first; this asserts the snapshot survives.
    os.environ["LPR_TEST_SNAPSHOT"] = "sk-real-shell-value"
    try:
        try:
            import streamlit as st

            _ = "LPR_TEST_SNAPSHOT" in st.secrets  # may mutate os.environ
        except Exception:
            pass
        assert _secret("LPR_TEST_SNAPSHOT") == "sk-real-shell-value", (
            "an exported key must not be lost to Streamlit's secrets injection"
        )

        # Simulate the injection itself: Streamlit stamps a placeholder over the
        # real value. The pristine copy must still recover it, and repeated calls
        # must agree — the original symptom was is_configured() and load_config()
        # returning contradictory answers in one process.
        _PRISTINE_ENV["LPR_TEST_CLOBBER"] = "sk-real-shell-value"
        os.environ["LPR_TEST_CLOBBER"] = "paste-your-key-here"
        assert _secret("LPR_TEST_CLOBBER") == "sk-real-shell-value", (
            "a clobbered shell value must be recovered from the pristine copy"
        )
        assert _secret("LPR_TEST_CLOBBER") == _secret("LPR_TEST_CLOBBER"), (
            "repeated lookups must agree"
        )
    finally:
        os.environ.pop("LPR_TEST_SNAPSHOT", None)
        os.environ.pop("LPR_TEST_CLOBBER", None)
        _PRISTINE_ENV.pop("LPR_TEST_CLOBBER", None)

    # ── config resolution, against a controlled source ───────────────────────
    #
    # ``_secret`` is stubbed rather than the environment being manipulated. The
    # earlier version popped env vars to simulate "no key", which quietly stopped
    # working the moment a real secrets.toml existed — Streamlit secrets are read
    # first, so the test was asserting against whatever the machine happened to be
    # configured with. Substituting the source makes these assertions deterministic
    # on any machine, configured or not.
    # Note: the global is rebound directly rather than via ``import app.llm``.
    # Under ``python -m app.llm`` this file executes as ``__main__``, so importing
    # ``app.llm`` yields a *second*, separate module object — patching that one
    # leaves the copy actually running untouched.
    _real_secret = _secret
    _fake: dict[str, str] = {}
    _secret = lambda name: _fake.get(name) or None  # noqa: E731
    try:
        # ── with nothing configured, the client is cleanly disabled ──────────
        _fake.clear()
        assert load_config() is None, "should be unconfigured with no key present"
        assert is_configured() is False
        assert provider_label() == "templates"

        text, error = complete("sys", "user")
        assert text is None and error == "no LLM configured", (text, error)

        # ── a key alone is enough; provider and model are inferred ───────────
        _fake.clear()
        _fake["LPR_LLM_API_KEY"] = "sk-test-not-a-real-key"
        config = load_config()
        assert config is not None
        assert config.provider == "openai", config.provider
        assert config.model == "gpt-4o-mini", config.model
        assert config.base_url == "https://api.openai.com/v1"
        assert is_configured() is True

        # A provider-standard key name is picked up on its own.
        _fake.clear()
        _fake["GEMINI_API_KEY"] = "gm-test-not-a-real-key"
        config = load_config()
        assert config is not None and config.provider == "gemini", config
        assert "generativelanguage" in config.base_url

        # An unedited placeholder counts as absent, even via the real reader.
        _fake.clear()
        _fake["LPR_LLM_API_KEY"] = "paste-your-key-here"
        assert _is_placeholder(_fake["LPR_LLM_API_KEY"])

        # Explicit settings win over the defaults, which is what lets a local
        # Ollama or LM Studio server be used.
        _fake.clear()
        _fake.update({
            "LPR_LLM_API_KEY": "local",
            "LPR_LLM_PROVIDER": "openai",
            "LPR_LLM_BASE_URL": "http://localhost:11434/v1/",
            "LPR_LLM_MODEL": "llama3",
        })
        config = load_config()
        assert config is not None
        assert config.model == "llama3"
        assert config.base_url == "http://localhost:11434/v1", "trailing slash should be trimmed"

        # An unknown provider disables rather than crashing.
        _fake["LPR_LLM_PROVIDER"] = "not-a-provider"
        assert load_config() is None, "an unknown provider must disable the client"

        # ── a real failure must be returned, never raised ────────────────────
        _fake["LPR_LLM_PROVIDER"] = "openai"
        _fake["LPR_LLM_BASE_URL"] = "http://127.0.0.1:9/v1"  # nothing listens here
        _fake["LPR_LLM_TIMEOUT"] = "2"
        text, error = complete("sys", "user")
        assert text is None, "no text should come back from a dead endpoint"
        assert error, "a failure must produce an error message"
        assert "local" not in error, "the key must not appear in the error"
        print(f"unreachable endpoint handled gracefully -> {error[:70]}")

        # ── retry policy: transient failures retry, permanent ones do not ────
        attempts = {"n": 0}

        def _always_timeout(*args, **kwargs):
            attempts["n"] += 1
            raise TimeoutError("read timed out")

        def _permanent(*args, **kwargs):
            attempts["n"] += 1
            raise _Permanent("HTTP 401 (check the API key): nope")

        _real_openai = _complete_openai
        try:
            _complete_openai = _always_timeout
            attempts["n"] = 0
            text, error = complete("sys", "user")
            assert text is None and error
            assert attempts["n"] == MAX_RETRIES + 1, (
                f"a timeout should be retried: {attempts['n']} attempt(s)"
            )

            _complete_openai = _permanent
            attempts["n"] = 0
            text, error = complete("sys", "user")
            assert text is None and "401" in error, error
            assert attempts["n"] == 1, (
                f"a permanent failure must not be retried: {attempts['n']} attempt(s)"
            )
            print(f"retry policy: transient retried {MAX_RETRIES + 1}x, "
                  f"permanent tried once")
        finally:
            _complete_openai = _real_openai
    finally:
        _secret = _real_secret

    print(f"\nconfigured right now: {is_configured()} ({provider_label()})")
    print("llm.py self-test passed: all assertions OK")
    sys.exit(0)
