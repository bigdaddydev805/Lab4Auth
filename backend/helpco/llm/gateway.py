"""The single door every model call goes through.

Responsibilities:
- route a task (onboard/decide/converse/compose/reflect) to a list of OpenRouter model slugs,
  with optional per-employee overrides (for model-comparison experiments);
- request structured output, validate it locally, repair once, then fall back;
- record every call (the "cassette") so runs can be replayed or forked cheaply;
- enforce a spending budget, degrading to the offline mock brain instead of failing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from . import mock
from .openrouter import OpenRouterClient, OpenRouterError, OutOfCredits
from .schema import extract_json, validate, wire_schema

log = logging.getLogger("helpco.llm")
VOLATILE = ("session_id", "user", "trace")


@dataclass
class LLMRequest:
    task: str
    system: list[str]
    user: str
    schema: dict | None = None
    schema_name: str = "reply"
    max_tokens: int = 500
    emp_id: str | None = None
    hire_no: int | None = None
    hint: dict = field(default_factory=dict)
    sim_ms: int = 0


@dataclass
class LLMResult:
    ok: bool
    data: Any = None
    raw: str = ""
    model: str = ""
    provider: str = ""
    source: str = ""          # openrouter | cassette | mock | fallback
    cost: float = 0.0
    error: str | None = None
    latency_ms: int = 0


class Gateway:
    def __init__(self, cfg, store, world_id: str = "default"):
        self.cfg = cfg
        self.store = store
        self.world_id = world_id
        self.provider = cfg.llm_provider()
        self.routes: dict = cfg.get("llm.routes", {})
        self.emp_routes: dict = cfg.get("llm.employees", {}) or {}
        self.cassette: str = cfg.get("llm.cassette", "record")
        self.budget = float(cfg.get("llm.budget_usd", 5.0))
        self.data_collection = cfg.get("llm.data_collection", "deny")
        self.spent = 0.0
        self.exhausted = False
        self.calls = Counter()
        self._occurrence: Counter = Counter()
        self.client: OpenRouterClient | None = None
        if self.provider == "openrouter":
            import os
            key = os.environ.get(cfg.get("llm.api_key_env", "OPENROUTER_API_KEY"), "")
            if not key:
                log.warning("OpenRouter selected but no API key found; using the mock brain")
                self.provider = "mock"
            else:
                self.client = OpenRouterClient(key, cfg.get("llm.base_url"), cfg.get("llm.app_url", ""),
                                               cfg.get("llm.app_title", "HelpCo"), cfg.get("llm.request_timeout_s", 60))

    # --- routing ----------------------------------------------------------------------------
    def models_for(self, task: str, hire_no: int | None) -> list[str]:
        if hire_no is not None:
            override = self.emp_routes.get(str(hire_no), {})
            if task in override:
                return list(override[task])
            if task == "converse" and "decide" in override:
                return list(override["decide"])
        route = self.routes.get(task) or self.routes.get("decide") or ["openrouter/auto"]
        return list(route)

    def status(self) -> dict:
        return {"provider": self.provider, "spent_usd": round(self.spent, 4), "budget_usd": self.budget,
                "exhausted": self.exhausted, "cassette": self.cassette, "calls": dict(self.calls),
                "routes": self.routes}

    # --- request building ---------------------------------------------------------------------
    def _body(self, req: LLMRequest, models: list[str], extra_messages: list | None = None) -> dict:
        system_blocks = [{"type": "text", "text": b} for b in req.system if b]
        if system_blocks:
            system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
        body: dict[str, Any] = {
            "model": models[0],
            "messages": [{"role": "system", "content": system_blocks}, {"role": "user", "content": req.user}]
                        + (extra_messages or []),
            "max_tokens": req.max_tokens,
            "provider": {"data_collection": self.data_collection},
            "session_id": f"{self.world_id}:{req.emp_id}:{req.task}"[:256],
            "user": self.world_id,
        }
        if len(models) > 1:
            body["models"] = models
        if req.schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": req.schema_name, "strict": True, "schema": wire_schema(req.schema)}}
            body["provider"]["require_parameters"] = True
            body["plugins"] = [{"id": "response-healing"}]
        return body

    @staticmethod
    def _key(body: dict) -> str:
        stable = {k: v for k, v in body.items() if k not in VOLATILE}
        return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    # --- the call -----------------------------------------------------------------------------
    def mock_result(self, req: LLMRequest, source: str = "mock", error: str | None = None) -> LLMResult:
        data = mock.respond(req.task, req.hint)
        self.calls[f"{source}:{req.task}"] += 1
        raw = json.dumps(data) if not isinstance(data, str) else data
        if self.cassette != "live":  # keep the prompt too, so the inspector can show what would be sent
            self.store.record_llm_call(
                key=f"{source}", task=req.task, emp_id=req.emp_id, sim_ms=req.sim_ms, wall=time.time(), model="mock",
                provider="mock", request=json.dumps({"system": req.system, "user": req.user}), response=None,
                content=raw, cost=0.0, prompt_tokens=None, completion_tokens=None, latency_ms=0, ok=1, error=error)
        return LLMResult(ok=True, data=data, raw=raw, model="mock", provider="mock", source=source, error=error)

    def _parse(self, req: LLMRequest, content: str) -> tuple[Any, list[str]]:
        if req.schema is None:
            text = (content or "").strip()
            return (text, []) if text else (None, ["empty reply"])
        try:
            data = extract_json(content)
        except (ValueError, json.JSONDecodeError) as e:
            return None, [f"not valid JSON ({e})"]
        return validate(data, req.schema)

    async def call(self, req: LLMRequest) -> LLMResult:
        if self.provider == "mock" or self.exhausted or self.client is None:
            return self.mock_result(req)

        models = self.models_for(req.task, req.hire_no)
        attempts = [models] + [[m] for m in models[1:]]
        last_error = "no attempt"
        for attempt_models in attempts:
            extra: list = []
            for repair in range(2):
                body = self._body(req, attempt_models, extra)
                result = await self._send(req, body)
                if result.error and not result.raw:
                    last_error = result.error
                    if self.exhausted:
                        return self.mock_result(req, "fallback", last_error)
                    break  # transport failure: try the next model
                data, errors = self._parse(req, result.raw)
                if not errors:
                    result.ok, result.data = True, data
                    self.calls[f"{result.source}:{req.task}"] += 1
                    return result
                last_error = "; ".join(errors)[:500]
                log.info("invalid %s output from %s: %s", req.task, result.model, last_error)
                extra = [{"role": "assistant", "content": result.raw[:4000]},
                         {"role": "user", "content": f"That reply didn't match the required format: {last_error}. "
                                                     "Reply again with only the corrected JSON."}]
        log.warning("%s for %s failed on all models (%s); using the mock brain", req.task, req.emp_id, last_error)
        return self.mock_result(req, "fallback", last_error)

    async def _send(self, req: LLMRequest, body: dict) -> LLMResult:
        base_key = self._key(body)
        self._occurrence[base_key] += 1
        replicate = self._occurrence[base_key] - 1
        if self.cassette in ("replay", "replay_then_live"):
            hit = self.store.find_llm_call(base_key, replicate)
            if hit:
                return LLMResult(ok=False, raw=hit["content"] or "", model=hit["model"] or "",
                                 provider=hit["provider"] or "", source="cassette", cost=0.0)
            if self.cassette == "replay":
                return LLMResult(ok=False, error=f"replay miss for {req.task}")
        started = time.time()
        try:
            data, latency = await self.client.chat(body)
        except OutOfCredits as e:
            self.exhausted = True
            self._record(req, base_key, body, None, "", 0.0, 0, str(e))
            return LLMResult(ok=False, error=str(e))
        except OpenRouterError as e:
            self._record(req, base_key, body, None, "", 0.0, int((time.time() - started) * 1000), str(e))
            return LLMResult(ok=False, error=str(e))
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):  # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        usage = data.get("usage") or {}
        cost = float(usage.get("cost") or 0.0)
        self.spent += cost
        if self.spent >= self.budget:
            self.exhausted = True
            log.warning("LLM budget of $%.2f reached; switching to the mock brain", self.budget)
        model = data.get("model", body["model"])
        provider = data.get("provider", "")
        error = None
        if choice.get("finish_reason") == "length":
            error = "reply was cut off (max_tokens)"
        self._record(req, base_key, body, data, content, cost, latency, error, model, provider, usage)
        return LLMResult(ok=False, raw=content if not error else "", model=model, provider=provider,
                         source="openrouter", cost=cost, latency_ms=latency, error=error)

    def _record(self, req, key, body, response, content, cost, latency, error, model="", provider="", usage=None):
        if self.cassette == "live":
            return
        usage = usage or {}
        self.store.record_llm_call(
            key=key, task=req.task, emp_id=req.emp_id, sim_ms=req.sim_ms, wall=time.time(), model=model or body["model"],
            provider=provider, request=json.dumps(body), response=json.dumps(response) if response else None,
            content=content, cost=cost, prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"), latency_ms=latency, ok=0 if error else 1, error=error)

    async def close(self) -> None:
        if self.client:
            await self.client.close()
