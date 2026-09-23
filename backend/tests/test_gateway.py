"""The OpenRouter gateway against a fake OpenRouter server (no network, no key needed)."""
import json

import pytest
from aiohttp import web

from helpco.config import load_config
from helpco.llm import Gateway, LLMRequest
from helpco.llm.schema import action_schema
from helpco.store import Store

GOOD = {"thought": "coffee", "action": "use", "target": "coffee machine", "item": None, "text": None, "minutes": 3}


async def fake_openrouter(aiohttp_server, replies):
    """replies: list of (status, content-or-dict). Records every request body."""
    seen = []

    async def handler(request):
        body = await request.json()
        seen.append({"body": body, "headers": dict(request.headers)})
        status, content = replies[min(len(seen) - 1, len(replies) - 1)]
        if status != 200:
            return web.json_response({"error": {"message": "nope"}}, status=status)
        return web.json_response({"model": body["model"], "provider": "FakeAI",
                                  "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001}})
    app = web.Application()
    app.router.add_post("/api/v1/chat/completions", handler)
    server = await aiohttp_server(app)
    return server, seen


def make_gateway(tmp_path, monkeypatch, server, **llm):
    monkeypatch.setenv("TEST_OR_KEY", "sk-or-test")
    cfg = load_config(overrides={"llm": {"provider": "openrouter", "api_key_env": "TEST_OR_KEY",
                                         "base_url": str(server.make_url("/api/v1")),
                                         "routes": {"decide": ["a/primary", "b/backup"]}, **llm}})
    return Gateway(cfg, Store(tmp_path / "w.db"), "test")


def req(**kw):
    return LLMRequest(task="decide", system=["rules", "identity"], user="what now?", schema=action_schema(),
                      emp_id="emp_1", hire_no=1, hint={"emp_id": "emp_1"}, **kw)


async def test_structured_request_shape_and_success(tmp_path, monkeypatch, aiohttp_server):
    server, seen = await fake_openrouter(aiohttp_server, [(200, json.dumps(GOOD))])
    gw = make_gateway(tmp_path, monkeypatch, server)
    res = await gw.call(req())
    assert res.ok and res.source == "openrouter" and res.data["action"] == "use"
    body = seen[0]["body"]
    assert body["models"] == ["a/primary", "b/backup"]
    assert body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["strict"]
    assert body["provider"]["require_parameters"] is True
    assert body["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert seen[0]["headers"]["Authorization"] == "Bearer sk-or-test"
    assert gw.spent == pytest.approx(0.001)
    await gw.close()


async def test_invalid_output_is_repaired_once(tmp_path, monkeypatch, aiohttp_server):
    server, seen = await fake_openrouter(aiohttp_server, [(200, "I think I'll get coffee!"), (200, json.dumps(GOOD))])
    gw = make_gateway(tmp_path, monkeypatch, server)
    res = await gw.call(req())
    assert res.ok and res.data["target"] == "coffee machine"
    assert "didn't match the required format" in seen[1]["body"]["messages"][-1]["content"]
    await gw.close()


async def test_falls_back_to_the_mock_brain_when_all_models_fail(tmp_path, monkeypatch, aiohttp_server):
    server, _ = await fake_openrouter(aiohttp_server, [(400, "")])
    gw = make_gateway(tmp_path, monkeypatch, server)
    res = await gw.call(req())
    assert res.ok and res.source == "fallback"
    await gw.close()


async def test_out_of_credits_switches_to_mock(tmp_path, monkeypatch, aiohttp_server):
    server, _ = await fake_openrouter(aiohttp_server, [(402, "")])
    gw = make_gateway(tmp_path, monkeypatch, server)
    res = await gw.call(req())
    assert gw.exhausted and res.source == "fallback"
    await gw.close()


async def test_cassette_replays_recorded_calls_without_the_network(tmp_path, monkeypatch, aiohttp_server):
    server, seen = await fake_openrouter(aiohttp_server, [(200, json.dumps(GOOD))])
    gw = make_gateway(tmp_path, monkeypatch, server)
    await gw.call(req())
    await gw.close()
    replay = make_gateway(tmp_path, monkeypatch, server, cassette="replay")
    res = await replay.call(req())
    assert res.ok and res.source == "cassette" and len(seen) == 1
    await replay.close()


async def test_per_employee_model_override(tmp_path, monkeypatch, aiohttp_server):
    server, seen = await fake_openrouter(aiohttp_server, [(200, json.dumps(GOOD))])
    gw = make_gateway(tmp_path, monkeypatch, server, employees={"1": {"decide": ["qwen/special"]}})
    await gw.call(req())
    assert seen[0]["body"]["model"] == "qwen/special"
    await gw.close()
