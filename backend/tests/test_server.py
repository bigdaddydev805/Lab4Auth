"""The HTTP + WebSocket protocol (docs/PROTOCOL.md)."""
import json

from helpco.engine import Engine
from helpco.server.app import create_app


async def test_snapshot_commands_and_art(cfg, aiohttp_client):
    cfg.set("clock.mode", "realtime")
    cfg.set("clock.scale", 600)
    client = await aiohttp_client(create_app(Engine(cfg)))

    ws = await client.ws_connect("/ws")
    hello = json.loads((await ws.receive()).data)
    snap = json.loads((await ws.receive()).data)
    assert hello["type"] == "hello" and hello["v"] == 1
    assert snap["type"] == "snapshot"
    p = snap["payload"]
    for key in ("office", "clock", "employees", "items", "questions", "art"):
        assert key in p
    assert p["art"]["character"]["animations"]["walk_down"]["frames"] == 4

    await ws.send_json({"type": "command", "id": "c1", "command": {"type": "submit_question", "text": "What is a rainbow?"}})
    for _ in range(200):
        msg = json.loads((await ws.receive()).data)
        if msg["type"] == "ack":
            assert msg["payload"] == {"id": "c1", "result": {"ok": True, "id": "Q1"}}
            break
    else:
        raise AssertionError("no ack")
    await ws.close()

    for path in ("/art/office_bg.png", "/art/office_atlas.png", "/art/snapshot.png"):
        r = await client.get(path)
        assert r.status == 200 and r.content_type == "image/png"
    r = await client.get("/api/state")
    assert (await r.json())["questions"][0]["text"] == "What is a rainbow?"
    r = await client.post("/api/command", json={"type": "clock", "action": "pause"})
    assert (await r.json())["clock"]["paused"] is True
    r = await client.get("/")
    assert r.status == 200 and "HelpCo" in await r.text()
