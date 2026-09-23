"""HTTP + WebSocket server: streams the living office to the Godot client and the browser inspector.

Protocol (see docs/PROTOCOL.md): every WebSocket message is a JSON envelope
    {"v": 1, "type": ..., "seq": n, "sim_ms": ..., "payload": {...}}
The server sends `hello` and a full `snapshot` on connect, then deltas (`event`, `employee`, `item`,
`question`, `whiteboard`, `decision`) and a `clock` tick twice a second. Clients send
`{"type": "command", "id": ..., "command": {...}}`; commands become engine inputs (never direct
state changes) and are answered with an `ack`.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from aiohttp import WSMsgType, web

from .. import PROTOCOL_VERSION, __version__
from ..art import character, office, scene
from ..engine import Engine

log = logging.getLogger("helpco.server")
STATIC = Path(__file__).with_name("static")
QUEUE_LIMIT = 5000


class Hub:
    """Fans engine messages out to connected clients (each with its own queue)."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.clients: set[asyncio.Queue] = set()
        self.seq = 0
        engine.listeners.append(self.publish)

    def envelope(self, type_: str, payload) -> str:
        self.seq += 1
        return json.dumps({"v": PROTOCOL_VERSION, "type": type_, "seq": self.seq, "sim_ms": self.engine.clock.ms,
                           "payload": payload}, default=str)

    def publish(self, msg: dict) -> None:
        kind = msg.get("type")
        payload = {k: v for k, v in msg.items() if k != "type"}
        if kind in ("event", "employee", "item", "question", "clock"):
            payload = payload.get(kind, payload)
        data = self.envelope(kind, payload)
        for q in list(self.clients):
            if q.qsize() > QUEUE_LIMIT:
                self.clients.discard(q)  # a stuck client; it will reconnect and resync
                q.put_nowait(None)
                continue
            q.put_nowait(data)


def _snapshot(engine: Engine) -> dict:
    st = engine.full_state()
    st["art"] = {"character": character.meta(), "office": office.office_meta()}
    st["protocol"] = PROTOCOL_VERSION
    return st


def create_app(engine: Engine) -> web.Application:
    app = web.Application()
    hub = Hub(engine)
    app["engine"], app["hub"] = engine, hub
    routes = web.RouteTableDef()

    # --------------------------------------------------------------- websocket
    @routes.get("/ws")
    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)
        q: asyncio.Queue = asyncio.Queue()
        hub.clients.add(q)
        await ws.send_str(hub.envelope("hello", {"protocol": PROTOCOL_VERSION, "server": __version__,
                                                  "world": engine.world_id}))
        await ws.send_str(hub.envelope("snapshot", _snapshot(engine)))

        async def sender():
            while True:
                data = await q.get()
                if data is None or ws.closed:
                    break
                await ws.send_str(data)

        send_task = asyncio.create_task(sender())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "command":
                    result = await engine.enqueue(data.get("command") or {})
                    await ws.send_str(hub.envelope("ack", {"id": data.get("id"), "result": result}))
                elif data.get("type") == "resync":
                    await ws.send_str(hub.envelope("snapshot", _snapshot(engine)))
        finally:
            hub.clients.discard(q)
            send_task.cancel()
        return ws

    # --------------------------------------------------------------- REST
    @routes.get("/api/state")
    async def state(request):
        return web.json_response(_snapshot(engine), dumps=lambda o: json.dumps(o, default=str))

    @routes.post("/api/command")
    async def command(request):
        body = await request.json()
        return web.json_response(await engine.enqueue(body))

    @routes.get("/api/events")
    async def events(request):
        after = int(request.query.get("after", 0))
        limit = min(2000, int(request.query.get("limit", 300)))
        types = [t for t in request.query.get("types", "").split(",") if t] or None
        rows = engine.store.events(after, limit, types)
        from ..events import Event
        from .. import narrate
        for r in rows:
            ev = Event(r["type"], r["sim_ms"], r["actor"], r["room"], r["targets"], r["payload"], r["witnesses"],
                       r["source"], r["decision_id"])
            r["summary"] = narrate.summary(engine, ev)
        return web.json_response(rows)

    @routes.get("/api/employees/{emp_id}")
    async def employee(request):
        emp_id = request.match_info["emp_id"]
        e = engine.world.employees.get(emp_id)
        if not e:
            raise web.HTTPNotFound()
        mem = engine.memory
        memories = mem.all_for(emp_id, tiers=("episode",), limit=200)
        today = mem.day_experiences(emp_id, engine.world.day_number)
        rels = mem.relationships(emp_id)
        for r in rels:
            r["name"] = engine.name_of(r["other_id"])
        beliefs_all = mem.beliefs(emp_id, active=False, limit=200)
        for b in beliefs_all:
            about = b["about"]
            b["about_name"] = engine.name_of(about.split(":", 1)[1]) if about.startswith("person:") else about
        return web.json_response({
            "employee": e.public(), "intro": e.intro, "last_decision": e.last_decision, "stats": e.stats,
            "days_worked": e.days_worked, "met": [engine.name_of(m) for m in e.met],
            "identity": mem.identity_history(emp_id), "goals": mem.goals(emp_id, active=False),
            "beliefs": beliefs_all, "relationships": rels, "memories": memories, "today": today,
            "llm_calls": engine.store.recent_llm_calls(30, emp_id),
        }, dumps=lambda o: json.dumps(o, default=str))

    @routes.get("/api/files")
    async def files(request):
        return web.json_response([f.to_dict() for f in engine.world.files.values()])

    @routes.get("/api/llm")
    async def llm(request):
        return web.json_response({"status": engine.gateway.status(),
                                  "recent": engine.store.recent_llm_calls(int(request.query.get("limit", 40)))},
                                 dumps=lambda o: json.dumps(o, default=str))

    # --------------------------------------------------------------- art
    @routes.get("/art/meta.json")
    async def art_meta(request):
        return web.json_response({"character": character.meta(), "office": office.office_meta()})

    @routes.get("/art/office_bg.png")
    async def art_bg(request):
        return web.Response(body=office.background_png(), content_type="image/png")

    @routes.get("/art/office_atlas.png")
    async def art_atlas(request):
        return web.Response(body=office.atlas_png(), content_type="image/png")

    @routes.get("/art/employee/{emp_id}.png")
    async def art_employee(request):
        e = engine.world.employees.get(request.match_info["emp_id"])
        if not e or not e.look:
            raise web.HTTPNotFound()
        png = await asyncio.get_running_loop().run_in_executor(None, character.sheet_png, e.look)
        return web.Response(body=png, content_type="image/png", headers={"Cache-Control": "max-age=86400"})

    @routes.get("/art/portrait/{emp_id}.png")
    async def art_portrait(request):
        e = engine.world.employees.get(request.match_info["emp_id"])
        if not e or not e.look:
            raise web.HTTPNotFound()
        return web.Response(body=character.portrait_png(e.look), content_type="image/png")

    @routes.get("/art/snapshot.png")
    async def art_snapshot(request):
        st = engine.full_state()
        scale = max(1, min(6, int(request.query.get("scale", 3))))
        png = await asyncio.get_running_loop().run_in_executor(None, scene.render, st, scale, 0)
        return web.Response(body=png, content_type="image/png")

    @routes.get("/")
    async def index(request):
        return web.FileResponse(STATIC / "index.html")

    app.router.add_routes(routes)
    app.router.add_static("/static/", STATIC)

    async def on_startup(app):
        engine.boot()
        app["stop"] = asyncio.Event()
        app["loop_task"] = asyncio.create_task(engine.run_realtime(app["stop"]))

        async def clock_ticker():
            while not app["stop"].is_set():
                await asyncio.sleep(0.5)
                hub.publish({"type": "clock", "clock": engine.clock_state()})
        app["clock_task"] = asyncio.create_task(clock_ticker())

    async def on_cleanup(app):
        app["stop"].set()
        engine.running = False
        for t in (app["loop_task"], app["clock_task"]):
            t.cancel()
        try:
            engine.snapshot("shutdown")
        except Exception:
            log.exception("final snapshot failed")
        await engine.gateway.close()
        engine.store.close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def serve(cfg, host: str = "127.0.0.1", port: int = 8765) -> None:
    engine = Engine(cfg)
    app = create_app(engine)
    print(f"HelpCo is open at http://{host}:{port}  (inspector)  ·  ws://{host}:{port}/ws  (Godot client)")
    print(f"Brain: {engine.gateway.provider}  ·  saves: {cfg.save_dir}")
    web.run_app(app, host=host, port=port, print=None)
