# HelpCo architecture (v0)

> **AI decides intention. Simulation determines reality. Game client displays what happened.**

```
 Godot 4.7 client ─┐                                     ┌─ OpenRouter (any model family)
 Browser inspector ┴─ WebSocket/HTTP ─ Python backend ───┤     ↑ via helpco/llm (one door for every model call)
                                        │                └─ offline mock brain (no key / budget hit / tests)
                                        └─ SQLite per office: events · snapshots · memories · beliefs · model calls
```

## Backend modules (`backend/helpco/`)

| Module | Role |
|---|---|
| `engine.py` | **The World Engine (Game Master).** Owns the clock and a schedule (event heap). Runs the day cycle (morning → arrivals → workday → leave → nightly reflection → skip the night). Executes plans step by step (walk → do → call), decides who perceives each event (witnesses), handles conversations and player commands. Nothing else mutates the world. |
| `actions.py` | Turns an intention (`go_to`, `use coffee machine`, `say`, `read /private/…`, …) into a plan, or refuses it with in-world feedback (`action.denied`). Permissions live here: private folders are off-limits, and denied attempts are recorded with `security: true`. |
| `cognition.py` | Decides *when* an employee thinks (arrived, finished something, was addressed, a new question, a denial, end of day) and *what they're told*: date/time/tenure facts, where they are, who's there, what they're holding, the question queue if they can see it, today's experiences, recalled memories, beliefs about the people present, and why they're being asked. Also runs day-one self-creation (name, pronouns, look, desk, introduction, personal item) and answer writing. |
| `reflection.py` | Nightly consolidation, per employee, over *their own* experiences: journal, long-term memories, beliefs (hearsay kept as hearsay, confidence capped by trust in the source), relationship changes, goals, a plan for tomorrow, and — rarely, rate-limited — a new self-concept version. |
| `memory.py` | Experiences (first-person), beliefs (never deleted: superseded beliefs are closed and linked), relationships, identity versions, goals. Retrieval = recency + importance + relevance (SQLite FTS5 BM25), with Ebbinghaus-style fading into `dormant`. |
| `narrate.py` | Event → first-person memory text for each witness (so memories are subjective by construction), and → an observer feed line. |
| `llm/` | `gateway.py` routes each task to an OpenRouter model list (per-employee overrides), requests strict JSON-schema output, validates locally, repairs once, falls back across models and finally to the mock brain, enforces a $ budget, and records every call to the cassette (`record` / `replay` / `replay_then_live`). `openrouter.py` is a ~100-line aiohttp client. `mock.py` is the deterministic offline brain. |
| `office.py`, `catalog.py`, `state.py`, `clock.py`, `store.py`, `events.py` | The map and furniture (A* pathfinding, rooms = earshot), what exists to choose from, world state, the real-calendar clock, persistence, the event record. |
| `art/` | Code-drawn pixel art: per-employee animated sprite sheets from their chosen look, the office background, a furniture/item atlas, and a server-side scene renderer. |
| `server/` | aiohttp HTTP + WebSocket (see [PROTOCOL.md](PROTOCOL.md)) and the browser inspector. |

## A decision, end to end

1. Something happens that deserves thought, e.g. June finishes her 45 minutes of work, so the engine calls `request_decision(june, "finished")`.
2. Cognition builds her point of view: "It's Tuesday, March 16, 2027, 10:38 AM… You arrived at 8:54 AM; this is your day 2 at HelpCo… You're at your desk… People here: Milo… The question queue: Q3 (open)… Earlier today… What comes to mind: (yesterday)…". The model gets this plus the stable rules and her identity block (cacheable), and returns `{"thought", "action", "target", "item", "text", "minutes"}`.
3. The gateway validates the reply against the schema (repairing or falling back if needed) and records the call.
4. Actions resolve the intention against the real world: "use coffee machine" becomes *walk* to the free spot, *do* "making coffee" for 3 min, then *call*: create a mug held by June and raise energy. Anything impossible is denied with a reason she'll see next time.
5. The engine executes the plan over sim time, emitting events (`action.started`, `coffee.made`, …). Each event records its witnesses, and each witness gets a first-person memory. Clients receive `employee`/`item`/`event` messages and animate it.
6. When the plan finishes, go back to step 1.

In **realtime** mode the clock keeps running while a model thinks; the employee shows "…". In **lockstep** mode (`helpco sim`, tests, experiments) time only advances when nobody is waiting on a model, so runs don't depend on latency.

## What's recorded

Every meaningful event, with actor, room, targets, payload, witnesses, source (llm / autopilot / schedule / player) and decision id. Also daily snapshots of the whole world and every model call (prompt, reply, model, provider, cost, latency). That gives save/load, the inspector's history, memory audits against ground truth, and cheap forks for experiments.

## Not yet built (see TECHNICAL_REQUIREMENTS.md §13)

- Embeddings / hybrid retrieval.
- Relationship-graph and biography views.
- The experiment runner (fork + N replicates).
- An answer output gate for real outside askers.
- The economy.
- Generated catalog items.
- Replay/rewind UI.
- The departure lifecycle.

## Godot client (`client/`)

The client renders what the server says and sends the owner's commands. It never changes the world.

| Part | Role |
|---|---|
| `scripts/net/`, `scripts/model/` | WebSocket link with reconnect, the client-side world state, and a smoothed sim clock that extrapolates between server ticks. |
| `scripts/world3d/office3d.gd` | Builds the 3D diorama from the snapshot's office layout, places items on surfaces, keeps one chibi per present employee, and animates the ambient details (plants, LEDs, monitors that switch on when someone sits down, coffee steam, the question board, the wall clock). |
| `scripts/world3d/chibi.gd` | One employee: body built from their look, following the server's path with the sim clock, animated procedurally. |
| `scripts/world3d/props.gd`, `kit.gd` | Furniture, items and clutter from primitives, with shared materials and meshes. |
| `scripts/world3d/atmosphere.gd` | Time-of-day lighting from the office's local time, rain and weather (ambience only, seeded by date), window shader, dust. |
| `scripts/world3d/camera_rig.gd` | Isometric orthographic camera: drag/pinch/wheel, follow the selected employee, framing around the HUD. |
| `scripts/world3d/ambience.gd` | Procedural sound: rain, room tone, keyboards, coffee machine, chimes. Off by default. |
| `scripts/ui/` | HUD (top bar, side panel with feed/team/questions/whiteboard, owner bar), name tags and speech bubbles projected over the 3D characters. |

