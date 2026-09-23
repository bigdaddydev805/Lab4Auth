# HelpCo client protocol (v1)

The Python backend is the only source of truth. Clients (the Godot game client and the browser inspector) render what it says and send commands; they never change the world directly.

- WebSocket: `ws://127.0.0.1:8765/ws`
- HTTP: `http://127.0.0.1:8765` (art, REST inspection, the browser inspector at `/`)

## Envelope

Every server → client WebSocket message is one JSON text frame:

```json
{"v": 1, "type": "event", "seq": 1234, "sim_ms": 1805100000000, "payload": { }}
```

- `seq` increases by one per message on this server (per process). Use it to detect gaps.
- `sim_ms` is the simulation clock (Unix epoch milliseconds, UTC) when the message was sent.
- **IDs are strings.** Godot parses every JSON number as a float, so never use numeric IDs as dictionary keys.
- Snapshots are about 12 KB. Godot's `WebSocketPeer.inbound_buffer_size` defaults to 64 KiB; set it to at least 1 MiB to be safe.

## Server → client messages

| type | payload | notes |
|---|---|---|
| `hello` | `{protocol, server, world}` | Sent first on connect. |
| `snapshot` | full state (below) | Sent on connect and in reply to `resync`. Replace all local state. |
| `clock` | clock state | Twice a second (real time). |
| `employee` | an employee (below) | Whenever anything about an employee changes. Replace the local copy. |
| `item` | an item (below) | Created, moved, picked up, and so on. |
| `question` | a question (below) | Submitted, claimed, answered. |
| `event` | an event (below) | Everything that happens, for feeds, logs and speech bubbles. |
| `decision` | `{employee, decision}` | What an employee just decided and their private `thought` (observer/debug view). |
| `whiteboard` | `{text}` | The meeting-room whiteboard changed. |
| `ack` | `{id, result}` | Reply to a client `command`. |

### Snapshot payload

```jsonc
{
  "office": {                // layout (logic): tile grid and furniture
    "tile": 16, "width": 24, "height": 14,
    "map": ["####…", "#mmmm…"],          // '#': wall; m/b/w/e: meeting room, break room, work area, entrance
    "rooms": {"m": {"id": "meeting_room", "name": "the meeting room"}, …},
    "door": [20, 13], "spawn": [20, 12],
    "objects": [{"id": "desk_1", "kind": "desk", "name": "Desk 1", "x": 2, "y": 9, "w": 3, "h": 1,
                 "surface": true, "wall": false, "spots": [{"x": 3, "y": 8, "facing": "down", "pose": "sit"}]}, …]
  },
  "clock": { },               // see "Clock"
  "employees": [ ],           // see "Employee"
  "items": [ ],               // see "Item"
  "questions": [ ],           // see "Question"
  "whiteboard": "text",
  "llm": {"provider": "mock" | "openrouter", "spent_usd": 0.0, "budget_usd": 5.0, "exhausted": false, …},
  "last_seq": 812,            // last event seq in the store
  "art": {"character": { }, "office": { }},   // see "Art"
  "protocol": 1
}
```

### Clock

```json
{"sim_ms": 1805100000000, "scale": 20.0, "effective_scale": 3.0, "mode": "realtime", "paused": false,
 "tz": "America/Los_Angeles", "iso": "2027-03-15T09:12:00-07:00", "date": "Monday, March 15, 2027",
 "time": "9:12 AM", "day": 1, "phase": "workday"}
```

`effective_scale` is the current sim-seconds-per-real-second (it slows down while a conversation is going on). To run a smooth local clock between ticks: `sim_now = last.sim_ms + (real_ms_since_tick × effective_scale)`, unless `paused`.

### Employee

```jsonc
{
  "id": "emp_1", "hire_no": 1, "name": "Marisol", "pronouns": "she/her",
  "status": "present",            // hired (still choosing who to be) | offsite | present | departed
  "look": {"skin": "tan", "hair_style": "bob", "hair_color": "brown", "top": "sweater", "top_color": "green",
           "pants_color": "navy", "shoes_color": "charcoal", "accessories": ["glasses"], "accessory_color": "red"},
  "look_key": "a1b2c3d4e5",       // changes whenever the look changes → refetch the sprite sheet
  "pos": [3, 8], "facing": "down",   // tile position when not walking; facing: down | up | left | right
  "path": [[20, 12], [19, 12], …],   // non-empty while walking
  "path_t0": 1805100000000,          // sim ms when the walk started
  "ms_per_tile": 6666,               // sim ms per tile
  "activity": {"kind": "work", "label": "working at Desk 1", "pose": "type", "until_ms": 1805100900000, "target": null},
  "holding": "i3",                  // item id or null
  "thinking": false,                // waiting on its model
  "desk_id": "desk_1",
  "needs": {"energy": 0.8, "hunger": 0.2, "social": 0.4}
}
```

**Movement.** While `path` is non-empty, the employee is at tile `path[i]` where `i = floor((sim_now − path_t0) / ms_per_tile)`. Interpolate between `path[i]` and `path[i+1]` for smooth motion, and face the direction of travel. When the walk ends the server sends a new `employee` with the final `pos` and an empty `path`.

**Pose → animation** (see the character sheet):

| condition | animation |
|---|---|
| walking | `walk_<direction of travel>` |
| `pose` = `type` | `type` (seated at a desk, facing down) |
| `pose` = `sit`, facing up | `sit_up` |
| `pose` = `sit` | `sit_down` |
| holding a `coffee` item, standing, facing down | `hold_down`; play `sip` now and then |
| just spoke (speech bubble showing), standing, facing down | `talk_down` |
| otherwise | `idle_<facing>` |

Seated poses are drawn 2 px lower so they sit in the chair.

### Item

```json
{"id": "i3", "kind": "coffee", "label": "a mug of coffee", "owner": "emp_1", "created_by": "emp_1",
 "created_ms": 1805100000000, "location": {"on": "desk_1", "slot": 0}, "text": "", "color": null}
```

`location` is one of `{"on": <object id>, "slot": n}` (on a surface; use `art.office.surfaces[object id][slot]` for the pixel position), `{"held_by": <employee id>}`, or `{"at": [x, y]}` (on the floor).

### Question

```json
{"id": "Q1", "text": "Why does Saturn have rings?", "submitted_ms": 1805100000000, "status": "answered",
 "claimed_by": "emp_1", "claimed_ms": 1805100100000, "answer": "…", "answered_by": "emp_1", "answered_ms": 1805100500000}
```

### Event

```jsonc
{
  "seq": 812, "type": "speech.said", "sim_ms": 1805100000000,
  "actor": "emp_1", "room": "break_room", "targets": ["emp_2"],
  "payload": {"to": "emp_2", "text": "did you change the printer settings", "conversation": "c3"},
  "witnesses": {"emp_1": "did", "emp_2": "heard"},   // who perceived it (limited knowledge)
  "source": "llm",          // llm | autopilot | schedule | player | engine | mock | fallback
  "decision_id": "d41", "visibility": "public",       // public | private | system
  "summary": "Marisol → Yusuf: \"did you change the printer settings\""   // ready-made feed line
}
```

Event types include `day.started`, `day.ended`, `clock.skipped`, `employee.hired`, `employee.identity_chosen`, `employee.arrived`, `employee.entered`, `employee.left`, `employee.introduced`, `speech.said`, `owner.message`, `owner.reply`, `owner.gift`, `action.started`, `action.denied`, `activity.completed`, `coffee.made`, `question.submitted`, `question.claimed`, `question.draft_updated`, `question.answered`, `note.written`, `note.read`, `whiteboard.read`, `file.written`, `file.read`, `item.given`, `item.put_down`, `item.picked_up`, `appearance.changed`, `printer.printed`, `conversation.ended`, `reflection.completed`, `identity.changed`, `office.lunch`, `office.workday_end`, `llm.fallback`.

**Speech bubbles.** Show a bubble over `actor` for `speech.said`, `employee.introduced` and `owner.reply`, with `payload.text`, for about 2.5 s + 60 ms per character (real time).

## Client → server

```json
{"type": "command", "id": "c1", "command": {"type": "submit_question", "text": "Why does Saturn have rings?"}}
```

| command | fields | effect |
|---|---|---|
| `submit_question` | `text` | Adds a question to the queue (employees in the work area see it). |
| `message` | `to` (id or name), `text` | Speaks to one employee over the intercom (private). |
| `give` | `to`, `kind` (mug, succulent, photo_frame, rubber_duck, books, snow_globe, coffee, note, paper), `label`? | Gives an employee an item. |
| `hire` | — | Hires a new employee (up to 4); they choose who to be on arrival. |
| `clock` | `action`: `pause` \| `resume` \| `scale`, `value` | Controls time. |
| `snapshot` | — | Saves the world now. |

Also: `{"type": "resync"}` asks for a fresh `snapshot`.

The same commands work over HTTP: `POST /api/command` with the `command` object as the body.

## Art

All art is generated by the backend from each employee's chosen look, so clients need no asset pipeline.

- `GET /art/meta.json` → `{character, office}` (also included in the snapshot as `art`).
- **Character sheet:** `GET /art/employee/<id>.png?v=<look_key>`. `art.character` describes the grid:

  ```json
  {"frame_w": 34, "frame_h": 34, "columns": 8, "anchor": [17, 32],
   "animations": {"idle_down": {"row": 0, "frames": 8, "fps": 5}, "walk_down": {"row": 1, "frames": 4, "fps": 8}, …}}
  ```

  Frame `n` of an animation is at `(n × frame_w, row × frame_h)`. `anchor` is the feet position inside a frame. Place it at the bottom-center of the employee's tile: `(x × 16 + 8, (y + 1) × 16)`.
- **Office background:** `GET /art/office_bg.png`, drawn at (0, 0).
- **Furniture and item atlas:** `GET /art/office_atlas.png`. `art.office`:
  - `sprites`: `{name: {x, y, w, h}}` rectangles in the atlas.
  - `placements`: `[{id, sprite, x, y, sort_y}]`. Draw each furniture sprite at world pixel `(x, y)`, y-sorted with the characters by `sort_y` (a character's `sort_y` is its feet y).
  - `surfaces`: `{object id: [[px, py], …]}` slot positions for items on that surface. Draw an item's bottom edge at `py + 4`.
  - `items`: `{item kind: sprite name}`.
- `GET /art/snapshot.png?scale=3` renders the whole office server-side (handy for previews).

Everything is 16 px per tile and meant to be scaled by integers with nearest-neighbour filtering.

## REST (inspection)

- `GET /api/state`: the snapshot.
- `GET /api/events?after=<seq>&limit=<n>&types=a,b`: the event log (with `summary`).
- `GET /api/employees/<id>`: everything about one employee: identity history, beliefs (including abandoned ones), relationships, goals, long-term memories, today's experiences, recent model calls.
- `GET /api/files`: notes and files, including private folders. The owner can see everything.
- `GET /api/llm`: model router status, spend, and recent calls.
