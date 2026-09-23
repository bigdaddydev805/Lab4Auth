# HelpCo — Technical Requirements & Research

*Research snapshot: September 2026. Companion to [VISION.md](VISION.md).*

This document turns the design brief into technical requirements. It collects what's known about each subsystem from papers, open-source projects and official docs, and makes a recommendation for each. Sources are linked inline and collected at the end.

> **How to read the confidence of claims.** Several primary sites were blocked from the research environment: openrouter.ai, arxiv.org, docs.godotengine.org and alibabacloud.com. Their content was confirmed through GitHub mirrors, source code, SDK specs, and search results.
> - **(preprint):** many 2025–2026 papers cited here have not been peer-reviewed.
> - **(indirect):** prices and a few platform details were only confirmed through third-party sources. Check them before relying on them.

---

## 0. Summary

### Decisions made so far

| Area | Decision |
|---|---|
| Client | **Godot 4** (current stable: 4.7.2, Aug 2026), plus a browser-based debug/inspector view |
| Backend | **Python** simulation engine, which owns reality |
| AI access | **OpenRouter** only. Model-agnostic: any family (OpenAI, Anthropic, Qwen, Gemini, DeepSeek, …) chosen by model slug |
| Clock | Real calendar, **configurable** time scale |
| Identity | The AI decides everything about itself on day one: **name (completely free, duplicates allowed), appearance, pronouns, which desk, a self-introduction, and a personal item** |

### Top recommendations

1. **The backend is the "Game Master".** Code, not an LLM, validates every action and decides what each employee perceives. Agents only emit structured intentions. This is the pattern [Concordia](https://github.com/google-deepmind/concordia) and [AI Town](https://github.com/a16z-infra/ai-town/blob/main/ARCHITECTURE.md) converge on, and it is the brief's "AI decides intention, simulation determines reality."
2. **Event-sourced world.** An append-only event log with witnesses/provenance, plus daily snapshots, in SQLite (WAL). This one decision gives save/load, rewind, replay, forking for experiments, and ground truth for memory audits.
3. **Call the LLM only on meaningful triggers.** Cheap "autopilot" behavior fills the gaps: walking, sitting, finishing a task. This is the biggest cost and latency lever: 10–100× cheaper in [Lyfe Agents](https://arxiv.org/abs/2310.02172) and [Affordable Generative Agents](https://arxiv.org/abs/2402.02053).
4. **Memory = subjective experiences + versioned beliefs with provenance.** Retrieval scores recency, importance and relevance ([Generative Agents](https://arxiv.org/abs/2304.03442)). Contradicted beliefs are invalidated, never deleted ([Zep/Graphiti](https://arxiv.org/abs/2501.13956)). Nightly consolidation must preserve hedges ("Deven *said* …").
5. **A thin in-house OpenRouter client** (≈300 lines on `httpx`). Every call is recorded to a replay store, so experiments can be re-run, and budgets are enforced from `usage.cost`.
6. **A modular, palette-ramped character system.** Appearance is data (part IDs + color ramps) that the AI picks from a catalog, which the engine validates. The art source is still an open question (see §9).
7. **The #1 design risk is homogenization.** With no seeded personalities, LLM employees drift toward the same generic, helpful persona and the same names ("Elara"). Counter it with context diversity, model diversity per employee, and diversity metrics from week one (§4.3).

---

## 1. Prior art: what exists and how HelpCo differs

| Project | What it is | Relevance |
|---|---|---|
| [Generative Agents / Smallville](https://github.com/joonspk-research/generative_agents) (Park et al. 2023) | 25 LLM agents in a pixel town. Memory stream, reflection, planning | The canonical memory/reflection design. Cost "thousands of dollars" for 2 game days. The LLM partly narrates world state, which is the opposite of HelpCo's principle |
| [AI Town](https://github.com/a16z-infra/ai-town) (a16z) | Open-source TypeScript remake on Convex | Clean "engine exclusively owns state; LLM work returns as *inputs*" architecture. Pauses when idle |
| [Concordia](https://github.com/google-deepmind/concordia) (DeepMind) | Python library for generative social simulation | "Game Master" pattern. v2.4 (Mar 2026) added async engines and checkpoints |
| [Project Sid / PIANO](https://arxiv.org/abs/2411.00114) (Altera) | 500–1,000 agents in Minecraft | Emergent roles, culture and religion spreading. Fast modules plus a slow "cognitive controller" |
| [AgentOffice](https://github.com/harishkotra/agent-office), [Pixel Office](https://pixeloffice.org/), [Agent Pixels](https://www.agent-pixels.com/blog/tutorials/agent-pixels-launch-day), [The Office](https://www.avilevi.co.il/en/projects/the-office/) | Pixel-art offices that visualize *productive* AI agents | Same visual idea. AgentOffice thinks every ~15 s with Phaser + Colyseus + SQLite memory. The Office runs 16 Claude agents on PixiJS with A* |

**HelpCo's differentiators:** none of these centers on *identity emerging from experience*, nightly consolidation into a biography, a real calendar with tenure and anniversaries, limited perception with provenance, or the physical world as memory. HelpCo also has a research layer for controlled counterfactual experiments. Those are the requirements to protect.

---

## 2. System architecture

```
┌────────────────────────────┐        WebSocket (JSON, versioned)       ┌──────────────────────────────────────────────┐
│  Godot 4.7 client          │  ◀── snapshot@seq, event deltas, clock ── │  Python backend (single asyncio process)     │
│  - office + characters     │  ──▶ commands (question, message, give,   │                                              │
│  - interpolated movement   │       hire, clock) → become *inputs*      │  World Engine / Game Master (single writer)  │
│  - speech bubbles, UI      │                                           │   ├─ SimClock + Scheduler (event heap)        │
│  - corner-of-monitor mode  │                                           │   ├─ Action pipeline (validate → schedule)    │
└────────────────────────────┘                                           │   ├─ Perception filter (who saw/heard what)   │
┌────────────────────────────┐   HTTP + WS                               │   ├─ Autopilot (needs/utility, no LLM)        │
│  Browser inspector         │ ◀──────────────────────────────────────── │   └─ Cognition (when to call the LLM)         │
│  timeline, memories, LLM   │                                           │  Memory (per employee) + Nightly Reflection  │
│  calls, experiments        │                                           │  LLM Gateway ──▶ OpenRouter (any model)      │
└────────────────────────────┘                                           │  Store: SQLite (WAL) events/snapshots/memory │
                                                                         └──────────────────────────────────────────────┘
```

**Hard rules:**
- The LLM never mutates state.
- The HTTP/WebSocket layer never mutates state directly. Player commands become recorded *inputs* that the engine processes.
- Anything random uses seeded RNG streams.
- Every LLM call goes through one gateway that records it.

---

## 3. Simulation engine

### 3.1 Loop and time

**SimClock:**
- Stores integer milliseconds since a UTC world epoch. Calendar math uses `zoneinfo`, and the [`holidays`](https://holidays.readthedocs.io/en/latest/) package supplies holidays (it can be subclassed for company holidays).
- Modes:

  | Mode | Behavior |
  |---|---|
  | `paused` | Time stands still |
  | `realtime(scale)` | Configurable: 1× up to accelerated |
  | `lockstep` | Advances only when no LLM decision is pending. Deterministic given recorded responses; used for experiments and tests |
  | `skip_idle` | Fast-forwards nights and weekends |

- The **engine** hands employees ready-made time facts: "Tuesday, March 16, 2027, 10:38 AM", "you arrived at 8:54", "175 days at HelpCo", "last spoke with June 47 min ago". Agents never calculate time.

**Scheduler:**
- A heap of timed events: arrivals, task completions, meetings, need thresholds, end of day.
- A 5–10 Hz movement tick runs only while someone is walking.
- Office life is sparse events, so this is cheaper and simpler than a fixed tick. SimPy's `RealtimeEnvironment` is synchronous and awkward alongside async LLM I/O ([docs](https://simpy.readthedocs.io/en/latest/topical_guides/real-time-simulations.html)).

**Latency vs. time scale is a design decision.** At 60×, a 5-second LLM call equals 5 sim-minutes, which breaks the rhythm of a conversation. Options:
- (a) A slower default scale (~10–20×).
- (b) **Conversation time dilation**: participants are "frozen" while they think.
- (c) `lockstep` for experiments.
- Recommendation: (a) + (b) live, (c) for research.

### 3.2 Cognition: when to call the LLM

- **Triggers:**
  - the current intention finished;
  - someone addressed the employee;
  - a new question arrived;
  - an action was denied;
  - a meeting is starting;
  - morning planning;
  - a coarse re-plan every 30–60 sim-minutes.
- **Priority queue** with a concurrency limit. Conversation replies go first.
- **One in-flight decision per employee** (as in AI Town).
- **Staleness check:** each decision records the last event seq it saw. On return the engine re-validates it. If the world changed and it no longer applies, the decision is recorded as `denied{reason: stale}` and the employee is re-triggered.
- **While thinking:** the employee shows a thinking state and autopilot keeps them plausible.

### 3.3 Autopilot (non-LLM behavior) without predefined personalities

- **Universal needs curves**, identical for every employee: energy, hunger, comfort, focus. There are no per-employee trait weights; that would be a hidden personality.
- **Sims-style "smart objects"** advertise what they satisfy ([GMTK on The Sims](https://gmtk.substack.com/p/the-genius-ai-behind-the-sims)).
- **Utility scoring** ([IAUS](https://en.wikipedia.org/wiki/Utility_system)) is used only to:
  1. execute the sub-steps of the LLM's current intention;
  2. fill gaps while the LLM is thinking;
  3. surface interrupts ("you're exhausted") into the next prompt.
- **Every action is tagged** `source: llm | autopilot | schedule`. Track the ratio: if autopilot dominates, it is the personality.

### 3.4 Action pipeline

- **The LLM returns a flat, schema-constrained `ActionRequest`:** `{reason, verb, target, item, text, minutes}`.
- **Engine checks, in order:**
  1. schema / known verb;
  2. IDs exist;
  3. **capability check in code** (file ACLs, rooms, ownership);
  4. physical preconditions (reachable, free seat, co-located, office open).
- **On accept:** record `action.started` and schedule `action.completed`.
- **On reject:** record `action.denied{reason, policy, security}` and give the employee *in-world* feedback ("This folder contains private employee information. You are not authorized to access it.").
- **v0 verbs** (from the brief):
  - `go_to`, `work`, `say`, `use` (coffee machine, printer, couch, meeting table), `take_break`, `wait`, `write_note`;
  - question handling: `claim_question`, `work_on_question`, `submit_answer`;
  - `read`, `give`, `put_down`, `pick_up`, `claim_desk`, `leave_office`.
- **Speech is data.** Claims inside speech become *beliefs with provenance* in listeners' memory, never world facts.

### 3.5 Perception (limited knowledge)

- When an event is written, a perception filter computes its witnesses: `saw | heard | told | read`. Rules are same-room/earshot, line of sight, and private channels (intercom, private files).
- Employees only ever receive events they witnessed. This makes the "Maya and June talk privately; Milo doesn't know" rule structural rather than prompt-based.

### 3.6 Conversations

- Short, casual turns. Hard length caps: ~1–2 lines; brief examples: "did you change the printer settings" / "no".
- Silence is a valid choice.
- A conversation is a lightweight session between co-located employees. It ends when someone stops, leaves or gets pulled away.
- The full transcript is stored and inspectable. The client only shows bubbles.

---

## 4. Self-authored identity (day one)

The employees decide everything about themselves. The engine only supplies options and enforces physical constraints.

### 4.1 What they choose, and what the engine does with it

| Choice | Requirement |
|---|---|
| **Name** | Completely free text; **duplicates allowed**. Implication: stable internal `employee_id`s everywhere. When two employees share a name, perception text disambiguates them ("Maya — the one at desk 2"). |
| **Appearance** | Chosen from the art catalog (part IDs + colors), as structured output validated by the engine. It can change later through actions (and later, purchases). |
| **Pronouns** | Free choice. The engine uses them in all narration and in every *other* employee's context. They can change later (versioned). |
| **Desk** | Picked from free desks after looking around. The engine resolves conflicts (first valid claim wins). |
| **Self-introduction** | A short line spoken on first meeting each coworker. Stored as the employee's first memory. |
| **Personal item** | One starting item from a starter catalog (mug, plant, photo, rubber duck…), placed on their desk. It persists and is the first seed of the office's physical history. |

### 4.2 Onboarding flow

1. **Hire:** the employee receives only the brief's first-day text, plus the rooms, resources and current coworkers.
2. **Choose:** they make the choices above as one structured output with a free-text "why" for each. The *why* becomes their first self-knowledge memory: an origin story.
3. **Arrive:** they walk in through the entrance, look around, claim a desk and meet coworkers, giving their self-introduction.
4. **First night:** the first nightly reflection.

### 4.3 Homogenization (top risk) and mitigations that don't assign personality

**The evidence:**
- LLMs reuse a tiny pool of names. In one test of 1,000 story openings, 76% used names like Elara, Kael, Lyra, Maya, Chen, Thorne ([study](https://huggingface.co/blog/ChuckMcSneed/name-diversity-in-llms-experiment), [the "Elara problem"](https://microblog.christhomas.co.uk/blog/the-elara-bias)).
- Long-running agents converge on one persona ([persona collapse](https://arxiv.org/abs/2604.24698), preprint).
- Lives shrink toward familiar themes ("self-locking", [AutoPersonas](https://arxiv.org/abs/2607.08252), preprint).
- Agent populations show 3–4× less variation than humans ([2608.06485](https://arxiv.org/abs/2608.06485), preprint).

**Mitigations, all compatible with "completely free":**
- **Never include example names or looks** in prompts.
- **Choose in context, not in a vacuum.** Seed *circumstances*, not traits: arrival order, which desks are free, the weather, today's question queue, who's already there.
- **Per-employee model diversity.** OpenRouter makes this trivial. Pin the model per employee; it becomes part of who they are.
- **Evidence-gated self-concept changes** (§5.4). This reduced repeated life themes from 61.8% to 36.3% in AutoPersonas.
- **Measure population diversity from week 1** (names, looks, vocabulary, activity mix, relationship patterns) and treat it as a health metric.

---

## 5. Memory and nightly reflection

### 5.1 Data model (SQLite)

| Table | Key fields |
|---|---|
| `experience` (episodic, append-only, first-person) | `agent_id, sim_t_start/end, location, participants[], text (gist), world_event_ids[] (dev-only ground truth), importance 1–10, valence, strength S, last_access, access_count, tier (raw → episode → chapter → dormant), parent_id, provenance, embedding` |
| `belief` (knowledge, relationships, self, norms) | `kind (world \| person:<id> \| self \| norm), subject, statement, confidence 0–1, valid_from/valid_to, adopted_at/abandoned_at, superseded_by, evidence_ids[], contradicts_ids[], provenance` |
| `provenance` (embedded) | `channel (witnessed \| told \| overheard \| rumor \| read_artifact \| player \| inferred \| reflected), source_agent_id, hops, source_artifact_id, original_hedge` |
| `relationship_card` (per ordered pair) | `familiarity, warmth, trust_competence, trust_honesty, last_seen, top beliefs` |
| `identity_version` | `name, pronouns, appearance, self_concept (≤300 tokens), values[], valid_from/valid_to, change_reason, evidence_ids[]` |
| `goal` | `text, horizon, status, origin_evidence, progress` |
| `biography_chapter` | `period, summary, key_event_ids` |
| `artifact` | Notes, files, objects: `content, author, created_at, location` (the world as memory; carries culture to new hires) |

- **Two employees remember an event differently** because each writes its own first-person record from its own perception.
- **"Old identity becomes history"** is `identity_version` with validity intervals, the bitemporal pattern from [Graphiti](https://github.com/getzep/graphiti).

### 5.2 Retrieval (every LLM call)

1. **Candidates:** BM25 (SQLite FTS5) plus vector kNN (sqlite-vec) plus an entity filter for the people and objects present. About 50 candidates, fused with Reciprocal Rank Fusion ([hybrid search in SQLite](https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html)).
2. **Score:** `0.5·recency + 3·relevance + 2·importance (+ activation)`, each min-max normalized.
   - These are the weights in the *released* Generative Agents code ([retrieve.py](https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/retrieve.py)).
   - Note the paper and the code differ: the paper uses 0.995^game-hours, the code uses rank-based 0.99^i. Choose sim-time recency deliberately.
3. **Weight by belief confidence.**
4. **Keep provenance visible:** render "Bo told you…" and "you heard a rumor…" in the prompt, never as flat facts.
5. **Retrieval reinforces:** `S += 1` and `last_access` is updated.

**Bounded prompt:** it should be the same size in year 3 as on day 1. Roughly:
- identity ≈300 tokens (a stable, cacheable prefix);
- goals ≈150;
- relationship cards of the people present;
- ≤800 tokens of memories;
- ≈300 tokens of needs and world state.

Old history is reachable only through retrieval over day → week → chapter summaries. [Lifelong Sotopia](https://arxiv.org/abs/2506.12666) (preprint) shows full history in context *reduces* believability over time.

**Embeddings:** OpenRouter now serves an OpenAI-compatible `/embeddings` endpoint. Store the embedding model ID on every vector, since changing models means re-embedding. Keyword search is better for names and objects, and vectors are better for paraphrase, so use both. v0 can start with FTS5 alone.

### 5.3 Nightly consolidation (per employee, after they go home)

1. **Importance:** batch-score the day's raw experiences (one call, not one per memory).
2. **Episodes:** group events into episodes and write gists. Keep raw text only for importance ≥7.
3. **Belief extraction:** Mem0-style add / update / invalidate / no-op against retrieved beliefs ([Mem0](https://arxiv.org/abs/2504.19413)). Invalidate, never delete.
   - **Preserve hedges and attribution** ("Deven *said* Maya criticized my work"), and cap confidence at the trust level of the source.
   - Without this, consolidation turns hearsay into "fact" ([Manufactured Confidence](https://arxiv.org/abs/2606.29279), preprint).
4. **Relationships:** update the relationship cards for everyone encountered today, with at most ~2 new beliefs per person per night.
5. **Reflection:** Park-style. Ask 3 focal questions, write insights with evidence citations ([reflect.py](https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/reflect.py)).
6. **Self-concept:** propose a change only when a theme is supported across ≥k days, then commit a new `identity_version`. At most ~1 per sim-week unless something major happens.
7. **Goals:** update goals and sketch tomorrow's intentions.
8. **Forgetting:** `R = exp(−Δt / (τ·S·(1+importance)))`, based on [MemoryBank](https://arxiv.org/abs/2305.10250) / Ebbinghaus.
   - Below threshold 1, the memory is folded into its parent gist.
   - Below threshold 2, it becomes `dormant`: the employee can't retrieve it, but it stays in the database for research. A physical note can "remind" them and re-activate it.
9. **Weekly/monthly:** write biography chapters and merge duplicate beliefs. Always summarize **from evidence links**, not from summaries of summaries, to avoid semantic drift ([SSGM](https://arxiv.org/abs/2603.11768), preprint).

**Related pattern:** Letta's [sleep-time agents](https://www.letta.com/blog/sleep-time-compute/) use a separate process that rewrites the main agent's memory while it's idle. HelpCo's night is exactly that window.

### 5.4 Culture emergence (requirement, not a feature)

- **Culture needs shared, persistent social memory.** Agent-only social networks without it showed "scalability without socialization" ([Moltbook study](https://arxiv.org/abs/2602.14299), preprint). A small shared, *decaying* artifact store did produce roles, rituals and myth-making ([2606.30668](https://arxiv.org/abs/2606.30668), preprint).
- **So artifacts are first-class:** notes, the shared drive, objects left behind, and a desk that keeps its plant after its owner leaves.
- **Measure it:** does a norm or joke persist after everyone who started it is gone?

---

## 6. LLM layer (OpenRouter)

### 6.1 Client

**Recommendation: a thin in-house async client on `httpx`** (~300 lines), with `uv lock` including hashes.
- **Why not the official [`openrouter` SDK](https://pypi.org/project/openrouter/)** (v1.2.x):
  - It is Speakeasy-generated and churns fast (hundreds of releases since Nov 2025).
  - It pins `pydantic<2.13`.
  - It is published without PyPI provenance attestations.
- **Why not gateway meta-libraries like LiteLLM:** the March 2026 [LiteLLM PyPI compromise](https://docs.litellm.ai/blog/security-update-march-2026) shipped a credential stealer.
- **What owning the request dict buys:** canonical hashing for record/replay and direct access to cost and metadata headers.

### 6.2 Request shape (decision call)

```json
{
  "model": "<primary slug>",
  "models": ["<primary slug>", "<fallback slug>"],
  "messages": [
    {"role": "system", "content": [
      {"type": "text", "text": "<office rules + action reference>"},
      {"type": "text", "text": "<identity + long-term memory>", "cache_control": {"type": "ephemeral"}}]},
    {"role": "user", "content": "<time facts, perception, retrieved memories, options>"}
  ],
  "response_format": {"type": "json_schema", "json_schema": {"name": "action", "strict": true, "schema": {}}},
  "plugins": [{"id": "response-healing"}],
  "provider": {"require_parameters": true, "data_collection": "deny"},
  "max_completion_tokens": 300,
  "session_id": "<run>:<employee>",
  "user": "<save id hash>"
}
```

**Key facts** (from OpenRouter's OpenAPI spec in [OpenRouterTeam/python-sdk](https://github.com/OpenRouterTeam/python-sdk) and the docs):
- `models[]` falls back on any error. You are billed for the model that answered.
- `provider.require_parameters: true` is **essential**. Without it, providers that don't support structured outputs silently drop them.
- `usage.cost` and cached/reasoning token counts come back on every response.
- `session_id` gives sticky routing, so the prompt cache gets hits.
- Prompt caching is automatic for OpenAI, DeepSeek and Gemini, and needs explicit `cache_control` for Anthropic and Qwen. Minimum cacheable prefixes vary, e.g. **4,096 tokens for Claude Haiku 4.5**, so 2–3k-token prompts never cache there.
- Default provider routing is load-balanced and **non-deterministic**. Pin providers for experiments.

### 6.3 Routing per task (overridable per employee)

| Task | Tier | Notes |
|---|---|---|
| Action decisions | cheap | JSON; minimal reasoning; trigger-based (§3.2) |
| Conversation turns | cheap | ≤ ~200 tokens, e.g. `{utterance, end}` |
| Answering player questions | mid | Plain text first; low/medium reasoning |
| Nightly reflection | mid (frontier optional) | Free-text reflection first, then JSON extraction ([format restrictions can hurt reasoning](https://arxiv.org/abs/2408.02442)) |
| Onboarding (identity choice) | employee's pinned model | Same model they'll live with |
| Embeddings | `/embeddings` | Cached locally forever |

**Budget guards:**
- One OpenRouter key per environment with a daily credit `limit`, which gives a hard stop (HTTP 402).
- A per-employee and per-run $ ledger built from `usage.cost`. At 80% of budget, degrade to the cheap tier or autopilot.
- Always set `max_completion_tokens`.
- Watch models with mandatory reasoning tokens.
- Read live prices from `/models` at startup; never hard-code them.

### 6.4 Structured output, defensively

A flat schema (≤ ~8 fields, `enum` verbs, nullable via `["string","null"]`, no `oneOf`), then:
- `strict` + `require_parameters`;
- local validation (Pydantic `extra="forbid"`) plus semantic checks;
- **one** repair retry that includes the validator error;
- then a different-family fallback model;
- then a deterministic default action logged as `llm_failure`.

The sim never blocks. The valid-output rate per model is itself a research metric.

### 6.5 Reproducibility: record, don't hope

- **temperature 0 is not deterministic** (batch-size effects on shared inference; [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)).
- `seed` is best-effort at most.
- Some models (e.g. Claude Sonnet 5 / Opus 5) reject `temperature` entirely, so sampling can't be held constant across families.

**Therefore:**
- **LLM cassette:** key = `sha256(canonical request − volatile fields + replicate index)`. Stored with the response, usage, cost, served provider/model, latency and generation ID.
- **Modes:**
  - `live`, `record`;
  - `replay-strict` (a cache miss is an error);
  - `replay-until-divergence`: a cheap counterfactual prefix; never use it for variance estimates.
- **Experiments:**
  - Exact model slugs (no `~latest`, no `openrouter/auto`).
  - Pinned providers.
  - **N replicates per condition, compared statistically.** [Emergence World](https://arxiv.org/abs/2606.08367) (preprint) found identical setups diverge radically across models.

### 6.6 Cost model (per employee per sim-day)

**Assumptions:**
- 60 decisions at 2.5k in / 150 out;
- 20 chat turns at 2k / 100;
- 5 answers at 3k / 400;
- 1 reflection at 8k / 1.5k.

That totals ≈213k input and 14.5k output tokens per employee-day. The caching estimate assumes 60% of input is a cacheable prefix.

**Prices are (indirect) list prices as of Sept 2026.** Re-check them via `/models`.

| Setup | $/M in / out | Per employee-day (cached) | 6 employees × 30 days |
|---|---|---|---|
| Cheap model everywhere (e.g. GPT-5.6 Luna class) | ~0.20 / 1.20 | ~$0.04 | ~$7 |
| Mid model everywhere (e.g. Claude Sonnet 5) | 2 / 10 | ~$0.36 | ~$66 |
| Frontier everywhere (e.g. Claude Opus 5) | 5 / 25 | ~$0.91 | ~$164 |
| **Mixed** (cheap decisions + chat; mid answers + reflection) | — | **~$0.10** | **~$18** |

- **Prototype** (2 employees, mixed tiers): ≈ $0.20 per sim-day.
- **Reasoning tokens** can double mid/frontier costs.
- **Caching** saves ~35–40%.

---

## 7. Persistence, replay and experiments

### 7.1 Storage

SQLite in WAL mode (single writer, concurrent readers), one file per run/branch. DuckDB `ATTACH`es these files for analytics.

| Table | Contents |
|---|---|
| `events` | `seq, sim_time_ms, wall_time, type, actor_id, targets, location_id, payload, cause_seq, decision_id, visibility, schema_version` |
| `event_witnesses` | `seq, agent_id, mode` |
| `inputs` | Untrusted external input with source and trust level |
| `llm_calls` | The cassette (§6.5) |
| `snapshots` | Every sim midnight, every ~5k events, and on shutdown. Stores `engine_version` and `state_hash`, zstd-compressed |
| `runs` | `parent_run, fork_seq` |

### 7.2 Replay, re-simulation and experiments

**Three different things:**

| Term | What it does | How exact |
|---|---|---|
| **Replay** | Fold recorded events back into state; "rewind and watch an old day" streams stored events to the client over the same protocol as live play | Exact; no LLM calls |
| **Re-simulation** | Re-run the engine against the cassette | Exact for the same engine version |
| **Experiment** | Fresh LLM calls | Statistical |

**Supporting requirements:**
- Version events, keep reducers backward-compatible, and CI-test that replaying between two snapshots reproduces the same `state_hash`.
- **RNG:** use NumPy `SeedSequence.spawn` per purpose. Python's `random.choice`/`shuffle` are not guaranteed stable across versions ([docs](https://docs.python.org/3/library/random.html)).

**Experiment runner:** a TOML spec defines:
- the base snapshot and the model/provider pins;
- conditions (e.g. control vs. inject "Maya said you're bad at your job" to Milo on Mon 10:00);
- N replicates, duration, $ cap, and metrics.

Runs are headless `lockstep` processes, each with its own SQLite file and a manifest (git SHA, prompt hashes, models and providers actually used).

**Metrics:**
- information diffusion of planted facts (reach, depth, time), traced through provenance ([OASIS](https://arxiv.org/abs/2411.11581));
- relationship graphs;
- denied-action rates;
- canary leaks;
- cost per sim-day;
- Sotopia-style judge rubrics ([Sotopia](https://arxiv.org/abs/2310.11667)) spot-checked by humans, because LLM judges have known biases.

---

## 8. Godot client

| Area | v0 requirement |
|---|---|
| Engine | Godot **4.7.2**, typed GDScript (no C#, which keeps web export possible), **Compatibility (OpenGL) renderer** (most reliable for transparent windows) |
| Look | **Decided: a fully 3D isometric low-poly diorama** ("option 9" of the look exploration): lo-fi rainy-dusk lighting, real shadows, orthographic camera from one corner with the two back walls tall and the rest cut away. Everything is built in code from primitives (no asset files), the same way the pixel art was. The 2D pixel-art direction below is kept for the browser inspector |
| World | Built from the server's office layout (map + objects): 1 tile = 1 world unit, floors per room, tall back walls, half walls with glass partitions, furniture per object kind, extra lived-in clutter on free tiles |
| Pathfinding | Owned by the **backend** (it decides reality); the client follows waypoints. `AStarGrid2D` is only for an offline demo mode |
| Networking | Raw `WebSocketPeer` (not `WebSocketMultiplayerPeer`). Envelope `{v, seq, type, sim_t, payload}`. Snapshot on connect, `resume{last_seq}` on reconnect, exponential backoff. **Raise `inbound_buffer_size` or chunk snapshots** (the 64 KiB default silently breaks the connection). **Send IDs as strings** (Godot parses all JSON numbers as floats). No msgpack in v0 |
| Movement | Backend sends `move{id, path, t0_sim, speed}`; the client interpolates at `sim_now − ~150 ms` using periodic `clock{sim_t, scale}` sync |
| Characters | 3D chibis assembled from each employee's chosen look (skin, hair style/color, sweater, pants, shoes, glasses, frog hat) with procedural animation: walk, sit, type, sip, carry, talk, glance, blink |
| Bubbles | Nine-patch PanelContainer + RichTextLabel, width-capped, 2–3 lines max, typewriter via `visible_ratio`, 4.7's `offset_transform` for the pop-in. **Emoji:** Godot doesn't render COLR fonts (Windows' system emoji font), so map a curated emoji set to pixel icons with `[img height=1em]` or bundle Noto Color Emoji |
| Inspector | Side panel: Profile · Now (intent → outcome) · Memory · Conversations; full logs in RichTextLabel |
| **Corner-of-monitor mode** | Per-pixel transparency + borderless + always-on-top + `window_set_mouse_passthrough(polygon)`; tray `StatusIndicator` (Windows/macOS); cap 30 fps, 10–15 when unfocused. **Wayland can't do always-on-top or click-through**. Test Windows/macOS/X11 early. Reference: [Desktop-Pet](https://github.com/phanstudio/Desktop-Pet/) |

---

## 9. Art pipeline: how to get good, high-definition pixel art

### 9.1 "High-def pixel art"

This needs pinning down, because it drives cost:
- **Native 32 px (or larger) sprites.** More detail per character, but every part × animation × direction costs ~4× as much to make.
- **16 px sprites + modern rendering.** Crisp UI, smooth lighting, particles and a day/night tint around 16 px art. This is how most cozy indie games get a "modern" look cheaply. LimeZu's 32/48 px versions are upscales of the 16 px art, not extra detail.
- **HD-2D / 2.5D.** Pixel-art characters in a lit 3D office (depth of field, real shadows). Godot can do it with `Sprite3D` billboards. Most striking, most work.

### 9.2 Ways to get the art

| Option | What you get | Cost | License / caveats |
|---|---|---|---|
| **Buy a pack:** [LimeZu Modern Interiors](https://limezu.itch.io/moderninteriors) + [Modern Office](https://limezu.itch.io/modernoffice) | A cozy modern office tileset plus a **character generator**: 100+ outfits, 200 hairstyles, 80 accessories, 9 skin tones, many animations (sit, phone, read). 16/32/48 px | Low (one-time) | Commercial use OK, **credit required, no redistribution**, so keep the assets out of a public repo |
| [Mana Seed](https://seliel-the-shaper.itch.io/character-base) paper-doll | A layered base built around color ramps (fits the recolor system well) | Low–mid | Check the license and whether it has office-appropriate animations |
| [LPC generator](https://github.com/liberatedpixelcup/Universal-LPC-Spritesheet-Character-Generator) | Huge free layered set | Free | CC-BY-SA/GPL mix; per-author credits; **risky for Steam** unless filtered to CC0/OGA-BY; medieval RPG look |
| **AI sprite tools:** [PixelLab](https://www.pixellab.ai/pixellab-api) | Characters, **4/8-direction rotations, skeleton animation**, inpainting for clothes, tilesets, **API** | $12–50/mo; API priced per call | Commercial use on paid plans |
| **AI sprite tools:** [Retro Diffusion](https://retrodiffusion.ai/) | Grid-aligned, palette-limited art; up to 9 reference images to lock a character; animations and tilesets; API + MCP | ~$0.015–0.18/image; animations ~$0.07–0.25 | Commercial use allowed |
| **Commission a pixel artist** ([itch.io](https://itch.io/t/189980/paid-pixel-artist-for-hire), [Upwork](https://www.upwork.com/hire/pixel-art-freelancers/), [Fiverr](https://block.fiverr.com/hire/pixel-art)) | Unique HelpCo style, exactly the parts you need | Simple characters ~$20–70 each; a key animated character **~$600–1,500**; a full modular system costs far more ([pricing overview](https://2dwillneverdie.com/blog/how-much-do-sprites-cost/)) | Negotiate full commercial rights |
| **Claude-made procedural art** | Code-drawn modular sprites: part masks + palette ramps, recolored from data. See the [sample](research/sprite-sample-claude.png) | Free | Yours. Good for a charming prototype and for building the catalog system. **Not** the polish of a skilled pixel artist (animation cycles, multiple directions, fine shading) |

![Claude-made sample: four employees built from the same parts with different catalog choices](research/sprite-sample-claude.png)

### 9.3 Recommendation

1. **Prototype:** use Claude-made procedural sprites *or* LimeZu. Either way, build the **catalog system** first (part IDs + color ramps + layer order), because that's what the AI chooses from and what the engine validates.
2. **Production:** a commissioned artist defines the HelpCo style and the base body and animations. The catalog then grows with parts drawn to that template: commissioned, or AI-generated with PixelLab/Retro Diffusion and touched up in Aseprite.
3. **Later ("generated extras"):** when an employee wants something that doesn't exist ("I desperately need a frog hat"), an image model generates a new catalog item, a human or automated check approves it, and the world's wardrobe grows. OpenRouter now has an [Image API](https://openrouter.ai/docs/guides/overview/multimodal/image-generation), and PixelLab/Retro Diffusion have APIs.

---

## 10. Security, privacy and safety

- **HelpCo has the full "lethal trifecta"** ([Willison](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)): private data (employee files), untrusted input (submitted questions), and an outbound channel (answers delivered to real people).
- **Agents get no real tools:** no network, no code execution. Every capability is a fixed verb the engine validates in code, in the spirit of [CaMeL](https://arxiv.org/abs/2503.18813) and [action-selector / plan-then-execute](https://arxiv.org/abs/2506.08837).
- **Taint labels:** every text blob is labeled `system | agent | external:<id>`. Untrusted text enters prompts only inside delimited data blocks.
- **Canary tokens** in private files; detect them in speech, notes and outbound answers.
- **Output gate** for answers delivered to real people:
  - moderation (e.g. [Llama Guard 4](https://openrouter.ai/meta-llama/llama-guard-4-12b) on OpenRouter);
  - PII and canary scan;
  - optional human review queue;
  - rate limits per submitter.
- **Research attacks run only in forks.** Prompt injection ([AgentDojo](https://arxiv.org/abs/2406.13352)-style corpus), agent-to-agent spread ([Prompt Infection](https://arxiv.org/abs/2410.07283)) and manipulation are measured as attack success rate vs. utility. Map findings to the [OWASP Agentic Top 10](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/).
- **Privacy:**
  - use `data_collection: "deny"`/ZDR routing for real submitters' text;
  - choose retention for questions and answers in logs and traces;
  - never put real personal data into the world;
  - keep OpenRouter keys in the OS keychain.

---

## 11. Observability and evaluation

- **The event log and `llm_calls` are the primary record.**
- **Optional:** OpenTelemetry GenAI spans (`invoke_agent` → `chat` → `execute_tool`) with `helpco.sim_time`/`event_seq`/`decision_id`, exported to a local [Phoenix](https://arize.com/docs/phoenix/) container. The spec is still marked "Development".
- **The browser inspector shows:**
  - "what this employee knows and from whom";
  - the prompt and response behind any decision;
  - a timeline scrubber;
  - the denial log;
  - the cost ledger.
- **Believability probes:**
  - Park-style interviews (self-knowledge, memory, plans, reactions, reflections), ranked by humans;
  - automated memory Q&A checked against the Game Master log (hallucination rate);
  - persona-consistency metrics.

---

## 12. Packaging and distribution

**Local desktop, recommended for v0:**
- Godot export plus a Python sidecar started with `OS.create_process`.
- Bind to `127.0.0.1` with a per-launch auth token.
- Kill the backend on window close, **and** have it exit on its own when its parent disappears. Child processes otherwise outlive Godot.
- Bundle Python with PyInstaller *one-folder* mode (onefile triggers antivirus false positives), Nuitka, or python-build-standalone plus a uv-locked environment.
- Code-sign everything; notarize on macOS.

**Keys:**
- Bring-your-own OpenRouter key via **OAuth PKCE**: the player authorizes in the browser and gets a user-controlled key with a credit limit.
- Localhost callbacks work on any port.

**Hosted backend (later):**
- Enables a web client and multi-device viewing.
- Costs: servers, accounts, and relaying LLM spend.

**Steam:**
- Requires disclosing live-generated AI content and its guardrails.
- Avoid CC-BY-SA art.

**Offline progression:**
- v0: pause when closed, plus `skip_idle` for nights and weekends.
- v1: an optional headless "office keeps running" service mode with a daily $ cap.
- Later: cheap one-call "summarized days", tagged `fidelity: summary` so experiments can exclude them.

---

## 13. Phased roadmap

### v0: the brief's first milestone

**World:**
- One office, 2 employees, 4 desks, coffee machine, printer, couch, meeting table, entrance, question queue.

**Simulation:**
- Event log, snapshots, perception filter, the action pipeline with v0 verbs, and autopilot needs.
- Clock modes, with a configurable scale.

**Employees:**
- Day-one self-authored identity (§4).

**AI and memory:**
- OpenRouter gateway with cassette and budgets.
- FTS5 retrieval.
- Nightly consolidation, steps 1–8.

**Client and tools:**
- Godot client: office, characters, movement, bubbles, HUD, inspector.
- Browser inspector.
- Mock "offline brain" for tests and key-less demo runs.

**Milestone test:** watch 2 employees through several workdays, and verify they remember yesterday.

### v1

- Embeddings + hybrid retrieval.
- Relationship cards in the UI.
- Biography chapters.
- Corner-of-monitor mode.
- Experiment runner (fork + N replicates).
- Answer output gate.
- More employees (~6).
- Hire/leave lifecycle.
- Economy basics: salary, shop, gifts.

### Later

- Generated catalog items.
- A graph layer for associative recall.
- Culture analytics.
- Summarized offline days.
- Hosted mode.
- Replay viewer with a scrubber.
- Security research scenarios.

---

## 14. Open questions

1. **Art target:** which "high-def" (§9.1)? Buy a pack, commission, or AI-generate for the base style? Claude-made sprites for the prototype?
2. **Where does the backend run?** Local sidecar (BYO OpenRouter key) vs. hosted.
3. **Default time scale and conversation time dilation** (§3.1).
4. **Does the office keep living while the app is closed?** Pause, service mode, or summarized days (§12).
5. **Who are the question askers?** Only the player, or real outside people? If outside people, answers need an output gate and possibly human review (§10).
6. **Model assignment:** one model for everyone, or a different model per employee (helps diversity; changes "who they are" if swapped)?
7. **Is an LLM ever allowed to resolve ambiguous outcomes** (Concordia-style), or is the world engine always pure code? If LLM resolution is allowed, those events must be logged with `resolver: "llm"`.

---

## Sources

**Agents, memory, societies**
- Generative Agents ([paper](https://arxiv.org/abs/2304.03442), [code](https://github.com/joonspk-research/generative_agents)); Generative Agent Simulations of 1,000 People ([arXiv 2411.10109](https://arxiv.org/abs/2411.10109))
- AI Town ([repo](https://github.com/a16z-infra/ai-town), [ARCHITECTURE.md](https://github.com/a16z-infra/ai-town/blob/main/ARCHITECTURE.md), [memory.ts](https://github.com/a16z-infra/ai-town/blob/main/convex/agent/memory.ts))
- Concordia ([repo](https://github.com/google-deepmind/concordia), [paper](https://arxiv.org/abs/2312.03664), [changelog](https://github.com/google-deepmind/concordia/blob/main/CHANGELOG.md))
- Project Sid / PIANO ([arXiv 2411.00114](https://arxiv.org/abs/2411.00114), [MIT Technology Review](https://www.technologyreview.com/2024/11/27/1107377/a-minecraft-town-of-ai-characters-made-friends-invented-jobs-and-spread-religion/))
- Humanoid Agents ([2310.05418](https://arxiv.org/abs/2310.05418)), AgentSociety ([2502.08691](https://arxiv.org/abs/2502.08691)), Sotopia ([2310.11667](https://arxiv.org/abs/2310.11667)), Lifelong Sotopia ([2506.12666](https://arxiv.org/abs/2506.12666)), Lyfe Agents ([2310.02172](https://arxiv.org/abs/2310.02172)), Affordable Generative Agents ([2402.02053](https://arxiv.org/abs/2402.02053))
- Letta sleep-time compute ([paper](https://arxiv.org/abs/2504.13171), [blog](https://www.letta.com/blog/sleep-time-compute/)), Zep/Graphiti ([paper](https://arxiv.org/abs/2501.13956), [repo](https://github.com/getzep/graphiti)), Mem0 ([2504.19413](https://arxiv.org/abs/2504.19413)), A-MEM ([2502.12110](https://arxiv.org/abs/2502.12110)), HippoRAG 2 ([2502.14802](https://arxiv.org/abs/2502.14802)), MemoryBank ([2305.10250](https://arxiv.org/abs/2305.10250))
- 2026 preprints: Moltbook ([2602.14299](https://arxiv.org/abs/2602.14299), [2602.20059](https://arxiv.org/abs/2602.20059)), minimal culture ([2606.30668](https://arxiv.org/abs/2606.30668)), Emergence World ([2606.08367](https://arxiv.org/abs/2606.08367)), Manufactured Confidence ([2606.29279](https://arxiv.org/abs/2606.29279)), belief-based memory ([2606.22030](https://arxiv.org/abs/2606.22030)), SSGM ([2603.11768](https://arxiv.org/abs/2603.11768)), persona collapse ([2604.24698](https://arxiv.org/abs/2604.24698)), AutoPersonas ([2607.08252](https://arxiv.org/abs/2607.08252)), personality change ([2608.06485](https://arxiv.org/abs/2608.06485)), ID-RAG ([2509.25299](https://arxiv.org/abs/2509.25299))
- Name bias: [HF name-diversity experiment](https://huggingface.co/blog/ChuckMcSneed/name-diversity-in-llms-experiment), [The Elara problem](https://microblog.christhomas.co.uk/blog/the-elara-bias), [Sci-fi naming problem](https://glaforge.dev/posts/2025/07/22/the-sci-fi-naming-problem-are-llms-less-creative-than-we-think/)

**LLM layer**
- OpenRouter: [API overview](https://openrouter.ai/docs/api-reference/overview), [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection), [model fallbacks](https://openrouter.ai/docs/guides/routing/model-fallbacks), [structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs), [prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching), [limits](https://openrouter.ai/docs/api_reference/limits), [ZDR](https://openrouter.ai/docs/guides/features/zdr), [embeddings](https://openrouter.ai/docs/api_reference/embeddings), [image generation](https://openrouter.ai/docs/guides/overview/multimodal/image-generation), [Python SDK](https://github.com/OpenRouterTeam/python-sdk) / [PyPI](https://pypi.org/project/openrouter/)
- [LiteLLM compromise](https://docs.litellm.ai/blog/security-update-march-2026), [Datadog analysis](https://securitylabs.datadoghq.com/articles/litellm-compromised-pypi-teampcp-supply-chain-campaign/)
- [Defeating nondeterminism in LLM inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), [JSONSchemaBench](https://arxiv.org/abs/2501.10868), [Let Me Speak Freely?](https://arxiv.org/abs/2408.02442)
- Prices (indirect): [OpenRouter pricing writeup](https://omidsaffari.com/blog/openrouter-pricing), [TrueFoundry](https://www.truefoundry.com/blog/openrouter-pricing), [Sonnet 5](https://openrouter.ai/anthropic/claude-sonnet-5), [Opus 5](https://openrouter.ai/anthropic/claude-opus-5)

**Engine, data, security**
- [SimPy realtime](https://simpy.readthedocs.io/en/latest/topical_guides/real-time-simulations.html), [Event Sourcing (Fowler)](https://martinfowler.com/eaaDev/EventSourcing.html), [SQLite WAL](https://sqlite.org/wal.html), [DuckDB SQLite extension](https://duckdb.org/docs/lts/core_extensions/sqlite), [sqlite-vec hybrid search](https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html), [NumPy parallel RNG](https://numpy.org/doc/2.2/reference/random/parallel.html)
- [OTel GenAI agent spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md), [Inspect AI](https://inspect.aisi.org.uk/), [OASIS](https://arxiv.org/abs/2411.11581)
- [Lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/), [CaMeL](https://arxiv.org/abs/2503.18813), [design patterns vs. prompt injection](https://arxiv.org/abs/2506.08837), [AgentDojo](https://arxiv.org/abs/2406.13352), [Prompt Infection](https://arxiv.org/abs/2410.07283), [OWASP Agentic Top 10](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)

**Client and art**
- Godot: [releases](https://github.com/godotengine/godot/releases), [4.7 features](https://github.com/godotengine/godot-website/blob/master/_data/release_4_7/features.yml), [ProjectSettings (4.7)](https://github.com/godotengine/godot/blob/4.7-stable/doc/classes/ProjectSettings.xml), [WebSocketPeer](https://github.com/godotengine/godot/blob/4.7-stable/modules/websocket/doc_classes/WebSocketPeer.xml), [AStarGrid2D](https://github.com/godotengine/godot/blob/4.7-stable/doc/classes/AStarGrid2D.xml), [DisplayServer](https://github.com/godotengine/godot/blob/4.7-stable/doc/classes/DisplayServer.xml), [multiple resolutions](https://github.com/godotengine/godot-docs/blob/master/tutorials/rendering/multiple_resolutions.rst), [inbound buffer issue #84423](https://github.com/godotengine/godot/issues/84423)
- [Aseprite Wizard](https://github.com/viniciusgerevini/godot-aseprite-wizard), [palette-swap shader](https://github.com/KoBeWi/Godot-Palette-Swap-Shader), [Desktop-Pet](https://github.com/phanstudio/Desktop-Pet/), [entity interpolation](https://www.gabrielgambetta.com/entity-interpolation.html)
- Art: [LimeZu Modern Interiors](https://limezu.itch.io/moderninteriors), [LimeZu Modern Office](https://limezu.itch.io/modernoffice), [Mana Seed](https://seliel-the-shaper.itch.io/character-base), [LPC generator](https://github.com/liberatedpixelcup/Universal-LPC-Spritesheet-Character-Generator), [PixelLab](https://www.pixellab.ai/pixellab-api), [Retro Diffusion](https://retrodiffusion.ai/), [AI sprite tools compared](https://ludo.ai/compare/best-ai-sprite-generators), [sprite commission costs](https://2dwillneverdie.com/blog/how-much-do-sprites-cost/), [Upwork pixel artists](https://www.upwork.com/hire/pixel-art-freelancers/)
- Similar projects: [AgentOffice](https://github.com/harishkotra/agent-office), [Pixel Office](https://pixeloffice.org/), [Agent Pixels](https://www.agent-pixels.com/blog/tutorials/agent-pixels-launch-day), [The Office](https://www.avilevi.co.il/en/projects/the-office/)
