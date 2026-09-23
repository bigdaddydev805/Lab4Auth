"""Command line: `helpco serve` runs the live office; `helpco sim` runs days headless."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .config import load_config


def _cfg(args):
    overrides: dict = {"world": {}, "llm": {}, "clock": {}}
    if getattr(args, "save", None):
        overrides["world"]["save_dir"] = str(Path(args.save).resolve())
    if getattr(args, "mock", False):
        overrides["llm"]["provider"] = "mock"
    if getattr(args, "seed", None) is not None:
        overrides["world"]["seed"] = args.seed
    if getattr(args, "start", None):
        overrides["world"]["start"] = args.start
    if getattr(args, "scale", None):
        overrides["clock"]["scale"] = args.scale
    if getattr(args, "model", None):
        overrides["llm"]["routes"] = {t: [args.model] for t in ("onboard", "decide", "converse", "compose", "reflect")}
        overrides["llm"]["provider"] = "openrouter"
    return load_config(args.config, overrides)


def cmd_sim(args) -> None:
    from .engine import Engine
    from .clock import MINUTE

    cfg = _cfg(args)
    cfg.set("clock.mode", "lockstep")

    async def main():
        eng = Engine(cfg)
        lines = []

        def show(msg):
            if msg.get("type") != "event":
                return
            ev = msg["event"]
            if ev["type"] in ("action.started", "employee.entered", "conversation.ended") and not args.verbose:
                return
            line = f"{eng.clock.fmt_stamp(ev['sim_ms'])}  {ev['summary']}"
            if ev["type"] == "day.started":
                line = f"\n=== {ev['summary']} ==="
            lines.append(line)
            if not args.quiet:
                print(line, flush=True)
        eng.listeners.append(show)
        eng.boot()
        for q in args.question or []:
            eng.submit_question(q)
        await eng.run_days(args.days)
        await eng.gateway.close()
        eng.snapshot("sim_end")
        if args.quiet:
            print("\n".join(lines[-40:]))
        print(f"\nsaved to {cfg.save_dir / 'world.db'} — LLM: {eng.gateway.status()}")
        eng.store.close()

    asyncio.run(main())


def cmd_serve(args) -> None:
    from .server.app import serve

    cfg = _cfg(args)
    serve(cfg, host=args.host, port=args.port)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="helpco", description="HelpCo — a tiny living office for autonomous AI employees")
    p.add_argument("--config", help="path to helpco.toml (default: ./helpco.toml if present)")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("serve", help="run the live office (HTTP + WebSocket for the Godot client and inspector)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--save", help="save directory (default from config)")
    s.add_argument("--mock", action="store_true", help="use the offline mock brain")
    s.add_argument("--model", help="use this OpenRouter model slug for every task")
    s.add_argument("--scale", type=float, help="sim seconds per real second")
    s.add_argument("--start", help="calendar start: 'today' or an ISO date")
    s.add_argument("--seed", type=int)
    s.set_defaults(func=cmd_serve)

    m = sub.add_parser("sim", help="run whole days headless and print the timeline")
    m.add_argument("--days", type=int, default=1)
    m.add_argument("--save", default="saves/sim")
    m.add_argument("--mock", action="store_true", help="use the offline mock brain")
    m.add_argument("--model", help="use this OpenRouter model slug for every task")
    m.add_argument("--start", default=None)
    m.add_argument("--seed", type=int)
    m.add_argument("--question", action="append", help="submit a question at the start (repeatable)")
    m.add_argument("--verbose", action="store_true")
    m.add_argument("--quiet", action="store_true")
    m.set_defaults(func=cmd_sim)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if not getattr(args, "func", None):
        args = p.parse_args(["serve"] + (argv or sys.argv[1:]))
    args.func(args)


if __name__ == "__main__":
    main()
