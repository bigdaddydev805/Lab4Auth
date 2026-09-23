"""The simulation clock. The engine owns time; employees are only ever told what time it is.

Time is stored as integer milliseconds since the Unix epoch (UTC) and rendered in the office's
timezone, so the office runs on a real calendar (weekdays, weekends, seasons, anniversaries).
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

MINUTE = 60_000
HOUR = 60 * MINUTE
DAY = 24 * HOUR


def parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def resolve_tz(name: str):
    if name in ("", "local"):
        return datetime.now().astimezone().tzinfo or timezone.utc
    return ZoneInfo(name)


class SimClock:
    def __init__(self, start_ms: int, tz, scale: float = 20.0, mode: str = "realtime"):
        self.ms = int(start_ms)
        self.tz = tz
        self.scale = float(scale)
        self.mode = mode
        self.paused = False

    # --- conversions -----------------------------------------------------------------
    def dt(self, ms: int | None = None) -> datetime:
        return datetime.fromtimestamp((self.ms if ms is None else ms) / 1000, tz=timezone.utc).astimezone(self.tz)

    def ms_of(self, d: date, t: time) -> int:
        local = datetime.combine(d, t).replace(tzinfo=self.tz)
        return int(local.timestamp() * 1000)

    def today(self) -> date:
        return self.dt().date()

    # --- calendar ---------------------------------------------------------------------
    @staticmethod
    def is_workday(d: date) -> bool:
        return d.weekday() < 5

    def next_workday(self, d: date, include_today: bool = False) -> date:
        cur = d if include_today else d + timedelta(days=1)
        while not self.is_workday(cur):
            cur += timedelta(days=1)
        return cur

    # --- formatting (what employees are told) --------------------------------------------
    def fmt_time(self, ms: int | None = None) -> str:
        return self.dt(ms).strftime("%I:%M %p").lstrip("0")

    def fmt_date(self, ms: int | None = None) -> str:
        d = self.dt(ms)
        return f"{d.strftime('%A, %B')} {d.day}, {d.year}"

    def fmt_stamp(self, ms: int | None = None) -> str:
        return self.dt(ms).strftime("%H:%M:%S")

    @staticmethod
    def fmt_span(delta_ms: int) -> str:
        minutes = max(0, int(delta_ms // MINUTE))
        if minutes < 1:
            return "just now"
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        hours = minutes // 60
        if hours < 24:
            rem = minutes % 60
            base = f"{hours} hour{'s' if hours != 1 else ''}"
            return base + (f" {rem} min" if rem and hours < 3 else "")
        days = hours // 24
        if days < 60:
            return f"{days} day{'s' if days != 1 else ''}"
        months = days // 30
        if months < 24:
            return f"{months} month{'s' if months != 1 else ''}"
        return f"{days // 365} years"

    def fmt_ago(self, then_ms: int) -> str:
        span = self.fmt_span(self.ms - then_ms)
        return span if span == "just now" else f"{span} ago"

    def relative_day(self, ms: int) -> str:
        """'today', 'yesterday', 'on Monday', 'on March 3' — how people refer to past days."""
        then = self.dt(ms).date()
        days = (self.today() - then).days
        if days == 0:
            return "today"
        if days == 1:
            return "yesterday"
        if days < 7:
            return f"on {then.strftime('%A')}"
        return f"on {then.strftime('%B')} {then.day}"

    def season(self) -> str:
        m = self.dt().month
        return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                6: "summer", 7: "summer", 8: "summer"}.get(m, "autumn")

    def snapshot(self) -> dict:
        return {"sim_ms": self.ms, "scale": self.scale, "mode": self.mode, "paused": self.paused,
                "tz": str(self.tz), "iso": self.dt().isoformat()}


def resolve_start(start: str, tz, day_starts: time) -> int:
    """'today' -> the next workday morning on the real calendar; otherwise an ISO date/time."""
    if start == "today":
        now = datetime.now(tz)
        d = now.date()
        if not (SimClock.is_workday(d) and now.time() < day_starts):
            d += timedelta(days=1)
            while not SimClock.is_workday(d):
                d += timedelta(days=1)
        return int(datetime.combine(d, day_starts).replace(tzinfo=tz).timestamp() * 1000)
    dt = datetime.fromisoformat(start)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    if len(start) <= 10:  # a bare date: start at the office's morning
        dt = datetime.combine(dt.date(), day_starts).replace(tzinfo=tz)
    return int(dt.timestamp() * 1000)
