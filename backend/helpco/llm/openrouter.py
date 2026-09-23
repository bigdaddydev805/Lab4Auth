"""A thin OpenRouter chat-completions client on aiohttp.

Deliberately small (no gateway meta-library): we own the request dict, which makes hashing for
record/replay trivial and keeps the dependency surface tiny.
"""
from __future__ import annotations

import asyncio
import time

import aiohttp


class OpenRouterError(Exception):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class OutOfCredits(OpenRouterError):
    pass


class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str, app_url: str = "", app_title: str = "HelpCo",
                 timeout_s: float = 60.0, max_retries: int = 2):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": app_url,
            "X-OpenRouter-Title": app_title,
            "X-Title": app_title,
        }
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self.max_retries = max_retries
        self._session: aiohttp.ClientSession | None = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout, headers=self.headers)
        return self._session

    async def chat(self, body: dict) -> tuple[dict, int]:
        """POST /chat/completions. Returns (response json, latency ms)."""
        attempt = 0
        while True:
            t0 = time.monotonic()
            try:
                s = await self.session()
                async with s.post(f"{self.base_url}/chat/completions", json=body) as resp:
                    latency = int((time.monotonic() - t0) * 1000)
                    data = await resp.json(content_type=None)
                    if resp.status == 200 and isinstance(data, dict) and "error" not in data:
                        return data, latency
                    msg = (data or {}).get("error", {}).get("message") if isinstance(data, dict) else str(data)
                    if resp.status == 402:
                        raise OutOfCredits(msg or "out of credits", 402)
                    retryable = resp.status in (408, 429, 500, 502, 503, 504)
                    err = OpenRouterError(f"HTTP {resp.status}: {msg}", resp.status, retryable)
                    if resp.status == 429:
                        await asyncio.sleep(float(resp.headers.get("retry-after", 2 ** attempt)))
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                err = OpenRouterError(f"network: {e!r}", None, True)
            if not err.retryable or attempt >= self.max_retries:
                raise err
            attempt += 1
            await asyncio.sleep(0.5 * 2 ** attempt)

    async def models(self) -> list[dict]:
        s = await self.session()
        async with s.get(f"{self.base_url}/models") as resp:
            data = await resp.json(content_type=None)
            return data.get("data", []) if isinstance(data, dict) else []

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
