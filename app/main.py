"""HTTP API + background pipeline runner.

Run:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import AsyncIterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import __version__, rag
from .companies import company_name
from .config import ROOT, Settings, load_settings
from .llm import build_llm
from .notifiers import Broadcaster, build_notifiers
from .pipeline import Pipeline
from .sources import DemoSource, build_sources
from .storage import Store

log = logging.getLogger("app")
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = Store(settings.db_path)
        self.llm = build_llm(settings)
        self.sources = build_sources(settings)
        self.pipeline = Pipeline(settings, self.store, self.sources, self.llm)
        self.notifiers = build_notifiers(settings)
        self.broadcaster = Broadcaster()
        self.task: asyncio.Task | None = None
        self.paused = False
        if not self.store.get_watchlist():
            for t in settings.default_watchlist:
                self.store.add_ticker(t)

    @property
    def interval(self) -> float:
        return self.settings.demo_interval_s if self.settings.is_demo else self.settings.poll_interval_s

    async def deliver(self, alerts: list[dict]) -> None:
        for al in alerts:
            if al["priority"] == "push":
                for n in self.notifiers:
                    try:
                        await asyncio.to_thread(n.send, al)
                        al["channels"].append(n.name)
                    except Exception as e:  # a failing webhook must not break the feed
                        log.warning("notifier %s failed: %s", n.name, e)
            self.broadcaster.publish("alert", al)

    async def run_and_deliver(self, backfill: list[str] | None = None) -> dict:
        if backfill:
            report = await asyncio.to_thread(self.pipeline.backfill, backfill)
        else:
            report = await asyncio.to_thread(self.pipeline.run_once)
        await self.deliver(report.new_alerts)
        if report.counts:
            self.broadcaster.publish("metrics", self.metrics())
        return {"counts": report.counts, "duration_ms": report.duration_ms, "new_alerts": len(report.new_alerts)}

    async def loop(self) -> None:
        await asyncio.sleep(0.5)
        while True:
            if not self.paused:
                try:
                    await self.run_and_deliver()
                except Exception:
                    log.exception("pipeline run failed")
            await asyncio.sleep(self.interval)

    def metrics(self) -> dict:
        demo = next((s for s in self.sources if isinstance(s, DemoSource)), None)
        return {
            "pipeline": dict(self.pipeline.totals),
            "store": self.store.counts(),
            "last_run": self.pipeline.last_report.started_at if self.pipeline.last_report else None,
            "demo_remaining": demo.remaining if demo else None,
            "paused": self.paused,
            "subscribers": self.broadcaster.subscribers,
        }


def create_app(settings: Settings | None = None, start_background: bool = True) -> Starlette:
    settings = settings or load_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    state = AppState(settings)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        log.info(
            "mode=%s llm=%s sources=%s notifiers=%s",
            settings.mode, state.llm.name if state.llm else "heuristic",
            [s.name for s in state.sources], [n.name for n in state.notifiers],
        )
        if start_background:
            state.task = asyncio.create_task(state.loop())
        yield
        if state.task:
            state.task.cancel()

    # ---- handlers -----------------------------------------------------------

    async def index(_: Request) -> Response:
        return FileResponse(ROOT / "app" / "static" / "index.html")

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok", "version": __version__})

    async def config(_: Request) -> Response:
        return JSONResponse(
            {
                "mode": settings.mode,
                "synthetic_data": settings.is_demo,
                "llm": state.llm.name if state.llm else "heuristic (no API key set)",
                "sources": [s.name for s in state.sources],
                "notifiers": ["in-app", *[n.name for n in state.notifiers]],
                "alert_min_impact": settings.alert_min_impact,
                "min_source_trust": settings.min_source_trust,
                "interval_s": state.interval,
            }
        )

    async def get_watchlist(_: Request) -> Response:
        return JSONResponse([{"ticker": t, "name": company_name(t)} for t in state.store.get_watchlist()])

    async def add_watchlist(request: Request) -> Response:
        body = await request.json()
        ticker = str(body.get("ticker", "")).strip().upper().lstrip("$")
        if not TICKER_RE.match(ticker):
            return JSONResponse({"error": "Enter a ticker symbol like AAPL or BRK.B."}, status_code=422)
        if len(state.store.get_watchlist()) >= 25:
            return JSONResponse({"error": "The watchlist holds up to 25 tickers."}, status_code=422)
        state.store.add_ticker(ticker)
        result = await state.run_and_deliver(backfill=[ticker])
        return JSONResponse({"ticker": ticker, "name": company_name(ticker), "backfill": result}, status_code=201)

    async def remove_watchlist(request: Request) -> Response:
        state.store.remove_ticker(request.path_params["ticker"])
        return Response(status_code=204)

    async def list_alerts(request: Request) -> Response:
        q = request.query_params
        alerts = state.store.list_alerts(limit=min(int(q.get("limit", 100)), 500), ticker=q.get("ticker"))
        if q.get("priority") in {"push", "digest"}:
            alerts = [a for a in alerts if a["priority"] == q["priority"]]
        return JSONResponse(alerts)

    async def read_alert(request: Request) -> Response:
        state.store.mark_alert_read(request.path_params["alert_id"])
        return Response(status_code=204)

    async def digest(_: Request) -> Response:
        """Lower-priority items grouped by ticker: the 'what else happened' view."""
        out: dict[str, list] = {}
        for a in state.store.list_alerts(limit=300):
            if a["priority"] == "digest":
                out.setdefault(a["ticker"], []).append(
                    {"title": a["article"]["title"], "url": a["article"]["url"], "source": a["article"]["source_name"],
                     "summary": (a["article"]["enrichment"] or {}).get("summary")}
                )
        return JSONResponse(out)

    async def chat(request: Request) -> Response:
        body = await request.json()
        question = str(body.get("question", "")).strip()
        if not question:
            return JSONResponse({"error": "Ask a question about your watchlist."}, status_code=422)
        t0 = time.perf_counter()
        result = await asyncio.to_thread(
            rag.answer, state.store, state.llm, question, state.store.get_watchlist(), body.get("history") or []
        )
        result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return JSONResponse(result)

    async def stream(request: Request) -> Response:
        q = state.broadcaster.subscribe()

        async def gen() -> AsyncIterator[str]:
            try:
                yield "event: hello\ndata: {}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        yield await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                state.broadcaster.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def refresh(_: Request) -> Response:
        return JSONResponse(await state.run_and_deliver())

    async def metrics(_: Request) -> Response:
        return JSONResponse(state.metrics())

    async def demo_reset(_: Request) -> Response:
        if not settings.is_demo:
            return JSONResponse({"error": "Reset is only available in demo mode."}, status_code=400)
        state.store.reset()
        state.pipeline.totals.clear()
        state.pipeline.policy._recent_push.clear()
        for s in state.sources:
            if isinstance(s, DemoSource):
                s.rewind()
        state.broadcaster.publish("reset", {})
        return JSONResponse(await state.run_and_deliver())

    async def pause(request: Request) -> Response:
        state.paused = bool((await request.json()).get("paused", True))
        return JSONResponse({"paused": state.paused})

    routes = [
        Route("/", index),
        Route("/api/health", health),
        Route("/api/config", config),
        Route("/api/watchlist", get_watchlist, methods=["GET"]),
        Route("/api/watchlist", add_watchlist, methods=["POST"]),
        Route("/api/watchlist/{ticker}", remove_watchlist, methods=["DELETE"]),
        Route("/api/alerts", list_alerts),
        Route("/api/alerts/{alert_id}/read", read_alert, methods=["POST"]),
        Route("/api/digest", digest),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/stream", stream),
        Route("/api/refresh", refresh, methods=["POST"]),
        Route("/api/metrics", metrics),
        Route("/api/demo/reset", demo_reset, methods=["POST"]),
        Route("/api/pause", pause, methods=["POST"]),
        Mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static"),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.assistant = state
    return app


app = create_app()  # uvicorn app.main:app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
