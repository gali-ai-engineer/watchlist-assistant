"""API tests through Starlette's test client (no network, no background loop)."""

from __future__ import annotations

import unittest

from starlette.testclient import TestClient

from app.config import Settings
from app.main import create_app


class TestAPI(unittest.TestCase):
    def setUp(self):
        app = create_app(Settings(db_path=":memory:"), start_background=False)
        self.addCleanup(app.state.assistant.store.close)
        self.ctx = TestClient(app)
        self.c = self.ctx.__enter__()
        for _ in range(12):
            self.c.post("/api/refresh")

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def test_health_and_config(self):
        self.assertEqual(self.c.get("/api/health").json()["status"], "ok")
        cfg = self.c.get("/api/config").json()
        self.assertTrue(cfg["synthetic_data"])
        self.assertIn("heuristic", cfg["llm"])

    def test_alerts_and_priority_filter(self):
        alerts = self.c.get("/api/alerts").json()
        push = self.c.get("/api/alerts?priority=push").json()
        self.assertGreater(len(alerts), len(push))
        self.assertTrue(all(a["priority"] == "push" for a in push))
        self.assertIn("enrichment", alerts[0]["article"])

    def test_mark_read(self):
        a = self.c.get("/api/alerts").json()[0]
        self.assertEqual(self.c.post(f"/api/alerts/{a['id']}/read").status_code, 204)
        self.assertTrue(next(x for x in self.c.get("/api/alerts").json() if x["id"] == a["id"])["read"])

    def test_watchlist_add_validate_remove(self):
        r = self.c.post("/api/watchlist", json={"ticker": "$tsla"})
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.json()["ticker"], "TSLA")
        self.assertEqual(r.json()["backfill"]["new_alerts"], 1)
        self.assertEqual(self.c.post("/api/watchlist", json={"ticker": "<script>"}).status_code, 422)
        self.assertEqual(self.c.delete("/api/watchlist/TSLA").status_code, 204)
        self.assertNotIn("TSLA", [w["ticker"] for w in self.c.get("/api/watchlist").json()])

    def test_chat(self):
        r = self.c.post("/api/chat", json={"question": "What's happening with Amazon?"}).json()
        self.assertTrue(r["citations"])
        self.assertEqual(r["tickers"], ["AMZN"])
        self.assertEqual(self.c.post("/api/chat", json={"question": "  "}).status_code, 422)

    def test_digest_groups_by_ticker(self):
        d = self.c.get("/api/digest").json()
        self.assertTrue(d)
        self.assertTrue(all(isinstance(v, list) for v in d.values()))

    def test_metrics_and_reset(self):
        m = self.c.get("/api/metrics").json()
        self.assertGreater(m["pipeline"]["fetched"], 0)
        self.c.post("/api/demo/reset")
        self.assertEqual(self.c.get("/api/metrics").json()["demo_remaining"], 10)

    def test_index_served(self):
        self.assertIn("Watchlist Assistant", self.c.get("/").text)


if __name__ == "__main__":
    unittest.main()
