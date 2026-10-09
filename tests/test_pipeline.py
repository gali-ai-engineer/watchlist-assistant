"""Unit and integration tests. Run: python -m unittest -v  (or: pytest)"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from app import dedupe, matcher, trust
from app.alerts import AlertPolicy
from app.config import Settings
from app.enrich import enrich, heuristic_enrich
from app.llm import LLMError, parse_json
from app.models import Article, EventType, RawArticle
from app.pipeline import Pipeline
from app.rag import answer, tickers_in
from app.sources import DemoSource
from app.storage import Store

NOW = datetime.now(timezone.utc)


class FakeLLM:
    """Stands in for a real provider: returns canned text and records prompts."""

    name = "fake:model"

    def __init__(self, reply: str = "", fail: bool = False):
        self.reply, self.fail, self.prompts = reply, fail, []

    def complete(self, system: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 600) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise LLMError("boom")
        return self.reply


def settings(**kw) -> Settings:
    s = Settings(db_path=":memory:")
    for k, v in kw.items():
        setattr(s, k, v)
    return s


_stores: list[Store] = []


def make_store(tickers=("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")) -> Store:
    st = Store(":memory:")
    _stores.append(st)
    for t in tickers:
        st.add_ticker(t)
    return st


# ---------------------------------------------------------------------------
class TestDedupe(unittest.TestCase):
    def test_canonical_url_strips_tracking_and_www(self):
        a = dedupe.canonical_url("https://www.reuters.com/markets/x/?utm_source=tw&id=7#top")
        b = dedupe.canonical_url("http://reuters.com/markets/x?id=7")
        self.assertEqual(a, b)

    def test_same_story_different_wording_is_duplicate(self):
        s = dedupe.similarity(
            "EU regulators open probe into cloud licensing practices of Microsoft",
            "European Commission opens investigation into Microsoft cloud licensing",
        )
        self.assertGreater(s, 0.6)

    def test_follow_up_story_is_not_duplicate(self):
        s = dedupe.similarity(
            "US weighs new export curbs on advanced Nvidia AI chips to more countries",
            "Nvidia says it will comply with any new US export rules, sees limited near-term revenue impact",
        )
        self.assertLess(s, 0.45)

    def test_find_duplicate_requires_shared_ticker_and_time_window(self):
        cand = [dedupe.Candidate("a1", "Apple beats quarterly revenue estimates", "", ["AAPL"], NOW)]
        self.assertIsNone(dedupe.find_duplicate("Apple beats quarterly revenue estimates", "", ["MSFT"], NOW, cand, 0.45))
        self.assertIsNone(dedupe.find_duplicate("Apple beats quarterly revenue estimates", "", ["AAPL"], NOW + timedelta(days=5), cand, 0.45))
        self.assertEqual(dedupe.find_duplicate("Apple tops quarterly revenue estimates", "", ["AAPL"], NOW, cand, 0.45)[0], "a1")


class TestTrust(unittest.TestCase):
    def test_known_and_subdomains(self):
        self.assertEqual(trust.score("https://uk.reuters.com/x")[1], "Reuters")
        self.assertGreaterEqual(trust.score("https://www.sec.gov/Archives/x")[2], 1.0)

    def test_unknown_is_low_and_name_fallback(self):
        self.assertLess(trust.score("https://randomblog.example/x", "Random Blog")[2], 0.6)
        # Finnhub returns its own redirect URLs; the provider-reported source name is used instead
        self.assertEqual(trust.score("https://finnhub.io/api/news?id=1", "Bloomberg")[1], "Bloomberg")


class TestMatcher(unittest.TestCase):
    WL = ["AAPL", "MSFT", "NVDA"]

    def test_company_name_in_headline(self):
        self.assertEqual(matcher.match("Apple unveils new iPhone", "", [], self.WL), ["AAPL"])

    def test_fruit_is_not_the_company(self):
        self.assertEqual(matcher.match("Washington apple growers expect record harvest", "fruit", [], self.WL), [])

    def test_cashtag_and_exchange_symbol(self):
        self.assertEqual(matcher.match("Chip stocks rally; $NVDA leads", "", [], self.WL), ["NVDA"])
        self.assertIn("MSFT", matcher.match("Deal announced", "The company (NASDAQ: MSFT) said", [], self.WL))

    def test_passing_body_mention_alone_is_not_enough(self):
        self.assertEqual(matcher.match("Chipmaker ABC raises guidance", "Customers include Apple.", [], self.WL), [])


class TestEnrichment(unittest.TestCase):
    def test_heuristic_classifies_common_events(self):
        cases = {
            "Amazon in talks to acquire AI chip startup for $4 billion": (EventType.M_AND_A, 5),
            "Microsoft beats quarterly revenue estimates as Azure growth accelerates": (EventType.EARNINGS, 4),
            "Alphabet announces date for third quarter earnings release": (EventType.EARNINGS, 1),
            "Apple COO to step down at year end": (EventType.LEADERSHIP, 4),
            "Alphabet downgraded to Neutral at brokerage": (EventType.ANALYST, 2),
        }
        for title, (event, impact) in cases.items():
            e = heuristic_enrich(title, "")
            self.assertEqual((e.event_type, e.impact), (event, impact), title)

    def test_sentiment_direction(self):
        self.assertGreater(heuristic_enrich("Tesla deliveries rise, topping estimates", "").sentiment, 0)
        self.assertLess(heuristic_enrich("US weighs new export curbs on Nvidia chips", "").sentiment, 0)

    def test_llm_output_is_parsed_and_clamped(self):
        llm = FakeLLM('```json\n{"summary":"S","sentiment":4,"event_type":"M&A","impact":9,"why_it_matters":"W"}\n```')
        e = enrich(llm, "Amazon to buy X", "", ["AMZN"], "Reuters", 0.95)
        self.assertEqual((e.sentiment, e.impact, e.event_type, e.model), (1.0, 5, EventType.M_AND_A, "fake:model"))

    def test_llm_failure_falls_back_to_heuristic(self):
        e = enrich(FakeLLM(fail=True), "Amazon to acquire X", "", ["AMZN"], "Reuters", 0.95)
        self.assertEqual(e.model, "heuristic")
        self.assertEqual(e.event_type, EventType.M_AND_A)

    def test_parse_json_tolerates_prose(self):
        self.assertEqual(parse_json('Sure! {"a": 1} hope that helps')["a"], 1)
        with self.assertRaises(LLMError):
            parse_json("no json here")


class TestAlertPolicy(unittest.TestCase):
    def _article(self, impact: int, trust_score: float = 0.9, ticker: str = "NVDA", i: int = 0) -> Article:
        e = heuristic_enrich("x", "")
        e.impact = impact
        return Article(f"id{i}", "Reuters", "reuters.com", trust_score, f"https://r.com/{i}", "t", "s", NOW, [ticker], "demo", e)

    def test_threshold_splits_push_and_digest(self):
        p = AlertPolicy(min_impact=3, min_trust=0.6)
        self.assertEqual(p.decide(self._article(4))[0].priority, "push")
        self.assertEqual(p.decide(self._article(2, i=1))[0].priority, "digest")

    def test_untrusted_never_alerts(self):
        self.assertEqual(AlertPolicy(3, 0.6).decide(self._article(5, trust_score=0.3)), [])

    def test_rate_limit_downgrades_flood_to_digest(self):
        p = AlertPolicy(3, 0.6, max_per_hour=2)
        pr = [p.decide(self._article(5, i=i))[0].priority for i in range(4)]
        self.assertEqual(pr, ["push", "push", "digest", "digest"])
        # an hour later pushes are allowed again
        self.assertEqual(p.decide(self._article(5, i=9), now=NOW + timedelta(hours=1, minutes=1))[0].priority, "push")


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        self.p = Pipeline(settings(), self.store, [DemoSource()], None)

    def drain(self):
        for _ in range(20):
            self.p.run_once()

    def test_end_to_end_on_demo_data(self):
        self.drain()
        t = self.p.totals
        total = len(json.loads(DemoSource().path.read_text())["items"])
        self.assertEqual(t["fetched"], total)
        self.assertEqual(t["untrusted"], 3)  # two tip sites + one rumour blog
        self.assertEqual(t["unmatched"], 3)  # Fed minutes, apple harvest, Tesla (not on watchlist)
        self.assertEqual(t["dup_story"], 3)  # EU probe, Apple COO, MSFT earnings repeats
        self.assertEqual(t["fetched"], t["untrusted"] + t["unmatched"] + t["dup_story"] + t["enriched"])
        self.assertGreaterEqual(t["alert_push"], 5)

    def test_duplicates_are_recorded_as_also_reported_by(self):
        self.drain()
        probe = [a for a in self.store.recent_articles(["MSFT"]) if "probe" in a.title]
        self.assertEqual(len(probe), 1)
        self.assertIn("CNBC", probe[0].also_reported_by)

    def test_rerun_does_not_realert(self):
        self.drain()
        before = self.store.counts()["alerts_push"]
        self.p.sources[0].rewind()
        self.drain()
        self.assertEqual(self.store.counts()["alerts_push"], before)

    def test_backfill_finds_news_for_newly_added_ticker(self):
        self.drain()
        self.store.add_ticker("TSLA")
        before = dict(self.p.totals)
        report = self.p.backfill(["TSLA"])
        self.assertEqual([a["ticker"] for a in report.new_alerts], ["TSLA"])
        # the funnel doesn't double-count re-read stories; the Tesla item moves from unmatched to enriched
        t = self.p.totals
        self.assertEqual(t["fetched"], before["fetched"])
        self.assertEqual((t["unmatched"], t["enriched"]), (before["unmatched"] - 1, before["enriched"] + 1))
        self.assertEqual(t["fetched"], t["untrusted"] + t["unmatched"] + t["dup_story"] + t["enriched"])

    def test_broken_source_does_not_stop_others(self):
        class Broken:
            name = "broken"

            def fetch(self, *a):
                raise RuntimeError("down")

        p = Pipeline(settings(), make_store(), [Broken(), DemoSource()], None)
        r = p.run_once()
        self.assertEqual(r.counts["source_errors"], 1)
        self.assertGreater(r.counts["fetched"], 0)


class TestChat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = make_store()
        p = Pipeline(settings(), cls.store, [DemoSource()], None)
        for _ in range(20):
            p.run_once()
        cls.wl = cls.store.get_watchlist()

    def test_question_scope_detects_company_names(self):
        self.assertEqual(tickers_in("what's new with Nvidia and Apple?", self.wl), ["AAPL", "NVDA"])

    def test_extractive_answer_cites_only_scoped_ticker(self):
        r = answer(self.store, None, "Any news on Nvidia?", self.wl)
        self.assertTrue(r["citations"])
        self.assertTrue(all("NVDA" in c["tickers"] for c in r["citations"]))
        self.assertIn("[1]", r["answer"])

    def test_advice_request_is_declined_but_informed(self):
        r = answer(self.store, None, "Should I sell MSFT?", self.wl)
        self.assertIn("can't tell you whether to buy or sell", r["answer"])
        self.assertTrue(r["citations"])

    def test_llm_answer_keeps_only_cited_sources(self):
        llm = FakeLLM("Export curbs are being weighed [1]; Nvidia expects limited impact [3].")
        r = answer(self.store, llm, "Nvidia export rules?", self.wl)
        self.assertEqual([c["n"] for c in r["citations"]], [1, 3])
        self.assertIn("Headline:", llm.prompts[0])  # context was supplied to the model

    def test_empty_store_explains_itself(self):
        r = answer(make_store(["ZZZZ"]), None, "news?", ["ZZZZ"])
        self.assertEqual(r["citations"], [])
        self.assertIn("don't have any news", r["answer"])


class TestRawArticle(unittest.TestCase):
    def test_id_is_stable(self):
        a = RawArticle("Reuters", "https://reuters.com/x", "t", "s", NOW)
        self.assertEqual(a.id, RawArticle("CNBC", "https://reuters.com/x", "u", "v", NOW).id)


def tearDownModule():
    for st in _stores:
        st.close()


if __name__ == "__main__":
    unittest.main()
