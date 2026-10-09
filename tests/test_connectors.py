"""Connector tests with the network mocked: request shape in, sample payload out."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from app import http, llm, sources
from app.config import Settings

SINCE = datetime.now(timezone.utc) - timedelta(days=2)
RECENT = datetime.now(timezone.utc) - timedelta(hours=1)


class TestLLMProviders(unittest.TestCase):
    def _call(self, provider, response):
        with mock.patch.object(http, "post_json", return_value=response) as m:
            out = provider.complete("sys", "hello", json_mode=True, max_tokens=50)
        url, payload = m.call_args.args
        return out, url, payload, m.call_args.kwargs["headers"]

    def test_openai(self):
        out, url, payload, headers = self._call(
            llm.OpenAIProvider("k", "gpt-4o-mini", 5), {"choices": [{"message": {"content": "{}"}}]}
        )
        self.assertEqual(out, "{}")
        self.assertTrue(url.endswith("/v1/chat/completions"))
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(headers["Authorization"], "Bearer k")

    def test_anthropic(self):
        out, url, payload, headers = self._call(
            llm.AnthropicProvider("k", "claude-haiku-4-5", 5), {"content": [{"type": "text", "text": "hi"}]}
        )
        self.assertEqual(out, "hi")
        self.assertTrue(url.endswith("/v1/messages"))
        self.assertIn("JSON", payload["system"])
        self.assertEqual(headers["x-api-key"], "k")
        self.assertIn("anthropic-version", headers)

    def test_gemini(self):
        out, url, payload, headers = self._call(
            llm.GeminiProvider("k", "gemini-2.0-flash", 5), {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        )
        self.assertEqual(out, "ok")
        self.assertIn("gemini-2.0-flash:generateContent", url)
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(headers["x-goog-api-key"], "k")

    def test_http_error_becomes_llm_error(self):
        with mock.patch.object(http, "post_json", side_effect=http.HttpError(401, "bad key")):
            with self.assertRaises(llm.LLMError):
                llm.OpenAIProvider("k", "m", 5).complete("s", "p")

    def test_build_llm_selection(self):
        self.assertIsNone(llm.build_llm(Settings()))
        self.assertIsInstance(llm.build_llm(Settings(anthropic_api_key="a")), llm.AnthropicProvider)
        self.assertIsInstance(llm.build_llm(Settings(openai_api_key="o", gemini_api_key="g", llm_provider="gemini")), llm.GeminiProvider)
        self.assertIsNone(llm.build_llm(Settings(llm_provider="openai")))  # chosen but no key


class TestNewsSources(unittest.TestCase):
    def test_finnhub(self):
        payload = [
            {"datetime": int(RECENT.timestamp()), "headline": "Nvidia news", "source": "Reuters",
             "summary": "s", "url": "https://finnhub.io/api/news?id=1", "related": "NVDA"},
            {"datetime": int((SINCE - timedelta(days=1)).timestamp()), "headline": "old", "source": "Reuters", "url": "https://x/2"},
        ]
        with mock.patch.object(http, "get_json", return_value=payload) as m:
            out = sources.FinnhubSource("tok").fetch(["NVDA"], SINCE)
        self.assertIn("symbol=NVDA", m.call_args.args[0])
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0].source_name, out[0].source_tickers), ("Reuters", ["NVDA"]))

    def test_yahoo_rss(self):
        rss = f"""<?xml version="1.0"?><rss><channel>
        <item><title>Apple news</title><link>https://finance.yahoo.com/news/apple-1</link>
        <description>&lt;p&gt;Body &amp;amp; more&lt;/p&gt;</description>
        <pubDate>{RECENT.strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate></item>
        </channel></rss>""".encode()
        with mock.patch.object(http, "request", return_value=rss):
            out = sources.YahooRSSSource().fetch(["AAPL"], SINCE)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].summary, "Body & more")

    def test_sec_edgar_8k_items_become_plain_english(self):
        atom = f"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
        <entry><title>8-K - Current report</title>
        <link href="https://www.sec.gov/Archives/edgar/data/320193/x-index.htm"/>
        <summary type="html">&lt;b&gt;Filed:&lt;/b&gt; 2026-10-08 &lt;b&gt;Items&lt;/b&gt; 2.02, 9.01</summary>
        <updated>{RECENT.isoformat()}</updated></entry></feed>""".encode()
        with mock.patch.object(http, "request", return_value=atom) as m:
            out = sources.SECEdgarSource("Test test@example.com").fetch(["AAPL"], SINCE)
        self.assertIn("CIK=0000320193", m.call_args.args[0])
        self.assertEqual(m.call_args.kwargs["headers"]["User-Agent"], "Test test@example.com")
        self.assertEqual(out[0].title, "Apple Inc. files 8-K: reported results of operations (earnings)")
        self.assertEqual(out[0].source_tickers, ["AAPL"])

    def test_source_failure_is_contained(self):
        with mock.patch.object(http, "request", side_effect=http.HttpError(503, "down")):
            self.assertEqual(sources.YahooRSSSource().fetch(["AAPL"], SINCE), [])

    def test_build_sources(self):
        self.assertEqual([s.name for s in sources.build_sources(Settings())], ["demo"])
        live = sources.build_sources(Settings(mode="live", finnhub_api_key="k"))
        self.assertEqual([s.name for s in live], ["finnhub", "yahoo_rss", "sec_edgar"])


class TestNotifiers(unittest.TestCase):
    def test_slack_payload(self):
        from app.notifiers import SlackNotifier

        alert = {"ticker": "NVDA", "article": {"title": "T", "url": "https://u", "source_name": "Reuters",
                                               "enrichment": {"summary": "S", "why_it_matters": "W", "sentiment": -0.6}}}
        with mock.patch.object(http, "post_json") as m:
            SlackNotifier("https://hooks.slack.com/x").send(alert)
        text = m.call_args.args[1]["text"]
        self.assertTrue(text.startswith("▼ NVDA | T"))
        self.assertIn("Why it matters: W", text)


if __name__ == "__main__":
    unittest.main()
