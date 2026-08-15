import csv
import json
import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import generate


def load_config():
    return json.loads((Path(__file__).parents[1] / "config.json").read_text(encoding="utf-8"))


def valid_manual(now):
    raw = {
        "forward_pe": {"value": 25, "percentile": 20, "as_of": now.date().isoformat(), "percentile_window": "10Y"},
        "ttm_pe": {"value": 30, "percentile": 40, "as_of": now.date().isoformat(), "percentile_window": "10Y"},
        "cnn_fear_greed": {"value": 30, "as_of": now.date().isoformat()},
        "naaim": {"value": 70, "as_of": now.date().isoformat()},
        "us10y_score": {"value": 60, "as_of": now.date().isoformat()},
    }
    return {name: generate.manual_metric(name, value, now) for name, value in raw.items()}


def valid_market():
    return {
        "ndx": {"daily_return": 1.2, "value": 20000, "freshness_status": "fresh"},
        "spx": {"daily_return": 0.8, "value": 6000, "freshness_status": "fresh"},
        "qqq": {"daily_return": 1.1, "rsi": 60, "ma200_dev": 7, "value": 500, "freshness_status": "fresh"},
        "vxn": {"value": 25, "percentile": 70, "freshness_status": "fresh"},
        "us10y": {"value": 4.2, "daily_change": -0.02, "freshness_status": "fresh"},
    }


class IndicatorTests(unittest.TestCase):
    def test_parse_chartrow_pe(self):
        html = """
        <h1>Nasdaq-100 P/E ratio: <span>30.9</span> · forward 20.5</h1>
        <p>as of market close, Aug 14, 2026</p>
        """
        self.assertEqual(
            generate.parse_chartrow_pe(html),
            {"ttm_pe": 30.9, "forward_pe": 20.5, "as_of": "2026-08-14"},
        )

    def test_parse_cnn_fear_greed(self):
        payload = json.dumps({
            "fear_and_greed": {
                "score": 64.9714,
                "rating": "greed",
                "timestamp": "2026-08-14T23:59:53+00:00",
            }
        })
        self.assertEqual(
            generate.parse_cnn_fear_greed(payload),
            {"value": 64.9714, "as_of": "2026-08-14", "rating": "greed"},
        )

    def test_parse_naaim_number(self):
        html = '<body><div class="h1 text-center">77.34</div></body>'
        self.assertEqual(generate.parse_naaim_number(html), 77.34)

    def test_dynamic_pe_percentile_uses_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            path.write_text(
                "date,forward_pe,ttm_pe\n"
                "2026-08-01,10,20\n"
                "2026-08-02,20,30\n"
                "2026-08-03,30,40\n",
                encoding="utf-8",
            )
            now = datetime(2026, 8, 7, tzinfo=ZoneInfo("Asia/Shanghai"))
            pct, window, as_of = generate.pe_percentile(
                "forward_pe", 25, {"percentile": 37.5, "as_of": "2026-07-27"},
                {"percentile_window_days": 3650, "min_percentile_observations": 2}, now, path,
            )
            self.assertEqual(pct, 75.0)
            self.assertEqual(window, "动态历史（N=4）")
            self.assertIsNone(as_of)

    def test_automatic_us10y_score_uses_yield_bands(self):
        config = load_config()
        metric, error = generate.automatic_us10y_score(
            config,
            {"us10y": {"value": 4.64, "as_of": "2026-08-14", "freshness_status": "fresh"}},
            None,
        )
        self.assertIsNone(error)
        self.assertEqual(metric["value"], 35.0)
        self.assertEqual(metric["freshness_status"], "fresh")

    def test_pct_return_uses_n_sessions_back(self):
        series = pd.Series([100, 102, 104, 110])
        self.assertAlmostEqual(generate.pct_return(series, 1), (110 / 104 - 1) * 100)
        self.assertAlmostEqual(generate.pct_return(series, 3), 10)

    def test_wilder_rsi_uses_sma_seed_then_recursive_smoothing(self):
        # The first 14 moves have 7 gains and 7 losses, so the seed RSI is 50.
        # One further gain produces Wilder averages 0.535714 / 0.464286.
        prices = pd.Series([10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11])
        self.assertAlmostEqual(generate.wilder_rsi(prices), 53.57142857, places=6)

    def test_percentile_and_ma200_boundaries(self):
        series = pd.Series([1, 2, 2, 4])
        self.assertEqual(generate.historical_percentile(series, 4), 100)
        bands = load_config()["model"]["ma200_bands"]
        self.assertEqual(generate.ma200_score(-1, bands), 100)
        self.assertEqual(generate.ma200_score(0, bands), 100)
        self.assertEqual(generate.ma200_score(5, bands), 80)
        self.assertEqual(generate.ma200_score(15.01, bands), 20)


class ModelTests(unittest.TestCase):
    def test_transparent_model_formula(self):
        config = load_config()
        now = datetime(2026, 8, 7, tzinfo=ZoneInfo("Asia/Shanghai"))
        result = generate.score_model(config, valid_market(), valid_manual(now))
        self.assertEqual(result["dimensions"], {
            "valuation": 72.5,
            "sentiment": 70.0,
            "trend": 50.0,
            "positioning": 30.0,
            "macro": 60.0,
        })
        self.assertEqual(result["score"], 62.5)
        self.assertEqual(result["state"], "正常")
        self.assertEqual(result["multiplier"], 1.0)

    def test_missing_dimension_does_not_invent_total(self):
        config = load_config()
        now = datetime(2026, 8, 7, tzinfo=ZoneInfo("Asia/Shanghai"))
        manual = valid_manual(now)
        manual["naaim"]["value"] = None
        manual["naaim"]["freshness_status"] = "unavailable"
        result = generate.score_model(config, valid_market(), manual)
        self.assertIsNone(result["dimensions"]["positioning"])
        self.assertIsNone(result["score"])
        self.assertEqual(result["state"], "数据不足")

    def test_temperature_band_boundaries(self):
        config = load_config()
        bands = sorted(config["model"]["temperature_bands"], key=lambda item: item["min_score"], reverse=True)
        def classify(score):
            return next(item["state"] for item in bands if score >= item["min_score"])
        self.assertEqual(classify(80), "冰点")
        self.assertEqual(classify(65), "偏冷")
        self.assertEqual(classify(50), "正常")
        self.assertEqual(classify(35), "偏热")
        self.assertEqual(classify(20), "过热")
        self.assertEqual(classify(19.99), "极热")

    def test_naaim_allows_real_world_extremes_but_score_is_clamped(self):
        config = load_config()
        now = datetime(2026, 8, 7, tzinfo=ZoneInfo("Asia/Shanghai"))
        manual = valid_manual(now)
        manual["naaim"] = generate.manual_metric(
            "naaim", {"value": 110, "as_of": now.date().isoformat()}, now
        )
        self.assertEqual(manual["naaim"]["freshness_status"], "manual")
        result = generate.score_model(config, valid_market(), manual)
        self.assertEqual(result["dimensions"]["positioning"], 0)


class PersistenceTests(unittest.TestCase):
    def test_history_upserts_same_date(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            first = {field: None for field in generate.HISTORY_FIELDS}
            first.update({"date": "2026-08-06", "score": 50})
            second = dict(first, score=62)
            generate.upsert_history(first, path)
            rows = generate.upsert_history(second, path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["score"], 62)
            with path.open(encoding="utf-8", newline="") as handle:
                persisted = list(csv.DictReader(handle))
            self.assertEqual(len(persisted), 1)
            self.assertEqual(persisted[0]["score"], "62")


if __name__ == "__main__":
    unittest.main()
