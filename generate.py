#!/usr/bin/env python3
"""Generate the Market Pulse dashboard, history and optional ServerChan alert."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"
HISTORY_PATH = ROOT / "data" / "history.csv"
LATEST_PATH = ROOT / "data" / "latest.json"
SITE_PATH = ROOT / "site" / "index.html"
DEFAULT_PE_SOURCE = "https://chartrow.com/nasdaq-100/pe-ratio"

HISTORY_FIELDS = [
    "date", "generated_at", "score", "state", "multiplier", "ndx_change",
    "spx_change", "qqq_change", "qqq_5d_change", "qqq_20d_change", "vxn",
    "us10y", "forward_pe", "forward_pe_percentile", "ttm_pe",
    "ttm_pe_percentile", "cnn_fear_greed", "naaim", "rsi", "ma200_dev",
    "valuation_score", "sentiment_score", "trend_score", "positioning_score",
    "macro_score", "data_quality", "one_line_summary",
]


@dataclass
class FetchResult:
    frame: pd.DataFrame | None
    error: str | None = None


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"警告：无法读取 {path}: {exc}", file=sys.stderr)
        return default


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temp_path.replace(path)


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def rounded(value: Any, digits: int = 2) -> float | None:
    number = finite_number(value)
    return round(number, digits) if number is not None else None


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return min(high, max(low, value))


def safe_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def get_now(timezone_name: str) -> datetime:
    try:
        return datetime.now(ZoneInfo(timezone_name))
    except Exception:
        return datetime.now(timezone.utc).astimezone()


def validate_config(config: dict[str, Any]) -> None:
    ticker_names = set(config.get("tickers", {}))
    expected_tickers = {"ndx", "spx", "qqq", "vxn", "us10y"}
    if ticker_names != expected_tickers:
        raise ValueError(f"tickers 必须恰好包含：{', '.join(sorted(expected_tickers))}")
    weights = config.get("model", {}).get("weights", {})
    expected = {"valuation", "sentiment", "trend", "positioning", "macro"}
    if set(weights) != expected:
        raise ValueError(f"model.weights 必须恰好包含：{', '.join(sorted(expected))}")
    if not math.isclose(sum(float(value) for value in weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("model.weights 权重之和必须为 1")
    for name in ("valuation_weights", "sentiment_weights", "trend_weights"):
        values = config.get("model", {}).get(name, {})
        if not values or not math.isclose(sum(float(v) for v in values.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"model.{name} 权重之和必须为 1")
    bands = config.get("model", {}).get("temperature_bands", [])
    if not bands or min(float(band["min_score"]) for band in bands) > 0:
        raise ValueError("temperature_bands 必须覆盖 0 分")


def download_ticker(ticker: str, period: str = "5y") -> FetchResult:
    try:
        import yfinance as yf

        frame = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=20,
        )
        if frame is None or frame.empty:
            return FetchResult(None, "数据源返回空结果")
        return FetchResult(frame)
    except Exception as exc:  # Every ticker degrades independently.
        return FetchResult(None, f"{type(exc).__name__}: {exc}")


def close_series(frame: pd.DataFrame) -> pd.Series:
    if "Close" not in frame.columns:
        raise ValueError("行情缺少 Close 字段")
    series = frame["Close"]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce").dropna().sort_index()


def pct_return(series: pd.Series, periods: int) -> float | None:
    if len(series) <= periods:
        return None
    previous = finite_number(series.iloc[-periods - 1])
    current = finite_number(series.iloc[-1])
    if previous in (None, 0) or current is None:
        return None
    return (current / previous - 1) * 100


def wilder_rsi(series: pd.Series, period: int = 14) -> float | None:
    if len(series) <= period:
        return None
    delta = pd.to_numeric(series, errors="coerce").diff().dropna()
    gains = delta.clip(lower=0).to_numpy(dtype=float)
    losses = (-delta.clip(upper=0)).to_numpy(dtype=float)
    if len(gains) < period:
        return None
    # Wilder's original smoothing: seed with an SMA, then update recursively.
    last_gain = float(np.mean(gains[:period]))
    last_loss = float(np.mean(losses[:period]))
    for gain, loss in zip(gains[period:], losses[period:]):
        last_gain = (last_gain * (period - 1) + gain) / period
        last_loss = (last_loss * (period - 1) + loss) / period
    if last_gain is None or last_loss is None:
        return None
    if last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0
    return 100 - (100 / (1 + last_gain / last_loss))


def historical_percentile(series: pd.Series, window: int) -> float | None:
    values = series.tail(window).dropna()
    if len(values) < 2:
        return None
    current = finite_number(values.iloc[-1])
    if current is None:
        return None
    # Percentage rank includes the current observation and is deterministic for ties.
    return float((values <= current).sum() / len(values) * 100)


def as_of_from_series(series: pd.Series) -> str:
    timestamp = pd.Timestamp(series.index[-1])
    return timestamp.date().isoformat()


def market_metric(name: str, ticker: str, frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    closes = close_series(frame)
    if len(closes) < 2:
        raise ValueError("有效收盘价不足 2 个交易日")
    metric: dict[str, Any] = {
        "ticker": ticker,
        "value": rounded(closes.iloc[-1]),
        "daily_return": rounded(pct_return(closes, 1)),
        "as_of": as_of_from_series(closes),
        "source": "Yahoo Finance",
        "freshness_status": "fresh",
        "error": None,
    }
    if name == "qqq":
        ma200 = finite_number(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None
        current = finite_number(closes.iloc[-1])
        metric.update({
            "return_5d": rounded(pct_return(closes, 5)),
            "return_20d": rounded(pct_return(closes, 20)),
            "rsi": rounded(wilder_rsi(closes)),
            "ma200": rounded(ma200),
            "ma200_dev": rounded((current / ma200 - 1) * 100) if current is not None and ma200 else None,
        })
    elif name == "vxn":
        window = int(config.get("vxn_percentile_days", 756))
        actual_window = min(window, len(closes))
        metric.update({
            "daily_change": rounded(finite_number(closes.iloc[-1]) - finite_number(closes.iloc[-2])),
            "percentile": rounded(historical_percentile(closes, window), 1),
            "percentile_window": f"{actual_window} 个交易日",
        })
    elif name == "us10y":
        # Yahoo's ^TNX close is conventionally displayed directly as a percentage yield.
        previous = finite_number(closes.iloc[-2])
        current = finite_number(closes.iloc[-1])
        metric["daily_change"] = rounded(current - previous, 3) if current is not None and previous is not None else None
        metric["unit"] = "%"
    return metric


def unavailable_metric(ticker: str, error: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "value": None,
        "as_of": None,
        "source": "Yahoo Finance",
        "freshness_status": "unavailable",
        "error": error,
    }


def fetch_market(config: dict[str, Any], previous: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    market: dict[str, Any] = {}
    failures: list[str] = []
    previous_market = (previous or {}).get("market", {})
    period = f'{max(3, int(config.get("history_years", 5)))}y'
    for name, ticker in config["tickers"].items():
        result = download_ticker(str(ticker), period=period)
        try:
            if result.frame is None:
                raise ValueError(result.error or "未知抓取错误")
            market[name] = market_metric(name, str(ticker), result.frame, config)
        except Exception as exc:
            error = str(exc)
            prior = previous_market.get(name)
            if prior and finite_number(prior.get("value")) is not None:
                market[name] = dict(prior)
                market[name]["freshness_status"] = "stale"
                market[name]["error"] = error
            else:
                market[name] = unavailable_metric(str(ticker), error)
            failures.append(f"{ticker}: {error}")
    return market, failures


def parse_chartrow_pe(payload: str) -> dict[str, Any]:
    """Extract Nasdaq-100 trailing and forward PE from ChartRow's public page."""
    text = html.unescape(re.sub(r"<!--.*?-->|<[^>]+>", " ", payload, flags=re.S))
    text = " ".join(text.split())
    match = re.search(
        r"Nasdaq-100 P/E ratio:\s*([0-9]+(?:\.[0-9]+)?)\s*\W+forward\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.I,
    )
    if not match:
        raise ValueError("ChartRow PE values not found")
    date_match = re.search(r"as of market close,\s*([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})", text)
    if not date_match:
        raise ValueError("ChartRow PE date not found")
    try:
        as_of = datetime.strptime(date_match.group(1), "%b %d, %Y").date().isoformat()
    except ValueError as exc:
        raise ValueError("ChartRow PE date is invalid") from exc
    return {
        "ttm_pe": float(match.group(1)),
        "forward_pe": float(match.group(2)),
        "as_of": as_of,
    }


def automatic_pe_metrics(config: dict[str, Any], now: datetime) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    """Fetch current PE values while retaining configured historical percentile ranks."""
    settings = config.get("automatic_inputs", {}).get("pe", {})
    if settings.get("enabled", True) is False:
        return None, None
    url = str(settings.get("url") or DEFAULT_PE_SOURCE)
    timeout = max(5, int(settings.get("timeout_seconds", 30)))
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Market-Pulse/1.0 (+https://github.com/maqianxiong/market-pulse)"},
            timeout=timeout,
        )
        response.raise_for_status()
        values = parse_chartrow_pe(response.text)
        raw_inputs = config.get("manual_inputs", {})
        metrics: dict[str, dict[str, Any]] = {}
        for name in ("forward_pe", "ttm_pe"):
            raw = raw_inputs.get(name, {})
            percentile = finite_number(raw.get("percentile"))
            percentile_note = f" ({raw.get('as_of')})" if raw.get("as_of") else ""
            metrics[name] = {
                "value": rounded(values[name]),
                "percentile": rounded(percentile, 1),
                "as_of": values["as_of"],
                "source": f"ChartRow Nasdaq-100 P/E (automatic); percentile retained from config{percentile_note}",
                "freshness_status": "fresh",
                "percentile_window": raw.get("percentile_window") or "10Y",
                "percentile_as_of": raw.get("as_of"),
                "error": None,
            }
        return metrics, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def manual_metric(name: str, raw: dict[str, Any], now: datetime) -> dict[str, Any]:
    value = finite_number(raw.get("value"))
    percentile = finite_number(raw.get("percentile"))
    required = (value, percentile) if name in {"forward_pe", "ttm_pe"} else (value,)
    as_of = safe_date(raw.get("as_of"))
    status = "manual"
    error = None
    if any(item is None for item in required):
        status = "unavailable"
        error = "尚未在 config.json 中配置"
    elif name in {"cnn_fear_greed", "naaim", "us10y_score", "forward_pe", "ttm_pe"}:
        check_value = percentile if name in {"forward_pe", "ttm_pe"} else value
        minimum, maximum = (-200, 200) if name == "naaim" else (0, 100)
        if check_value is not None and not minimum <= check_value <= maximum:
            status = "unavailable"
            error = f"数值必须在 {minimum}–{maximum} 之间"
        elif as_of is not None:
            max_age = int(raw.get("max_age_days", 30))
            if (now.date() - as_of).days > max_age:
                status = "stale"
        else:
            error = "未填写数据日期"
    return {
        "value": rounded(value),
        "percentile": rounded(percentile, 1),
        "as_of": as_of.isoformat() if as_of else None,
        "source": str(raw.get("source") or "手工配置"),
        "freshness_status": status,
        "percentile_window": raw.get("percentile_window"),
        "error": error,
    }


def parse_manual_inputs(config: dict[str, Any], now: datetime) -> dict[str, Any]:
    raw_inputs = config.get("manual_inputs", {})
    names = ["forward_pe", "ttm_pe", "cnn_fear_greed", "naaim", "us10y_score"]
    return {name: manual_metric(name, raw_inputs.get(name, {}), now) for name in names}


def available(metric: dict[str, Any], field: str = "value") -> float | None:
    if metric.get("freshness_status") == "unavailable":
        return None
    return finite_number(metric.get(field))


def ma200_score(deviation: float | None, bands: Iterable[dict[str, Any]]) -> float | None:
    if deviation is None:
        return None
    for band in bands:
        maximum = finite_number(band.get("max_deviation"))
        if maximum is None or deviation <= maximum:
            return finite_number(band.get("score"))
    return None


def weighted_score(parts: Iterable[tuple[float | None, float]]) -> float | None:
    values = list(parts)
    if any(value is None for value, _ in values):
        return None
    return clamp(sum(float(value) * float(weight) for value, weight in values))


def score_model(config: dict[str, Any], market: dict[str, Any], manual: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    forward_pct = available(manual["forward_pe"], "percentile")
    ttm_pct = available(manual["ttm_pe"], "percentile")
    vxn_pct = available(market["vxn"], "percentile")
    cnn = available(manual["cnn_fear_greed"])
    rsi = available(market["qqq"], "rsi")
    deviation = available(market["qqq"], "ma200_dev")
    naaim = available(manual["naaim"])
    macro = available(manual["us10y_score"])

    valuation = weighted_score([
        ((100 - forward_pct) if forward_pct is not None else None, model["valuation_weights"]["forward_pe"]),
        ((100 - ttm_pct) if ttm_pct is not None else None, model["valuation_weights"]["ttm_pe"]),
    ])
    sentiment = weighted_score([
        (vxn_pct, model["sentiment_weights"]["vxn"]),
        ((100 - cnn) if cnn is not None else None, model["sentiment_weights"]["cnn"]),
    ])
    ma_score = ma200_score(deviation, model["ma200_bands"])
    trend = weighted_score([
        ((100 - rsi) if rsi is not None else None, model["trend_weights"]["rsi"]),
        (ma_score, model["trend_weights"]["ma200"]),
    ])
    positioning = clamp(100 - naaim) if naaim is not None else None
    macro_score_value = clamp(macro) if macro is not None else None
    dimensions = {
        "valuation": rounded(valuation, 1),
        "sentiment": rounded(sentiment, 1),
        "trend": rounded(trend, 1),
        "positioning": rounded(positioning, 1),
        "macro": rounded(macro_score_value, 1),
    }
    total = weighted_score([
        (dimensions[name], float(weight)) for name, weight in model["weights"].items()
    ])
    total = rounded(total, 1)
    if total is None:
        state = "数据不足"
        multiplier = None
        tone = "unavailable"
    else:
        band = next(
            band for band in sorted(model["temperature_bands"], key=lambda item: item["min_score"], reverse=True)
            if total >= float(band["min_score"])
        )
        state = str(band["state"])
        multiplier = rounded(band["multiplier"], 1)
        tone = str(band.get("tone", "normal"))
    missing = [name for name, score in dimensions.items() if score is None]
    return {
        "score": total,
        "state": state,
        "multiplier": multiplier,
        "tone": tone,
        "dimensions": dimensions,
        "ma200_score": rounded(ma_score, 1),
        "missing_dimensions": missing,
    }


def make_summary(model_result: dict[str, Any], market: dict[str, Any]) -> str:
    ndx = available(market["ndx"], "daily_return")
    spx = available(market["spx"], "daily_return")
    if ndx is not None and spx is not None:
        lead = "科技股今日跑赢大盘" if ndx > spx else "大盘今日相对更强" if spx > ndx else "科技股与大盘表现接近"
    else:
        lead = "指数强弱暂无法完整比较"
    if model_result["score"] is None:
        return f"{lead}；综合温度因配置项缺失暂不可用，请先查看数据状态。"
    state = model_result["state"]
    multiplier = model_result["multiplier"]
    return f"{lead}；市场处于{state}区间，模型参考定投倍率为 {multiplier:g}x。"


def collect_alerts(config: dict[str, Any], market: dict[str, Any], manual: dict[str, Any], model_result: dict[str, Any]) -> list[str]:
    thresholds = config.get("alerts", {})
    alerts: list[str] = []
    score = model_result.get("score")
    vxn = available(market["vxn"])
    forward_pct = available(manual["forward_pe"], "percentile")
    naaim = available(manual["naaim"])
    yield_change = available(market["us10y"], "daily_change")
    if score is not None and score >= 80:
        alerts.append("市场温度进入冰点区间")
    if vxn is not None and vxn > float(thresholds.get("vxn_high", 35)):
        alerts.append(f"恐慌指数升至 {vxn:.1f}")
    if forward_pct is not None and forward_pct <= float(thresholds.get("forward_pe_percentile_low", 10)):
        alerts.append("未来估值进入历史低分位")
    if naaim is not None:
        if naaim <= float(thresholds.get("naaim_low", 20)):
            alerts.append("机构仓位处于极低水平")
        elif naaim >= float(thresholds.get("naaim_high", 100)):
            alerts.append("机构仓位处于极高水平")
    if yield_change is not None and yield_change <= float(thresholds.get("us10y_daily_drop_pct_points", -0.15)):
        alerts.append("美债 10 年期收益率单日明显下行")
    drop_threshold = float(thresholds.get("index_drop_pct", -3))
    for key, label in (("ndx", "纳指100"), ("spx", "标普500")):
        change = available(market[key], "daily_return")
        if change is not None and change <= drop_threshold:
            alerts.append(f"{label}单日下跌 {abs(change):.2f}%")
    return alerts


def report_date(market: dict[str, Any], now: datetime) -> str:
    dates = [safe_date(metric.get("as_of")) for metric in market.values() if metric.get("freshness_status") == "fresh"]
    valid = [value for value in dates if value]
    if valid:
        return max(valid).isoformat()
    prior_dates = [safe_date(metric.get("as_of")) for metric in market.values()]
    valid_prior = [value for value in prior_dates if value]
    return (max(valid_prior) if valid_prior else now.date()).isoformat()


def data_quality(market: dict[str, Any], manual: dict[str, Any]) -> str:
    metrics = [*market.values(), *manual.values()]
    statuses = [item.get("freshness_status") for item in metrics]
    if "unavailable" in statuses:
        return "incomplete"
    if any(item.get("error") for item in metrics):
        return "incomplete"
    if "stale" in statuses:
        return "stale"
    return "complete"


def build_snapshot(config: dict[str, Any], now: datetime, previous: dict[str, Any] | None) -> dict[str, Any]:
    market, failures = fetch_market(config, previous)
    manual = parse_manual_inputs(config, now)
    automatic_pe, pe_error = automatic_pe_metrics(config, now)
    if automatic_pe:
        manual.update(automatic_pe)
    elif pe_error:
        failures.append(f"PE: {pe_error}")
        for name in ("forward_pe", "ttm_pe"):
            metric = manual[name]
            if metric.get("freshness_status") not in {"unavailable", "stale"}:
                metric["freshness_status"] = "stale"
                metric["error"] = pe_error
    model_result = score_model(config, market, manual)
    summary = make_summary(model_result, market)
    return {
        "report_date": report_date(market, now),
        "generated_at": now.isoformat(timespec="seconds"),
        "timezone": config.get("timezone", "Asia/Shanghai"),
        "market": market,
        "manual": manual,
        "model": model_result,
        "summary": summary,
        "alerts": collect_alerts(config, market, manual, model_result),
        "fetch_failures": failures,
        "data_quality": data_quality(market, manual),
        "model_weights": config["model"]["weights"],
    }


def snapshot_to_row(snapshot: dict[str, Any]) -> dict[str, Any]:
    market = snapshot["market"]
    manual = snapshot["manual"]
    model = snapshot["model"]
    dimensions = model["dimensions"]
    return {
        "date": snapshot["report_date"],
        "generated_at": snapshot["generated_at"],
        "score": model["score"],
        "state": model["state"],
        "multiplier": model["multiplier"],
        "ndx_change": market["ndx"].get("daily_return"),
        "spx_change": market["spx"].get("daily_return"),
        "qqq_change": market["qqq"].get("daily_return"),
        "qqq_5d_change": market["qqq"].get("return_5d"),
        "qqq_20d_change": market["qqq"].get("return_20d"),
        "vxn": market["vxn"].get("value"),
        "us10y": market["us10y"].get("value"),
        "forward_pe": manual["forward_pe"].get("value"),
        "forward_pe_percentile": manual["forward_pe"].get("percentile"),
        "ttm_pe": manual["ttm_pe"].get("value"),
        "ttm_pe_percentile": manual["ttm_pe"].get("percentile"),
        "cnn_fear_greed": manual["cnn_fear_greed"].get("value"),
        "naaim": manual["naaim"].get("value"),
        "rsi": market["qqq"].get("rsi"),
        "ma200_dev": market["qqq"].get("ma200_dev"),
        "valuation_score": dimensions["valuation"],
        "sentiment_score": dimensions["sentiment"],
        "trend_score": dimensions["trend"],
        "positioning_score": dimensions["positioning"],
        "macro_score": dimensions["macro"],
        "data_quality": snapshot["data_quality"],
        "one_line_summary": snapshot["summary"],
    }


def read_history(path: Path = HISTORY_PATH) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def upsert_history(row: dict[str, Any], path: Path = HISTORY_PATH) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {item.get("date", ""): item for item in read_history(path) if item.get("date")}
    clean = {field: "" if row.get(field) is None else row.get(field) for field in HISTORY_FIELDS}
    rows[str(clean["date"])] = clean
    ordered = [rows[key] for key in sorted(rows)]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    temp_path.replace(path)
    return ordered


STATUS_LABELS = {"fresh": "最新", "manual": "手工", "stale": "已过期", "unavailable": "不可用"}
DIMENSION_META = {
    "valuation": ("估值", "未来与当前估值的历史位置", "40%"),
    "sentiment": ("情绪", "恐慌指数与市场情绪", "25%"),
    "trend": ("趋势", "短期热度与长期均线", "20%"),
    "positioning": ("资金", "机构仓位拥挤程度", "10%"),
    "macro": ("宏观", "利率环境对成长股的支持度", "5%"),
}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    number = finite_number(value)
    return "—" if number is None else f"{number:.{digits}f}{suffix}"


def fmt_change(value: Any) -> str:
    number = finite_number(value)
    return "—" if number is None else f"{number:+.2f}%"


def change_class(value: Any) -> str:
    number = finite_number(value)
    return "flat" if number is None or number == 0 else "up" if number > 0 else "down"


def freshness_badge(metric: dict[str, Any]) -> str:
    status = str(metric.get("freshness_status", "unavailable"))
    label = STATUS_LABELS.get(status, status)
    metadata = [metric.get("source"), metric.get("as_of"), metric.get("error")]
    title = " · ".join(str(item) for item in metadata if item) or "暂无日期"
    return f'<span class="status {esc(status)}" title="{esc(title)}">{esc(label)}</span>'


def market_cards(snapshot: dict[str, Any]) -> str:
    market = snapshot["market"]
    cards = [
        ("纳指100", "ndx", fmt_change(market["ndx"].get("daily_return")), "今日"),
        ("标普500", "spx", fmt_change(market["spx"].get("daily_return")), "今日"),
        ("恐慌指数", "vxn", fmt(market["vxn"].get("value"), 1), f'{fmt(market["vxn"].get("percentile"), 0, "%")} 分位'),
        ("美债10年", "us10y", fmt(market["us10y"].get("value"), 2, "%"), f'{fmt(market["us10y"].get("daily_change"), 2)} 点'),
    ]
    parts = []
    for label, key, value, detail in cards:
        metric = market[key]
        trend_value = metric.get("daily_return") if key in {"ndx", "spx"} else metric.get("daily_change")
        parts.append(f"""
        <article class="market-card">
          <div class="card-head"><span>{esc(label)}</span>{freshness_badge(metric)}</div>
          <div class="market-value {change_class(trend_value)}">{esc(value)}</div>
          <div class="market-detail">{esc(detail)} · {esc(metric.get('as_of') or '日期未知')}</div>
        </article>""")
    return "".join(parts)


def qqq_strip(snapshot: dict[str, Any]) -> str:
    metric = snapshot["market"]["qqq"]
    items = [
        ("收盘", fmt(metric.get("value"), 2), "flat"),
        ("今日", fmt_change(metric.get("daily_return")), change_class(metric.get("daily_return"))),
        ("5日", fmt_change(metric.get("return_5d")), change_class(metric.get("return_5d"))),
        ("20日", fmt_change(metric.get("return_20d")), change_class(metric.get("return_20d"))),
        ("短期热度", fmt(metric.get("rsi"), 1), "flat"),
        ("距 MA200", fmt(metric.get("ma200_dev"), 1, "%"), change_class(metric.get("ma200_dev"))),
    ]
    values = "".join(
        f'<div><span>{esc(label)}</span><strong class="{css_class}">{esc(value)}</strong></div>'
        for label, value, css_class in items
    )
    return (
        '<article class="qqq-strip"><div class="qqq-title"><div><strong>QQQ</strong>'
        '<span>纳指100 ETF</span></div>' + freshness_badge(metric) + '</div>'
        f'<div class="qqq-values">{values}</div></article>'
    )


def dimension_cards(snapshot: dict[str, Any]) -> str:
    scores = snapshot["model"]["dimensions"]
    parts = []
    for key, (label, explanation, weight) in DIMENSION_META.items():
        score = finite_number(scores.get(key))
        width = 0 if score is None else clamp(score)
        value = "—" if score is None else f"{score:.1f}"
        parts.append(f"""
        <article class="dimension-card">
          <div class="card-head"><span>{esc(label)}</span><span class="weight">权重 {weight}</span></div>
          <div class="dimension-value">{value}<small>/100</small></div>
          <div class="meter"><span style="width:{width:.1f}%"></span></div>
          <p>{esc(explanation)}</p>
        </article>""")
    return "".join(parts)


def detail_rows(snapshot: dict[str, Any]) -> str:
    market, manual = snapshot["market"], snapshot["manual"]
    rows = [
        ("未来估值", fmt(manual["forward_pe"].get("value"), 2), f'{fmt(manual["forward_pe"].get("percentile"), 0, "%")} · {manual["forward_pe"].get("percentile_window") or "窗口未填"}', manual["forward_pe"]),
        ("当前估值", fmt(manual["ttm_pe"].get("value"), 2), f'{fmt(manual["ttm_pe"].get("percentile"), 0, "%")} · {manual["ttm_pe"].get("percentile_window") or "窗口未填"}', manual["ttm_pe"]),
        ("市场情绪", fmt(manual["cnn_fear_greed"].get("value"), 0), "CNN Fear & Greed", manual["cnn_fear_greed"]),
        ("机构仓位", fmt(manual["naaim"].get("value"), 1), "NAAIM Exposure Index", manual["naaim"]),
        ("短期热度", fmt(market["qqq"].get("rsi"), 1), "QQQ RSI(14)", market["qqq"]),
        ("长期趋势", fmt(market["qqq"].get("ma200_dev"), 1, "%"), "QQQ 距 MA200", market["qqq"]),
        ("利率环境评分", fmt(manual["us10y_score"].get("value"), 0), "手工 0–100", manual["us10y_score"]),
    ]
    return "".join(
        f'<div class="detail-row"><div><strong>{esc(label)}</strong><span>{esc(description)} · {esc(metric.get("source") or "来源未知")} · {esc(metric.get("as_of") or "日期未知")}</span></div>'
        f'<div class="detail-number">{esc(value)}</div>{freshness_badge(metric)}</div>'
        for label, value, description, metric in rows
    )


def chart_payload(history: list[dict[str, Any]]) -> str:
    points = []
    for row in history[-180:]:
        score = finite_number(row.get("score"))
        if score is not None and row.get("date"):
            points.append({"date": row["date"], "score": score})
    payload = json.dumps(points, ensure_ascii=False, separators=(",", ":"))
    return payload.replace("<", "\\u003c")


def render_html(snapshot: dict[str, Any], history: list[dict[str, Any]], output: Path = SITE_PATH) -> None:
    model = snapshot["model"]
    score = model["score"]
    score_text = "—" if score is None else f"{score:.1f}"
    multiplier = model["multiplier"]
    multiplier_text = "—" if multiplier is None else f"{multiplier:g}x"
    failures = [*snapshot.get("fetch_failures", [])]
    failures.extend(
        f"{name}: {metric.get('error')}" for name, metric in snapshot["manual"].items()
        if metric.get("freshness_status") == "unavailable"
    )
    issue_html = ""
    if failures:
        items = "".join(f"<li>{esc(item)}</li>" for item in failures)
        issue_html = f'<aside class="notice"><strong>数据提示</strong><ul>{items}</ul></aside>'
    alerts = snapshot.get("alerts", [])
    alert_html = ""
    if alerts:
        alert_html = '<div class="alerts">' + "".join(f'<span>⚑ {esc(item)}</span>' for item in alerts) + "</div>"
    weights = snapshot["model_weights"]
    weight_text = " · ".join(f"{DIMENSION_META[key][0]} {float(value) * 100:.0f}%" for key, value in weights.items())

    template = Template(r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
  <meta name="theme-color" content="#edf3f8">
  <meta name="description" content="Market Pulse：面向长期投资者的透明市场温度仪表盘">
  <title>Market Pulse · $report_date</title>
  <style>
    :root{--ink:#12202f;--muted:#687889;--line:#dbe5ec;--panel:rgba(255,255,255,.88);--blue:#1769e0;--green:#16876c;--red:#dc4a58;--amber:#c77812;--shadow:0 18px 50px rgba(42,65,82,.09);--radius:22px}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:radial-gradient(circle at 85% -10%,#d9eafe 0,transparent 32rem),linear-gradient(150deg,#f5f8fa,#eaf1f6);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;min-height:100vh}
    body:before{content:"";position:fixed;inset:0;pointer-events:none;background-image:linear-gradient(rgba(255,255,255,.2) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.2) 1px,transparent 1px);background-size:40px 40px;mask-image:linear-gradient(to bottom,black,transparent 48%)}
    .shell{width:min(1120px,calc(100% - 32px));margin:auto;padding:34px 0 54px;position:relative}.topbar{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:24px}.brand{font-size:22px;font-weight:760;letter-spacing:-.04em}.brand i{display:inline-block;width:9px;height:9px;margin-right:9px;border-radius:50%;background:var(--blue);box-shadow:0 0 0 6px rgba(23,105,224,.1)}.meta{text-align:right;color:var(--muted);font-size:12px;line-height:1.7}
    .hero{position:relative;overflow:hidden;padding:36px;min-height:310px;border-radius:30px;background:linear-gradient(130deg,#10253b 0,#123e64 55%,#146b89 100%);box-shadow:0 26px 70px rgba(18,55,82,.24);color:white}.hero:after{content:"";position:absolute;width:390px;height:390px;border:1px solid rgba(255,255,255,.11);border-radius:50%;right:-110px;top:-180px;box-shadow:0 0 0 45px rgba(255,255,255,.025),0 0 0 95px rgba(255,255,255,.018)}.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#b7d3e7}.hero-grid{display:grid;grid-template-columns:1fr auto;gap:34px;align-items:end;position:relative;z-index:1}.score-line{display:flex;align-items:baseline;gap:13px;margin:16px 0 4px}.score{font-size:92px;font-weight:780;letter-spacing:-.075em;line-height:.9}.score-unit{color:#b7d3e7}.state{display:inline-flex;align-items:center;padding:7px 12px;border-radius:999px;background:rgba(255,255,255,.12);font-size:13px}.summary{max-width:660px;font-size:17px;line-height:1.7;color:#e8f3fa;margin:26px 0 0}.allocation{text-align:right;padding:21px 23px;border:1px solid rgba(255,255,255,.15);background:rgba(255,255,255,.08);backdrop-filter:blur(14px);border-radius:18px;min-width:190px}.allocation span{display:block;color:#b7d3e7;font-size:12px}.allocation strong{display:block;font-size:42px;margin:5px 0 2px}.allocation small{color:#d6e7f2}
    .alerts{display:flex;gap:8px;flex-wrap:wrap;margin-top:22px;position:relative;z-index:1}.alerts span{font-size:12px;padding:7px 10px;background:rgba(255,255,255,.1);border-radius:999px;color:#dcecf5}
    section{margin-top:32px}.section-head{display:flex;justify-content:space-between;align-items:end;margin-bottom:13px}.section-head h2{font-size:18px;margin:0;letter-spacing:-.02em}.section-head p{font-size:12px;color:var(--muted);margin:0}.market-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.market-card,.qqq-strip,.dimension-card,.chart-card,.detail-card{background:var(--panel);border:1px solid rgba(255,255,255,.75);border-radius:var(--radius);box-shadow:var(--shadow);backdrop-filter:blur(12px)}.market-card{padding:19px}.card-head{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:13px}.market-value{font-size:29px;font-weight:720;letter-spacing:-.04em;margin:15px 0 5px}.market-detail{font-size:11px;color:var(--muted)}.up{color:var(--red)}.down{color:var(--green)}.flat{color:var(--ink)}.qqq-strip{display:grid;grid-template-columns:150px 1fr;gap:20px;margin-top:12px;padding:18px 20px;align-items:center}.qqq-title{display:flex;align-items:center;justify-content:space-between;gap:10px}.qqq-title strong,.qqq-title span{display:block}.qqq-title strong{font-size:22px}.qqq-title div>span{font-size:10px;color:var(--muted);margin-top:2px}.qqq-values{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}.qqq-values div{padding-left:14px;border-left:1px solid var(--line)}.qqq-values span,.qqq-values strong{display:block}.qqq-values span{font-size:10px;color:var(--muted);margin-bottom:5px}.qqq-values strong{font-size:14px;font-variant-numeric:tabular-nums}
    .status{padding:3px 7px;border-radius:999px;font-size:10px;background:#edf2f6;color:#6e7c87}.status.fresh{background:#e4f5ef;color:#14755f}.status.manual{background:#e6effe;color:#2765b7}.status.stale{background:#fff0d7;color:#9a5d0c}.status.unavailable{background:#f7e7e9;color:#a83d49}.notice{margin-top:16px;padding:15px 18px;border:1px solid #ead6b9;background:#fff7e9;border-radius:16px;color:#704d20;font-size:12px}.notice ul{margin:7px 0 0;padding-left:18px;line-height:1.7}
    .dimension-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.dimension-card{padding:18px}.weight{font-size:10px}.dimension-value{font-size:31px;font-weight:710;margin-top:20px}.dimension-value small{font-size:11px;color:var(--muted);font-weight:500}.dimension-card p{font-size:11px;line-height:1.5;color:var(--muted);margin:12px 0 0}.meter{height:5px;border-radius:9px;background:#e8eef2;margin-top:10px;overflow:hidden}.meter span{display:block;height:100%;background:linear-gradient(90deg,#1d80e2,#28af9a);border-radius:inherit}
    .lower-grid{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(280px,.75fr);gap:14px}.chart-card,.detail-card{padding:20px}.chart-wrap{position:relative;height:250px}.chart-wrap canvas{width:100%;height:100%;display:block}.chart-empty{position:absolute;inset:0;display:grid;place-items:center;color:var(--muted);font-size:13px}.chart-legend{display:flex;gap:16px;color:var(--muted);font-size:10px;margin-top:10px}.chart-legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.detail-row{display:grid;grid-template-columns:1fr auto auto;gap:10px;align-items:center;padding:12px 0;border-bottom:1px solid var(--line)}.detail-row:last-child{border:0}.detail-row strong,.detail-row span{display:block}.detail-row strong{font-size:12px}.detail-row div>span{font-size:10px;color:var(--muted);margin-top:3px}.detail-number{font-variant-numeric:tabular-nums;font-weight:650;font-size:13px}
    details{margin-top:14px;background:rgba(255,255,255,.6);border-radius:14px;padding:0 14px}summary{padding:13px 0;cursor:pointer;color:var(--muted);font-size:12px}.formula{padding:0 0 14px;font-size:11px;line-height:1.7;color:var(--muted)}footer{margin-top:35px;padding:22px 3px;border-top:1px solid var(--line);color:var(--muted);font-size:11px;line-height:1.75}footer strong{color:var(--ink)}
    @media(max-width:860px){.market-grid{grid-template-columns:repeat(2,1fr)}.qqq-strip{grid-template-columns:1fr}.qqq-values{grid-template-columns:repeat(3,1fr)}.dimension-grid{grid-template-columns:repeat(2,1fr)}.dimension-card:last-child{grid-column:span 2}.lower-grid{grid-template-columns:1fr}.hero-grid{grid-template-columns:1fr}.allocation{text-align:left;display:flex;align-items:center;gap:13px;min-width:0}.allocation strong{font-size:34px;margin:0}.score{font-size:78px}}
    @media(max-width:520px){.shell{width:min(100% - 20px,1120px);padding-top:20px}.topbar{align-items:flex-start}.meta{max-width:160px}.hero{padding:26px 22px;border-radius:24px;min-height:0}.score{font-size:70px}.summary{font-size:15px;margin-top:20px}.market-card{padding:16px}.market-value{font-size:25px}.qqq-strip{padding:16px}.qqq-values{gap:13px 4px}.qqq-values div{padding-left:9px}.dimension-grid{grid-template-columns:1fr 1fr;gap:9px}.dimension-card{padding:15px}.dimension-card p{min-height:33px}.section-head{align-items:flex-start}.section-head p{text-align:right;max-width:180px}.chart-card,.detail-card{padding:16px}}
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar"><div class="brand"><i></i>Market Pulse</div><div class="meta">数据日 $report_date<br>生成于 $generated_at</div></header>
    <section class="hero">
      <div class="hero-grid">
        <div><div class="eyebrow">Market temperature · 市场温度</div><div class="score-line"><strong class="score">$score</strong><span class="score-unit">/ 100</span><span class="state">$state</span></div><p class="summary">$summary</p></div>
        <div class="allocation"><span>模型参考定投倍率</span><strong>$multiplier</strong><small>先结论，再看原因</small></div>
      </div>
      $alerts
    </section>
    $issues
    <section><div class="section-head"><h2>今日市场</h2><p>红涨绿跌 · 收盘行情</p></div><div class="market-grid">$market_cards</div>$qqq_strip</section>
    <section><div class="section-head"><h2>五个维度</h2><p>分数越高，逆向配置吸引力越高</p></div><div class="dimension-grid">$dimension_cards</div></section>
    <section class="lower-grid">
      <article class="chart-card"><div class="section-head"><h2>历史温度</h2><p>最近 180 个有效记录</p></div><div class="chart-wrap"><canvas id="historyChart" aria-label="市场温度历史曲线"></canvas><div class="chart-empty" id="chartEmpty">积累第二个有效交易日后显示趋势</div></div><div class="chart-legend"><span><i style="background:#1b78d0"></i>市场温度</span><span><i style="background:#e7bb71"></i>正常区间 50–65</span></div></article>
      <article class="detail-card"><div class="section-head"><h2>指标状态</h2><p>不隐藏缺失与过期</p></div>$detail_rows<details><summary>查看透明模型权重</summary><div class="formula">$weight_text<br>温度越高表示估值、情绪、趋势、仓位与利率环境的逆向吸引力越高。缺少任一维度时不强行计算总分。</div></details></article>
    </section>
    <footer><strong>数据说明</strong><br>行情来自 Yahoo Finance；估值、CNN 市场情绪、NAAIM 与利率环境评分来自 config.json 手工配置，页面会标明新鲜度。VXN 分位使用最多 756 个最近交易日。<br><br><strong>免责声明</strong><br>本站仅作个人研究和记录，模型倍率不是买卖指令，不构成任何投资建议。市场有风险，请独立判断。</footer>
  </main>
  <script>
  (()=>{const data=$chart_data,canvas=document.getElementById('historyChart'),empty=document.getElementById('chartEmpty');if(data.length<2)return;empty.hidden=true;const ctx=canvas.getContext('2d'),dpr=Math.min(devicePixelRatio||1,2);function draw(){const box=canvas.getBoundingClientRect(),w=Math.max(280,box.width),h=Math.max(180,box.height);canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const p={l:34,r:12,t:16,b:25},cw=w-p.l-p.r,ch=h-p.t-p.b,x=i=>p.l+i/(data.length-1)*cw,y=v=>p.t+(100-v)/100*ch;ctx.fillStyle='rgba(224,164,69,.08)';ctx.fillRect(p.l,y(65),cw,y(50)-y(65));ctx.strokeStyle='rgba(61,82,98,.11)';ctx.lineWidth=1;ctx.font='10px -apple-system,sans-serif';ctx.fillStyle='#82909c';ctx.textAlign='right';[0,20,40,60,80,100].forEach(v=>{ctx.beginPath();ctx.moveTo(p.l,y(v));ctx.lineTo(w-p.r,y(v));ctx.stroke();ctx.fillText(v,p.l-8,y(v)+3)});const grad=ctx.createLinearGradient(0,p.t,0,h-p.b);grad.addColorStop(0,'rgba(23,105,224,.22)');grad.addColorStop(1,'rgba(23,105,224,0)');ctx.beginPath();data.forEach((v,i)=>i?ctx.lineTo(x(i),y(v.score)):ctx.moveTo(x(i),y(v.score)));ctx.lineTo(x(data.length-1),h-p.b);ctx.lineTo(x(0),h-p.b);ctx.closePath();ctx.fillStyle=grad;ctx.fill();ctx.beginPath();data.forEach((v,i)=>i?ctx.lineTo(x(i),y(v.score)):ctx.moveTo(x(i),y(v.score)));ctx.strokeStyle='#1769e0';ctx.lineWidth=2.2;ctx.lineJoin='round';ctx.stroke();ctx.fillStyle='#687889';ctx.textAlign='left';ctx.fillText(data[0].date.slice(5),p.l,h-5);ctx.textAlign='right';ctx.fillText(data.at(-1).date.slice(5),w-p.r,h-5)}draw();new ResizeObserver(draw).observe(canvas)})();
  </script>
</body>
</html>''')
    rendered = template.substitute(
        report_date=esc(snapshot["report_date"]),
        generated_at=esc(snapshot["generated_at"].replace("T", " ")),
        score=score_text,
        state=esc(model["state"]),
        summary=esc(snapshot["summary"]),
        multiplier=multiplier_text,
        alerts=alert_html,
        issues=issue_html,
        market_cards=market_cards(snapshot),
        qqq_strip=qqq_strip(snapshot),
        dimension_cards=dimension_cards(snapshot),
        detail_rows=detail_rows(snapshot),
        weight_text=esc(weight_text),
        chart_data=chart_payload(history),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_suffix(output.suffix + ".tmp")
    temp_path.write_text(rendered, encoding="utf-8")
    temp_path.replace(output)


def notification_markdown(snapshot: dict[str, Any]) -> tuple[str, str]:
    model, market = snapshot["model"], snapshot["market"]
    score = "不可用" if model["score"] is None else f'{model["score"]:.1f}（{model["state"]}）'
    multiplier = "不可用" if model["multiplier"] is None else f'{model["multiplier"]:g}x'
    title = f'Market Pulse｜{snapshot["report_date"]}｜{model["state"]}'
    lines = [
        f'## 市场温度：{score}',
        f'**模型参考定投倍率：{multiplier}**',
        '',
        f'- 纳指100：{fmt_change(market["ndx"].get("daily_return"))}',
        f'- 标普500：{fmt_change(market["spx"].get("daily_return"))}',
        f'- 恐慌指数：{fmt(market["vxn"].get("value"), 1)}',
        f'- 美债10年：{fmt(market["us10y"].get("value"), 2, "%")}',
        f'- QQQ 短期热度：{fmt(market["qqq"].get("rsi"), 0)}',
        '',
        f'- Nasdaq-100 Forward PE: {fmt(snapshot["manual"]["forward_pe"].get("value"), 2)}',
        f'- Nasdaq-100 TTM PE: {fmt(snapshot["manual"]["ttm_pe"].get("value"), 2)}',
        snapshot["summary"],
    ]
    if snapshot["alerts"]:
        lines.extend(['', '**事件提醒**', *[f'- {item}' for item in snapshot["alerts"]]])
    unavailable = [
        f'{metric.get("ticker", name)}（{metric.get("error") or "不可用"}）'
        for group in (snapshot["market"], snapshot["manual"])
        for name, metric in group.items()
        if metric.get("freshness_status") == "unavailable"
    ]
    stale = [
        metric.get("ticker", name) for group in (snapshot["market"], snapshot["manual"])
        for name, metric in group.items() if metric.get("freshness_status") == "stale"
    ]
    if unavailable:
        lines.extend(['', '**不可用项**', *[f'- {item}' for item in unavailable]])
    if stale:
        lines.extend(['', f'过期/回退数据：{"、".join(stale)}'])
    lines.extend(['', '> 仅作个人研究和记录，不构成投资建议。'])
    return title, '\n'.join(lines)


def send_serverchan(snapshot: dict[str, Any], sendkey: str | None) -> bool | None:
    if not sendkey:
        print("未设置 SERVERCHAN_SENDKEY，跳过微信推送。")
        return None
    title, description = notification_markdown(snapshot)
    try:
        if "\n" in title or "\r" in title:
            raise ValueError("ServerChan title must not contain newlines")
        endpoint = f"https://sctapi.ftqq.com/{sendkey}.send"
        if sendkey.lower().startswith("sctp"):
            uid_match = re.match(r"sctp(\d+)t", sendkey, flags=re.I)
            if uid_match:
                endpoint = f"https://{uid_match.group(1)}.push.ft07.com/send/{sendkey}.send"
        response = requests.post(
            endpoint,
            data={"title": title, "desp": description},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        code = payload.get("code")
        if str(code) != "0":
            raise RuntimeError(payload.get("message") or payload.get("data") or "Server酱返回失败")
        message = str(payload.get("message") or "accepted")
        print(f"ServerChan accepted: code={code}, message={message}")
        return True
    except Exception as exc:
        print(f"警告：Server酱推送失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 Market Pulse 静态仪表盘")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="配置文件路径")
    parser.add_argument("--no-push", action="store_true", help="即使存在 SendKey 也不推送")
    args = parser.parse_args(argv)
    config = load_json(args.config)
    if not isinstance(config, dict):
        print(f"错误：无法读取配置文件 {args.config}", file=sys.stderr)
        return 2
    try:
        validate_config(config)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"错误：配置无效：{exc}", file=sys.stderr)
        return 2
    now = get_now(str(config.get("timezone", "Asia/Shanghai")))
    previous = load_json(LATEST_PATH, {})
    snapshot = build_snapshot(config, now, previous)
    history = upsert_history(snapshot_to_row(snapshot))
    save_json(LATEST_PATH, snapshot)
    render_html(snapshot, history)
    print(f"已生成 {SITE_PATH.relative_to(ROOT)}，数据日 {snapshot['report_date']}，质量 {snapshot['data_quality']}。")
    if snapshot["fetch_failures"]:
        print("行情失败项：" + "；".join(snapshot["fetch_failures"]), file=sys.stderr)
    if not args.no_push:
        send_serverchan(snapshot, os.environ.get("SERVERCHAN_SENDKEY"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
