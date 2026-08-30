# Market Pulse

**English** | [简体中文](README.zh-cn.md)

[![Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-2ea44f)](https://imvictorma.github.io/market-pulse/)
[![Daily Update](https://github.com/imvictorma/market-pulse/actions/workflows/daily.yml/badge.svg)](https://github.com/imvictorma/market-pulse/actions/workflows/daily.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Market Pulse is a Nasdaq-100 market-temperature dashboard for long-term investors. After the US market close it automatically fetches quotes, computes a transparent five-factor temperature model, generates a mobile-first static page, and can push updates to WeChat via ServerChan.

It is not an up/down predictor. A higher temperature means higher "contrarian allocation appeal" across valuation, sentiment, trend, positioning, and the rate environment. The DCA multiplier shown on the page is model output, not investment advice.

## Features

- Automatic quotes: Nasdaq-100, S&P 500, QQQ, VXN, US 10-year Treasury yield
- Automatic indicators: 1/5/20-day changes, Wilder RSI(14), MA200 deviation, VXN percentile over the past 756 trading days
- Manual indicators: Forward PE, TTM PE, CNN Fear & Greed, NAAIM, rate-environment score
- Transparent five-factor scoring, temperature bands, reference DCA multiplier, and event alerts
- `data/history.csv` upserts by data date — rerunning the same day creates no duplicates
- On failure it reuses the last successful snapshot and marks it `stale`; with no previous value it shows "unavailable" — it never fabricates data
- Self-contained HTML/CSS/JS dashboard; no frontend framework or CDN
- GitHub Actions updates automatically on trading days, commits history, deploys Pages, and sends ServerChan notifications

## Local Setup

Requires Python 3.12.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate.py --no-push
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python generate.py --no-push
```

The output is `site/index.html`. From the repository root, preview it locally with:

```bash
python -m http.server 8000 --directory site
```

Then open `http://localhost:8000` in your browser.

## Configuring Manual Inputs

Before the first run, edit `manual_inputs` in `config.json`. Defaults are intentionally `null`: until you enter real data, the page shows "insufficient data" instead of passing sample numbers off as market data.

Recommended fields for each manual item:

- `value`: current absolute value; required for both PE items
- `percentile`: only for the two PE items, range 0–100
- `as_of`: the data date, format `YYYY-MM-DD`
- `source`: name or link describing the data source
- `percentile_window`: PE percentile window, e.g. `10Y`
- `max_age_days`: days after which the item is marked stale

Example structure (numbers illustrate the format only — replace with values you have verified):

```json
"forward_pe": {
  "value": 25.0,
  "percentile": 60,
  "as_of": "2026-08-06",
  "source": "your data source",
  "percentile_window": "10Y",
  "max_age_days": 45
}
```

Model thresholds, factor weights, MA200 bands, temperature multipliers, and alert thresholds all live in `config.json` — no Python changes needed. After changing weights, the program validates that each weight group sums to 1.

## Deploying to GitHub Pages from Scratch

1. Create a new GitHub repository and push everything in this directory to the default branch.
2. In `Settings → Pages`, set Source under *Build and deployment* to **GitHub Actions**.
3. In `Settings → Secrets and variables → Actions`, create a repository secret:
   - Name: `SERVERCHAN_SENDKEY`
   - Value: a freshly regenerated SendKey from the ServerChan console
4. Do not reuse old keys that have appeared in chats or screenshots, and never put the key into `config.json`, code, or commit history.
5. Go to `Actions → Daily Market Pulse → Run workflow` and run it once manually.
6. After the workflow succeeds, the Pages URL appears in the deployment job and in the repository Pages settings.

The workflow uses `30 23 * * 1-5` — UTC Mon–Fri 23:30, i.e. 07:30 Beijing time the next day. GitHub scheduled jobs may lag a few minutes. On US market holidays it usually upserts the most recent data date instead of fabricating that day's quotes.

> If the default branch is protected, the GitHub Actions bot may be unable to push generated files. Grant the workflow write access, or adjust branch rules to allow `github-actions[bot]` writes. In `Settings → Actions → General → Workflow permissions`, choose **Read and write permissions**.

## Data Failures & Freshness

The page uses four states:

- `最新` Fresh: quotes fetched successfully this run
- `手工` Manual: from `config.json` and still within its configured freshness window
- `已过期` Stale: quotes reused after a fetch failure, or manual data past `max_age_days`
- `不可用` Unavailable: no data this run and no old value to fall back on

The last complete quote snapshot is stored in `data/latest.json`. If a single ticker fails, the others still generate; failures are listed in the push notification. If any of the five factor scores is missing, the overall temperature explicitly shows unavailable — it never silently recomputes with the remaining weights.

## Model Formulas

```text
Valuation  = (100 − Forward PE percentile) × 62.5%
           + (100 − TTM PE percentile) × 37.5%

Sentiment  = VXN percentile × 60%
           + (100 − CNN Fear & Greed) × 40%

Trend      = (100 − RSI) × 50%
           + MA200 deviation score × 50%

Positioning = 100 − NAAIM
Macro       = manual rate-environment score

Temperature = Valuation × 40% + Sentiment × 25% + Trend × 20%
            + Positioning × 10% + Macro × 5%
```

Default temperature bands: Freezing ≥80 (3×), Cold ≥65 (2×), Neutral ≥50 (1×), Warm ≥35 (0.6×), Hot ≥20 (0.3×), Extreme <20 (0×).

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover indicator calculation, model formulas, key threshold boundaries, missing-data behavior, and history upserts.
