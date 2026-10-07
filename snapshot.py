"""
Run the full scan and save everything the dashboard needs to data/snapshot.json.

    python snapshot.py

GitHub Actions runs this on a schedule (see .github/workflows/scan.yml), so the
deployed dashboard loads instantly and never has to call Yahoo itself.
"""
import datetime as dt
import json
import math
import os

import numpy as np
import pandas as pd

import call_scanner as cs
from indicators import macd, rsi, ttm_squeeze

CHART_DAYS = 180       # price history shown on charts
FORECAST_DAYS = 60     # GARCH forecast horizon shown (trading days)
SAMPLE_PATHS = 100     # Monte Carlo paths drawn on the chart
SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "snapshot.json")


def _clean(o):
    """Make numpy/pandas objects JSON-safe (NaN/inf become null)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (dt.date, dt.datetime, pd.Timestamp)):
        return o.isoformat()
    return o


def _dates(index):
    return [d.strftime("%Y-%m-%d") for d in index]


def ticker_payload(x):
    """Everything the dashboard draws for one ticker."""
    full = x["df"]
    df = full.tail(CHART_DAYS)
    n = len(df)
    m_line, m_sig, m_hist = macd(full["Close"])
    sq_on, mom = ttm_squeeze(full)
    g, b = x["garch"], x["best"]

    # in-sample GARCH volatility (annualized); sigma2 aligns with returns, i.e. full.index[1:]
    garch_vol = np.sqrt(g.sigma2 * 252) / 100
    garch_vol = pd.Series(garch_vol, index=full.index[1:]).reindex(df.index)

    # forecast of daily volatility for each future day (annualized)
    p = g.persistence
    vl = g.omega / max(1 - p, 1e-8)
    s1 = g.next_sigma2()
    fc = np.sqrt((vl + p ** np.arange(FORECAST_DAYS) * (s1 - vl)) * 252) / 100
    fc_dates = pd.bdate_range(full.index[-1].tz_localize(None) + pd.Timedelta(days=1),
                              periods=FORECAST_DAYS)

    out = {
        "ticker": x["ticker"], "price": x["price"], "grade": cs.grade(x), "score": x["score"],
        "checks": x["checks"], "squeeze_txt": x["squeeze_txt"], "rsi": x["rsi"],
        "target": x["target"], "stop": x["stop"], "earnings": x["earnings"],
        "levels": {k: x["vp"][k] for k in ("poc", "vah", "val")},
        "profile": {"centers": x["vp"]["centers"], "volume": x["vp"]["volume"]},
        "ohlc": {"dates": _dates(df.index), "open": df["Open"].values, "high": df["High"].values,
                 "low": df["Low"].values, "close": df["Close"].values, "volume": df["Volume"].values},
        "ind": {"rsi": rsi(full["Close"]).tail(n).values, "macd": m_line.tail(n).values,
                "signal": m_sig.tail(n).values, "hist": m_hist.tail(n).values,
                "sq_on": sq_on.tail(n).values, "sq_mom": mom.tail(n).values},
        "garch": {"mu": g.mu, "omega": g.omega, "alpha": g.alpha, "beta": g.beta, "nu": g.nu,
                  "se": g.std_errors, "loglik": g.loglik, "persistence": g.persistence,
                  "half_life": g.half_life_days, "long_run_vol": g.long_run_vol,
                  "vol_hist": garch_vol.values, "fc_dates": _dates(fc_dates), "fc_vol": fc,
                  "vol_to_expiry": x["garch_vol"]},
        "market_iv": x["iv_ref"],
        "candidates": sorted(x["candidates"], key=lambda c: -c["ev_pct"]),
        "best": None, "mc": None,
    }

    if b:
        mc = b["mc"]
        pnl = mc["pnl"] * 100
        counts, edges = np.histogram(pnl, bins=60)
        prices = x["prices"]
        out["best"] = {k: b[k] for k in ("exp", "dte", "strike", "entry", "iv", "delta", "gamma",
                                         "theta", "vega", "garch_vol", "earnings_in_window")}
        out["mc"] = {
            **{k: mc[k] for k in ("p_target", "p_stop", "p_time", "ev", "ev_se", "ev_ci",
                                  "ev_pct", "p_profit", "median", "var5", "cvar5")},
            "n_paths": len(pnl), "hold_td": x["hold_td"],
            "paths": prices[:SAMPLE_PATHS],
            "bands": {str(q): np.percentile(prices, q, axis=0) for q in (5, 25, 50, 75, 95)},
            "hist_counts": counts, "hist_edges": edges,
        }
    return out


def summary_row(t):
    b, mc = t["best"] or {}, t["mc"] or {}
    return {"Ticker": t["ticker"], "Price": t["price"], "Grade": t["grade"], "Score": t["score"],
            **t["checks"],
            "Best call": f"{b['exp']} {b['strike']:g}C" if b else None,
            "EV/contract": mc.get("ev", 0) * 100 if mc else None,
            "EV/premium": mc.get("ev_pct") if mc else None,
            "P(profit)": mc.get("p_profit") if mc else None,
            "IV/GARCH": (t["market_iv"] / t["garch"]["vol_to_expiry"])
            if t["market_iv"] else None}


def build_snapshot(tickers=None, run_tests=True, **settings):
    if tickers:
        settings["TICKERS"] = list(tickers)
    settings.setdefault("MAKE_PLOTS", False)
    cs.configure(**settings)
    rate = cs.risk_free_rate()

    payloads, errors = [], []
    for t in cs.TICKERS:
        try:
            payloads.append(ticker_payload(cs.scan(t, rate)))
        except Exception as e:
            errors.append({"ticker": t, "error": f"{type(e).__name__}: {e}"})

    order = {"A": 0, "B": 1, "W": 2, "-": 3}
    payloads.sort(key=lambda t: (order.get(t["grade"][0], 9), -t["score"],
                                 -(t["mc"]["ev_pct"] if t["mc"] else -9)))
    tests = []
    if run_tests:
        from test_models import run_all
        tests = run_all()

    return _clean({
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "risk_free_rate": rate,
        "settings": {k: getattr(cs, k) for k in sorted(cs.SETTABLE) if k != "MAKE_PLOTS"},
        "summary": [summary_row(t) for t in payloads],
        "tickers": payloads, "errors": errors, "tests": tests,
    })


if __name__ == "__main__":
    snap = build_snapshot()
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snap, f, separators=(",", ":"))
    ok = sum(t["passed"] for t in snap["tests"])
    print(f"Saved {SNAPSHOT_PATH}  ({len(snap['tickers'])} tickers, "
          f"{len(snap['errors'])} errors, tests {ok}/{len(snap['tests'])})")
    for e in snap["errors"]:
        print(f"  {e['ticker']}: {e['error']}")
