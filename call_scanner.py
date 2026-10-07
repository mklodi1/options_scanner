"""
Call-options setup scanner
Squeeze + MACD + RSI + Volume Profile  ->  GARCH volatility model  ->  Monte Carlo trade simulation

Setup:  pip install -r requirements.txt
Run:    python call_scanner.py

Data: Yahoo via yfinance (unofficial, ~15 min delayed). Analysis tool, not financial advice.
"""
import datetime as dt
import math
import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from garch import fit_garch
from indicators import macd, rsi, ttm_squeeze, volume_profile
from montecarlo import evaluate_call, plot_trade, simulate_paths
from pricing import bs_call, implied_vol

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "AMZN", "NVDA", "MSFT", "AMD", "MSTR"]

HOLD_DAYS = 10                 # calendar days (~7 trading days)
DTE_MIN, DTE_MAX = 21, 50
DELTA_MIN, DELTA_MAX = 0.50, 0.70
RSI_LOW, RSI_HIGH = 50, 70
PROFILE_DAYS, PROFILE_BINS = 60, 50
MIN_TARGET_PCT = 0.03          # target = first high-volume node >= 3% above price
MAX_STOP_PCT = 0.08            # stop never more than 8% below price
MIN_OPEN_INTEREST = 100
MAX_SPREAD_PCT = 0.15
FALLBACK_RATE = 0.04

# models
HISTORY = "2y"                 # price history used to fit GARCH
N_PATHS = 20000                # Monte Carlo paths per ticker
SIM_METHOD = "fhs"             # "fhs" (resample real residuals) or "t" (Student-t draws)
IV_SENSITIVITY = 0.5           # how much IV follows simulated GARCH vol (0 = fixed, 1 = 1:1)
MAX_IV_TO_GARCH = 1.15         # "IV cheap" = market IV <= 1.15x GARCH-forecast vol
SEED = 42
MAKE_PLOTS = True


SETTABLE = {"TICKERS", "HOLD_DAYS", "N_PATHS", "SIM_METHOD", "IV_SENSITIVITY", "MAX_IV_TO_GARCH",
            "DTE_MIN", "DTE_MAX", "DELTA_MIN", "DELTA_MAX", "SEED", "MAKE_PLOTS"}


def configure(**kwargs):
    """Override settings at runtime (used by the dashboard and the snapshot job)."""
    for k, v in kwargs.items():
        if k not in SETTABLE:
            raise KeyError(f"unknown setting {k}")
        globals()[k] = v


# ---------------------------------------------------------------------------
def risk_free_rate():
    try:
        return float(yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1]) / 100
    except Exception:
        return FALLBACK_RATE


def next_earnings(tk):
    try:
        cal = tk.calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        future = [pd.Timestamp(d).date() for d in (dates or [])
                  if pd.Timestamp(d).date() >= dt.date.today()]
        return min(future) if future else None
    except Exception:
        return None


def dividend_yield(tk, price):
    try:
        d = tk.dividends
        if d is None or len(d) == 0:
            return 0.0
        return float(d[d.index > d.index[-1] - pd.Timedelta(days=365)].sum()) / price
    except Exception:
        return 0.0


def levels(df, S, vp):
    floor = S * (1 + MIN_TARGET_PCT)
    above = [h for h in vp["hvns"] if h > floor]
    if above:
        target = min(above)
    elif vp["vah"] > floor:
        target = vp["vah"]
    else:
        target = max(float(df["High"].tail(PROFILE_DAYS).max()), floor)
    supports = [x for x in [vp["poc"], vp["val"], vp["vah"]] + vp["hvns"] if x < S * 0.99]
    stop = max(max(supports) if supports else 0, S * (1 - MAX_STOP_PCT))
    return target, stop


# ---------------------------------------------------------------------------
def scan(ticker, r):
    tk = yf.Ticker(ticker)
    df = tk.history(period=HISTORY, auto_adjust=False).dropna()
    if len(df) < 250:
        return {"ticker": ticker, "error": "not enough price history"}

    S = float(df["Close"].iloc[-1])
    q = dividend_yield(tk, S)
    today = dt.date.today()

    # --- technicals ---
    rsi_now = float(rsi(df["Close"]).iloc[-1])
    m_line, m_sig, m_hist = macd(df["Close"])
    sq_on, mom = ttm_squeeze(df)
    vp = volume_profile(df, PROFILE_DAYS, PROFILE_BINS)
    target, stop = levels(df, S, vp)

    fired = bool(sq_on.iloc[-6:-1].any() and not sq_on.iloc[-1])
    mom_up = bool(mom.iloc[-1] > 0 and mom.iloc[-1] > mom.iloc[-2])
    squeeze_txt = ("FIRED, mom up" if fired and mom_up else "ON (coiling)" if sq_on.iloc[-1]
                   else "fired, mom weak" if fired else "none")
    checks = {
        "Squeeze": fired and mom_up,
        "MACD": bool(m_line.iloc[-1] > m_sig.iloc[-1] and m_hist.iloc[-1] > m_hist.iloc[-2]),
        "RSI": RSI_LOW <= rsi_now <= RSI_HIGH,
        "VolProfile": S > vp["poc"],
    }

    # --- GARCH volatility model ---
    rets = 100 * np.log(df["Close"] / df["Close"].shift()).dropna().values
    g = fit_garch(rets)
    hold_td = max(round(HOLD_DAYS * 252 / 365), 1)

    # --- Monte Carlo: simulate once, test every contract on the same paths ---
    prices, sig2 = simulate_paths(g, S, hold_td, N_PATHS, SIM_METHOD, SEED)
    earnings = next_earnings(tk)

    best, atm, candidates = None, {}, []
    for exp in tk.options:
        exp_d = dt.datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_d - today).days
        if not DTE_MIN <= dte <= DTE_MAX:
            continue
        T = dte / 365
        try:
            chain = tk.option_chain(exp).calls
        except Exception:
            continue
        for _, c in chain.iterrows():
            bid, ask, K = c.get("bid") or 0, c.get("ask") or 0, float(c["strike"])
            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2
            if (ask - bid) / mid > MAX_SPREAD_PCT or (c.get("openInterest") or 0) < MIN_OPEN_INTEREST:
                continue
            iv = implied_vol(mid, S, K, T, r, q)
            if iv is None:
                continue
            if exp not in atm or abs(K - S) < abs(atm[exp][1] - S):
                atm[exp] = (iv, K)
            gk = bs_call(S, K, T, r, iv, q)
            if not DELTA_MIN <= gk["delta"] <= DELTA_MAX:
                continue

            mc = evaluate_call(prices, sig2, K, dte, ask, iv, target, stop, r, q,
                               IV_SENSITIVITY, exit_cost_frac=(ask - bid) / 2 / mid)
            cand = {"exp": exp, "dte": dte, "strike": K, "entry": ask, "iv": iv, **gk,
                    "garch_vol": g.forecast_vol(dte * 252 / 365),
                    "earnings_in_window": earnings is not None and earnings <= exp_d, "mc": mc}
            candidates.append({k: cand[k] for k in ("exp", "dte", "strike", "entry", "iv", "delta",
                                                     "theta", "vega", "earnings_in_window")}
                              | {k: mc[k] for k in ("ev", "ev_pct", "p_profit", "p_target", "p_stop")})
            key = (not cand["earnings_in_window"], mc["ev_pct"])
            if best is None or key > (not best["earnings_in_window"], best["mc"]["ev_pct"]):
                best = cand

    if best and best["exp"] in atm:
        iv_ref, gv = atm[best["exp"]][0], best["garch_vol"]
    elif atm:
        e = min(atm)
        iv_ref = atm[e][0]
        gv = g.forecast_vol(((dt.date.fromisoformat(e) - today).days) * 252 / 365)
    else:
        iv_ref, gv = None, g.forecast_vol(21)
    checks["IV cheap"] = iv_ref is not None and iv_ref / gv <= MAX_IV_TO_GARCH

    return {"ticker": ticker, "price": S, "checks": checks, "score": sum(checks.values()),
            "squeeze_txt": squeeze_txt, "rsi": rsi_now, "vp": vp, "target": target, "stop": stop,
            "garch": g, "garch_vol": gv, "iv_ref": iv_ref, "earnings": earnings,
            "best": best, "prices": prices, "hold_td": hold_td, "df": df, "candidates": candidates}


# ---------------------------------------------------------------------------
def grade(x):
    b = x["best"]
    ev_sig = b is not None and b["mc"]["ev_ci"][0] > 0   # EV positive at 95% confidence
    earn_ok = b is not None and not b["earnings_in_window"]
    if x["score"] == 5 and ev_sig and earn_ok:
        return "A  all checks + positive EV"
    if x["score"] >= 4 and ev_sig:
        return "B  close, review the miss"
    if x["squeeze_txt"].startswith("ON") and x["checks"]["VolProfile"]:
        return "W  watchlist: coiling"
    return "-  no setup"


def report(results):
    ok = sorted([x for x in results if "error" not in x],
                key=lambda x: (grade(x)[0] not in "A", grade(x)[0] not in "AB", -x["score"],
                               -(x["best"]["mc"]["ev_pct"] if x["best"] else -9)))
    line = "=" * 82
    print(f"\n{line}\n CALL SETUP SCAN  {dt.datetime.now():%Y-%m-%d %H:%M}   "
          f"hold {HOLD_DAYS}d   {N_PATHS:,} GARCH paths ({SIM_METHOD})\n{line}")
    print(f"{'Ticker':<7}{'Price':>9}  {'Score':<6}{'Grade':<30}{'EV/premium':>11}  {'P(profit)':>9}")
    for x in ok:
        b = x["best"]
        ev = f"{b['mc']['ev_pct']*100:+.1f}%" if b else "n/a"
        pp = f"{b['mc']['p_profit']*100:.0f}%" if b else "n/a"
        print(f"{x['ticker']:<7}{x['price']:>9.2f}  {x['score']}/5   {grade(x):<30}{ev:>11}  {pp:>9}")

    for x in ok:
        g, b, vp = x["garch"], x["best"], x["vp"]
        print(f"\n{'-'*82}\n{x['ticker']}  ${x['price']:.2f}   {grade(x)}")
        print("  " + "  ".join(f"[{'x' if v else ' '}] {k}" for k, v in x["checks"].items()))
        print(f"  Squeeze {x['squeeze_txt']} | RSI {x['rsi']:.1f} | POC {vp['poc']:.2f} "
              f"VAL {vp['val']:.2f} VAH {vp['vah']:.2f}")
        print(f"  Target {x['target']:.2f} ({(x['target']/x['price']-1)*100:+.1f}%)  "
              f"Stop {x['stop']:.2f} ({(x['stop']/x['price']-1)*100:+.1f}%)"
              + (f"  | earnings {x['earnings']}" if x["earnings"] else ""))
        se = g.std_errors
        print(f"  GARCH(1,1)-t: alpha {g.alpha:.3f} (±{se['alpha']:.3f})  beta {g.beta:.3f} "
              f"(±{se['beta']:.3f})  nu {g.nu:.1f}  persistence {g.persistence:.3f}  "
              f"half-life {g.half_life_days:.0f}d")
        iv_txt = (f"market IV {x['iv_ref']*100:.0f}% = {x['iv_ref']/x['garch_vol']:.2f}x GARCH"
                  if x["iv_ref"] else "no market IV")
        print(f"  Vol: long-run {g.long_run_vol*100:.0f}%  forecast to expiry "
              f"{x['garch_vol']*100:.0f}%  {iv_txt}")
        if not b:
            print("  No liquid call in the delta/DTE range.")
            continue
        mc = b["mc"]
        print(f"  Call {b['exp']} {b['strike']:g}C ({b['dte']} DTE) ask ${b['entry']:.2f} "
              f"(${b['entry']*100:.0f})  delta {b['delta']:.2f}  theta {b['theta']:.3f}/d  "
              f"vega {b['vega']:.3f}")
        print(f"  Monte Carlo over {x['hold_td']} trading days:  P(target) {mc['p_target']*100:.0f}%  "
              f"P(stop) {mc['p_stop']*100:.0f}%  P(time exit) {mc['p_time']*100:.0f}%")
        lo, hi = mc["ev_ci"]
        print(f"    EV ${mc['ev']*100:+.0f}/contract  (95% CI ${lo*100:+.0f} to ${hi*100:+.0f})  "
              f"P(profit) {mc['p_profit']*100:.0f}%  median ${mc['median']*100:+.0f}")
        print(f"    5% VaR ${mc['var5']*100:.0f}   CVaR ${mc['cvar5']*100:.0f}")
        if b["earnings_in_window"]:
            print("    ! earnings before expiration - IV crush risk")

    for x in results:
        if "error" in x:
            print(f"\n{x['ticker']}: skipped ({x['error']})")
    print("\nNot financial advice. Model outputs depend on the assumptions listed in README.md.\n")


def save_outputs(results):
    rows = []
    for x in results:
        if "error" in x:
            continue
        b, g = x["best"] or {}, x["garch"]
        mc = b.get("mc", {})
        rows.append({"ticker": x["ticker"], "price": round(x["price"], 2), "grade": grade(x),
                     "score": x["score"], **x["checks"], "rsi": round(x["rsi"], 1),
                     "target": round(x["target"], 2), "stop": round(x["stop"], 2),
                     "garch_alpha": g.alpha, "garch_beta": g.beta, "garch_nu": g.nu,
                     "garch_vol": x["garch_vol"], "market_iv": x["iv_ref"],
                     "expiry": b.get("exp"), "strike": b.get("strike"), "entry": b.get("entry"),
                     "delta": b.get("delta"), "p_target": mc.get("p_target"),
                     "p_stop": mc.get("p_stop"), "ev_per_contract": mc.get("ev", 0) * 100,
                     "p_profit": mc.get("p_profit")})
        if MAKE_PLOTS and b:
            os.makedirs("plots", exist_ok=True)
            plot_trade(x["ticker"], x["prices"], mc["pnl"], x["target"], x["stop"],
                       b["entry"], f"plots/{x['ticker']}.png")
    pd.DataFrame(rows).to_csv("scan_results.csv", index=False)
    print("Saved scan_results.csv" + (" and plots/" if MAKE_PLOTS else ""))


if __name__ == "__main__":
    rate = risk_free_rate()
    print(f"Risk-free rate {rate*100:.2f}%. Fitting models for {', '.join(TICKERS)} ...")
    results = []
    for t in TICKERS:
        try:
            results.append(scan(t, rate))
            print(f"  {t} done")
        except Exception as e:
            results.append({"ticker": t, "error": f"{type(e).__name__}: {e}"})
    report(results)
    save_outputs(results)
