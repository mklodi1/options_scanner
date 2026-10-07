"""
Streamlit dashboard.

    Local:   streamlit run app.py
    Deploy:  Streamlit Community Cloud -> point it at this file (see README).
"""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, "data", "snapshot.json")
README = os.path.join(HERE, "README.md")

GREEN, RED, BLUE, GRAY, ORANGE = "#2e9e5b", "#d64545", "#3b6fd4", "#8a8f98", "#e08a1e"

st.set_page_config(page_title="Options Call Scanner", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_snapshot(path, mtime):
    # mtime is part of the cache key, so a new snapshot from GitHub Actions is picked up
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=900, show_spinner=False)
def live_scan(tickers, hold_days, n_paths, iv_sens, method):
    from snapshot import build_snapshot
    return build_snapshot(list(tickers), run_tests=False, HOLD_DAYS=hold_days,
                          N_PATHS=n_paths, IV_SENSITIVITY=iv_sens, SIM_METHOD=method)


@st.cache_data(show_spinner=False)
def run_tests_live():
    from test_models import run_all
    return run_all()


def money(x, sign=True):
    if x is None:
        return "n/a"
    return f"{'+' if sign and x >= 0 else ''}{'-' if x < 0 else ''}${abs(x):,.0f}"


def pct(x, sign=False):
    return "n/a" if x is None else (f"{x*100:+.1f}%" if sign else f"{x*100:.0f}%")


# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------
def price_chart(t):
    o = t["ohlc"]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.82, 0.18],
                        horizontal_spacing=0.01)
    fig.add_trace(go.Candlestick(x=o["dates"], open=o["open"], high=o["high"], low=o["low"],
                                 close=o["close"], name="price", showlegend=False,
                                 increasing_line_color=GREEN, decreasing_line_color=RED),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=t["profile"]["volume"], y=t["profile"]["centers"], orientation="h",
                         marker_color=GRAY, opacity=0.6, name="volume profile",
                         showlegend=False, hovertemplate="%{y:.2f}: %{x:,.0f}<extra></extra>"),
                  row=1, col=2)
    for y, label, color, dash in [(t["target"], "target", GREEN, "dash"),
                                  (t["stop"], "stop", RED, "dash"),
                                  (t["levels"]["poc"], "POC", ORANGE, "dot"),
                                  (t["levels"]["vah"], "VAH", GRAY, "dot"),
                                  (t["levels"]["val"], "VAL", GRAY, "dot")]:
        fig.add_hline(y=y, line_color=color, line_dash=dash, line_width=1,
                      annotation_text=f"{label} {y:.2f}", annotation_position="top left",
                      row=1, col=1)
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_rangeslider_visible=False, title="Price & volume profile")
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    return fig


def indicator_chart(t):
    o, i = t["ohlc"], t["ind"]
    d = o["dates"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("RSI (14)", "MACD (12, 26, 9)", "Squeeze momentum"))
    fig.add_trace(go.Scatter(x=d, y=i["rsi"], line_color=BLUE, name="RSI"), row=1, col=1)
    for y in (30, 50, 70):
        fig.add_hline(y=y, line_dash="dot", line_color=GRAY, line_width=1, row=1, col=1)
    fig.add_trace(go.Bar(x=d, y=i["hist"], name="histogram", marker_color=[
        GREEN if (h or 0) >= 0 else RED for h in i["hist"]], opacity=0.5), row=2, col=1)
    fig.add_trace(go.Scatter(x=d, y=i["macd"], line_color=BLUE, name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=d, y=i["signal"], line_color=ORANGE, name="signal"), row=2, col=1)
    mom = [m if m is not None else 0 for m in i["sq_mom"]]
    colors = []
    for k, m in enumerate(mom):
        prev = mom[k - 1] if k else m
        colors.append(GREEN if m >= 0 and m >= prev else "#9fd8b4" if m >= 0
                      else RED if m < prev else "#f0a3a3")
    fig.add_trace(go.Bar(x=d, y=mom, marker_color=colors, name="momentum"), row=3, col=1)
    fig.add_trace(go.Scatter(x=d, y=[0] * len(d), mode="markers", name="squeeze on/off",
                             marker=dict(size=5, color=[RED if on else GREEN for on in i["sq_on"]])),
                  row=3, col=1)
    fig.update_layout(height=560, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def vol_chart(t):
    g = t["garch"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["ohlc"]["dates"], y=[v * 100 if v is not None else None
                                                      for v in g["vol_hist"]],
                             name="GARCH conditional vol", line_color=BLUE))
    fig.add_trace(go.Scatter(x=g["fc_dates"], y=[v * 100 for v in g["fc_vol"]],
                             name="GARCH forecast", line=dict(color=BLUE, dash="dash")))
    fig.add_hline(y=g["long_run_vol"] * 100, line_dash="dot", line_color=GRAY,
                  annotation_text=f"long-run {g['long_run_vol']*100:.0f}%")
    if t["market_iv"]:
        fig.add_hline(y=t["market_iv"] * 100, line_color=ORANGE,
                      annotation_text=f"market IV {t['market_iv']*100:.0f}%",
                      annotation_position="bottom right")
    fig.update_layout(height=380, title="Volatility: GARCH(1,1)-t vs market implied",
                      yaxis_title="annualized vol (%)", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def paths_chart(t):
    mc = t["mc"]
    days = list(range(len(mc["paths"][0])))
    fig = go.Figure()
    b = mc["bands"]
    fig.add_trace(go.Scatter(x=days + days[::-1], y=b["95"] + b["5"][::-1], fill="toself",
                             fillcolor="rgba(59,111,212,0.12)", line=dict(width=0),
                             name="5–95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days + days[::-1], y=b["75"] + b["25"][::-1], fill="toself",
                             fillcolor="rgba(59,111,212,0.22)", line=dict(width=0),
                             name="25–75%", hoverinfo="skip"))
    for p in mc["paths"]:
        fig.add_trace(go.Scatter(x=days, y=p, mode="lines", line=dict(width=0.6, color=BLUE),
                                 opacity=0.25, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days, y=b["50"], line=dict(color="#1d3c7a", width=2.5),
                             name="median"))
    fig.add_hline(y=t["target"], line_color=GREEN, line_dash="dash",
                  annotation_text=f"target {t['target']:.2f}")
    fig.add_hline(y=t["stop"], line_color=RED, line_dash="dash",
                  annotation_text=f"stop {t['stop']:.2f}", annotation_position="bottom left")
    fig.update_layout(height=400, title=f"{mc['n_paths']:,} simulated GARCH paths",
                      xaxis_title="trading days", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def pnl_chart(t):
    mc = t["mc"]
    e = np.array(mc["hist_edges"])
    mids = (e[:-1] + e[1:]) / 2
    fig = go.Figure(go.Bar(x=mids, y=mc["hist_counts"], width=np.diff(e),
                           marker_color=[GREEN if m > 0 else RED for m in mids], opacity=0.7,
                           hovertemplate="$%{x:,.0f}: %{y} paths<extra></extra>"))
    fig.add_vline(x=mc["ev"] * 100, line_color=BLUE, line_dash="dash",
                  annotation_text=f"EV {money(mc['ev']*100)}")
    fig.add_vline(x=mc["var5"] * 100, line_color=RED, line_dash="dot",
                  annotation_text=f"5% VaR {money(mc['var5']*100)}",
                  annotation_position="top left")
    fig.update_layout(height=400, title="P&L per contract at exit", xaxis_title="$",
                      yaxis_title="paths", bargap=0, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ---------------------------------------------------------------------------
# PAGE SECTIONS
# ---------------------------------------------------------------------------
def overview(snap):
    rows = snap["summary"]
    if not rows:
        st.warning("No tickers scanned successfully.")
        return
    df = pd.DataFrame(rows)
    for c in ("Squeeze", "MACD", "RSI", "VolProfile", "IV cheap"):
        df[c] = df[c].map({True: "✅", False: "—"})
    st.dataframe(
        df, hide_index=True, width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "EV/contract": st.column_config.NumberColumn(format="$%.0f"),
            "EV/premium": st.column_config.NumberColumn(format="percent"),
            "P(profit)": st.column_config.NumberColumn(format="percent"),
            "IV/GARCH": st.column_config.NumberColumn(format="%.2fx"),
        })
    st.caption("Grades: **A** all five checks + EV positive at 95% confidence + no earnings "
               "before expiry · **B** four checks + significant positive EV · "
               "**W** squeeze coiling above POC · **—** no setup")


def ticker_tab(t):
    b, mc, g = t["best"], t["mc"], t["garch"]
    c = st.columns(5)
    c[0].metric("Price", f"${t['price']:,.2f}")
    c[1].metric("Grade", t["grade"].split()[0], t["grade"][2:].strip(), delta_color="off")
    c[2].metric("EV / contract", money(mc["ev"] * 100) if mc else "n/a",
                pct(mc["ev_pct"], sign=True) + " of premium" if mc else None)
    c[3].metric("P(profit)", pct(mc["p_profit"]) if mc else "n/a")
    ratio = t["market_iv"] / g["vol_to_expiry"] if t["market_iv"] else None
    c[4].metric("IV / GARCH", f"{ratio:.2f}x" if ratio else "n/a",
                "cheap" if t["checks"]["IV cheap"] else "rich", delta_color="off")

    st.markdown("  ".join(f"{'✅' if v else '❌'} **{k}**" for k, v in t["checks"].items())
                + f"  ·  squeeze: {t['squeeze_txt']}  ·  RSI {t['rsi']:.1f}"
                + (f"  ·  next earnings {t['earnings']}" if t["earnings"] else ""))

    st.plotly_chart(price_chart(t), width="stretch")
    with st.expander("Indicators (RSI, MACD, squeeze)"):
        st.plotly_chart(indicator_chart(t), width="stretch")

    st.subheader("Volatility model")
    left, right = st.columns([2, 1])
    left.plotly_chart(vol_chart(t), width="stretch")
    se = g["se"]
    right.markdown("**GARCH(1,1), Student-t, fit by MLE**")
    right.dataframe(pd.DataFrame({
        "parameter": ["ω (omega)", "α (alpha)", "β (beta)", "ν (tail df)"],
        "estimate": [g["omega"], g["alpha"], g["beta"], g["nu"]],
        "std. error": [se["omega"], se["alpha"], se["beta"], se["nu"]],
    }), hide_index=True, width="stretch",
        column_config={"estimate": st.column_config.NumberColumn(format="%.4f"),
                       "std. error": st.column_config.NumberColumn(format="%.4f")})
    right.markdown(f"Persistence α+β **{g['persistence']:.3f}** · shock half-life "
                   f"**{g['half_life']:.0f} days** · long-run vol **{g['long_run_vol']*100:.0f}%** · "
                   f"forecast to expiry **{g['vol_to_expiry']*100:.0f}%**")

    if not b:
        st.info("No liquid call in the delta / expiration range, so no trade was simulated.")
        return

    st.subheader("Monte Carlo trade simulation")
    st.markdown(f"**{b['exp']} {b['strike']:g}C** ({b['dte']} DTE) · ask ${b['entry']:.2f} "
                f"(${b['entry']*100:,.0f}/contract) · delta {b['delta']:.2f} · "
                f"theta {b['theta']:.3f}/day · vega {b['vega']:.3f} · IV {b['iv']*100:.0f}%"
                + ("  \n⚠️ Earnings before expiration: IV crush risk" if b["earnings_in_window"] else ""))
    m = st.columns(6)
    m[0].metric("P(target first)", pct(mc["p_target"]))
    m[1].metric("P(stop first)", pct(mc["p_stop"]))
    m[2].metric("P(time exit)", pct(mc["p_time"]))
    lo, hi = mc["ev_ci"]
    m[3].metric("EV 95% CI", f"{money(lo*100)} to {money(hi*100)}")
    m[4].metric("5% VaR", money(mc["var5"] * 100, sign=False))
    m[5].metric("CVaR (worst 5%)", money(mc["cvar5"] * 100, sign=False))
    left, right = st.columns(2)
    left.plotly_chart(paths_chart(t), width="stretch")
    right.plotly_chart(pnl_chart(t), width="stretch")

    with st.expander(f"All {len(t['candidates'])} contracts tested on the same paths"):
        cdf = pd.DataFrame(t["candidates"])
        cdf["ev"] = cdf["ev"] * 100
        cdf = cdf.rename(columns={"exp": "expiry", "ev": "EV/contract", "ev_pct": "EV/premium",
                                  "earnings_in_window": "earnings before expiry"})
        st.dataframe(cdf, hide_index=True, width="stretch", column_config={
            "entry": st.column_config.NumberColumn(format="$%.2f"),
            "iv": st.column_config.NumberColumn("IV", format="percent"),
            "delta": st.column_config.NumberColumn(format="%.2f"),
            "theta": st.column_config.NumberColumn(format="%.3f"),
            "vega": st.column_config.NumberColumn(format="%.3f"),
            "EV/contract": st.column_config.NumberColumn(format="$%.0f"),
            "EV/premium": st.column_config.NumberColumn(format="percent"),
            "p_profit": st.column_config.NumberColumn("P(profit)", format="percent"),
            "p_target": st.column_config.NumberColumn("P(target)", format="percent"),
            "p_stop": st.column_config.NumberColumn("P(stop)", format="percent"),
        })


def validation_tab(snap):
    st.markdown("Each test checks a model against a known answer or an independent method.")
    tests = snap.get("tests") or []
    if not tests:
        if st.button("Run validation tests"):
            with st.spinner("Running tests..."):
                tests = run_tests_live()
    if tests:
        n_ok = sum(x["passed"] for x in tests)
        (st.success if n_ok == len(tests) else st.error)(f"{n_ok}/{len(tests)} tests passed")
        st.dataframe(pd.DataFrame([{
            "test": x["name"].replace("test_", "").replace("_", " "),
            "result": "✅ pass" if x["passed"] else "❌ fail", "detail": x["message"]}
            for x in tests]), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
st.sidebar.title("📈 Call Scanner")
mode = st.sidebar.radio("Data", ["Latest snapshot", "Run live scan"],
                        help="Snapshots are generated daily after the close by GitHub Actions. "
                             "A live scan fetches fresh Yahoo data and refits every model.")
snap = None

if mode == "Run live scan":
    raw = st.sidebar.text_input("Tickers (comma separated, max 6)", "AAPL, AMZN, NVDA, MSFT, AMD, MSTR")
    tickers = tuple(dict.fromkeys(s.strip().upper() for s in raw.split(",") if s.strip()))[:6]
    hold = st.sidebar.slider("Holding period (calendar days)", 5, 20, 10)
    n_paths = st.sidebar.select_slider("Monte Carlo paths", [2000, 5000, 10000], 5000)
    iv_sens = st.sidebar.slider("IV sensitivity k", 0.0, 1.0, 0.5, 0.1,
                                help="How much implied vol follows simulated GARCH vol. "
                                     "IV_exit = IV_0 × (σ_exit/σ_0)^k")
    method = st.sidebar.radio("Shock distribution", ["fhs", "t"], horizontal=True,
                              format_func={"fhs": "Historical residuals", "t": "Student-t"}.get)
    if st.sidebar.button("Run scan", type="primary"):
        st.session_state["live_args"] = (tickers, hold, n_paths, iv_sens, method)
    args = st.session_state.get("live_args")   # keep showing the last scan until rerun
    if not args:
        st.info("Set the parameters in the sidebar and press **Run scan**.")
    else:
        with st.spinner(f"Fitting GARCH and simulating {n_paths:,} paths per ticker..."):
            try:
                snap = live_scan(*args)
            except Exception as e:
                st.error(f"Live scan failed ({e}). Yahoo may be rate-limiting this server; "
                         "the latest snapshot still works.")
else:
    if os.path.exists(SNAPSHOT):
        snap = load_snapshot(SNAPSHOT, os.path.getmtime(SNAPSHOT))
    else:
        st.info("No snapshot yet. Run `python snapshot.py` (or the GitHub Action), "
                "or switch to **Run live scan** in the sidebar.")

st.sidebar.divider()
st.sidebar.caption("For research and education. Not financial advice. "
                   "Data from Yahoo Finance via yfinance, delayed.")

st.title("Options Call Scanner")
st.caption("Technical screen → GARCH(1,1)-t volatility model → Monte Carlo trade simulation")

if snap:
    s = snap["settings"]
    when = dt.datetime.fromisoformat(snap["generated_at"])
    st.caption(f"Generated {when:%Y-%m-%d %H:%M} UTC · hold {s['HOLD_DAYS']} days · "
               f"{s['N_PATHS']:,} paths ({s['SIM_METHOD']}) · IV sensitivity {s['IV_SENSITIVITY']} · "
               f"risk-free {snap['risk_free_rate']*100:.2f}%")
    for e in snap["errors"]:
        st.warning(f"{e['ticker']}: {e['error']}")

    names = [t["ticker"] for t in snap["tickers"]]
    tabs = st.tabs(["Overview"] + names + ["Model validation", "Methodology"])
    with tabs[0]:
        overview(snap)
    for tab, t in zip(tabs[1:1 + len(names)], snap["tickers"]):
        with tab:
            ticker_tab(t)
    with tabs[-2]:
        validation_tab(snap)
    with tabs[-1]:
        if os.path.exists(README):
            st.markdown(open(README).read())
