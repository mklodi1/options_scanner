"""
Model validation. Run:  python test_models.py   (or: pytest test_models.py)
Each test checks a model against a known answer or an independent method.
"""
import math
import numpy as np

from garch import fit_garch, simulate_garch
from montecarlo import evaluate_call, mc_price_gbm, simulate_paths
from pricing import bs_call, bs_call_vec, implied_vol


def test_bs_textbook_value():
    # Hull, Options Futures & Other Derivatives: S=K=100, T=1, r=5%, sigma=20% -> 10.4506
    assert abs(bs_call(100, 100, 1, 0.05, 0.20)["price"] - 10.4506) < 1e-3


def test_put_call_parity_and_delta_bounds():
    S, K, T, r, s, q = 100, 95, 0.5, 0.04, 0.3, 0.01
    c = bs_call(S, K, T, r, s, q)
    # put from the closed form, then check C - P = S e^{-qT} - K e^{-rT}
    sq = s * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * s * s) * T) / sq
    nc = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    p = K * math.exp(-r * T) * nc(-(d1 - sq)) - S * math.exp(-q * T) * nc(-d1)
    assert abs((c["price"] - p) - (S * math.exp(-q * T) - K * math.exp(-r * T))) < 1e-10
    assert 0 < c["delta"] < 1 and c["gamma"] > 0 and c["vega"] > 0 and c["theta"] < 0


def test_greeks_match_finite_differences():
    S, K, T, r, s = 100, 105, 0.25, 0.04, 0.35
    h = 1e-3
    g = bs_call(S, K, T, r, s)
    fd_delta = (bs_call(S + h, K, T, r, s)["price"] - bs_call(S - h, K, T, r, s)["price"]) / (2 * h)
    fd_vega = (bs_call(S, K, T, r, s + h)["price"] - bs_call(S, K, T, r, s - h)["price"]) / (2 * h) / 100
    assert abs(g["delta"] - fd_delta) < 1e-5 and abs(g["vega"] - fd_vega) < 1e-5


def test_implied_vol_round_trip():
    for s in (0.15, 0.4, 0.9):
        price = bs_call(120, 110, 40 / 365, 0.04, s)["price"]
        assert abs(implied_vol(price, 120, 110, 40 / 365, 0.04) - s) < 1e-4


def test_vectorized_matches_scalar():
    S = np.array([90.0, 100.0, 110.0])
    v = bs_call_vec(S, 100, 0.3, 0.04, np.array([0.2, 0.3, 0.4]))
    for i in range(3):
        assert abs(v[i] - bs_call(S[i], 100, 0.3, 0.04, [0.2, 0.3, 0.4][i])["price"]) < 1e-10


def test_monte_carlo_converges_to_black_scholes():
    bs = bs_call(100, 105, 0.5, 0.04, 0.3)["price"]
    mc, se = mc_price_gbm(100, 105, 0.5, 0.04, 0.3, n_paths=400000)
    assert abs(mc - bs) < 3 * se, (mc, bs, se)   # within 3 standard errors


def test_garch_recovers_known_parameters():
    true = dict(mu=0.05, omega=0.05, alpha=0.08, beta=0.90)
    r = simulate_garch(4000, **true, nu=6, seed=1)
    fit = fit_garch(r)
    assert abs(fit.alpha - true["alpha"]) < 0.03, fit.alpha
    assert abs(fit.beta - true["beta"]) < 0.04, fit.beta
    assert abs(fit.persistence - 0.98) < 0.02
    assert 3.5 < fit.nu < 10


def test_garch_forecast_reverts_to_long_run():
    fit = fit_garch(simulate_garch(3000, 0.0, 0.05, 0.08, 0.90, nu=8, seed=2))
    assert abs(fit.forecast_vol(5000) - fit.long_run_vol) / fit.long_run_vol < 0.05


def test_simulated_paths_have_garch_volatility():
    fit = fit_garch(simulate_garch(3000, 0.0, 0.05, 0.08, 0.90, nu=8, seed=3))
    prices, _ = simulate_paths(fit, 100, 21, n_paths=40000, method="t", seed=0)
    realized = np.log(prices[:, -1] / prices[:, 0]).std() * math.sqrt(252 / 21)
    assert abs(realized - fit.forecast_vol(21)) / fit.forecast_vol(21) < 0.05


def test_trade_outcome_probabilities_sum_to_one():
    fit = fit_garch(simulate_garch(2000, 0.0, 0.05, 0.08, 0.90, nu=8, seed=4))
    prices, sig2 = simulate_paths(fit, 100, 7, n_paths=10000, seed=0)
    mc = evaluate_call(prices, sig2, K=97, dte_calendar=35, entry=5.0, iv0=0.35,
                       target=104, stop=96)
    assert abs(mc["p_target"] + mc["p_stop"] + mc["p_time"] - 1) < 1e-12
    assert mc["cvar5"] <= mc["var5"] <= mc["median"]


def run_all():
    """Run every test; returns [{name, passed, message}] (used by the dashboard)."""
    out = []
    for name, f in [(n, f) for n, f in globals().items() if n.startswith("test_")]:
        try:
            f(); out.append({"name": name, "passed": True, "message": ""})
        except Exception as e:
            out.append({"name": name, "passed": False, "message": f"{type(e).__name__}: {e}"})
    return out


if __name__ == "__main__":
    import sys
    res = run_all()
    for r in res:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['name']}  {r['message']}")
    n_ok = sum(r["passed"] for r in res)
    print(f"\n{n_ok}/{len(res)} passed")
    sys.exit(0 if n_ok == len(res) else 1)
