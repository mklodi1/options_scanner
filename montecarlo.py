"""
Monte Carlo simulation of the trade.

Price paths follow the fitted GARCH(1,1) dynamics, so volatility clusters and changes
along each path (unlike constant-vol Black-Scholes). Shocks come from either:
  - "fhs": Filtered Historical Simulation - resample the stock's own standardized residuals
           (keeps its real fat tails and skew), or
  - "t":   Student-t draws with the fitted degrees of freedom.

Each path is then run through the trading rules: exit at the target, at the stop
(both checked on daily closes), or at the end of the holding period. The option is
repriced at exit with Black-Scholes, using an IV that moves with the path's GARCH vol.
"""
import math
import numpy as np

from pricing import bs_call_vec

TRADING_DAYS = 252


def simulate_paths(fit, S0, n_days, n_paths=20000, method="fhs", seed=42):
    """Returns (prices, sigma2): both shape (n_paths, n_days + 1). sigma2 in percent^2."""
    rng = np.random.default_rng(seed)
    if method == "fhs":
        z_pool = (fit.std_resid - fit.std_resid.mean()) / fit.std_resid.std()
        draw = lambda: rng.choice(z_pool, n_paths)
    else:
        scale = math.sqrt((fit.nu - 2) / fit.nu)
        draw = lambda: rng.standard_t(fit.nu, n_paths) * scale

    prices = np.empty((n_paths, n_days + 1))
    sigma2 = np.empty((n_paths, n_days + 1))
    prices[:, 0] = S0
    sigma2[:, 0] = fit.next_sigma2()
    log_s = np.full(n_paths, math.log(S0))
    for t in range(n_days):
        eps = np.sqrt(sigma2[:, t]) * draw()
        log_s += (fit.mu + eps) / 100.0
        prices[:, t + 1] = np.exp(log_s)
        sigma2[:, t + 1] = fit.omega + fit.alpha * eps ** 2 + fit.beta * sigma2[:, t]
    return prices, sigma2


def first_hit(mask):
    """Index (1-based day) of the first True in each row; inf where never True."""
    hit = mask.any(axis=1)
    return np.where(hit, mask.argmax(axis=1) + 1, np.inf)


def evaluate_call(prices, sigma2, K, dte_calendar, entry, iv0, target, stop,
                  r=0.04, q=0.0, iv_sensitivity=0.5, exit_cost_frac=0.0):
    """
    Simulate the full trade on every path. All paths share the same random draws for every
    contract tested (common random numbers), so comparisons between strikes are low-noise.

    iv_sensitivity: how strongly IV follows GARCH vol. IV_exit = IV0 * (sigma_exit/sigma_0)^k.
                    0 = IV never changes, 1 = moves one-for-one. 0.5 is a moderate assumption.
    exit_cost_frac: fraction of the option value lost when selling (half the bid/ask spread).
    """
    n_paths, n_steps = prices.shape[0], prices.shape[1] - 1
    t_tgt = first_hit(prices[:, 1:] >= target)
    t_stp = first_hit(prices[:, 1:] <= stop)
    exit_day = np.minimum(np.minimum(t_tgt, t_stp), n_steps).astype(int)
    rows = np.arange(n_paths)

    S_exit = prices[rows, exit_day]
    T_rem = (dte_calendar - exit_day * 365 / TRADING_DAYS) / 365
    iv_exit = iv0 * (sigma2[rows, exit_day] / sigma2[:, 0]) ** (iv_sensitivity / 2)
    value = bs_call_vec(S_exit, K, T_rem, r, iv_exit, q) * (1 - exit_cost_frac)
    pnl = value - entry

    ev = pnl.mean()
    se = pnl.std(ddof=1) / math.sqrt(n_paths)
    var5 = np.percentile(pnl, 5)
    return {
        "p_target": float(np.mean(t_tgt <= np.minimum(t_stp, n_steps))),
        "p_stop": float(np.mean(t_stp < np.minimum(t_tgt, n_steps + 1))),
        "p_time": float(np.mean((t_tgt > n_steps) & (t_stp > n_steps))),
        "ev": float(ev), "ev_se": float(se),
        "ev_ci": (float(ev - 1.96 * se), float(ev + 1.96 * se)),
        "ev_pct": float(ev / entry),
        "p_profit": float(np.mean(pnl > 0)),
        "median": float(np.median(pnl)),
        "var5": float(var5),                          # 5th percentile P&L
        "cvar5": float(pnl[pnl <= var5].mean()),     # average of the worst 5%
        "pnl": pnl,
    }


def mc_price_gbm(S, K, T, r, sigma, q=0.0, n_paths=200000, seed=0):
    """
    Risk-neutral Monte Carlo price of a European call under geometric Brownian motion,
    with antithetic variates. Should match Black-Scholes; used as a validation test.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths // 2)
    z = np.concatenate([z, -z])
    ST = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)
    payoff = np.exp(-r * T) * np.maximum(ST - K, 0)
    pair_means = 0.5 * (payoff[: n_paths // 2] + payoff[n_paths // 2:])
    return float(payoff.mean()), float(pair_means.std(ddof=1) / math.sqrt(len(pair_means)))


def plot_trade(ticker, prices, pnl, target, stop, entry, path_out, n_show=150):
    """Two-panel chart: simulated price paths with target/stop, and the P&L distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    days = np.arange(prices.shape[1])
    for p in prices[:n_show]:
        a1.plot(days, p, lw=0.5, alpha=0.35, color="steelblue")
    lo, mid, hi = np.percentile(prices, [5, 50, 95], axis=0)
    a1.fill_between(days, lo, hi, color="steelblue", alpha=0.12, label="5-95% band")
    a1.plot(days, mid, color="navy", lw=1.5, label="median")
    a1.axhline(target, color="green", ls="--", label=f"target {target:.2f}")
    a1.axhline(stop, color="red", ls="--", label=f"stop {stop:.2f}")
    a1.set(title=f"{ticker}: GARCH Monte Carlo paths", xlabel="trading days", ylabel="price")
    a1.legend(fontsize=8)

    per_contract = pnl * 100
    a2.hist(per_contract, bins=80, color="gray", alpha=0.8)
    a2.axvline(0, color="black", lw=1)
    a2.axvline(per_contract.mean(), color="blue", ls="--",
               label=f"EV ${per_contract.mean():.0f}")
    a2.axvline(np.percentile(per_contract, 5), color="red", ls=":",
               label=f"5% VaR ${np.percentile(per_contract, 5):.0f}")
    a2.set(title=f"P&L per contract (entry ${entry*100:.0f})", xlabel="$", ylabel="paths")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_out, dpi=130)
    plt.close(fig)
