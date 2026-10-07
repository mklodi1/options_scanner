"""
GARCH(1,1) with Student-t innovations, fit by maximum likelihood.

    r_t       = mu + eps_t,           eps_t = sigma_t * z_t,   z_t ~ standardized Student-t(nu)
    sigma2_t  = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}

Returns are in PERCENT (daily log return * 100) for numerical stability.
Stationarity requires alpha + beta < 1; long-run variance = omega / (1 - alpha - beta).
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

TRADING_DAYS = 252


@dataclass
class GarchFit:
    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float
    std_errors: dict
    loglik: float
    sigma2: np.ndarray          # in-sample conditional variances (percent^2)
    std_resid: np.ndarray       # standardized residuals z_t = eps_t / sigma_t
    last_eps: float
    last_sigma2: float

    @property
    def persistence(self):
        return self.alpha + self.beta

    @property
    def half_life_days(self):
        """Days for a volatility shock to decay halfway back to the long-run level."""
        p = self.persistence
        return math.log(0.5) / math.log(p) if 0 < p < 1 else float("inf")

    @property
    def long_run_vol(self):
        """Annualized long-run volatility (decimal)."""
        var = self.omega / max(1 - self.persistence, 1e-8)
        return math.sqrt(var * TRADING_DAYS) / 100

    def next_sigma2(self):
        """One-step-ahead conditional variance for tomorrow."""
        return self.omega + self.alpha * self.last_eps ** 2 + self.beta * self.last_sigma2

    def forecast_vol(self, horizon_days):
        """
        Annualized average volatility over the next `horizon_days` trading days.
        Uses E[sigma2_{t+h}] = VL + (alpha+beta)^(h-1) * (sigma2_{t+1} - VL).
        This is the number to compare against an option's implied volatility.
        """
        h = max(int(horizon_days), 1)
        p = self.persistence
        vl = self.omega / max(1 - p, 1e-8)
        s1 = self.next_sigma2()
        path = vl + p ** np.arange(h) * (s1 - vl)
        return math.sqrt(path.mean() * TRADING_DAYS) / 100


def _filter(params, r):
    mu, omega, alpha, beta = params[:4]
    eps = r - mu
    sigma2 = np.empty_like(r)
    sigma2[0] = np.var(r)
    for t in range(1, len(r)):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return eps, sigma2


def _neg_loglik(params, r):
    mu, omega, alpha, beta, nu = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999 or nu <= 2.05:
        return 1e10
    eps, sigma2 = _filter(params, r)
    if np.any(sigma2 <= 0):
        return 1e10
    # standardized Student-t log density (unit variance)
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    ll = c - 0.5 * np.log(sigma2) - (nu + 1) / 2 * np.log1p(eps ** 2 / (sigma2 * (nu - 2)))
    return -float(ll.sum())


def _numerical_hessian(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i] = h * max(abs(x[i]), 1e-2)
            ej[j] = h * max(abs(x[j]), 1e-2)
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej)
                                 - f(x - ei + ej) + f(x - ei - ej)) / (4 * ei[i] * ej[j])
    return H


def fit_garch(returns_pct):
    """Fit GARCH(1,1)-t by MLE. `returns_pct` = daily log returns * 100."""
    r = np.asarray(returns_pct, dtype=float)
    var = np.var(r)
    best = None
    # a few starting points guard against local optima
    for a0, b0 in [(0.05, 0.90), (0.10, 0.85), (0.03, 0.95)]:
        x0 = np.array([r.mean(), var * (1 - a0 - b0), a0, b0, 8.0])
        res = minimize(_neg_loglik, x0, args=(r,), method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-7})
        if best is None or res.fun < best.fun:
            best = res
    x = best.x
    names = ["mu", "omega", "alpha", "beta", "nu"]
    try:
        cov = np.linalg.inv(_numerical_hessian(lambda p: _neg_loglik(p, r), x))
        se = {n: float(math.sqrt(cov[i, i])) if cov[i, i] > 0 else float("nan")
              for i, n in enumerate(names)}
    except np.linalg.LinAlgError:
        se = {n: float("nan") for n in names}
    eps, sigma2 = _filter(x, r)
    x = [float(v) for v in x]
    return GarchFit(mu=x[0], omega=x[1], alpha=x[2], beta=x[3], nu=x[4], std_errors=se,
                    loglik=-best.fun, sigma2=sigma2, std_resid=eps / np.sqrt(sigma2),
                    last_eps=float(eps[-1]), last_sigma2=float(sigma2[-1]))


def simulate_garch(n, mu, omega, alpha, beta, nu=None, seed=0):
    """Simulate a GARCH(1,1) return series (used to test that fit_garch recovers parameters)."""
    rng = np.random.default_rng(seed)
    if nu:
        z = rng.standard_t(nu, n) * math.sqrt((nu - 2) / nu)
    else:
        z = rng.standard_normal(n)
    r = np.empty(n)
    s2 = omega / (1 - alpha - beta)
    eps_prev = 0.0
    for t in range(n):
        s2 = omega + alpha * eps_prev ** 2 + beta * s2
        eps_prev = math.sqrt(s2) * z[t]
        r[t] = mu + eps_prev
    return r
