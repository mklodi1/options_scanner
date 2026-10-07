"""Black-Scholes pricing, Greeks and implied volatility (scalar and vectorized)."""
import math
import numpy as np
from scipy.special import ndtr  # standard normal CDF, works on arrays


def norm_pdf(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2.0 * math.pi)


def bs_call(S, K, T, r, sigma, q=0.0):
    """European call price + Greeks. Theta is per calendar day, vega per 1 vol point."""
    if T <= 0 or sigma <= 0:
        return {"price": max(S - K, 0.0), "delta": float(S > K),
                "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    price = S * dq * ndtr(d1) - K * dr * ndtr(d2)
    return {
        "price": float(price),
        "delta": float(dq * ndtr(d1)),
        "gamma": float(dq * norm_pdf(d1) / (S * sigma * sqT)),
        "theta": float((-S * dq * norm_pdf(d1) * sigma / (2 * sqT)
                        - r * K * dr * ndtr(d2) + q * S * dq * ndtr(d1)) / 365.0),
        "vega": float(S * dq * norm_pdf(d1) * sqT / 100.0),
    }


def bs_call_vec(S, K, T, r, sigma, q=0.0):
    """Vectorized call price. S, T, sigma may be arrays (one entry per Monte Carlo path)."""
    S, T, sigma = (np.asarray(a, dtype=float) for a in (S, T, sigma))
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-6)
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    return S * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)


def implied_vol(price, S, K, T, r, q=0.0, tol=1e-6):
    """Back out implied volatility by bisection (price is monotonic in sigma)."""
    if T <= 0:
        return None
    intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    if price <= intrinsic:
        return None
    lo, hi = 1e-3, 5.0
    if bs_call(S, K, T, r, hi, q)["price"] < price:
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, r, mid, q)["price"] > price:
            hi = mid
        else:
            lo = mid
    iv = 0.5 * (lo + hi)
    return iv if 0.02 < iv < 4.9 else None
