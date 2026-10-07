"""Technical indicators: RSI, MACD, TTM Squeeze, volume profile."""
import numpy as np
import pandas as pd


def rsi(close, n=14):
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def macd(close, fast=12, slow=26, signal=9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def linreg_end(series, n):
    """Value of a rolling least-squares line at its last point."""
    x = np.arange(n)
    return series.rolling(n).apply(lambda y: np.polyval(np.polyfit(x, y, 1), n - 1), raw=True)


def ttm_squeeze(df, n=20, bb_mult=2.0, kc_mult=1.5):
    """Squeeze is ON when Bollinger Bands sit inside Keltner Channels."""
    close, high, low = df["Close"], df["High"], df["Low"]
    sma = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    on = (bb_mult * std) < (kc_mult * atr)
    mid = ((high.rolling(n).max() + low.rolling(n).min()) / 2 + sma) / 2
    return on, linreg_end(close - mid, n)


def volume_profile(df, days=60, bins=50):
    """Spread each day's volume across its high-low range. Returns POC, VAH, VAL, HVNs."""
    d = df.tail(days)
    edges = np.linspace(d["Low"].min(), d["High"].max(), bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol = np.zeros(bins)
    for lo, hi, c, v in zip(d["Low"], d["High"], d["Close"], d["Volume"]):
        mask = (centers >= lo) & (centers <= hi)
        if not mask.any():
            mask[np.argmin(np.abs(centers - c))] = True
        vol[mask] += v / mask.sum()

    poc = int(np.argmax(vol))
    lo_i = hi_i = poc
    covered, total = vol[poc], vol.sum()
    while covered < 0.70 * total and (lo_i > 0 or hi_i < bins - 1):
        down = vol[lo_i - 1] if lo_i > 0 else -1
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        if up >= down:
            hi_i += 1; covered += vol[hi_i]
        else:
            lo_i -= 1; covered += vol[lo_i]

    sm = np.convolve(vol, np.ones(3) / 3, mode="same")
    hvns = [centers[i] for i in range(1, bins - 1)
            if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > np.median(sm)]
    return {"poc": centers[poc], "vah": centers[hi_i], "val": centers[lo_i], "hvns": hvns,
            "centers": centers.tolist(), "volume": vol.tolist()}
