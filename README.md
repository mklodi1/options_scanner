# Options Call Scanner: GARCH Volatility + Monte Carlo Trade Simulation

A scanner that screens stocks for short-term (1–2 week) bullish setups, then decides whether a specific call option is worth buying by simulating the trade under a fitted volatility model rather than relying on a single Black-Scholes price.

## Pipeline

1. **Technical screen.** TTM Squeeze, MACD, RSI and a 60-day volume profile (POC, value area, high-volume nodes). The volume profile also sets the trade's price target and stop.
2. **Volatility model.** A GARCH(1,1) model with Student-t innovations is fit to two years of daily log returns by maximum likelihood (`garch.py`). Standard errors come from the inverse numerical Hessian.
3. **Is implied volatility cheap?** The GARCH forecast of average volatility through the option's expiration is compared with the market's implied volatility. A ratio of IV / GARCH ≤ 1.15 counts as cheap.
4. **Monte Carlo trade simulation** (`montecarlo.py`).
   - 20,000 price paths follow the fitted GARCH dynamics, so volatility clusters and evolves along each path.
   - Shocks come from **Filtered Historical Simulation**: the stock's own standardized residuals are resampled, preserving its empirical fat tails and skew.
   - Every path is run through the trading rules: exit at the target, at the stop, or when the holding period ends.
   - The call is repriced at exit with Black-Scholes, using an IV that moves with the path's simulated volatility.
   - Outputs: P(target first), P(stop first), expected P&L with a 95% confidence interval, P(profit), 5% VaR and CVaR.
5. **Contract selection.** Every liquid call with 0.50–0.70 delta and 21–50 days to expiration is evaluated on the *same* simulated paths (common random numbers, which reduces noise when comparing strikes). The contract with the highest expected return on premium is selected.

A setup is graded **A** only when all five screens pass, the lower bound of the 95% CI on expected value is above zero, and no earnings report falls before expiration.

## Validation (`python test_models.py`)

| Test | What it checks |
|---|---|
| Textbook value | Black-Scholes matches Hull's reference value (10.4506) |
| Put-call parity | Call/put relationship holds to 1e-10 |
| Greeks | Analytic delta and vega match finite differences |
| IV round trip | Price → IV → price recovers the input volatility |
| MC vs Black-Scholes | Risk-neutral Monte Carlo (antithetic variates) lands within 3 SE of the closed form |
| GARCH parameter recovery | MLE recovers known α, β, ν from a simulated GARCH-t series |
| Forecast mean reversion | Long-horizon GARCH forecast converges to the long-run volatility |
| Path volatility | Realized volatility of simulated paths matches the GARCH forecast |
| Outcome probabilities | P(target) + P(stop) + P(time exit) = 1; CVaR ≤ VaR ≤ median |

## Assumptions and limitations

- **Barriers use daily closes.** Target and stop are checked at each close, not intraday.
- **IV dynamics are assumed.** At exit, IV = IV₀ × (σ_exit / σ₀)^k with k = 0.5. This sensitivity parameter is set by assumption, not estimated from data.
- **Flat volatility.** Black-Scholes uses one IV per contract, with no skew or term-structure surface.
- **Real-world drift.** The simulation uses the GARCH-estimated mean return. That is appropriate for evaluating a trade, but it's noisy, and it is not a risk-neutral price.
- **Costs.** Entry assumes paying the ask; exit subtracts half the bid/ask spread.
- **No backtest yet.** The technical signal has not been tested out-of-sample. Positive simulated EV depends on the model being correct, and it doesn't show that the strategy has an edge.
- **Data.** Prices come from Yahoo via yfinance, which is unofficial and delayed about 15 minutes.
- **Live mode isn't built for heavy traffic.** Settings are module-level, so two simultaneous live scans with different settings could interfere. The snapshot mode is unaffected.

## Dashboard

**Live demo:** *(https://mklodi-options-scanner.streamlit.app/)*

The Streamlit dashboard (`app.py`) has:

- An overview table ranking every ticker.
- One tab per ticker:
  - Candlestick chart with the volume profile, target and stop.
  - RSI, MACD and squeeze panels.
  - GARCH conditional volatility with its forecast, compared against market IV.
  - Parameter estimates with standard errors.
  - Monte Carlo path fan and P&L distribution, plus every contract tested on the same paths.
- A model-validation tab with live test results.

It has two data modes:

- **Latest snapshot** (default): reads `data/snapshot.json`. A GitHub Actions job (`.github/workflows/scan.yml`) produces this file every weekday after the close. The job runs the validation tests first and only publishes if all of them pass. The dashboard loads instantly and never depends on Yahoo being reachable from the hosting server.
- **Run live scan**: fetches fresh data and refits every model. Viewers can change the holding period, path count, IV sensitivity and shock distribution.

## Project structure

```
pricing.py        Black-Scholes, Greeks, implied volatility
garch.py          GARCH(1,1)-t maximum likelihood, forecasting, simulation
montecarlo.py     GARCH path simulation (FHS / Student-t), trade evaluation, risk metrics
indicators.py     RSI, MACD, TTM Squeeze, volume profile
call_scanner.py   Scan logic + console report
snapshot.py       Runs the scan and saves dashboard data
app.py            Streamlit dashboard
test_models.py    Model validation suite
.github/workflows/scan.yml   Scheduled scan + test gate
```

## Run locally

```
pip install -r requirements.txt
python test_models.py      # validate the models
python snapshot.py         # build data/snapshot.json
streamlit run app.py       # open the dashboard
python call_scanner.py     # or: console report + CSV + PNG charts
```

## Deploy

1. Push this folder to a public GitHub repository.
2. In the repo's **Actions** tab, enable workflows, open **Daily scan** and click **Run workflow** to create the first snapshot.
3. Go to share.streamlit.io, sign in with GitHub, choose **Create app**, select the repo, and set the main file to `app.py`.
4. Paste the app URL at the top of this README and on your CV.

*For research and education. Not financial advice.*
