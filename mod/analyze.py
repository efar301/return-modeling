import pandas as pd
import yfinance as yf
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from data.distributions import simulate_future

def fit_to_past(stock: str, prediction_period: str, n_simulations: int):
    if stock is None: 
        return "Ticker must be preset."
    
    if prediction_period.lower() not in ["day", "week", "month", "year"]:
        return "Not a valid period."
    
    n_days = {
        "day": 1,
        "week": 7,
        "month": 30,
        "year": 365
    }
    
    start_date = None
    if prediction_period == "day": 
        start_date = datetime.today() - timedelta(weeks=16)
    elif prediction_period == "week":
        start_date = datetime.today() - timedelta(weeks=44)
    elif prediction_period == "month":
        start_date = datetime.today() - timedelta(weeks=96)
    else: 
        start_date = datetime.today() - timedelta(weeks=156)

    data = yf.Ticker(stock).history(start=start_date, end=datetime.today())
    latest_close = data["Close"].iloc[-1]
    print(f"latest closing value: {latest_close}")

    data["daily_simple_return"] = round((data["Close"] / data["Close"].shift(1)) - 1, 5)

    data = data.dropna()


    print(f"fitting distribution...")
    distributions = ["norm", "t", "norminvgauss", "genhyperbolic", 
                     "laplace", "skewnorm", "nct"]
    
    results = []
    for d in distributions:
        dist = getattr(stats, d)
        params = dist.fit(data["daily_simple_return"])


        loglikelihood = np.sum(dist.logpdf(data["daily_simple_return"], *params))
        k = len(params)

        aic = 2 * k - 2 * loglikelihood
        bic = np.log(len(data["daily_simple_return"])) * k - 2 * loglikelihood

        results.append({"distribution": dist.name, "aic": aic, "bic": bic, "params": params})

    results.sort(key=lambda x: x["aic"])

    print(f"best distriution for simple returns: {results[0]["distribution"]}")

    print(f"simulating n futures...")

    sim_results = simulate_future(results[0], n_simulations, n_days[prediction_period], latest_close)
    print(f"done!")
    
    return sim_results[0], sim_results[1], results[0]["distribution"]



