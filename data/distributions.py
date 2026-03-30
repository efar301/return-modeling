from scipy import stats
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

def get_best_distribution(df: pd.DataFrame = None, csv: str = None, column: str = None):
    if column is None:
        return "Must specify a column"
    
    if df is None and csv is None:
        return "No data provided"
    
    data = None
    if csv is not None:
        data = pd.read_csv(csv)
    else: 
        data = df

    data = data.dropna()
    col = data[column]



    distributions = ["norm", "t", "norminvgauss", "genhyperbolic", 
                     "laplace", "skewnorm", "nct"]
    
    results = []
    for d in distributions:
        dist = getattr(stats, d)
        params = dist.fit(col)


        loglikelihood = np.sum(dist.logpdf(col, *params))
        k = len(params)

        aic = 2 * k - 2 * loglikelihood
        bic = np.log(len(col)) * k - 2 * loglikelihood

        results.append({"distribution": dist.name, "aic": aic, "bic": bic, "params": params})

    results.sort(key=lambda x: x["aic"])


    return results


def plot_fitted_distributions(
    df: pd.DataFrame = None,
    csv: str = None,
    column: str = None,
    bins: int = 80,
    distributions: list[str] | None = None,
    fit_results: list[dict] | None = None,
):
    if column is None:
        return "Must specify a column"

    if df is None and csv is None:
        return "No data provided"

    data = pd.read_csv(csv) if csv is not None else df
    data = data.dropna()
    col = data[column].astype(float)

    if distributions is None:
        distributions = ["norm", "t", "norminvgauss", "genhyperbolic", "laplace", "skewnorm", "nct"]

    x_min, x_max = np.quantile(col, [0.001, 0.999])
    x = np.linspace(x_min, x_max, 600)

    fig = go.Figure()
    fig.add_histogram(x=col, nbinsx=bins, histnorm="probability density", name="returns", opacity=0.55)

    if fit_results is not None:
        for r in fit_results:
            dist = getattr(stats, r["distribution"])
            params = r["params"]
            pdf = dist.pdf(x, *params)
            fig.add_trace(go.Scatter(x=x, y=pdf, mode="lines", name=dist.name))
    else:
        for d in distributions:
            dist = getattr(stats, d)
            params = dist.fit(col)
            pdf = dist.pdf(x, *params)
            fig.add_trace(go.Scatter(x=x, y=pdf, mode="lines", name=dist.name))

    fig.update_layout(
        title=f"Fitted distributions vs {column}",
        xaxis_title=column,
        yaxis_title="Density",
        bargap=0.02,
        legend_title="Series",
    )
    return fig

def simulate_future(distribution: dict, n_runs: int = 10000, n_days: int = 100, curr_price: float = 0):
    
    dist = getattr(stats, distribution["distribution"])
    returns = dist.rvs(*distribution["params"], size=(n_days, n_runs))

    paths = np.zeros((n_days + 1, n_runs))
    paths[0, :] = curr_price

    for i in range(n_days):
        paths[i + 1, :] = paths[i, :] * (1 + returns[i, :])

    final_prices = paths[-1, :]
    count_above_start = (final_prices > curr_price).sum()
    percent_above_start = count_above_start / n_runs



    fig = go.Figure()
    for i in range(paths.shape[1]):
        fig.add_trace(go.Scatter(y=paths[:, i], mode="lines",line=dict(width=0.25) ,showlegend=False))

    fig.update_layout(
        title="Monte Carlo Price Paths",
        xaxis_title="Day",
        yaxis_title="Price"
    )

    return fig, round(percent_above_start, 3)

    

    







if __name__ == "__main__":
    fig = plot_fitted_distributions(csv="csvs/NVDA_historical_prices.csv", column="daily_simple_return")
    fig.show()
