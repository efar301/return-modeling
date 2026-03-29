import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def pull_ticker(ticker: str, start: str | datetime = datetime.today() - datetime.timedelta(weeks=52 * 10), end: str | datetime = datetime.today()):

    if (type(start) == str):
        start = datetime.strptime(start, "%m-%d-%Y")
    if (type(end) == str):
        end = datetime.strptime(end, "%m-%d-%Y")

    data = yf.Ticker(ticker).history(start=start, end=end)

    # simple returns
    data["daily_simple_return"] = round(data["Close"] / data["Close"].shift(1), 5)
    data["weekly_simple_return"] = round(data["Close"] / data["Close"].shift(7), 5)

    # moving averages
    data["ma5"] = round(data["Close"].rolling(window=5).mean(), 5)
    data["ma30"] = round(data["Close"].rolling(window=30).mean(), 5)

    filename = f"./csvs/{ticker}_historical_prices.csv"
    data.to_csv(filename)


if __name__ == "__main__":
    # pass
   pull_ticker("NVDA")