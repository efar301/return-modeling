from scipy import stats
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def add_features(data: pd.DataFrame, period: int):

    spy_data = pd.read_csv("csvs/SPY_raw.csv", parse_dates=["Date"])
    spy_data["Date"] = (
        pd.to_datetime(spy_data["Date"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    spy_data = spy_data.sort_values("Date").set_index("Date")
    spy_data["log_returns"] = np.log(spy_data["Close"] / spy_data["Close"].shift(1))
    # extra features
    if "Date" in data.columns:
        data["Date"] = (
            pd.to_datetime(data["Date"], utc=True)
            .dt.tz_convert(None)
            .dt.normalize()
        )
        data = data.sort_values("Date").set_index("Date")

    data["price_increased"] = (data["Close"].shift(-period) > data["Close"]).astype(int)

    # log returns for 1, 2, 3 days
    data["log_returns_1"] = np.log(data["Close"] / data["Close"].shift(1))
    data["log_returns_5"] = np.log(data["Close"] / data["Close"].shift(5))
    data["log_returns_30"] = np.log(data["Close"] / data["Close"].shift(30))

    data["spy_comp"] = data["log_returns_1"] - spy_data["log_returns"].reindex(data.index)

    # 5 and 30 day moving averages
    data["ma5"] = data["Close"].rolling(window=5).mean()
    data["ma5_ratio"] = data["ma5"] / data["Close"]
    data["ma30"] = data["Close"].rolling(window=30).mean()
    data["ma30_ratio"] = data["ma30"] / data["Close"]

    # roll_win = 20
    # vol_mean = data["Volume"].rolling(window=roll_win, min_periods=1).mean()
    # vol_std = data["Volume"].rolling(window=roll_win, min_periods=1).std()
    # data["normalized_volume"] = (data["Volume"] - vol_mean) / vol_std
    data["volatility_20"] = data["log_returns_1"].rolling(window=20).std()

    return data

import pandas as pd

def print_feature_importance(pipeline_model, feature_names):
    lr_model = pipeline_model.named_steps['logisticregression']
    
    coefficients = lr_model.coef_[0]
    
    # Create a DataFrame to view them easily
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Weight': coefficients,
        'Absolute Weight': abs(coefficients)
    })
    
    importance = importance.sort_values(by='Absolute Weight', ascending=False).reset_index(drop=True)
    
    print("\n--- Model Feature Weights ---")
    print(importance.to_string())
    print("-----------------------------\n")
    
    return importance

def fit_lg(training_data: pd.DataFrame):

    feature_names = ["log_returns_1", "log_returns_5", "ma5_ratio", "ma30_ratio", "volatility_20", "spy_comp"]

    features = training_data[feature_names]
    predicted = training_data["price_increased"]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(l1_ratio=1, solver='saga', max_iter=200),
    )
    model.fit(features, predicted)

    # print_feature_importance(model, feature_names)
    return model


def predict(stock: str, prediction_period: str):
    if stock is None: 
        return "Ticker must be preset."
    
    if prediction_period.lower() not in ["day", "week", "month", "year"]:
        return "Not a valid period."
    
    n_days = {
        "day": 1,
        "week": 5,
        "month": 30,
        "year": 365
    }

    period = n_days[prediction_period.lower()]
    
    start_date = None
    if prediction_period.lower() == "day": 
        start_date = datetime.today() - timedelta(weeks=26)
    elif prediction_period.lower() == "week":
        start_date = datetime.today() - timedelta(weeks=52)
    elif prediction_period.lower() == "month":
        start_date = datetime.today() - timedelta(weeks=104)
    else: 
        start_date = datetime.today() - timedelta(weeks=156)

    data = yf.Ticker(stock).history(start=start_date, end=datetime.today())
    data = data.reset_index()
    latest_close = data["Close"].iloc[-1]
    print(f"latest closing value: {latest_close}")

    data = add_features(data, period)

    feature_cols = ["log_returns_1", "log_returns_5", "ma5_ratio", "ma30_ratio", "volatility_20", "spy_comp"]
    target_col = "price_increased"

    train = data.dropna(subset=feature_cols + [target_col])
    curr = data.dropna(subset=feature_cols).iloc[-1:]

    model = fit_lg(train)

    curr_features = curr[feature_cols]
    increase_probability = model.predict_proba(curr_features)[0, 1]

    return increase_probability

def backtest(stock: str, prediction_period: str, start: datetime, end: datetime):
    if stock is None: 
        return "Ticker must be preset."
    
    if prediction_period.lower() not in ["day", "week", "month", "year"]:
        return "Not a valid period."
    
    n_days = {
        "day": 1,
        "week": 5,
        "month": 30,
        "year": 365
    }

    period = n_days[prediction_period.lower()]

    scores = {"num_increase_correct": 0, 
              "num_decrease_correct": 0, 
              "num_increase_incorrect": 0, 
              "num_decrease_incorrect": 0
              }

    csv_path = f"{stock.upper()}_raw.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("csvs", f"{stock.upper()}_raw.csv")

    data = pd.read_csv(csv_path, parse_dates=["Date"])
    data["Date"] = (
        pd.to_datetime(data["Date"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    data = data.sort_values("Date").set_index("Date")
    data = add_features(data, period)

    feature_cols = ["log_returns_1", "log_returns_5", "ma5_ratio", "ma30_ratio", "volatility_20", "spy_comp"]
    target_col = "price_increased"

    dates = pd.date_range(start, end, freq="B")

    for day in dates:
        if day not in data.index:
            day = data.index.asof(day)
            if pd.isna(day):
                continue

        if prediction_period.lower() == "day": 
            start_date = day - timedelta(weeks=26)
        elif prediction_period.lower() == "week":
            start_date = day - timedelta(weeks=52)
        elif prediction_period.lower() == "month":
            start_date = day - timedelta(weeks=104)
        else: 
            start_date = day - timedelta(weeks=156)

        window = data.loc[start_date:day]
        if window.empty:
            continue

        cutoff = day - pd.tseries.offsets.BDay(period)
        train = window.loc[:cutoff].dropna(subset=feature_cols + [target_col])
        if train.empty:
            continue

        curr = window.loc[day]
        if curr[feature_cols].isna().any():
            continue

        model = fit_lg(train)

        probability = model.predict_proba(curr[feature_cols].to_frame().T)[0, 1]
        curr_increase = int(curr[target_col])
        
        if probability >= 0.5 and curr_increase == 1:
            scores["num_increase_correct"] += 1
        elif probability < 0.5 and curr_increase == 0:
            scores["num_decrease_correct"] += 1
        elif probability >= 0.5 and curr_increase == 0:
            scores["num_decrease_incorrect"] += 1
        else:
            scores["num_increase_incorrect"] += 1

    return scores







        
