import pandas as pd
import plotly.express as px

def plot_basic(df: pd.DataFrame = None, csv: str = None, columns: list = None):
    if not columns:
        return "Must plot at least 1 column"
    
    if df is None and csv is None:
        return "No data provided"
    
    data = None
    if csv is not None:
        data = pd.read_csv(csv)
    else: 
        data = df

    data = data.dropna()

    fig = px.line(data, x="Date", y=columns)
    fig.show()
