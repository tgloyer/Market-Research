import os
import joblib
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

client = StockHistoricalDataClient(api_key, secret_key)
daily_request = StockBarsRequest(
  symbol_or_symbols=["SPY"],
  timeframe=TimeFrame.Day,
  start=datetime(2019, 1, 1),
  end=datetime(2024, 12, 31)
)

daily_bars = client.get_stock_bars(daily_request).df
daily_bars = daily_bars.reset_index()
daily_bars["timestamp"] = (daily_bars["timestamp"].dt.tz_convert("America/New_York"))
daily_bars["date"] = daily_bars["timestamp"].dt.date
daily_bars = daily_bars.set_index("date").sort_index()

daily_bars.to_csv("data/spy_daily.csv", index=False)

minute_request = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Minute,
    start=datetime(2019, 1, 1),
    end=datetime(2024, 12, 31)
)

minute_bars = client.get_stock_bars(minute_request)
minute_bars = minute_bars.df
minute_bars = minute_bars.reset_index()

minute_bars["timestamp"] = (minute_bars["timestamp"].dt.tz_convert("America/New_York"))
minute_bars.to_csv("data/spy_minute.csv", index=False)