import os
import joblib
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

daily_bars = pd.read_csv("data/spy_daily.csv")
daily_bars["timestamp"] = pd.to_datetime(daily_bars["timestamp"], utc= True)
daily_bars["timestamp"] = (daily_bars["timestamp"].dt.tz_convert("America/New_York"))
daily_bars["date"] = daily_bars["timestamp"].dt.date
daily_bars = daily_bars.set_index("date").sort_index()

minute_bars = pd.read_csv("data/spy_minute.csv")
minute_bars["timestamp"] = pd.to_datetime(minute_bars["timestamp"], utc = True)
minute_bars["timestamp"] = (minute_bars["timestamp"].dt.tz_convert("America/New_York"))
minute_bars["date"] = minute_bars["timestamp"].dt.date

#Calculate features
#Trend
daily_bars["50_day_avg"] = daily_bars["close"].rolling(50).mean().shift(1)
daily_bars["200_day_avg"] = daily_bars["close"].rolling(200).mean().shift(1)
daily_bars["price_vs_50ma"] = daily_bars["close"].shift(1) / daily_bars["50_day_avg"]
daily_bars["price_vs_200ma"] = daily_bars["close"].shift(1) / daily_bars["200_day_avg"]
daily_bars["50ma_vs_200ma"] = daily_bars["50_day_avg"] / daily_bars["200_day_avg"]

#Momentum
daily_bars["return_1d"] = daily_bars["close"].shift(1).pct_change(1)
daily_bars["return_5d"] = daily_bars["close"].shift(1).pct_change(5)
daily_bars["return_20d"] = daily_bars["close"].shift(1).pct_change(20)

#Volatility
daily_bars["daily_range"] = (daily_bars["high"].shift(1) - daily_bars["low"].shift(1)) / daily_bars["close"].shift(1)
daily_bars["volatility_20d"] = daily_bars["return_1d"].rolling(20).std()

#Volume
daily_bars["avg_volume_20d"] = daily_bars["volume"].shift(1).rolling(20).mean()
daily_bars["volume_ratio"] = (daily_bars["volume"].shift(1) / daily_bars["avg_volume_20d"])

#VWAP
daily_bars["price_vs_vwap"] = daily_bars["close"].shift(1) / daily_bars["vwap"].shift(1)

#Candle
daily_bars["body_pct"] = (daily_bars["close"].shift(1) - daily_bars["open"].shift(1)) / daily_bars["open"].shift(1)
daily_bars["range_pct"] = (daily_bars["high"].shift(1) - daily_bars["low"].shift(1)) / daily_bars["close"].shift(1)

#Trade count
daily_bars["avg_trade_size"] = (daily_bars["volume"].shift(1) / daily_bars["trade_count"].shift(1))

# Overnight Gap
daily_bars["yesterday_close"] = (daily_bars["close"].shift(1))
daily_bars["gap_return"] = (daily_bars["open"] / daily_bars["yesterday_close"] - 1)

# Minute Features for intraday
minute_bars["time"] = minute_bars["timestamp"].dt.time
minute_bars["date"] = minute_bars["timestamp"].dt.date
morning = minute_bars[
    (minute_bars["time"] >= pd.to_datetime("09:30").time()) &
    (minute_bars["time"] < pd.to_datetime("11:30").time())
].copy()

# First hour
first_hour = morning[morning["time"] < pd.to_datetime("10:30").time()].copy()
first_hour_return = (first_hour.groupby("date")["close"].last()/ first_hour.groupby("date")["open"].first()- 1)

# Second hour
second_hour = morning[morning["time"] >= pd.to_datetime("10:30").time()].copy()
second_hour_return = (second_hour.groupby("date")["close"].last()/ second_hour.groupby("date")["open"].first()- 1)

# Two hour
morning_return = (morning.groupby("date")["close"].last()/ morning.groupby("date")["open"].first()- 1)

# Morning range
morning_high = morning.groupby("date")["high"].max()
morning_low = morning.groupby("date")["low"].min()
morning_open = morning.groupby("date")["open"].first()
morning_range = ((morning_high - morning_low)/ morning_open)

# Morning volatility
morning["minute_return"] = (morning.groupby("date")["close"].pct_change())
morning_volatility = (morning.groupby("date")["minute_return"].std())

# Morning Volume
morning_volume = (morning.groupby("date")["volume"].sum())
avg_morning_volume_20d = (morning_volume.rolling(20).mean().shift(1))
morning_volume_ratio = (morning_volume / avg_morning_volume_20d)

# Morning VWAP
morning["price_volume"] = (morning["vwap"] * morning["volume"])
morning_vwap = (morning.groupby("date")["price_volume"].sum()/ morning.groupby("date")["volume"].sum())
morning_close = morning.groupby("date")["close"].last()
price_vs_morning_vwap = (morning_close / morning_vwap - 1)

# Position within morning range
close_position_in_morning_range = ((morning_close - morning_low) / (morning_high - morning_low))

entry_price = minute_bars[
    minute_bars["time"] == pd.to_datetime("11:30").time()
].groupby("date")["open"].first()

close_price = minute_bars[
    minute_bars["time"] == pd.to_datetime("16:00").time()
].groupby("date")["close"].last()

# Target Return
target_return = close_price / entry_price - 1
target = "target_return"
feature_columns = [
    
    "gap_return",
    "first_hour_return",
    "second_hour_return",
    "morning_return",
    "morning_volume_ratio",
    "morning_volatility",
    "morning_range",
    "price_vs_morning_vwap",
    "close_position_in_morning_range"
]

daily_feature_columns = [
    "50_day_avg",
    "200_day_avg",
    "price_vs_50ma",
    "price_vs_200ma",
    "50ma_vs_200ma",
    "return_1d",
    "return_5d",
    "return_20d",
    "daily_range",
    "volatility_20d",
    "avg_volume_20d",
    "volume_ratio",
    "price_vs_vwap",
    "body_pct",
    "range_pct",
    "avg_trade_size"
]
morning_features = pd.DataFrame({
    "gap_return": daily_bars["gap_return"],
    "first_hour_return": first_hour_return,
    "second_hour_return": second_hour_return,
    "morning_return": morning_return,
    "morning_volume_ratio": morning_volume_ratio,
    "morning_volatility": morning_volatility,
    "morning_range": morning_range,
    "price_vs_morning_vwap": price_vs_morning_vwap,
    "close_position_in_morning_range": close_position_in_morning_range
})
previous_day_features = daily_bars[daily_feature_columns]
data = pd.concat([ morning_features, target_return.rename("target_return")], axis=1)
data = data[feature_columns + [target]].dropna()
test_data = data[(data.index >= pd.to_datetime("2023-01-01").date()) & (data.index <= pd.to_datetime("2024-12-31").date())]
X_test = test_data[feature_columns]
y_test = test_data["target_return"]

model = joblib.load("spy_model_ridge.pkl")

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)

results = test_data.copy()
results["predicted_return"] = predictions

print(results[[
    "target_return",
    "predicted_return"
]].head(20))

actual_direction = y_test > 0
predicted_direction = predictions > 0

directional_accuracy = (
    actual_direction == predicted_direction
).mean()

print("Directional accuracy:", directional_accuracy)

baseline_predictions = [0] * len(y_test)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

print("Model MAE:", mae)
print("Baseline MAE:", baseline_mae)

correlation = results["target_return"].corr(
    results["predicted_return"]
)

print("Prediction correlation:", correlation)
