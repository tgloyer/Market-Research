import os
import joblib
import pandas as pd
import numpy as np
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

# Morning momentum
morning_momentum_change = (second_hour_return - first_hour_return)

# Morning range
morning_high = morning.groupby("date")["high"].max()
morning_low = morning.groupby("date")["low"].min()
morning_open = morning.groupby("date")["open"].first()
morning_close = morning.groupby("date")["close"].last()
morning_range = ((morning_high - morning_low)/ morning_open)

# Morning volatility
morning["minute_return"] = (morning.groupby("date")["close"].pct_change())
morning_volatility = (morning.groupby("date")["minute_return"].std())

# Morning Volume
morning_volume = (morning.groupby("date")["volume"].sum())
avg_morning_volume_20d = (morning_volume.rolling(20).mean().shift(1))
morning_volume_ratio = (morning_volume / avg_morning_volume_20d)

first_hour_volume = (first_hour.groupby("date")["volume"].sum())
second_hour_volume = (second_hour.groupby("date")["volume"].sum())
second_vs_first_volume = (second_hour_volume / first_hour_volume)

# Morning VWAP
morning["price_volume"] = (morning["vwap"] * morning["volume"])
morning_vwap = (morning.groupby("date")["price_volume"].sum()/ morning.groupby("date")["volume"].sum())
price_vs_morning_vwap = (morning_close / morning_vwap - 1)

# Position within morning range
close_position_in_morning_range = ((morning_close - morning_low) / (morning_high - morning_low))

# First/second hour range
first_hour_high = first_hour.groupby("date")["high"].max()
first_hour_low = first_hour.groupby("date")["low"].min()
first_hour_open = first_hour.groupby("date")["open"].first()

first_hour_range = (
    (first_hour_high - first_hour_low)
    / first_hour_open
)

second_hour_high = second_hour.groupby("date")["high"].max()
second_hour_low = second_hour.groupby("date")["low"].min()
second_hour_open = second_hour.groupby("date")["open"].first()

second_hour_range = (
    (second_hour_high - second_hour_low)
    / second_hour_open
)

# New High/low
second_hour_new_high = ((second_hour_high > first_hour_high).astype(int))
second_hour_new_low = ((second_hour_low < first_hour_low).astype(int))

# Max upside/downside
max_upside_from_open = (
    (morning_high - morning_open)
    / morning_open
)
max_downside_from_open = (
    (morning_low - morning_open)
    / morning_open
)

# Gap
gap_return = (daily_bars["open"]/ daily_bars["close"].shift(1)- 1)
gap_morning_interaction = (gap_return * morning_return)
gap_momentum_interaction = (gap_return * morning_momentum_change)

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
    "morning_momentum_change",
    "morning_volume_ratio",
    "second_vs_first_volume",
    "morning_volatility",
    "morning_range",
    "first_hour_range",
    "second_hour_range",
    "price_vs_morning_vwap",
    "close_position_in_morning_range",
    "second_hour_new_high",
    "second_hour_new_low",
    "max_upside_from_open",
    "max_downside_from_open",
    "gap_morning_interaction",
    "gap_momentum_interaction"
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
    "morning_momentum_change": morning_momentum_change,
    "morning_volume_ratio": morning_volume_ratio,
    "second_vs_first_volume": second_vs_first_volume,
    "morning_volatility": morning_volatility,
    "morning_range": morning_range,
    "first_hour_range": first_hour_range,
    "second_hour_range": second_hour_range,
    "price_vs_morning_vwap": price_vs_morning_vwap,
    "close_position_in_morning_range": close_position_in_morning_range,
    "second_hour_new_high": second_hour_new_high,
    "second_hour_new_low": second_hour_new_low,
    "max_upside_from_open": max_upside_from_open,
    "max_downside_from_open": max_downside_from_open,
    "gap_morning_interaction": gap_morning_interaction,
    "gap_momentum_interaction": gap_momentum_interaction
})
previous_day_features = daily_bars[daily_feature_columns]
data = pd.concat([previous_day_features, morning_features, target_return.rename("target_return")], axis=1)
data = data[feature_columns + [target]].dropna()
test_data = data[(data.index >= pd.to_datetime("2023-01-01").date()) & (data.index <= pd.to_datetime("2024-12-31").date())]
X_test = test_data[feature_columns]
y_test = test_data["target_return"]

model = joblib.load("spy_model_ridge_morefeats.pkl")

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
results = test_data.copy()
results["predicted_return"] = predictions


# =========================
# BASIC TRADING BACKTEST
# =========================

# Long if prediction is positive
# Short if prediction is negative

results["position"] = np.where(
    results["predicted_return"] > 0,
    1,
    -1
)

# Strategy return
results["strategy_return"] = (
    results["position"] *
    results["target_return"]
)

# Starting capital
initial_capital = 10000

# Equity curve
results["equity"] = (
    initial_capital *
    (1 + results["strategy_return"]).cumprod()
)

# Running maximum
results["running_max"] = (
    results["equity"].cummax()
)

# Drawdown
results["drawdown"] = (
    results["equity"] /
    results["running_max"] - 1
)

# =========================
# PERFORMANCE
# =========================

final_value = results["equity"].iloc[-1]

total_return = (
    final_value / initial_capital - 1
)

win_rate = (
    results["strategy_return"] > 0
).mean()

# =========================
# RISK METRICS
# =========================

max_drawdown = results["drawdown"].min()

average_daily_return = (
    results["strategy_return"].mean()
)

daily_volatility = (
    results["strategy_return"].std()
)

sharpe_ratio = (
    average_daily_return /
    daily_volatility
) * np.sqrt(252)

# =========================
# ADDITIONAL METRICS
# =========================

winning_returns = results.loc[
    results["strategy_return"] > 0,
    "strategy_return"
]

losing_returns = results.loc[
    results["strategy_return"] < 0,
    "strategy_return"
]

best_day = results["strategy_return"].max()

worst_day = results["strategy_return"].min()
# =========================
# AFTERNOON DATA
# =========================

afternoon = minute_bars[
    (minute_bars["time"] >= pd.to_datetime("11:30").time()) &
    (minute_bars["time"] < pd.to_datetime("16:00").time())
].copy()


# =========================
# ENTRY PRICE
# =========================

entry_prices = (
    afternoon[
        afternoon["time"] == pd.to_datetime("11:30").time()
    ]
    .groupby("date")["open"]
    .first()
)

results["entry_price"] = entry_prices


# ============================================================
# TRADE-RELATIVE MORNING FEATURES
# ============================================================

results["aligned_first_hour_return"] = (
    results["position"] * results["first_hour_return"]
)

results["aligned_morning_return"] = (
    results["position"] * results["morning_return"]
)

results["adverse_morning_move"] = np.where(
    results["position"] == 1,
    results["max_downside_from_open"],
    -results["max_upside_from_open"]
)

results["favorable_morning_move"] = np.where(
    results["position"] == 1,
    results["max_upside_from_open"],
    -results["max_downside_from_open"]
)

# =========================
# ACTUAL STOP-LOSS BACKTEST
# =========================

stop_levels = [
    None,
    0.0040,
    0.0050,
    0.0075,
    0.0100
]

stop_results = []

for stop in stop_levels:

    trade_returns = []

    for date, trade in results.iterrows():

        day_data = afternoon[
            afternoon["date"] == date
        ].sort_values("timestamp")

        if day_data.empty:
            continue

        position = trade["position"]
        entry_price = trade["entry_price"]

        exit_return = None

        # No stop
        if stop is None:

            exit_price = day_data["close"].iloc[-1]

            if position == 1:
                exit_return = exit_price / entry_price - 1
            else:
                exit_return = entry_price / exit_price - 1

        # With stop
        else:

            for _, minute in day_data.iterrows():

                if position == 1:

                    stop_price = entry_price * (1 - stop)

                    if minute["low"] <= stop_price:
                        exit_return = -stop
                        break

                else:

                    stop_price = entry_price * (1 + stop)

                    if minute["high"] >= stop_price:
                        exit_return = -stop
                        break

            # Stop never hit
            if exit_return is None:

                exit_price = day_data["close"].iloc[-1]

                if position == 1:
                    exit_return = exit_price / entry_price - 1
                else:
                    exit_return = entry_price / exit_price - 1

        trade_returns.append(exit_return)

    trade_returns = pd.Series(trade_returns)

    initial_capital = 10000

    equity = (
        initial_capital *
        (1 + trade_returns).cumprod()
    )

    running_max = equity.cummax()

    drawdown = (
        equity / running_max - 1
    )

    final_value = equity.iloc[-1]

    total_return = (
        final_value / initial_capital - 1
    )

    win_rate = (
        trade_returns > 0
    ).mean()

    average_win = trade_returns[
        trade_returns > 0
    ].mean()

    average_loss = trade_returns[
        trade_returns < 0
    ].mean()

    max_drawdown = drawdown.min()

    daily_volatility = trade_returns.std()
    average_return = trade_returns.mean()

    sharpe_ratio = (
        average_return / daily_volatility
    ) * np.sqrt(252)

    worst_day = trade_returns.min()

    stop_results.append({
        "stop": (
            "No stop"
            if stop is None
            else f"{stop:.2%}"
        ),
        "final_value": final_value,
        "total_return": total_return,
        "win_rate": win_rate,
        "average_win": average_win,
        "average_loss": average_loss,
        "max_drawdown": max_drawdown,
        "worst_day": worst_day,
        "sharpe": sharpe_ratio
    })
def backtest_stop(results, afternoon, stop_loss):
    """
    Backtest one stop-loss percentage.

    Entry: 11:30 AM open
    Exit: stop-loss or 4:00 PM close
    """

    output = results.copy()

    stop_returns = []
    stopped = []

    for date, trade in output.iterrows():

        entry_price = trade["entry_price"]
        position = trade["position"]

        day = afternoon[afternoon["date"] == date]

        if day.empty:
            stop_returns.append(np.nan)
            stopped.append(False)
            continue

        if position == 1:
            stop_price = entry_price * (1 - stop_loss)

            stop_hit = day["low"] <= stop_price

            if stop_hit.any():
                stop_returns.append(-stop_loss)
                stopped.append(True)
            else:
                final_price = day["close"].iloc[-1]

                trade_return = (
                    final_price / entry_price - 1
                )

                stop_returns.append(trade_return)
                stopped.append(False)

        else:
            stop_price = entry_price * (1 + stop_loss)

            stop_hit = day["high"] >= stop_price

            if stop_hit.any():
                stop_returns.append(-stop_loss)
                stopped.append(True)
            else:
                final_price = day["close"].iloc[-1]

                trade_return = (
                    entry_price / final_price - 1
                )

                stop_returns.append(trade_return)
                stopped.append(False)

    output["strategy_return"] = stop_returns
    output["stopped"] = stopped

    return output
strong_features = [
    "aligned_first_hour_return",
    "aligned_morning_return",
    "adverse_morning_move"
]

stop_levels = [
    None,
    0.005,
    0.0075,
    0.01
]
for feature in strong_features:

    print("\n")
    print("=" * 70)
    print(f"FEATURE: {feature}")
    print("=" * 70)

    results["feature_bucket"] = pd.qcut(
        results[feature],
        q=5,
        duplicates="drop"
    )

    for bucket in results["feature_bucket"].dropna().unique():

        bucket_data = results[
            results["feature_bucket"] == bucket
        ].copy()

        print("\n" + "-" * 70)
        print(f"BUCKET: {bucket}")
        print(f"TRADES: {len(bucket_data)}")
        print("-" * 70)

        for stop in stop_levels:

            if stop is None:

                test = bucket_data.copy()

                test["strategy_return"] = (
                    test["position"] *
                    test["target_return"]
                )

                stop_name = "No stop"

            else:

                test = backtest_stop(
                    bucket_data,
                    afternoon,
                    stop
                )

                stop_name = f"{stop:.2%} stop"

            total_return = (
                (1 + test["strategy_return"]).prod() - 1
            )

            win_rate = (
                test["strategy_return"] > 0
            ).mean()

            average_return = (
                test["strategy_return"].mean()
            )

            stopped_trades = (
                test["stopped"].sum()
                if "stopped" in test.columns
                else 0
            )

            print(
                f"{stop_name:<12} | "
                f"Return: {total_return:>7.2%} | "
                f"Win: {win_rate:>6.2%} | "
                f"Avg: {average_return:>7.3%} | "
                f"Stopped: {stopped_trades}"
            )

# =========================
# PRINT RESULTS
# =========================

print("\n===== STOP-LOSS BACKTEST RESULTS =====")

for result in stop_results:

    print(f"\nStop-loss: {result['stop']}")

    print(
        f"Final value: "
        f"${result['final_value']:,.2f}"
    )

    print(
        f"Total return: "
        f"{result['total_return']:.2%}"
    )

    print(
        f"Win rate: "
        f"{result['win_rate']:.2%}"
    )

    print(
        f"Average win: "
        f"{result['average_win']:.2%}"
    )

    print(
        f"Average loss: "
        f"{result['average_loss']:.2%}"
    )

    print(
        f"Max drawdown: "
        f"{result['max_drawdown']:.2%}"
    )

    print(
        f"Worst day: "
        f"{result['worst_day']:.2%}"
    )

    print(
        f"Sharpe ratio: "
        f"{result['sharpe']:.2f}"
    )

# =========================
# PRINT RESULTS
# =========================

print("\n===== BACKTEST RESULTS =====")

print(
    f"Final value: ${final_value:,.2f}"
)

print(
    f"Total return: {total_return:.2%}"
)

print(
    f"Trades: {len(results)}"
)

print(
    f"Win rate: {win_rate:.2%}"
)

print("\n===== RISK METRICS =====")

print(
    f"Max drawdown: {max_drawdown:.2%}"
)

print(
    f"Average daily return: {average_daily_return:.4%}"
)

print(
    f"Daily volatility: {daily_volatility:.4%}"
)

print(
    f"Sharpe ratio: {sharpe_ratio:.2f}"
)

print("\n===== ADDITIONAL METRICS =====")

print(
    f"Average winning day: {winning_returns.mean():.4%}"
)

print(
    f"Average losing day: {losing_returns.mean():.4%}"
)

print(
    f"Best day: {best_day:.4%}"
)

print(
    f"Worst day: {worst_day:.4%}"
)

print(
    f"Winning days: {(results['strategy_return'] > 0).sum()}"
)

print(
    f"Losing days: {(results['strategy_return'] < 0).sum()}"
)
# =========================
# INTRADAY MOVEMENT ANALYSIS
# =========================

afternoon = minute_bars[
    (minute_bars["time"] >= pd.to_datetime("11:30").time()) &
    (minute_bars["time"] < pd.to_datetime("16:00").time())
].copy()

# Get the 11:30 entry price
entry_prices = (
    afternoon[afternoon["time"] == pd.to_datetime("11:30").time()]
    .groupby("date")["open"]
    .first()
)

# Get the highest and lowest price after entry
afternoon_high = (
    afternoon.groupby("date")["high"].max()
)

afternoon_low = (
    afternoon.groupby("date")["low"].min()
)

# Add these to the results dataframe
results["entry_price"] = entry_prices

results["afternoon_high"] = afternoon_high
results["afternoon_low"] = afternoon_low

# =========================
# MAX FAVORABLE / ADVERSE MOVEMENT
# =========================

# For a LONG:
# favorable = price going UP
# adverse = price going DOWN

long_mfe = (
    results["afternoon_high"] /
    results["entry_price"] - 1
)

long_mae = (
    results["afternoon_low"] /
    results["entry_price"] - 1
)

# For a SHORT:
# favorable = price going DOWN
# adverse = price going UP

short_mfe = (
    results["entry_price"] /
    results["afternoon_low"] - 1
)

short_mae = (
    results["afternoon_high"] /
    results["entry_price"] - 1
)

# Maximum favorable excursion
results["mfe"] = np.where(
    results["position"] == 1,
    (results["afternoon_high"] / results["entry_price"]) - 1,
    (results["entry_price"] / results["afternoon_low"]) - 1
)

# Maximum adverse excursion
results["mae"] = np.where(
    results["position"] == 1,
    (results["afternoon_low"] / results["entry_price"]) - 1,
    (results["entry_price"] / results["afternoon_high"]) - 1
)

winning_trades = results[
    results["strategy_return"] > 0
]

losing_trades = results[
    results["strategy_return"] < 0
]

print("\n===== INTRADAY MOVEMENT ANALYSIS =====")

print("\nALL TRADES")

print(
    f"Average MFE: "
    f"{results['mfe'].mean():.2%}"
)

print(
    f"Average MAE: "
    f"{results['mae'].mean():.2%}"
)

print("\nWINNING TRADES")

print(
    f"Average MFE: "
    f"{winning_trades['mfe'].mean():.2%}"
)

print(
    f"Average MAE: "
    f"{winning_trades['mae'].mean():.2%}"
)

print("\nLOSING TRADES")

print(
    f"Average MFE: "
    f"{losing_trades['mfe'].mean():.2%}"
)

print(
    f"Average MAE: "
    f"{losing_trades['mae'].mean():.2%}"
)

actual_direction = y_test > 0
predicted_direction = predictions > 0

directional_accuracy = (actual_direction == predicted_direction).mean()

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
