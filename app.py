from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import holidays
from datetime import datetime

# ==== Load models ====
xgb_model = joblib.load('final_tuned_xgboost.pkl')
meta_model = joblib.load('final_hybrid_stack_model.pkl')
lstm_model = tf.keras.models.load_model('final_lstm_model.h5',compile=False)

# ==== Initialize Flask ====
app = Flask(__name__)

# ==== Prediction Endpoint ====
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received request for prediction")
        payload = request.get_json()
        transactions = payload['data']
        print("Data received:", transactions)
        df = pd.DataFrame(transactions)
        if df.empty:
            return jsonify({
                "next_month_budget": float(0),
                "note": "Estimate based on limited data."
            })
        print("DataFrame created:", df)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        df.dropna(subset=['Date'], inplace=True)
        df['Amount'] = df['Amount'].astype(float)
        print("DataFrame after processing:", df)

        daily = df.groupby('Date')['Amount'].sum().reset_index()
        daily['Date'] = pd.to_datetime(daily['Date'])
        daily.sort_values('Date', inplace=True)
        daily.set_index('Date', inplace=True)

        print("Daily DataFrame:", daily)
        current_date = datetime.today().date()
        next_month_start = (current_date.replace(day=1) + pd.DateOffset(months=1)).replace(day=1)
        next_month_end = (next_month_start + pd.DateOffset(months=1) - pd.DateOffset(days=1))
        print("Next month start:", next_month_start)
        print("Next month end:", next_month_end)    
        future_dates = pd.date_range(start=next_month_start, end=next_month_end)

        if daily.shape[0] < 3:
            avg = df['Amount'].mean()
            days_in_next_month = future_dates.size
            est_budget = avg * days_in_next_month
            return jsonify({
                "next_month_budget": float(round(est_budget, 2)),
                "note": "Estimate based on limited data."
            })
        
        daily['DayOfWeek'] = daily.index.dayofweek
        daily['Month'] = daily.index.month
        daily['Day'] = daily.index.day
        daily['IsWeekend'] = daily['DayOfWeek'].isin([5, 6]).astype(int)
        daily['IsHoliday'] = daily.index.isin(holidays.India()).astype(int)
        daily['sin_day'] = np.sin(2 * np.pi * daily['Day'] / 31)
        daily['cos_day'] = np.cos(2 * np.pi * daily['Day'] / 31)
        daily['sin_month'] = np.sin(2 * np.pi * daily['Month'] / 12)
        daily['cos_month'] = np.cos(2 * np.pi * daily['Month'] / 12)
        daily['Prev'] = daily['Amount'].shift(1)
        daily['Rolling_3'] = daily['Amount'].rolling(3).mean()
        daily['EMA_3'] = daily['Amount'].ewm(span=3).mean()
        daily['PctChange'] = daily['Amount'].pct_change().fillna(0)
        daily['LogAmount'] = np.log1p(daily['Amount'])
        print("Daily DataFrame after feature engineering:", daily)

        # last_date = daily.index.max()
        # next_month_start = (last_date + pd.offsets.MonthBegin(1)).date()
        # next_month_end = (next_month_start + pd.offsets.MonthEnd(1)).date()
        # future_dates = pd.date_range(start=next_month_start, end=next_month_end)
        # print("Future dates:", future_dates)

        future_preds = []
        log_values = daily['LogAmount'].values.tolist()

        # Pad with mean if not enough history
        if len(log_values) < 14:
            pad_value = np.mean(log_values) if log_values else 0.0
            log_values = [pad_value] * (14 - len(log_values)) + log_values
        else:
            log_values = log_values[-14:]
        print("Log values for LSTM:", log_values)

        for date in future_dates:
            day = date.day
            month = date.month
            day_of_week = date.weekday()
            is_weekend = int(day_of_week in [5, 6])
            is_holiday = int(date in holidays.India())

            sin_day = np.sin(2 * np.pi * day / 31)
            cos_day = np.cos(2 * np.pi * day / 31)
            sin_month = np.sin(2 * np.pi * month / 12)
            cos_month = np.cos(2 * np.pi * month / 12)

            # Estimate Prev, Rolling_3, EMA_3, PctChange
            prev = np.expm1(log_values[-1])
            rolling_3 = np.mean(np.expm1(log_values[-3:])) if len(log_values) >= 3 else prev
            ema_3 = pd.Series(np.expm1(log_values)).ewm(span=3).mean().iloc[-1]
            pct_change = (prev - np.expm1(log_values[-2])) / np.expm1(log_values[-2]) if len(log_values) > 1 else 0

            # Features for XGBoost
            features = np.array([[
                day_of_week, month, day, is_weekend, is_holiday,
                sin_day, cos_day, sin_month, cos_month,
                prev, rolling_3, ema_3, pct_change
            ]])

            # LSTM dummy seq
            lstm_seq = np.array(log_values[-14:]).reshape(1, 14, 1)
            lstm_pred = np.expm1(lstm_model.predict(lstm_seq, verbose=0)[0][0])
            xgb_pred = np.expm1(xgb_model.predict(features)[0])
            hybrid = meta_model.predict([[xgb_pred, lstm_pred]])[0]

            log_values.append(np.log1p(hybrid))  # simulate next day's log for future predictions
            future_preds.append({
                "date": str(date.date()),
                "predicted_expense": round(hybrid, 2)
            })

        total_predicted_budget = sum([entry['predicted_expense'] for entry in future_preds])
        print("Total predicted budget:", total_predicted_budget)
        
        return jsonify({
            "next_month_budget": float(f"{total_predicted_budget:.2f}"),
        })

    except Exception as e:
        print("IN EXCEPTION",e)
        return jsonify({"error": str(e)}), 500

# ==== Start Flask Server ====
if __name__ == '__main__':
    app.run(debug=True, port=5000)
