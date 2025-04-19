from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from prophet import Prophet
from keras.models import Sequential
import keras
from keras.layers import LSTM, Dense, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



def run_arima(df,steps=7):
    model = ARIMA(df['Close'], order=(1, 1, 1))
    model_fit = model.fit()
    in_sample_preds = model_fit.predict(start=len(df)-75, end=len(df)-1)
    forecast = model_fit.forecast(steps=steps)
    return in_sample_preds.round(3) , forecast.round(3)


def run_xgboost(df,steps=7):
    df['5-Day MA'] = df['Close'].rolling(window=5).mean()
    df['10-Day MA'] = df['Close'].rolling(window=10).mean()
    df['Pct Change'] = df['Close'].pct_change()
    df['Prev Close'] = df['Close'].shift(1)
    df['Prev 5-Day MA'] = df['5-Day MA'].shift(1)
    df['Prev 10-Day MA'] = df['10-Day MA'].shift(1)
    df['Prev Pct Change'] = df['Pct Change'].shift(1)
    df.dropna(inplace=True)
    X = df[['Prev Close', 'Prev 5-Day MA', 'Prev 10-Day MA', 'Prev Pct Change']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = []
    last_row = df.iloc[-1]
    current_row = last_row
    closing_prices = df.iloc[-10:,0].values.tolist()
    for i in range(steps):
        input_data = current_row[['Prev Close', 'Prev 5-Day MA', 'Prev 10-Day MA', 'Prev Pct Change']].values.reshape(1, -1)

        predicted_close = model.predict(input_data)

        predictions.append(predicted_close[0])

        closing_prices.append(predicted_close[0])
        prev_5_day_ma = sum(closing_prices[-5:]) / 5 
        prev_10_day_ma = sum(closing_prices[-10:]) / 10
        prev_pct_change = (predicted_close[0] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) > 1 else 0
        current_row['Prev Close'] = predicted_close[0]
        current_row['Prev 5-Day MA'] = prev_5_day_ma
        current_row['Prev 10-Day MA'] = prev_10_day_ma
        current_row['Prev Pct Change'] = prev_pct_change
        new_row = pd.Series(current_row, name=pd.to_datetime(df.index[-1]) + pd.Timedelta(days=1))
        df = pd.concat([df, new_row.to_frame().T])

    forecast_dates = pd.date_range(df.index[-steps], periods=(steps+1), freq='D')[1:]
    forecast = pd.Series(predictions, index=forecast_dates).round(3).to_frame()
    forecast.columns = ['Forecast']
    y_pred = pd.Series(y_pred, index=X_test.index).round(3).to_frame()
    y_pred.columns = ['Forecast']
    return  y_pred , forecast

def run_prophet(df,steps=7):
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    fig = model.plot(forecast)
    forecast = forecast[['ds', 'yhat']].set_index('ds').round(3)
    forecast.columns = ['Forecast']
    return forecast, fig

def run_lstm(df,model_path,steps=7):
    df['Close_smooth'] = df['Close'].rolling(window=3).mean()
    df['day_of_week'] = df.index.dayofweek
    df = df.dropna()
    df = df[['Close_smooth', 'Volume', 'day_of_week']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    def create_seq2seq_dataset(data, time_step=60, forecast_horizon=20):
        X, y = [], []
        for i in range(len(data) - time_step - forecast_horizon):
            X.append(data[i:i + time_step])
            y.append(data[i + time_step:i + time_step + forecast_horizon, 0]) 
        return np.array(X), np.array(y)
    time_step = 60
    forecast_horizon = steps
    X, y = create_seq2seq_dataset(scaled_data, time_step, forecast_horizon)
    X = X.reshape(X.shape[0], X.shape[1], scaled_data.shape[1])  
    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    y_test = y[train_size:]
    def inverse_transform_sequence(seq_pred):
        padded = np.concatenate([seq_pred.reshape(-1, 1), np.zeros((seq_pred.size, 2))], axis=1)
        return scaler.inverse_transform(padded)[:, 0]
    
    model = keras.models.load_model(model_path)
    y_pred = model.predict(X_test)
    y_pred = np.array([inverse_transform_sequence(seq) for seq in y_pred])
    in_sample_preds =  np.concatenate([y_pred[:, 0], y_pred[-1, 1:steps]])
    in_sample_preds = pd.Series(in_sample_preds, index=df.index[-len(in_sample_preds):]).round(3).to_frame()
    in_sample_preds.columns = ['Forecast']
    last_input_seq = scaled_data[-time_step:]
    last_input_seq = last_input_seq.reshape(1, time_step, scaled_data.shape[1]) 
    future_prediction_scaled = model.predict(last_input_seq) 
    future_prediction_scaled = future_prediction_scaled[0]  
    padded_future_preds = np.concatenate([
        future_prediction_scaled.reshape(-1, 1), 
        np.zeros((forecast_horizon, 2))], axis=1)
    future_prediction_actual = scaler.inverse_transform(padded_future_preds)[:, 0]    
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    future_prediction_actual = pd.Series(future_prediction_actual, index=future_dates)
    future_prediction_actual = future_prediction_actual.to_frame()
    future_prediction_actual.columns = ['Forecast']
    return in_sample_preds , future_prediction_actual

