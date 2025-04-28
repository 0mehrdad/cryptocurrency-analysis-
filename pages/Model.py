import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
import yfinance as yf
import time
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import models
from sklearn.metrics import mean_squared_error, mean_absolute_error

start_date = datetime(2023, 10, 1).strftime('%Y-%m-%d')
end_date = datetime(2024, 10, 1).strftime('%Y-%m-%d')
forcast_date = datetime(2025, 2, 1).strftime('%Y-%m-%d')

st.title ("Predicting and forecasting")
select_crypto = st.selectbox(
    "Select the cryptocurrency you want to predict",
      ("ZEC-USD", "XMR-USD", "LTC-USD", "AXS-USD"))
select_model = st.selectbox(
    "Select the model you want to use",
      ("ARIMA", "LSTM", "XGBoost","Prophet"))
select_forcast = st.selectbox(
    "Select the forecast you want to use",
      ("7 days", "14 days", "30 days"))
if select_forcast == '7 days':
    select_forcast = 7
elif select_forcast == '14 days': 
    select_forcast = 14
elif select_forcast == '30 days': 
    select_forcast = 30
if select_model == 'ARIMA': 
    df = yf.download(select_crypto, start=start_date, end=end_date)
    forecast_original = yf.download(select_crypto, start=end_date, end=forcast_date)['Close']
    in_sample_preds , forecast = models.run_arima(df,steps=select_forcast)
    last_actual_value = pd.Series([df['Close'].iloc[-1][select_crypto]], index=[df.index[-1]])
    concat = pd.concat([ in_sample_preds , forecast])
    original = df['Close']
    combined = pd.DataFrame({
    'Original': df['Close'][select_crypto],
    'Prediction': concat
}
    , index = original.index.union(forecast.index))
    combined.columns = ['Actual', 'Forecast']
    st.line_chart(combined, use_container_width=True, color=['#0000FF', "#FF0000"])
    cl1 , cl2 = st.columns(2)
    with cl1:
        st.subheader("Forecast")
        forecast = forecast.to_frame()
        forecast.columns = ['Forecast']
        st.write(forecast)
    with cl2:
        st.subheader("Buy and Sell Signals")
        selected_signal = st.selectbox(
            "Select the signal you want to use",
              ("Buy Signal", "Sell Signal"))
        if selected_signal == 'Buy Signal':
            st.write(f'Buy Signal Price: {combined['Forecast'].min():.3f}')
            st.write('Buy Signal Date:' , combined['Forecast'].idxmin().strftime('%Y-%m-%d'))
        elif selected_signal == 'Sell Signal':
            st.write(f'Sell Signal Price: {combined['Forecast'].max():.3f}')
            st.write('Sell Signal Date:' , combined['Forecast'].idxmax().strftime('%Y-%m-%d'))

    #model evaluation
    st.subheader("Model Evaluation")
    st.write("In-sample predictions:")
    st.write(f"Mean Absolute Error: {mean_absolute_error(original.loc[in_sample_preds.index], in_sample_preds):.3f}")
    st.write(f"Mean Squared Error: {mean_squared_error(original.loc[in_sample_preds.index], in_sample_preds):.3f}")
    st.write("Forecast:")
    st.write(f"Mean Absolute Error: {mean_absolute_error(forecast_original.loc[forecast.index], forecast):.3f}")
    st.write(f"Mean Squared Error: {mean_squared_error(forecast_original.loc[forecast.index], forecast):.3f}")

elif select_model == 'XGBoost':
    df = yf.download(select_crypto, start=start_date, end=end_date)
    forecast_original = yf.download(select_crypto, start=end_date, end=forcast_date)['Close']
    in_sample_preds , forecast = models.run_xgboost(df,steps=select_forcast)
    last_actual_value = pd.Series([df['Close'].iloc[-1][select_crypto]], index=[df.index[-1]])
    forecast_ = forecast.iloc[:, 0]
    concat = pd.concat([ in_sample_preds , forecast_])
    original = df['Close']
    combined = pd.DataFrame({
    'Original': df['Close'][select_crypto],
    'Prediction': concat['Forecast']
}
    , index = original.index.union(forecast.index))
    combined.columns = ['Actual', 'Forecast']
    st.line_chart(combined, use_container_width=True, color=['#0000FF', "#FF0000"])
    cl1 , cl2 = st.columns(2)
    with cl1:
        st.subheader("Forecast")
        st.write(forecast)
    with cl2:
        st.subheader("Buy and Sell Signals")
        selected_signal = st.selectbox(
            "Select the signal you want to use",
              ("Buy Signal", "Sell Signal"))
        if selected_signal == 'Buy Signal':
            st.write(f'Buy Signal Price: {combined['Forecast'].min():.3f}')
            st.write('Buy Signal Date:' , combined['Forecast'].idxmin().strftime('%Y-%m-%d'))
        elif selected_signal == 'Sell Signal':
            st.write(f'Sell Signal Price: {combined['Forecast'].max():.3f}')
            st.write('Sell Signal Date:' , combined['Forecast'].idxmax().strftime('%Y-%m-%d'))

    #model evaluation
    st.subheader("Model Evaluation")
    st.write("In-sample predictions:")
    st.write(f"Mean Absolute Error: {mean_absolute_error(original.loc[in_sample_preds.index], in_sample_preds):.3f}")
    st.write(f"Mean Squared Error: {mean_squared_error(original.loc[in_sample_preds.index], in_sample_preds):.3f}")
    st.write("Forecast:")
    st.write(f"Mean Absolute Error: {mean_absolute_error(forecast_original.loc[forecast.index], forecast):.3f}")
    st.write(f"Mean Squared Error: {mean_squared_error(forecast_original.loc[forecast.index], forecast):.3f}")

elif select_model == 'LSTM':
    df = yf.download(select_crypto, start=start_date, end=end_date)
    forecast_original = yf.download(select_crypto, start=end_date, end=forcast_date)['Close']
    file_path = os.path.dirname(os.path.abspath(__file__))
    model_path = file_path + f"\models\{select_crypto}_{select_forcast}.keras" 
    in_sample_preds , forecast = models.run_lstm(df,model_path,steps=select_forcast)
    last_actual_value = pd.Series([df['Close'].iloc[-1][select_crypto]], index=[df.index[-1]])
    forecast_ = forecast.iloc[:, 0]
    concat = pd.concat([ in_sample_preds , forecast_])
    original = df['Close']
    combined = pd.DataFrame({
    'Original': df['Close'][select_crypto],
    'Prediction': concat['Forecast']
}
    , index = original.index.union(forecast.index))
    combined.columns = ['Actual', 'Forecast']
    st.line_chart(combined, use_container_width=True, color=['#0000FF', "#FF0000"])
    cl1 , cl2 = st.columns(2)
    with cl1:
        st.subheader("Forecast")
        forecast = forecast.rename(columns={forecast.columns[0]: 'Forecast'})
        st.write(forecast)
    with cl2:
        st.subheader("Buy and Sell Signals")
        selected_signal = st.selectbox(
            "Select the signal you want to use",
              ("Buy Signal", "Sell Signal"))
        if selected_signal == 'Buy Signal':
            st.write(f'Buy Signal Price: {combined['Forecast'].min():.3f}')
            st.write('Buy Signal Date:' , combined['Forecast'].idxmin().strftime('%Y-%m-%d'))
        elif selected_signal == 'Sell Signal':
            st.write(f'Sell Signal Price: {combined['Forecast'].max():.3f}')
            st.write('Sell Signal Date:' , combined['Forecast'].idxmax().strftime('%Y-%m-%d'))
        
    #model evaluation
    st.subheader("Model Evaluation")
    st.write("In-sample predictions:")
    st.write(f"Mean Absolute Error: {mean_absolute_error(original.loc[in_sample_preds.index], in_sample_preds):.3f}")
    st.write(f"Mean Squared Error: {mean_squared_error(original.loc[in_sample_preds.index], in_sample_preds):.3f}")
    st.write("Forecast:")
    st.write(f"Mean Absolute Error: {mean_absolute_error(forecast_original.loc[forecast.index], forecast):.3f}")
    st.write(f"Mean Squared Error: {mean_squared_error(forecast_original.loc[forecast.index], forecast):.3f}")

elif select_model == 'Prophet':
    df = yf.download(select_crypto, start=start_date, end=end_date)
    df_copy = df.copy()
    forecast_original = yf.download(select_crypto, start=end_date, end=forcast_date)['Close']
    df.columns = df.columns.get_level_values(0)
    forecast , fig = models.run_prophet(df,steps=select_forcast)
    st.pyplot(fig)
    cl1 , cl2 = st.columns(2)
    with cl1:
        st.subheader("Forecast")
        st.write(forecast)
    with cl2:
        st.subheader("Buy and Sell Signals")
        selected_signal = st.selectbox(
            "Select the signal you want to use",
              ("Buy Signal", "Sell Signal"))
        if selected_signal == 'Buy Signal':
            st.write(f'Buy Signal Price: {forecast['Forecast'].min():.3f}')
            st.write('Buy Signal Date:' , forecast['Forecast'].idxmin().strftime('%Y-%m-%d'))
        elif selected_signal == 'Sell Signal':
            st.write(f'Sell Signal Price: {forecast['Forecast'].max():.3f}')
            st.write('Sell Signal Date:' , forecast['Forecast'].idxmax().strftime('%Y-%m-%d'))
    df_concat = pd.concat([df_copy['Close'], forecast_original], axis=0)
    #model evaluation
    st.subheader("Model Evaluation")
    st.write("In-sample predictions:")
    st.write(f"Mean Absolute Error: {mean_absolute_error(df_copy['Close'], forecast['Forecast'].loc[df_copy.index]):.3f}")
    st.write(f"Mean Squared Error: {mean_squared_error(df_copy['Close'], forecast['Forecast'].loc[df_copy.index]):.3f}")
    st.write("Forecast:")
    st.write(f"Mean Absolute Error: {mean_absolute_error(forecast_original[-select_forcast:], 
                                                         forecast['Forecast'][-select_forcast:]):.3f}")
    st.write(f"Mean Squared Error: {mean_squared_error(forecast_original[-select_forcast:],
                                                         forecast['Forecast'][-select_forcast:]):.3f}")


# Display similar cryptocurrencies
st.subheader("Similar Cryptocurrencies")
file_path = os.path.dirname(os.path.abspath(__file__))
df_similar= pd.read_csv(f'{file_path}\df_clusters.csv')
similar = df_similar.Cluster[df_similar['Crypto']== select_crypto]
similar = df_similar[df_similar['Cluster'] == similar.values[0]]
similar = similar.drop(columns=['Cluster'])
cryptos = similar['Crypto'].tolist()
cryptos = [c for c in cryptos if c != select_crypto][:5]
joined = ", ".join(cryptos)
if cryptos!= []:
    st.write(f"🔍 Cryptocurrencies similar to **{select_crypto}**: {joined}")
    st.write('💰 You can buy and sell these cryptocurrencies with the same signals as the selected cryptocurrency.')
