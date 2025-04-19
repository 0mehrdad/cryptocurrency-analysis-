import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
import yfinance as yf
import time
from datetime import datetime
import torch
from darts.models import NBEATSModel
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel

start_date = datetime(2023, 10, 1).strftime('%Y-%m-%d')
end_date = datetime(2024, 10, 1).strftime('%Y-%m-%d')

st.title ("Predicting and forecasting with N-BEATS and Transformers")
select_crypto = st.selectbox(
    "Select the cryptocurrency you want to predict",
      ("ZEC-USD", "XMR-USD", "LTC-USD", "AXS-USD"))
select_model = st.selectbox(
    "Select the model you want to use",
      ("N-BEATS", "Transformer"))
select_forcast = st.slider(
    'Select the number of days to forecast',1, 100,7)
if select_model == 'N-BEATS':
    df = yf.download(select_crypto, start=start_date, end=end_date)
    file_path = os.path.dirname(os.path.abspath(__file__))
    model_path = file_path + f"\models_extra\{select_model}_{select_crypto}"
    model = NBEATSModel.load(model_path)
    forecast = model.predict(n=select_forcast+75)
    scaler = Scaler()
    df.columns = df.columns.get_level_values(0)
    df1 = df.reset_index()
    series = TimeSeries.from_dataframe(df1, 'Date', 'Close')
    series_scaled = scaler.fit_transform(series)
    forecast = scaler.inverse_transform(forecast)
    forecast = forecast.pd_dataframe()
    forecast.columns = ['Forecast']
    original = df['Close']
    combined = pd.DataFrame({
    'Original': df['Close'],
    'Prediction': forecast['Forecast']}
    , index = original.index.union(forecast.index))
    combined.columns = ['Actual', 'Forecast']
    st.line_chart(combined, use_container_width=True, color=['#0000FF', "#FF0000"])
    cl1 , cl2 = st.columns(2)
    with cl1:
        st.subheader("Forecast")
        forecast = forecast.rename(columns={forecast.columns[0]: 'Forecast'})
        st.write(forecast[75:])
    with cl2:
        st.subheader("Buy and Sell Signals")
        selected_signal = st.selectbox(
            "Select the signal you want to use",
              ("Buy Signal", "Sell Signal"))
        target = forecast[75:]
        if selected_signal == 'Buy Signal':
            st.write(f'Buy Signal Price: {target['Forecast'].min():.3f}')
            st.write('Buy Signal Date:' , target['Forecast'].idxmin().strftime('%Y-%m-%d'))
        elif selected_signal == 'Sell Signal':
            st.write(f'Sell Signal Price: {target['Forecast'].max():.3f}')
            st.write('Sell Signal Date:' , target['Forecast'].idxmax().strftime('%Y-%m-%d'))


if select_model == 'Transformer':
    df = yf.download(select_crypto, start=start_date, end=end_date)
    file_path = os.path.dirname(os.path.abspath(__file__))
    model_path = file_path + f"\models_extra\{select_model}_{select_crypto}"
    model = TransformerModel.load(model_path)
    forecast = model.predict(n=select_forcast+75)
    scaler = Scaler()
    df.columns = df.columns.get_level_values(0)
    series = TimeSeries.from_dataframe(df, value_cols="Close")
    series_scaled = scaler.fit_transform(series)
    forecast = scaler.inverse_transform(forecast)
    forecast = forecast.pd_dataframe()
    last_date = df.index[-75] 
    new_index = pd.date_range(
    start=last_date + series.freq,  
    periods=len(forecast),
    freq=series.freq)
    forecast.columns = ['Forecast']
    forecast.index = new_index
    original = df['Close']
    combined = pd.DataFrame({
    'Original': df['Close'],
    'Prediction': forecast['Forecast']}
    , index = original.index.union(forecast.index))
    combined.columns = ['Actual', 'Forecast']
    st.line_chart(combined, use_container_width=True, color=['#0000FF', "#FF0000"])
    cl1 , cl2 = st.columns(2)
    with cl1:
        st.subheader("Forecast")
        forecast = forecast.rename(columns={forecast.columns[0]: 'Forecast'})
        st.write(forecast[75:])
    with cl2:
        st.subheader("Buy and Sell Signals")
        selected_signal = st.selectbox(
            "Select the signal you want to use",
              ("Buy Signal", "Sell Signal"))
        target = forecast[75:]
        if selected_signal == 'Buy Signal':
            st.write(f'Buy Signal Price: {target['Forecast'].min():.3f}')
            st.write('Buy Signal Date:' , target['Forecast'].idxmin().strftime('%Y-%m-%d'))
        elif selected_signal == 'Sell Signal':
            st.write(f'Sell Signal Price: {target['Forecast'].max():.3f}')
            st.write('Sell Signal Date:' , target['Forecast'].idxmax().strftime('%Y-%m-%d'))

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
