import streamlit as st
import yfinance as yf
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tick = ['ZEC-USD','XMR-USD','LTC-USD','AXS-USD']
# setting date
start_date = datetime(2023, 10, 1).strftime('%Y-%m-%d')
end_date = datetime(2024, 10, 1).strftime('%Y-%m-%d')
df1 = yf.download(tick, start= start_date , end = end_date )
df = df1['Close']
st.title('Exploratory Data Analysis')
option_1 = st.selectbox(
    "Choose a cryptocurrency",
    ("ZEC-USD", "XMR-USD", "LTC-USD", "AXS-USD"))
st.write(f"Selected cryptocurrency: {option_1}")
option_2 = st.selectbox(
    "Choose a plot",
    ( "Candlestick" , "Line Plot", "Moving Average", "Histogram", "Volume"))
st.write(f"Selected plot: {option_2}")
df1.columns = df1.columns.swaplevel(0, 1)
if option_2 == "Candlestick":
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(x=df1.index,
                                         open=df1[option_1]['Open'],
                                         high=df1[option_1]['High'],
                                         low=df1[option_1]['Low'],
                                         close=df1[option_1]['Close'],
                                         )])
    fig.update_layout(title=f'{option_1} Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)
elif option_2 == "Line Plot":
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df[option_1], label='Close Price')
    plt.title(f'{option_1} Price Chart')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)
elif option_2 == "Moving Average":
    window = st.slider("Select moving average window", 1, 30, 5)
    df['Moving Average'] = df[option_1].rolling(window=window).mean()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df[option_1], label='Close Price')
    plt.plot(df['Moving Average'], label=f'{window}-Day Moving Average', color='orange')
    plt.title(f'{option_1} Price Chart with Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)
elif option_2 == "Histogram":
    fig = plt.figure(figsize=(10, 5))
    plt.hist(df[option_1], bins=30, alpha=0.7, color='blue')
    plt.title(f'{option_1} Histogram')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    st.pyplot(fig)
elif option_2 == "Volume":
    fig = plt.figure(figsize=(10, 5))
    plt.bar(df1.index, df1[option_1]['Volume'], color='blue', alpha=0.7)
    plt.title(f'{option_1} Volume Chart')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    st.pyplot(fig)
st.write("## Statistical Summary")
st.dataframe(df1[option_1].describe(), height=300, width=1000)
st.write("## Dataframe")
st.dataframe(df1[option_1], height=300, width=1000)
