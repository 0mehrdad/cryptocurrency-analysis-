import streamlit as st
import pandas as pd
import yfinance as yf
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tick = [
     'SOL-USD', 'MATIC-USD',
    'ARB-USD', 'OP-USD', 'EGLD-USD', 'KAS-USD', 'NEAR-USD',
    'UNI-USD', 'AAVE-USD', 'SNX-USD',
    'SAND-USD', 'AXS-USD', 'MANA-USD', 'ILV-USD',
    'XMR-USD', 'ZEC-USD', 'DASH-USD', 'KMD-USD', 'SCRT-USD',
    'OKB-USD', 'HT-USD', 'FTT-USD', 'CRO-USD',
    'USDC-USD', 'DAI-USD',"TRX-USD",'INJ-USD','FET-USD','LTC-USD','FIL-USD'
]
start_date = datetime(2023, 10, 1).strftime('%Y-%m-%d')
end_date = datetime(2024, 10, 1).strftime('%Y-%m-%d')
df = yf.download(tick, start= start_date , end = end_date )['Close']
st.title ("Correlation Analysis")
corr_matrix = df.corr()
selected = st.selectbox(
    "Choose a Cryptocurrency",
    ('ZEC-USD','XMR-USD','LTC-USD','AXS-USD'))
selected1 = st.selectbox(
    'choose an Option',
    ('Most Positively Correlated', 'Most Negatively Correlated'))
if selected1 == 'Most Positively Correlated':
    top_corr = corr_matrix[selected].drop(selected).nlargest(4)
    selected_cryptos = [selected] + top_corr.index.tolist()
    filtered_corr = corr_matrix.loc[selected_cryptos, selected_cryptos]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(filtered_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax) 
    ax.set_title(f"Correlation Matrix for {selected} and its Top 4 Correlated Cryptocurrencies")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig)
elif selected1 == 'Most Negatively Correlated':
    bottom_corr = corr_matrix[selected].drop(selected).nsmallest(4)
    selected_cryptos = [selected] + bottom_corr.index.tolist()
    filtered_corr = corr_matrix.loc[selected_cryptos, selected_cryptos]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(filtered_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax) 
    ax.set_title(f"Correlation Matrix for {selected} and its Bottom 4 Correlated Cryptocurrencies")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig)