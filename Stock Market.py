import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock

# Parameters
ticker = 'AAPL'  # Apple Inc. stock ticker
start_date = '2023-01-01'
end_date = '2024-01-01'

# Fetch stock data
df = fetch_stock_data(ticker, start_date, end_date)

# Basic statistical analysis
print("Basic Statistical Analysis:")
print(df.describe())

# Visualization
plt.figure(figsize=(14, 7))

# Closing price over time
plt.subplot(2, 2, 1)
plt.plot(df['Close'], label='Closing Price')
plt.title(f'{ticker} Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Volume over time
plt.subplot(2, 2, 2)
plt.plot(df['Volume'], label='Volume', color='orange')
plt.title(f'{ticker} Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()

# Moving Averages
df['20_day_MA'] = df['Close'].rolling(window=20).mean()
df['50_day_MA'] = df['Close'].rolling(window=50).mean()

plt.subplot(2, 2, 3)
plt.plot(df['Close'], label='Closing Price')
plt.plot(df['20_day_MA'], label='20 Day MA', linestyle='--')
plt.plot(df['50_day_MA'], label='50 Day MA', linestyle='--')
plt.title(f'{ticker} Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Daily returns
df['Daily Return'] = df['Close'].pct_change()

plt.subplot(2, 2, 4)
sns.histplot(df['Daily Return'].dropna(), bins=50, kde=True)
plt.title(f'{ticker} Daily Returns Distribution')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Correlation with other stocks (Example: Google and Amazon)
tickers = ['AAPL', 'GOOGL', 'AMZN']
data = yf.download(tickers, start=start_date, end=end_date)['Close']

correlation_matrix = data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Stock Correlation Matrix')
plt.show()
