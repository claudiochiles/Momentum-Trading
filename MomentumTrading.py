#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 22:58:39 2018

@author: Claudio
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import seaborn as sns; sns.set()

#start date
yearS=2016
monthS=1
dayS=1

#end date
yearE=2018
monthE=6
dayE=24

#stock ticker
ticker = ['ENELAM.SN','ECL.SN','CAP.SN','FALABELLA.SN','LTM.SN',
          'BCI.SN','FORUS.SN','SQM-B.SN','CCU.SN','^IPSA']

#Recommended Stocks to Buy
stock = pdr.get_data_yahoo(ticker[0], 
                          start=datetime.datetime(yearS,monthS,dayS), 
                          end=datetime.datetime(yearE, monthE, dayE))

## Sample and Filtering data
sample = stock.sample(20)
monthly_sample = stock.resample('M').apply(lambda x: x[-1])
monthly_sample.pct_change()
# Resample `stock` to business months, take last observation as value 
monthlyBM = stock.resample('BM').apply(lambda x: x[-1])
monthlyBM.pct_change()
# Resample `aapl` to quarters, take the mean as value per quarter
quarter = stock.resample("4M").mean()
# Calculate the quarterly percentage change
quarter.pct_change()

## Returns for Stock
# Assign `Adj Close` to `daily_close`
daily_close = stock[['Adj Close']]
# Daily returns
daily_pct_c = daily_close.pct_change()
# Replace NA values with 0
daily_pct_c.fillna(0, inplace=True)
# Daily log returns
daily_log_returns = np.log(daily_close.pct_change()+1)
daily_log_returns.fillna(0, inplace=True)
# Daily returns not log returns
daily_pct_c = daily_close / daily_close.shift(1) - 1
# Calculate the cumulative daily returns# Calcul 
cum_daily_return = (1 + daily_pct_c).cumprod()
# Resample the cumulative daily return to cumulative monthly return 
cum_monthly_return = cum_daily_return.resample("M").mean()

## Plot and Histogram of daily returns
#linear plot of daily returns
# Just a figure and one subplot
plt.subplot(211)
plt.plot(stock['Adj Close'])
plt.title("Adjusted Price of Stock")
plt.subplot(212)
plt.plot(daily_pct_c)
plt.title("Returns of Stock")
plt.subplots_adjust(hspace=.5)
daily_pct_c.hist(bins=50,sharex=True)
# Show the plot
plt.show()
# Pull up summary statistics
print(daily_pct_c.describe())

## Moving windows (Moving Average)
# Isolate the adjusted closing prices 
adj_close_px = stock['Adj Close']

## Calculate the moving average
moving_avg = adj_close_px.rolling(window=50).mean()
# Inspect the result
moving_avg[-10:]
# Short moving window rolling mean
stock['50 MA'] = adj_close_px.rolling(window=50).mean()
# Long moving window rolling mean
stock['200 MA'] = adj_close_px.rolling(window=200).mean()
# Plot the adjusted closing price, the short and long windows of rolling means
stock[['Adj Close', '50 MA', '200 MA']].plot()
plt.title("Moving Averages")
plt.show()

## Volatility Calculation
# Define the minumum of periods to consider 
min_periods = 75 
# Calculate the volatility
vol = daily_pct_c.rolling(min_periods).std() * np.sqrt(min_periods) 
# Plot the volatility
vol.plot(figsize=(10, 8))
# Show the plot
plt.title("Volatility of Stock")
plt.show()

################# Trading Strategies #################

##Moving Average Strategy (Trend following trading)
Current = stock['Adj Close']
MA50 = stock['50 MA']
MA200 = stock['50 MA']
if (MA50[-1] > MA200[-1]) & (MA50[-1] > Current[-1]):
    print('Buy that stock')
else:
    print('Do not buy that stock')

# Initialize the short and long windows
short_window = 50
long_window = 200
# Initialize the `signals` DataFrame with the `signal` column
signals = pd.DataFrame(index=stock.index)
signals['signal'] = 0.0
# Create short simple moving average over the short window
signals['short_mavg'] = stock['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
# Create long simple moving average over the long window
signals['long_mavg'] = stock['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)   
# Generate trading orders
signals['positions'] = signals['signal'].diff()
# Initialize the plot figure
fig = plt.figure()
# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price in $')
# Plot the closing price
stock['Close'].plot(ax=ax1, color='r', lw=2.)
# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index, 
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')        
# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index, 
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')         
# Show the plot
plt.show()

