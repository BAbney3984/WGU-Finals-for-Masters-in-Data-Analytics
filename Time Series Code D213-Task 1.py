#!/usr/bin/env python
# coding: utf-8

# # #NLM2 - Task 1: Time Series Modeling
# 
# Brittany Abney
# Student ID: 01024609
# WGU
# Representation and Reporting
# D213
# June 7, 2023

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


# In[2]:


#load the dataset
df = pd.read_csv('medical time series.csv')


# In[3]:


# Visually inspect dataframe to facilitate exploration, spot problems
pd.set_option("display.max_columns", None)
df


# In[4]:


#check for nulls
print(df.isnull().sum())


# In[5]:


df.info()


# In[6]:


print(df.describe())


# In[7]:


# Correct the day column to show dates, in datetime format 
start_date = pd.to_datetime('2022-01-01')
# Convert Day column to differences in time, by subtracting one (to count from 0, rather than 1) and then add the difference
# (the time_delta) to the previously established start date (so 1 day from Jan 1 2022, 2 days from Jan 1 2022, etc.
df['Day'] = pd.to_timedelta(df['Day']-1, unit='D') + start_date
# Rename columns to be Pythonic 
df.columns = ['date', 'revenue']
# With datetime column properly established, set this as index
df.set_index('date', inplace=True)
# Visually inspect final dataframe to verify appearance
df


# In[8]:


# Long X and small Y dictate a wide graph figure
plt.figure(figsize = [16,5])
# Prettify the graph
plt.title("Hospital Daily Revenue, 2022 - 2023")
plt.xlabel("Date")
plt.ylabel("Daily Revenue (in Millions USD")
# Plot time series data
plt.plot(df)
# Generate trend line
x = mdates.date2num(df.index)
y = df.revenue
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
# Plot trendline
plt.plot(x, p(x), "r--")
plt.show()


# 

# In[447]:


#Augmented Dickey-Fuller on data
#run Dickey-Fuller 
from statsmodels.tsa.stattools import adfuller
new_result = adfuller(df.revenue)
print('ADF Stat: %f' % new_result[0])
print('p-value: %f' % new_result[1])
print('Critical Values:')
for key, value in new_result[4].items():
    print('\t%s: %.3f' % (key, value))


# In[448]:


df_trans = df.diff().dropna()
# Perform Augmented Dicky-Fuller on data
adfuller_results = adfuller(df_trans.revenue)
# Print resulting test-statistic and p-value
print(f"The test statistic of an augmented Dicky-Fuller test on this data is {round(adfuller_results[0], 4)}, which has a p-value of {round(adfuller_results[1], 8)}")
# Plot to verify stationarity
df_trans.plot();


# In[449]:


#the above shows that the data is now stationary. Next I will split the data into a training and test set. I will include shuffle=False to make sure that the time series data is not rearranged and maintains its order. 
# Split time series into a training set and a test set
train, test = train_test_split(df_trans, test_size=0.2, shuffle=False, random_state=369)
train


# In[450]:


test


# In[451]:


#Save data
train.to_csv('train_clean.csv')
test.to_csv('test_clean.csv')
train.to_csv('Documents/Train_Task1.csv')
test.to_csv('Documents/Test_Task1.csv')

#save prepared data 
df.to_csv('Documents/PreparedData D213 Task1.csv')


# In[452]:


#Seasonality
# Decompose the transformed data
decomposed_data = seasonal_decompose(df_trans)
# Long X and small Y dictate a wide graph figure
plt.figure(figsize = [16,5])
# Plot seasonal component of the data
plt.plot(decomposed_data.seasonal);


# In[453]:


#Seasonality
# Long X and small y dictate a wide graph figure
plt.figure(figsize = [16,5])
# Plot seasonal component of data
plt.plot(decomposed_data.seasonal, marker='o')
plt.xlim(pd.to_datetime('2023-01-01'), pd.to_datetime('2023-02-01'))
# Draw red lines on Mondays
plt.axvline(x=pd.to_datetime('2023-01-05'), color='red')
plt.axvline(x=pd.to_datetime('2023-01-12'), color='red')
plt.axvline(x=pd.to_datetime('2023-01-19'), color='red')
plt.axvline(x=pd.to_datetime('2023-01-26'), color='red');


# In[454]:


#Trends
# Long X and small Y dictate a wide graph figure
plt.figure(figsize = [16,5])
# Plot trend component of the data
plt.plot(decomposed_data.trend);


# In[455]:


#Auto Correlation - plot autocorrelation and partial autocorrelation in one figure - share a y axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[16,5], sharey=True)
# Plot ACF to 8 lags (only 7 days in a week), ignore zero (zero always = 1)
plot_acf(df_trans, lags=8, zero=False, ax=ax1)
# Plot PACF to 8 lags (only 7 days in a week), ignore zero (zero always = 1)
plot_pacf(df_trans, lags=8, zero=False,ax=ax2)
# Zoom in on y axis 
plt.ylim(-0.6, 0.6);


# In[456]:


#spectral density

plt.psd(x=df_trans.revenue);


# In[457]:


#Decomposed Time Series
decomposed_data.plot()


# In[458]:


#Trends in Residuals of Decomp
# Long X and small Y dictate a wide graph figure
plt.figure(figsize = [16,5])
# Plot residual component of the data
plt.plot(decomposed_data.resid);


# In[459]:


#Arima Model of Time Series Data
#Above ACF and PACF plots indicate best data suited is AR(1) model
model = ARIMA(train, order=(1, 0, 0), freq='D')
results = model.fit()
print(results.summary())


# In[460]:


forecasted = results.get_prediction(start = 584, end = 729, dynamic = True)
plt.plot(test)
plt.plot(forecasted.predicted_mean);


# In[461]:


print(forecasted.predicted_mean)


# In[462]:


# Place the forecast differences into a temp df
forecast_temp = pd.DataFrame(forecasted.predicted_mean)
# Make consistent names for df for concatenation
forecast_temp.rename(columns={'predicted_mean' : 'revenue'}, inplace=True)
# Concat a copy of Train 
df_w_forecast = pd.concat([train.copy(), forecast_temp.copy()])
# We've generated one DF with the differences in daily revenue for the entire 2-year period, invert the differences using cumsum
df_w_forecast = df_w_forecast.cumsum()
# Check output to verify expected values 
df_w_forecast


# In[463]:


# Calculate confidence intervals from forecasted data
confidence_intervals = forecasted.conf_int()
# Like the forecast, these confidence limits are also differences in daily revenue, these need transformed back to daily revenue
confidence_intervals


# In[464]:


# Establish a df to match the confidence intervals dataframe, including the UNTRANSFORMED data 
previous_row = pd.DataFrame({'lower revenue': [19.312734], 'upper revenue' : [19.312734], 'date' : ['2023-08-07']})
# Convert given date string to datetime and then set as index
previous_row['date'] = pd.to_datetime(previous_row['date'])
previous_row.set_index('date', inplace=True)
previous_row


# In[465]:


# Concatenate the prior row and the confidence intervals data
confidence_intervals = pd.concat([previous_row, confidence_intervals])
# Un-transform the confidence intervals using cumsum()
confidence_intervals = confidence_intervals.cumsum()
# Make sure first row (data preceding the forecast) is omitted
confidence_intervals = confidence_intervals.loc['2023-08-08' : '2023-12-31']
# Verify un-transformed confidence intervals
confidence_intervals


# In[466]:


# Long X and small Y dictate a wide graph figure
plt.figure(figsize = [16,5])
# Prettify the graph
plt.title("Hospital  Daily Revenue, 2022 - 2023")
plt.xlabel("Date")
plt.ylabel("Daily Revenue (in Millions USD")
# Plot the forecasted data
plt.plot(df_w_forecast, color = 'green', linestyle = 'dashed')
# Plot the original data (includes both the train set and the test set, untransformed - their actual observed values)
plt.plot(df, color = 'blue')
# Plot the confidence intervals
plt.fill_between(confidence_intervals.index, confidence_intervals['lower revenue'], confidence_intervals['upper revenue'], color = 'pink')
# Keep the y-axis zoomed in, without expanding to fit the full confidence interval values
plt.ylim(-7, 27)
# Provide legend to distinguish predicted values from observed values
plt.legend(['Predicted', 'Observed'])
plt.show();


# In[467]:


# Calculate root mean squared error of forecasted data against the observed data (both untransformed)
rmse = mean_squared_error(df.loc['2023-08-08' : '2023-12-31'], df_w_forecast.revenue.loc['2023-08-08' : '2023-12-31'], squared=False)
print(f"The root mean squared error of this forecasting model is {round(rmse, 5)}")


# In[468]:


plt.figure(figsize = [16,16])
results.plot_diagnostics();


# In[ ]:




