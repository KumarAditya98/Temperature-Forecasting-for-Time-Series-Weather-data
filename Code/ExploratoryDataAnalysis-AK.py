import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox import cal_autocorr, ADF_Cal, kpss_test, Cal_rolling_mean_var, ACF_PACF_Plot, diff
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets

# import os
# os.getcwd()

# Retrieving feature set and target set created from previous file.

df_target = pd.read_csv('Dataset/target_series.csv',index_col='DateTime')
df_features = pd.read_csv('Dataset/feature_series.csv',index_col='DateTime')

# Making a time-series plot for target variable
fig, ax = plt.subplots(figsize=(16,8))
df_target['T(degC)'].plot(label="Daily time-series Data")
plt.grid()
plt.legend()
plt.title("Plotting the daily aggregated time-series")
plt.xlabel("Time")
plt.ylabel("Temperature (degC)")
plt.show()

# It appears stationary at first glance. Performing an ACF function and plotting it.
# lags calculation = Number of observations/50 = 92 lags
# df_target.shape
lags = 92
# target = np.array(df_target)
ACF_PACF_Plot(df_target['T(degC)'],lags)
plt.show()

# The ACF-PACF plot of the time-series shows high correlation between the lagged values. However, judging by the tail-off in ACF and cut-off in PACF after lag = 6, this appears to be the charateristics of a typical AR process. However, there does appear to be seasonality in the data judging by the ACF plot, with mild peaks in the ACF values at regular intervals. But figuring out the period of this seasonality is difficult due to overall high correlation between the lags.
# The ACF plot seems to have a sinusoidal decay. with a seasonality period of 365 (plotted for 500 lags)

# Doing a correlation matrix plot between all the variables (target+features) in the dataset.
df = pd.concat((df_features,df_target),axis=1)
correlation_matrix = df.corr(method='pearson')
fig, ax = plt.subplots(figsize = (16,12))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',ax=ax,fmt='.2f',annot_kws={"size": 35 / np.sqrt(len(correlation_matrix)),"fontweight":'bold'},cbar=True)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=35 / np.sqrt(len(correlation_matrix)),width=2)
fig.suptitle('Correlation Matrix for all variables (Pearson)',fontweight='bold',size=24)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.tight_layout()
plt.show()

# Here we see the presence of very strong multi-collinearity within features, as well as high correlation between features and the target variable.

# Performing a train-test split for further analysis.
X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, shuffle=False, test_size=0.2, random_state=6313)

# Saving out these train and test datasets for model building later on.
X_test.to_csv('Dataset/X_test.csv')
y_test.to_csv('Dataset/y_test.csv')
X_train.to_csv('Dataset/X_train.csv')
y_train.to_csv('Dataset/y_train.csv')

# Stationarity tests
Cal_rolling_mean_var(df_target["T(degC)"])
# The rolling mean and rolling mean variance tend to stabilize after all the sample have been added to the calculation

ADF_Cal(df_target['T(degC)'])
# Stationary

kpss_test(df_target['T(degC)'])
# Stationary

# Looking at above ACF/PACF plots for stationarity we can see a clear AR pattern, indicating that the process is stationary. However, the ACF seems to decay very slowly suggesting that the process could be non-stationary, but this is because of high correlation between the lags.

# Therefore, since process is stationary, no transformations are required.

# Time series decomposition
# Since the magnitude of my variance remains the same and is not changing, I will consider an additive decomposition for my time series and not a multiplicative one.

STL = STL(df_target["T(degC)"],period=365)
# Choosing period as 365 since the expectation with weather data is seasonality of 365 days
res = STL.fit()
T = res.trend
S = res.seasonal
R = res.resid
Seasonally_adjusted = df_target['T(degC)'] - S
# Plotting seasonally adjusted data
fig, ax = plt.subplots(figsize=(16,8))
Seasonally_adjusted.plot(label="Daily time-series Data - Seasonally adjusted")
plt.grid()
plt.legend()
plt.title("Plotting the daily aggregated time-series after seasonal adjustment")
plt.xlabel("Time")
plt.ylabel("Temperature (degC)")
plt.show()
# No apparent trend, presence of residual makes it look like noise
trend_adjusted = df_target['T(degC)'] - T
# Plotting trend adjusted data
fig, ax = plt.subplots(figsize=(16,8))
trend_adjusted.plot(label="Daily time-series Data - Trend adjusted")
plt.grid()
plt.legend()
plt.title("Plotting the daily aggregated time-series after trend adjustment")
plt.xlabel("Time")
plt.ylabel("Temperature (degC)")
plt.show()
# Presence of high seasonality
# Over all breakdown
fig = res.plot()
plt.show()
SOT = max(0,(1-((np.var(R))/(np.var(R+T)))))
print(SOT)
# Very less trended
SOS = max(0,(1-((np.var(R))/(np.var(R+S)))))
print(SOS)
# Very highly seasonal

# Doubts - If Seasonal data but stationary, do we need to perform differencing??? Do we need to remove seasonality before GPAC and model building? Seasonal differencing removes seasonality but its to make time series stationary, so if its already stationary, do we need to do differencing??
# Back-transformation for differencing doubt. When we do differencing, we're transforming the data so the forecast we make is on this transformed data. Do we then require a back transformation on the forecasted values?

# For testing purpose, I will perform a 1 order seasonal differencing (365) in the dataset
df_target_diff = diff(df_target,'T(degC)',365)
# Maintaining original df by removing the new column
df_target.drop(columns=['T(degC)_365_Diff'],axis=1,inplace=True)

Cal_rolling_mean_var(df_target_diff['T(degC)_365_Diff'].dropna())
ADF_Cal(df_target_diff['T(degC)_365_Diff'].dropna())
kpss_test(df_target_diff['T(degC)_365_Diff'].dropna())
# The process still remains stationary judging by above tests, in fact becomes more stationary based on ADF test statistic.
ACF_PACF_Plot(df_target_diff['T(degC)_365_Diff'].dropna(),lags)
plt.show()
# ACF-PACF plot display a much more cleaner trend with decay in ACF plot and cut-off in PACF plot. This is a typical AR process behavior. This makes my assumption that the seasonality is 365 correct.
# I can make use of this seasonally differenced dataset for modeling as well since everything is right. However, one disadvantage is the loss of data points due to the seasonal differencing of 365. I will first proceed with my original dataset and then shift to this if that is not producing good results. Train-test split pending for this differenced data.

# This will be the end for my Exploratory Data Analysis. Continued in pre-processing and modeling python file

