import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox import cal_autocorr, ADF_Cal, kpss_test, Cal_rolling_mean_var
import seaborn as sns
from sklearn.model_selection import train_test_split
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
lags = 50
target = np.array(df_target)
cal_autocorr(target,lags,'Temperature (degC)')
plt.show()

# This plot shows a typical behavior of a non-stationary dataset due to the very slow decay of ACF towards zero. However, this just indicates the lagged values are highly correlated.

# Doing a correlation matrix plot between all the variables in the dataset.
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

# Performing a train-test split before proceeding with further analysis.
X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, shuffle=False, test_size=0.2, random_state=6313)




from toolbox import ADF_Cal
ADF_Cal(target)

from toolbox import kpss_test
kpss_test(target)

Temp_1 = pd.Series(y_train['T(degC)'].values,index = y_train.index,
                 name = 'Temp (degC)')
from toolbox import Cal_rolling_mean_var
Cal_rolling_mean_var(Temp_1)
from statsmodels.tsa.seasonal import STL
STL = STL(Temp_1,period=365)
res = STL.fit()
T = res.trend
S = res.seasonal
R = res.resid
fig = res.plot()
plt.show()
SOT = max(0,(1-((np.var(R))/(np.var(R+T)))))
SOT
SOS = max(0,(1-((np.var(T))/(np.var(R+S)))))
SOS

# from toolbox import cal_autocorr
# lags = 90
# Temp_1_arr = np.array(Temp_1)
# cal_autocorr(Temp_1_arr,lags,'Temperature (degC)')
# plt.show()

# from toolbox import ADF_Cal
# ADF_Cal(Temp_1_arr)
#
# from toolbox import kpss_test
# kpss_test(Temp_1_arr)
#
# Temp_1 = pd.Series(df_2['T(degC)'].values,index = df_2.index,
#                  name = 'Temp (degC)')
# from toolbox import Cal_rolling_mean_var
# Cal_rolling_mean_var(Temp_1)
# from statsmodels.tsa.seasonal import STL
# STL = STL(Temp_1,period=365)
# res = STL.fit()
# T = res.trend
# S = res.seasonal
# R = res.resid
# fig = res.plot()
# plt.show()
# SOT = max(0,(1-((np.var(R))/(np.var(R+T)))))
# SOT
# SOS = max(0,(1-((np.var(T))/(np.var(R+S)))))
# SOS
# from statsmodels.tsa.seasonal import STL
# Temp = pd.Series(df['Tlog(degC)'].values,index = df_new.index,
#                  name = 'temp')
# STL = STL(Temp,period=144)
# res = STL.fit()
# T = res.trend
# S = res.seasonal
# R = res.resid
# fig = res.plot()
# plt.show()
# SOT = max(0,(1-((np.var(R))/(np.var(R+T)))))
# SOT


