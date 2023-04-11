import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox import cal_autocorr, ADF_Cal, kpss_test, Cal_rolling_mean_var
# import os
# os.getcwd()

# Retrieving feature set and target set created from previous file.

df_target = pd.read_csv('Dataset/target_series.csv',index_col='DateTime')
df_features = pd.read_csv('Dataset/feature_series.csv',index_col='DateTime')

lags = 50
target = np.array(y_train)
cal_autocorr(target,lags,'Temperature (degC)')
plt.show()

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


