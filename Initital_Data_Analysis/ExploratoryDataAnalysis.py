import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.getcwd()


X_train = pd.read_csv('Dataset/X_train.csv')
y_train = pd.read_csv('Dataset/y_train.csv')
X_test = pd.read_csv('Dataset/X_test.csv')
X_test = pd.read_csv('Dataset/X_test.csv')


#
# # Plotting the newly aggregated data along with the drift line used for imputation
# fig, ax = plt.subplots(figsize=(16,8))
# df_target['T(degC)'].plot(label="Daily time-series Data")
# Temp['T(degC)'].plot(label="Interpolation Line")
# plt.grid()
# plt.legend()
# plt.title("Plotting the daily aggregated time-series along with the interpolation line used to impute missing values")
# plt.xlabel("Time")
# plt.ylabel("Temperature (degC)")
# plt.show()
#
# from toolbox import cal_autocorr
# lags = 90
# Temp_1_arr = np.array(Temp_1)
# cal_autocorr(Temp_1_arr,lags,'Temperature (degC)')
# plt.show()
#
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
#
#
#
# # # Subsetting only the time stamps and target variable for plotting.
# # df_new = df.iloc[:,0:3:2]
# # print(df_new.head())
# #
# # type(df_new.DateTime)
# # df_new.index = pd.to_datetime(df_new.DateTime)
# # df_new = df_new.drop('DateTime',axis = 1)
# #
# # fig, ax = plt.subplots(figsize=(30,16))
# # df['Tpot(K)'].plot()
# # plt.tight_layout()
# # plt.grid()
# # plt.show()
