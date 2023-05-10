import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox import cal_autocorr, ADF_Cal, kpss_test, Cal_rolling_mean_var, ACF_PACF_Plot, diff, Cal_GPAC,lm_param_estimate,autocorrelation,drift_forecast_test
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import scipy

X_train = pd.read_csv('Dataset/X_train.csv',index_col='DateTime')
X_test = pd.read_csv('Dataset/X_test.csv',index_col='DateTime')
y_train = pd.read_csv('Dataset/y_train.csv',index_col='DateTime')
y_test = pd.read_csv('Dataset/y_test.csv',index_col='DateTime')
df_target = pd.read_csv('Dataset/target_series.csv',index_col='DateTime')
df_features = pd.read_csv('Dataset/feature_series.csv',index_col='DateTime')

# Holt-Winter Method
holt1 = ets.ExponentialSmoothing(y_train['T(degC)'],trend='add',damped_trend=True,seasonal='add',freq='D',seasonal_periods=365).fit()
holt2 = ets.ExponentialSmoothing(y_train['T(degC)'],trend='add',seasonal='add',freq='D',seasonal_periods=365).fit()
holt1f = holt1.forecast(steps=len(y_test))
holt2f = holt2.forecast(steps=len(y_test))
holt1f = pd.DataFrame(holt1f,columns=['Holt-Winter with damping']).set_index(y_test.index)
holt2f = pd.DataFrame(holt2f,columns=['Holt-Winter without damping']).set_index(y_test.index)

RMSE_hw1 = np.sqrt(np.square(np.subtract(y_test.values,np.ndarray.flatten(holt1f.values))).mean())
RMSE_hw2 = np.sqrt(np.square(np.subtract(y_test.values,np.ndarray.flatten(holt2f.values))).mean())
print("Root mean square error for Holt-Winter with damping is ", RMSE_hw1,"\nAIC value of this model is",holt1.aic,"\nBIC value of this model is",holt2.bic)
print("Root mean square error for Holt-Winter without damping is ", RMSE_hw2,"\nAIC value of this model is",holt2.aic,"\nBIC value of this model is",holt2.bic)

fig, ax = plt.subplots(figsize=(16,8))
y_test['T(degC)'].plot(ax=ax,label='Test Data')
holt1f['Holt-Winter with damping'].plot(ax=ax,label="with damping (RMSE={:0.2f}, AIC={:0.2f})".format(RMSE_hw1, holt1.aic))
holt2f['Holt-Winter without damping'].plot(ax=ax,label="w/o damping (RMSE={:0.2f}, AIC={:0.2f})".format(RMSE_hw2, holt2.aic))
plt.legend(loc='lower right')
plt.title(f'Holt-Winters Seasonal Smoothing')
plt.xlabel('Time')
plt.ylabel('Temperature (degC)')
plt.grid()
plt.show()

MAE_hw1 = np.abs(np.subtract(y_test.values,np.ndarray.flatten(holt1f.values))).mean()
MAE_hw2 = np.abs(np.subtract(y_test.values,np.ndarray.flatten(holt2f.values))).mean()
MAPE_hw1 = np.abs(np.subtract(y_test.values,np.ndarray.flatten(holt1f.values))/y_test.values).mean()
MAPE_hw2 = np.abs(np.subtract(y_test.values,np.ndarray.flatten(holt2f.values))/y_test.values).mean()
print(f"With damping the Mean Absolute Error is: {MAE_hw1} and MAPE is: {MAPE_hw1}\nWithout damping the Mean Absolute Error: {MAE_hw2} and MAPE is: {MAPE_hw2}")

# The results of Holt-Winter are only partially satisfactory. The forecasted values are more or less following the actual temperatures but not completely able to cover the correct values.
# Also it remains uncertain as to whether damping positively affects the model or not. As the MSE value seems to go up but the AIC value goes down upon introducing damping.

# I'll evaluate the residuals of the fitted models for further clarity on model bias.
cal_autocorr(holt1.resid,50,'Holt-Winter model (damped) residuals')
plt.show()
cal_autocorr(holt2.resid,50,'Holt-Winter model (w/o damp) residuals')
plt.show()
# Performing the ljung -box test to mathematically confirm non-whiteness of the model.
test_results = sm.stats.diagnostic.acorr_ljungbox(holt1.resid, lags=[365])
print(test_results)
# P-value of 0 indicates that we can reject the null hypothesis that the residual is white.

# We can clearly see that the ACF plot does not resemble an impulse, indicating that all the correlations in the train data have not been captured by these Holt-Winter models. Damping does not enhance or degrade the performance of the fitted models as it is clear with the ACF plot for residuals.

# Feature Selection/Elimination based on Multi-Collinearity
# From the previous analysis of correlation matrix, we did observe a high degree of multi-collinearity between the features of the dataset. In this section i'll try eliminate all such features that have multi collinearity using SVD, condition number and VIF methods.

# Before performing the multi-collinearity tests, it will be a good idea to standardise my dataset as the range for features differ drastically.
scaler = StandardScaler()
scaled_xtrain = scaler.fit_transform(X_train)
scaled_xtest = scaler.transform(X_test)
scaled_xtrain_df = pd.DataFrame(scaled_xtrain,columns=X_train.columns)

# Working only with train set to avoid leakage problems
H = np.dot(scaled_xtrain.T,scaled_xtrain)
s, d, v = np.linalg.svd(H)
print("SingularValues = ", d)

# From the SVD analysis, we see that a few singular values are very close to 0 indicating the presence of multi-collinearity in the feature space. The features causing these values need to be removed from the feature set.

# Q3b
Cond = np.linalg.cond(scaled_xtrain_df)
print("The Condition Number of the feature space is: ",Cond)
# From this we can see that the condition number is way above 1000 indicating a high degree of multi-collinearity in the dataset. This needs to be removed.

# Next I will perform a VIF test to identify the features that result in high multi-collinearity.
pd.options.display.float_format = '{:.4f}'.format
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain_df, i) for i in range(scaled_xtrain_df.shape[1])]
vif["features"] = X_train.columns
print(vif)

# With this, we can see the presence of extremely large VIF values in few variables. I'll proceed analysis by dropping the highest VIF value and then running the VIF test again to see the new VIF values. First identifying 'VPmax(mbar)' as the column to drop.

scaled_xtrain_df.drop(columns='VPmax(mbar)',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain_df, i) for i in range(scaled_xtrain_df.shape[1])]
vif["features"] = scaled_xtrain_df.columns
print(vif)

# Next identified variable is 'H2OC(mmol/mol)'
scaled_xtrain_df.drop(columns='H2OC(mmol/mol)',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain_df, i) for i in range(scaled_xtrain_df.shape[1])]
vif["features"] = scaled_xtrain_df.columns
print(vif)

# Next identified variable is 'VPact(mbar)'
scaled_xtrain_df.drop(columns='VPact(mbar)',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain_df, i) for i in range(scaled_xtrain_df.shape[1])]
vif["features"] = scaled_xtrain_df.columns
print(vif)

# Next identified variable is 'max.wv(m/s)'
scaled_xtrain_df.drop(columns='max.wv(m/s)',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain_df, i) for i in range(scaled_xtrain_df.shape[1])]
vif["features"] = scaled_xtrain_df.columns
print(vif)

# Next identified variable is 'rho(g/m**3)'
scaled_xtrain_df.drop(columns='rho(g/m**3)',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain_df, i) for i in range(scaled_xtrain_df.shape[1])]
vif["features"] = scaled_xtrain_df.columns
print(vif)

# Next identified variable is 'PAR(�mol/m�/s)'
scaled_xtrain_df.drop(columns='PAR(�mol/m�/s)',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain_df, i) for i in range(scaled_xtrain_df.shape[1])]
vif["features"] = scaled_xtrain_df.columns
print(vif)

# Next identified variable is 'SWDR(W/m�)'
scaled_xtrain_df.drop(columns='SWDR(W/m�)',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain_df, i) for i in range(scaled_xtrain_df.shape[1])]
vif["features"] = scaled_xtrain_df.columns
print(vif)

# Next identified variable is 'sh(g/kg)'
scaled_xtrain_df.drop(columns='sh(g/kg)',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain_df, i) for i in range(scaled_xtrain_df.shape[1])]
vif["features"] = scaled_xtrain_df.columns
print(vif)

# After dropping this variable, I see that the VIF values all my features are now less than 10. However, to make my model more robust, I will choose a strict cut-off keeping VIF values less than 5 to avoid issues with multicollinearity in my model.
# Therefore, the next identified variable is 'VPdef(mbar)'
scaled_xtrain_df.drop(columns='VPdef(mbar)',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain_df, i) for i in range(scaled_xtrain_df.shape[1])]
vif["features"] = scaled_xtrain_df.columns
print(vif)

# Now all my features have VIF value of less than 5 which indicates I've dropped all the columns that cause multi-collinearity. Now I can proceed with other analysis.

# Multiple linear regression analysis.
#--------------------------DECIDED NOT TO DO THIS--------------------------------------------------
# Before fitting my new scaled train features into a multiple linear regression model, I will perform a slight data augmentation to treat for any outlier values that may affect my model predictions drastically. Since my data is scaled to 0 mean and unit variance, any value less tha  -5 and beyond 5 will be treated as an outlier that may skew my linear regression. To deal with such values, I will clip these values to -5 and 5 respectively.
# First inspecting whether such values exist in my dataset
# print(np.min(scaled_xtrain_df))
# print(np.max(scaled_xtrain_df))
# # 3 columns seems to have high outlier values
#
# # Performing data augmentation
# # scaled_xtrain_df = np.clip(scaled_xtrain_df, -5, 5)
# # Checking to see if augmentation was successful
# print(np.min(scaled_xtrain_df))
# print(np.max(scaled_xtrain_df))
# # It worked.
# The training data has been augmented.
#---------------------------------NOT RUNNING ABOVE--------------------------------------

# Before proceeding, I will also drop the same columns in X_test that i dropped from my X_train so that i can make predictions later on
scaled_xtest_df = pd.DataFrame(scaled_xtest,columns=X_train.columns)
scaled_xtest_df.drop(columns=['VPmax(mbar)','H2OC(mmol/mol)','VPact(mbar)','max.wv(m/s)','rho(g/m**3)','PAR(�mol/m�/s)','SWDR(W/m�)','sh(g/kg)','VPdef(mbar)'],axis=1,inplace=True)

# I am deciding not to standard scale my y_train and y_test as a matter of choice. Expectation is that the estimated parameters will be sufficiently large to reproduce the required target values.

# Now fitting an OLS model to my data
scaled_xtrain_df = sm.add_constant(scaled_xtrain_df) # To add constant intercept
y_train_df = y_train.reset_index()
y_train_df = y_train_df.drop(columns='DateTime',axis=1)
model = sm.OLS(y_train_df,scaled_xtrain_df).fit()
print(model.params)
print(model.summary())

# I get a very good adjusted R-squared value of 99.7%. Using the features in my training data, I'm able to explain 99.7% of the variability in my target.
# However, looking closely at all the parameter values, I do see that 3 parameters have a high P-value > 0.05 and the confidence interval indicates that these parameters can have a coefficient of 0 as well. Therefore, I'll remove the first highest P-value and rebuild my model to see the effect on AIC, BIC and Adjusted-R Squared value.
# The first identified variable is 'CO2(ppm)'
scaled_xtrain_df1 = scaled_xtrain_df.drop(columns='CO2(ppm)',axis=1)
model1 = sm.OLS(y_train_df,scaled_xtrain_df1).fit()
print(model1.params)
print(model1.summary())
# The AIC has gone down along with BIC indicating a better performance, there appears to be no change in adjusted r-squared value. I'll retain dropping this feature.
# Another feature that still has an insiginifcant p-value and confidence interval containing a 0 is 'wv(m/s)'. I'll proceed with dropping this feature as well.
scaled_xtrain_df2 = scaled_xtrain_df1.drop(columns='wv(m/s)',axis=1)
model2 = sm.OLS(y_train_df,scaled_xtrain_df2).fit()
print(model2.params)
print(model2.summary())
# The AIC has gone down along with BIC indicating a better performance, there appears to be no change in adjusted r-squared value. I'll retain dropping this feature.
# Another feature that still has an insiginifcant p-value and confidence interval containing a 0 is 'rain(mm)'. I'll proceed with dropping this feature as well.
scaled_xtrain_df3 = scaled_xtrain_df2.drop(columns='rain(mm)',axis=1)
model3 = sm.OLS(y_train_df,scaled_xtrain_df3).fit()
print(model3.params)
print(model3.summary())
# After removing this feature, the model performance has improved further. The AIC has decreased and BIC has dropped as well. Adjusted R-squared value remains the same indicating this model performs well! All the parameters are also withinin the significant range. Additionnally, the F-test statistic indicates that we can reject the null hypothesis that this model and an intercept only model perform the same indicating this model with the given features performs much better. I'll consider this model to be my final model.

# Removing the same features from my test set to perform forecasts.
scaled_xtest_df = sm.add_constant(scaled_xtest_df)
# Dropping the columns based on above analysis in test set as well.
scaled_xtest_df = scaled_xtest_df.drop(columns=['CO2(ppm)','rain(mm)','wv(m/s)'],axis=1)
predictions = model3.predict(scaled_xtest_df)

predictions.index = y_test.index
fig, ax = plt.subplots(figsize = (16,8))
predictions.plot(ax=ax,label="Predicted T(degC)")
y_test['T(degC)'].plot(ax=ax,label="Actual T(degC)",linestyle='--')
plt.legend(loc='lower right')
plt.grid()
plt.title('OLS Model - Forecast vs Actual T(degC)')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()

# Looking at the model performance
RMSE_ols = np.sqrt(np.square(np.subtract(y_test['T(degC)'],predictions)).mean())
MAE_ols = np.abs(np.subtract(y_test['T(degC)'],predictions)).mean()
MAPE_ols = (np.abs(np.subtract(y_test['T(degC)'],predictions))/np.abs(y_test['T(degC)'])).mean()
print(f"The RMSE of this model is: {RMSE_ols}\nThe MAE of this model is: {MAE_ols}\nThe MAPE os this model is: {MAPE_ols}")
# According to this metric, the model is very performing well on the unseen test set.

# Testing out the whiteness of residuals, first visually
residuals = model3.resid
cal_autocorr(residuals,62,'Residual ACF for OLS Model')
plt.show()
# Research: Looking at this ACF plot, we can say that the correlations within the target series have not been fully captured by this OLS model since the residuals are not white. Possible reason behind this is potentially the dataset i'm working with is non-linear and yet i'm trying to fit into a linear model which may cause the residuals to not be white. Alternatively, it could also be due the fact that my data has high autocorrelation which is not being accounted for by this linear model.

# Find out the Q-value and performing box-pierce test
Ry = autocorrelation(residuals,60)
Q = len(residuals) * np.sum(np.square(Ry[60+1:]))
DOF = 60 - 7
alfa = 0.01
chi_critical = scipy.stats.chi2.ppf(1 - alfa, DOF)
print(f"Q is {Q} and chi critical is {chi_critical}")
if Q < chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")

# Performing the Chi-square test using ljung-box test to mathematically find out the whiteness of the residuals.
test_results = sm.stats.diagnostic.acorr_ljungbox(residuals, lags=[1])
print(test_results)
# Now checking for lag for 365 since the data has seasonality of 365
test_results1 = sm.stats.diagnostic.acorr_ljungbox(residuals, lags=[365])
print(test_results1)
# Both these test results state that we can reject the null hypothesis that the residuals come from an independently and identical distibution and hence reinforces our visual observation that the residuals are not white.

# The mean and variance of the residuals
print(f"The mean of the residuals is: {np.mean(residuals)}")
print(f"The variance of the residuals is: {np.var(residuals)}")
# the residuals do seem to have a mean of 0 and constant variance of 0.155 which is desirable, however, the residuals are not white indicating the presence of uncaptured correlations in the target series using a simple OLS model.


# Building the base models now - Average, Drift, Naive, SES using original y_train and y_test
# Average method
y_pred1 = np.mean(y_train.values)
y_test_avg = pd.DataFrame(np.array([y_pred1]*len(y_test)),index = y_test.index,columns = ["Average Forecast"])
y_pred_avg = np.array([y_pred1]*len(y_test))
error_avg = y_test.values.ravel() - y_pred_avg
RMSE_avg = np.sqrt(np.square(error_avg).mean())
MAE_avg = np.abs(error_avg).mean()
MAPE_avg = np.abs(y_test.values.ravel() - y_pred_avg/y_test.values).mean()
print(f"The RMSE of Average model is: {RMSE_avg}\nThe MAE of this model is: {MAE_avg}\nThe MAPE of this model is: {MAPE_avg}")
# Plotting the Average forecast vs the test set.
fig, ax = plt.subplots(figsize = (16,8))
y_test_avg["Average Forecast"].plot(ax=ax,label="Average Forecast")
y_test['T(degC)'].plot(ax=ax,label="Actual T(degC)")
plt.legend(loc='lower right')
plt.grid()
plt.title('Average Forecast Model - Forecast vs Actual T(degC)')
plt.xlabel('Date')
plt.ylabel('Temperature (degC)')
plt.show()

# Naive method
y_pred2 = y_train.values[-1]
y_pred_naive = np.array([y_pred2]*len(y_test))
y_test_naive = pd.DataFrame(y_pred_naive,index = y_test.index,columns = ["Naive Forecast"])
error_naive = y_test.values.ravel() - y_pred_naive.ravel()
RMSE_naive = np.sqrt(np.square(error_naive).mean())
MAE_naive = np.abs(error_naive).mean()
MAPE_naive = np.abs(y_test.values.ravel() - y_pred_naive.ravel()/y_test.values).mean()
print(f"The RMSE of Naive model is: {RMSE_naive}\nThe MAE of this model is: {MAE_naive}\nThe MAPE of this model is: {MAPE_naive}")
# Plotting the Naive forecast vs the test set.
fig, ax = plt.subplots(figsize = (16,8))
y_test_naive["Naive Forecast"].plot(ax=ax,label="Naive Forecast")
y_test['T(degC)'].plot(ax=ax,label="Actual T(degC)")
plt.legend(loc='lower right')
plt.grid()
plt.title('Naive Forecast Model - Forecast vs Actual T(degC)')
plt.xlabel('Date')
plt.ylabel('Temperature (degC)')
plt.show()

# Drift method
y_pred3 = drift_forecast_test(y_train.values.ravel(),len(y_test))
y_test_drift = pd.DataFrame(np.array(y_pred3),index = y_test.index, columns=["Drift Forecast"])
y_pred_drift = np.array([y_pred3]).reshape(-1,1).ravel()
error_drift = y_test.values.ravel() - y_pred_drift.ravel()
RMSE_drift = np.sqrt(np.square(error_drift).mean())
MAE_drift = np.abs(error_drift).mean()
MAPE_drift = np.abs(y_test.values.ravel() - y_pred_drift.ravel()/y_test.values).mean()
print(f"The RMSE of Drift model is: {RMSE_drift}\nThe MAE of this model is: {MAE_drift}\nThe MAPE of this model is: {MAPE_drift}")
fig, ax = plt.subplots(figsize = (16,8))
y_test_drift["Drift Forecast"].plot(ax=ax,label="Drift Forecast")
y_test['T(degC)'].plot(ax=ax,label="Actual T(degC)")
plt.legend(loc='lower right')
plt.grid()
plt.title('Drift Forecast Model - Forecast vs Actual T(degC)')
plt.xlabel('Date')
plt.ylabel('Temperature (degC)')
plt.show()


# Simple and Exponential Smoothing
SES_model = ets.ExponentialSmoothing(y_train,trend=None,damped=False,seasonal=None).fit()
y_pred_ses = SES_model.forecast(steps=len(y_test))
y_test_ses = pd.DataFrame(y_pred_ses,index = y_test.index, columns=["SES Forecast"])
error_ses = y_test.values.ravel() - y_pred_ses.values.ravel()
RMSE_ses = np.sqrt(np.square(error_ses).mean())
MAE_ses = np.abs(error_ses).mean()
MAPE_ses = np.abs(y_test.values.ravel() - y_pred_ses.values.ravel()/y_test.values).mean()
print(f"The RMSE of SES model is: {RMSE_ses}\nThe MAE of this model is: {MAE_ses}\nThe MAPE of this model is: {MAPE_ses}")
fig, ax = plt.subplots(figsize = (16,8))
y_test_ses["SES Forecast"].plot(ax=ax,label="SES Forecast")
y_test['T(degC)'].plot(ax=ax,label="Actual T(degC)")
plt.legend(loc='lower right')
plt.grid()
plt.title('SES Forecast Model - Forecast vs Actual T(degC)')
plt.xlabel('Date')
plt.ylabel('Temperature (degC)')
plt.show()

# Starting the ARIMA/SARIMA model development
# Although initially i had considered my data to be stationary and determined that no differencing needs to be applied. Looking at the ACF/PACF plot of the data again, there appears to be high seasonality in the data. This is also reinforced by the Strength of seasonality using STL decomposition previously. A consistent pattern in the ACF/PACF suggests seasonality. As a consequence of this observation, instead of the ARIMA model, I will investigate the SARIMA model. There is no need for non-seasonal differencing since the data is already stationary. However, before proceeding now (based on several research papers referenced: https://article.sciencepublishinggroup.com/pdf/10.11648.j.ijema.20210906.17.pdf, I will perform the seasonal differencing to eliminate this high seasonality. This is because high correlations in the time series can make it difficult to build accurate ARIMA models because the long-term dependencies in the time series are difficult to capture using only a few lags.

# Hence, i need to perform order one seasonal differencing (seasonality index as 365 as decided before) here before proceeding and then do my train-test spit. I will have to trade-off my loss of datapoints before proceeding.

y_train_diff = diff(y_train,'T(degC)',365).copy()
# Maintaining original df by removing the new column
y_train.drop(columns=['T(degC)_365_Diff'],axis=1,inplace=True)
# Dropping the previous target column from new dataframe along with the null rows introduced after differencing
y_train_diff.drop(columns='T(degC)',axis=1,inplace=True)
y_train_diff = y_train_diff.dropna()
# From prior analysis, i already know that this new differenced time series is also stationary, infact it is more stationary as indicated by ADF test. Now i'll proceed with the train-test split and further analysis.

# target_train, target_test = train_test_split(df_target_diff, shuffle=False, test_size=0.2, random_state=6313)
ry = sm.tsa.stattools.acf(y_train_diff['T(degC)_365_Diff'].values, nlags=100)
ryy = ry[::-1]
Ry = np.concatenate((ryy, ry[1:]))
Cal_GPAC(Ry,30,30)

# Keeping the ACF/PACF plot handy
ACF_PACF_Plot(y_train_diff['T(degC)_365_Diff'].values,100)
# Judging by just the ACF/PACF, i can guess that my AR only process order is 3 since the PACF cuts off after 3 lags. But i'll consider GPAC for time being

# A preliminary order i'm deciding to select is ARMA(1,0); Another possible order i'll select is (3,0)
lm_param_estimate(y_train_diff,1,0)
# Algorithm converges very quickly
lm_param_estimate(y_train_diff,3,0)
# Algorithm converges very quickly

# ARIMA(1,0,0)
arima_model1 = sm.tsa.arima.ARIMA(y_train_diff,order=(1, 0, 0),trend='n',freq='D').fit()
print(arima_model1.summary())
model_hat1 = arima_model1.predict(start=0, end=len(y_train_diff) - 1)
e1 = y_train_diff.reset_index()['T(degC)_365_Diff'] - model_hat1.reset_index()['predicted_mean']
Re1 = autocorrelation(np.array(e1), 100)
ACF_PACF_Plot(e1,100)
cal_autocorr(e1,100,"ACF of Residuals with ARIMA(1,0,0)xSARIMA(0,1,0,365)")
plt.show()
Q = len(e1) * np.sum(np.square(Re1[100+1:]))
DOF = 100 - 1 - 0
alfa = 0.01
chi_critical = scipy.stats.chi2.ppf(1 - alfa, DOF)
print(f"Q is {Q} and chi critical is {chi_critical}")
if Q < chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")
y_train_diff.index = pd.to_datetime(y_train_diff.index)
model_hat1.index = y_train_diff.index
fig, ax = plt.subplots(figsize=(16,8))
y_train_diff['T(degC)_365_Diff'].plot(ax=ax,label="True data")
model_hat1.plot(ax=ax,label="Fitted data")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.legend()
plt.title(" Train vs One-Step Prediction - ARIMA(1,0,0)xSARIMA(0,1,0,365)")
plt.tight_layout()
plt.show()
# Judging by the residual analysis, we don't get a white residual hence new order must be considered.

# ARIMA(3,0,0)
arima_model2 = sm.tsa.arima.ARIMA(y_train_diff,order=(3, 0, 0),trend='n',freq='D').fit()
print(arima_model2.summary())
model_hat2 = arima_model2.predict(start=0, end=len(y_train_diff) - 1)
e2 = y_train_diff.reset_index()['T(degC)_365_Diff'] - model_hat2.reset_index()['predicted_mean']
test_results_sarima = sm.stats.diagnostic.acorr_ljungbox(e2, lags=[25])
Re2 = autocorrelation(np.array(e2),100)
ACF_PACF_Plot(e2,100)
cal_autocorr(e2,100,"ACF of Residuals with ARIMA(3,0,0)xSARIMA(0,1,0,365)")
plt.show()
Q = len(e2) * np.sum(np.square(Re2[100+1:]))
DOF = 100 - 3 - 0
alfa = 0.01
chi_critical = scipy.stats.chi2.ppf(1 - alfa, DOF)
print(f"Q is {Q} and chi critical is {chi_critical}")
if Q < chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")
model_hat2.index = y_train_diff.index
fig, ax = plt.subplots(figsize=(16,8))
y_train_diff['T(degC)_365_Diff'].plot(ax=ax,label="True data")
model_hat2.plot(ax=ax,label="Fitted data")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.legend()
plt.title(" Train vs One-Step Prediction - ARIMA(3,0,0)xSARIMA(0,1,0,365)")
plt.tight_layout()
plt.show()

# With the residual anlaysis for this model, i can see that the residuals have become white. Hence the final model that i will select is ARIMA(3,0,0)xSARIMA(0,1,0,365). Since the model was built on top of seasonally differenced data, there is a component of SARIMA as well.
lm_param_estimate(y_train_diff,3,0)
# The estimated parameter match between the SARIMAX model and custom-developed lm_algorithm_estimation
residual_variance = np.var(e2)
forecast_values = arima_model2.forecast(steps=len(y_train_diff))
# The forecasted values that i've gotten from this model are transformed. I'll have to back-transform these values to compare with my test set.
y_test_orig = []
Num_observations = len(y_train_diff)
s = 365
for i in range(len(y_test)):
    if i < s:
        y_test_orig.append(forecast_values[i] + y_train.iloc[- s + i])
    else:
        y_test_orig.append(forecast_values[i] + y_test_orig[i - s])

y_test_orig = np.array(y_test_orig).ravel()
e_forecast = y_test.reset_index()['T(degC)'].ravel() - y_test_orig
forecast_variance = np.var(e_forecast)
y_testdf = pd.DataFrame(y_test_orig.reshape(-1,1),columns=["Forecast"],index=y_test.index)

fig, ax = plt.subplots(figsize=(16,8))
y_test['T(degC)'].plot(ax=ax,label='Test Data')
y_testdf.plot(ax=ax,label="Forecast")
plt.legend(loc='lower right')
plt.title(f'ARIMA(3,0,0)xSARIMA(0,1,0,365) Model Results')
plt.xlabel('Time')
plt.ylabel('Temperature (degC)')
plt.grid()
plt.show()

Generalization = forecast_variance/residual_variance
print(Generalization)

# Plotting the distribution of the residual error to assess bias in the model
e2.hist()
plt.title("Residual Histogram for ARIMA(3,0,0)xSARIMA(0,1,0,365) Model")
plt.ylabel("Frequency")
plt.xlabel("Error Value")
plt.show()

# Calculating the Model performance through forecast RMSE
RMSE_sarima = np.sqrt(np.square(e_forecast).mean())
MAE_sarima = np.abs(e_forecast).mean()
MAPE_sarima = np.abs(e_forecast/y_test.values.ravel()).mean()
print(f"The RMSE of ARIMA(3,0,0)xSARIMA(0,1,0,365) is: {RMSE_sarima}\nThe MAE of ARIMA(3,0,0)xSARIMA(0,1,0,365) is: {MAE_sarima}\nThe MAPE of ARIMA(3,0,0)xSARIMA(0,1,0,365) is: {MAPE_sarima}")

# Developing the custom forecast function for SARIMA model
# Retrieving the model parameters from the model summary
lm_param_estimate(y_train_diff,3,0) #[-1.011398710003359, 0.2983713542290045, -0.09062909533692821]
arima_model2.summary() # ar.L1 1.0111     ar.L2 -0.2981   ar.L3 0.0905
# The coefficients obtained from my custom model and package are similar. I'll utilize the package coefficients as my final model paramter.
def custom_forecast_function(data,Step):
    y_hat = []
    for i in range(1, Step+1):
        if i == 1:
            y_hat.append(1.0111 * data.values.ravel().tolist()[-1] - 0.2981 * data.values.ravel().tolist()[-2] + 0.0905 * data.values.ravel().tolist()[-3])
        elif i == 2:
            y_hat.append(1.0111 * y_hat[0] - 0.2981 * data.values.ravel().tolist()[-1] + 0.0905 * data.values.ravel().tolist()[-2])
        elif i == 3:
            y_hat.append(1.0111 * y_hat[1] - 0.2981 * y_hat[0] + 0.0905 * data.values.ravel().tolist()[-1])
        else:
            y_hat.append(1.0111 * y_hat[i-2] - 0.2981 * y_hat[i-3] + 0.0905 * y_hat[i-4])
    y_hat = np.array(y_hat)
    return y_hat

# Testing custom function to forecast values
custom_forecast_values = custom_forecast_function(y_train_diff,len(y_test))
# These forecasts are made on the transformed data. I'll have to back transform this to perform a comparison with original test data. Utilizing the reverse transformation used above.

y_test_orig_cust = []
s = 365
for i in range(len(y_test)):
    if i < s:
        y_test_orig_cust.append(custom_forecast_values[i] + y_train.iloc[- s + i])
    else:
        y_test_orig_cust.append(custom_forecast_values[i] + y_test_orig_cust[i - s])

# Plotting this against test set
y_test_orig_cust = np.array(y_test_orig_cust).ravel()
e_forecast = y_test.reset_index()['T(degC)'].ravel() - y_test_orig_cust
y_testdf_cust = pd.DataFrame(y_test_orig_cust.reshape(-1,1),columns=["Custom Forecast"],index=y_test.index)

fig, ax = plt.subplots(figsize=(16,8))
y_test['T(degC)'].plot(ax=ax,label='Test Data')
y_testdf_cust.plot(ax=ax,label="Forecast")
plt.legend(loc='lower right')
plt.title(f'ARIMA(3,0,0)xSARIMA(0,1,0,365) Model Results')
plt.xlabel('Time')
plt.ylabel('Temperature (degC)')
plt.grid()
plt.show()

# Base model Comparisons with SARIMA model

MAPE_avg = np.abs(y_test.values.ravel() - y_pred_avg/y_test.values).mean()
print(f"The RMSE of Average model is: {RMSE_avg}\nThe MAE of this model is: {MAE_avg}\nThe MAPE of this model is: {MAPE_avg}")
# Plotting the Average forecast vs the test set.
fig, ax = plt.subplots(3,2,figsize = (16,8))
y_test_avg["Average Forecast"].plot(ax=ax[0,0],label="Average Forecast")
y_test['T(degC)'].plot(ax=ax[0,0],label="Actual T(degC)")
ax[0,0].legend(loc='lower right')
ax[0,0].grid()
ax[0,0].set_title('Average Forecast Model - Forecast vs Actual T(degC)')
ax[0,0].set_xlabel('Date')
ax[0,0].set_ylabel('Temperature (degC)')
y_test_naive["Naive Forecast"].plot(ax=ax[0,1],label="Naive Forecast")
y_test['T(degC)'].plot(ax=ax[0,1],label="Actual T(degC)")
ax[0,1].legend(loc='lower right')
ax[0,1].grid()
ax[0,1].set_title('Naive Forecast Model - Forecast vs Actual T(degC)')
ax[0,1].set_xlabel('Date')
ax[0,1].set_ylabel('Temperature (degC)')
y_test_drift["Drift Forecast"].plot(ax=ax[1,0],label="Drift Forecast")
y_test['T(degC)'].plot(ax=ax[1,0],label="Actual T(degC)")
ax[1,0].legend(loc='lower right')
ax[1,0].grid()
ax[1,0].set_title('Drift Forecast Model - Forecast vs Actual T(degC)')
ax[1,0].set_xlabel('Date')
ax[1,0].set_ylabel('Temperature (degC)')
y_test_ses["SES Forecast"].plot(ax=ax[1,1],label="SES Forecast")
y_test['T(degC)'].plot(ax=ax[1,1],label="Actual T(degC)")
ax[1,1].legend(loc='lower right')
ax[1,1].grid()
ax[1,1].set_title('SES Forecast Model - Forecast vs Actual T(degC)')
ax[1,1].set_xlabel('Date')
ax[1,1].set_ylabel('Temperature (degC)')
y_test['T(degC)'].plot(ax=ax[2,1],label='Actual T(degC)')
y_testdf['Forecast'].plot(ax=ax[2,1],label="SARIMA Forecast")
ax[2,1].legend(loc='lower right')
ax[2,1].set_title(f'ARIMA(3,0,0)xSARIMA(0,1,0,365) Model Results')
ax[2,1].set_xlabel('Time')
ax[2,1].set_ylabel('Temperature (degC)')
ax[2,1].grid()
fig.suptitle("Base Model Comparison with SARIMA Model")
plt.tight_layout()
plt.show()

# Plotting the metrics in a table.
from tabulate import tabulate
basemodel_table = [['Model Type', 'Root-Mean-Square-Error','Mean-Absolute-Error','Mean-Absolute-Percent-Error'],
         ["Average Method Forecast",round(RMSE_avg,3),round(MAE_avg,3),round(MAPE_avg,3)],["Naive Method Forecast",round(RMSE_naive,3),round(MAE_naive,3),round(MAPE_naive,3)],["Drift Method Forecast",round(RMSE_drift,3),round(MAE_drift,3),round(MAPE_drift,3)],["SES Method Forecast",round(RMSE_ses,3),round(MAE_ses,3),round(MAPE_ses,3)],["SARIMA Model Forecast",round(RMSE_sarima,3),round(MAE_sarima,3),round(MAPE_sarima,3)]]
print(tabulate(basemodel_table,headers='firstrow', tablefmt = 'fancy_grid'))

# We can see the SARIMA Model significantly outperforms the base models that we have.
