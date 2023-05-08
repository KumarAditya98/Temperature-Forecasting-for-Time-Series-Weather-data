import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox import cal_autocorr, ADF_Cal, kpss_test, Cal_rolling_mean_var, ACF_PACF_Plot, diff
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

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

MSE1 = np.square(np.subtract(y_test.values,np.ndarray.flatten(holt1f.values))).mean()
MSE2 = np.square(np.subtract(y_test.values,np.ndarray.flatten(holt2f.values))).mean()
print("Mean square error for Holt-Winter with damping is ", MSE1,"\nAIC value of this model is",holt1.aic,"\nBIC value of this model is",holt2.bic)
print("Mean square error for Holt-Winter without damping is ", MSE2,"\nAIC value of this model is",holt2.aic,"\nBIC value of this model is",holt2.bic)

fig, ax = plt.subplots(figsize=(16,8))
y_test['T(degC)'].plot(ax=ax,label='Test Data')
holt1f['Holt-Winter with damping'].plot(ax=ax,label="with damping (MSE={:0.2f}, AIC={:0.2f})".format(MSE1, holt1.aic))
holt2f['Holt-Winter without damping'].plot(ax=ax,label="w/o damping (MSE={:0.2f}, AIC={:0.2f})".format(MSE2, holt2.aic))
plt.legend(loc='lower right')
plt.title(f'Holt-Winters Seasonal Smoothing')
plt.xlabel('Time')
plt.ylabel('Temperature (degC)')
plt.grid()
plt.show()

# The results of Holt-Winter are only partially satisfactory. The forecasted values are more or less following the actual temperatures but not completely able to cover the correct values.
# Also it remains uncertain as to whether damping positively affects the model or not. As the MSE value seems to go up but the AIC value goes down upon introducing damping.

# I'll evaluate the residuals of the fitted models for further clarity on model performance.
cal_autocorr(holt1.resid,50,'ACF of Holt-Winter model (damped) residuals')
plt.show()
cal_autocorr(holt2.resid,50,'ACF of Holt-Winter model (w/o damp) residuals')
plt.show()
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
# Before fitting my new scaled train features into a multiple linear regression model, I will perform a slight data augmentation to treat for any outlier values that may affect my model predictions drastically. Since my data is scaled to 0 mean and unit variance, any value less tha  -5 and beyond 5 will be treated as an outlier that may skew my linear regression. To deal with such values, I will clip these values to -5 and 5 respectively.
# First inspecting whether such values exist in my dataset
print(np.min(scaled_xtrain_df))
print(np.max(scaled_xtrain_df))
# 3 columns seems to have high outlier values

# Performing data augmentation
scaled_xtrain_df = np.clip(scaled_xtrain_df, -5, 5)
# Checking to see if augmentation was successful
print(np.min(scaled_xtrain_df))
print(np.max(scaled_xtrain_df))
# It worked.
# The training data has been augmented.

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
# However, looking closely at all the parameter values, I do see that 2 parameters have a high P-value > 0.05 and the confidence interval indicates that these parameters can have a coefficient of 0 as well. Therefore, I'll remove the first highest P-value and rebuild my model to see the effect on AIC, BIC and Adjusted-R Squared value.
# The first identified variable is 'CO2(ppm)'
scaled_xtrain_df1 = scaled_xtrain_df.drop(columns='CO2(ppm)',axis=1)
model1 = sm.OLS(y_train_df,scaled_xtrain_df1).fit()
print(model1.params)
print(model1.summary())
# The AIC has only slightly increased by a unit of 1, however, there appears to be no change in adjusted r-squared value. The BIC value has gone down indicating an improved model performance. I'll retain dropping this feature.
# Another feature that still has an insiginifcant p-value and confidence interval containing a 0 is 'rain(mm)'. I'll proceed with dropping this feature as well.
scaled_xtrain_df2 = scaled_xtrain_df1.drop(columns='rain(mm)',axis=1)
model2 = sm.OLS(y_train_df,scaled_xtrain_df2).fit()
print(model2.params)
print(model2.summary())
# After removing this feature, the model performance has not changed. The AIC has increased by 1unit but BIC has dropped. Adjusted R-squared value remains the same indicating this model performs well! All the parameters are also withinin the significant range. Additionnally, the F-test statistic indicates that we can reject the null hypothesis that this model and an intercept only model perform the same indicating this model with the given features performs much better. I'll consider this model to be my final model.

# Removing the same features from my test set to perform forecasts.
scaled_xtest_df = sm.add_constant(scaled_xtest_df)
# Dropping the columns based on above analysis in test set as well.
scaled_xtest_df = scaled_xtest_df.drop(columns=['CO2(ppm)','rain(mm)'],axis=1)
predictions = model2.predict(scaled_xtest_df)

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

# Testing out my residuals

# Calculating the forecast error
predictions = pd.Series(predictions,name='Forecast')
Error = np.array(y_test['T(degC)']-predictions)
lags = 50
cal_autocorr(Error,lags,'Prediction Error')
plt.show()

MSE = np.square(np.subtract(y_test['T(degC)'],predictions)).mean()
print(MSE)