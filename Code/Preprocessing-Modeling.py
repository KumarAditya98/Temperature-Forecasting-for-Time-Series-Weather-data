import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox import cal_autocorr, ADF_Cal, kpss_test, Cal_rolling_mean_var, ACF_PACF_Plot, diff
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets


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
