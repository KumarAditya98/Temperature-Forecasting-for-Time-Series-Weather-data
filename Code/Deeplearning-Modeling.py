import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
# from keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dense, LSTM, Dropout  # ,CuDNNLSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
from toolbox import autocorrelation, cal_autocorr
import statsmodels.api as sm
from tabulate import tabulate

df_target = pd.read_csv('Dataset/target_series.csv',index_col='DateTime')
df_features = pd.read_csv('Dataset/feature_series.csv',index_col='DateTime')
df = pd.concat((df_features,df_target),axis=1)
data = df.copy()

scaler1 = StandardScaler()
scaled_data = scaler1.fit_transform(data.values)
df_temp = df['T(degC)'].values
dataset = data.values
training_data_len = np.math.ceil(len(df_temp) *.8)
train_data = scaled_data[0:training_data_len,:]

x_train = []
y_train = []
x_test = []
y_test = []
n_past = len(dataset) - training_data_len

for i in range(n_past,len(train_data)):
    x_train.append(train_data[i-365:i,0:train_data.shape[1]-1])
    y_train.append(train_data[i,train_data.shape[1]-1])

x_train, y_train = np.array(x_train), np.array(y_train)
print(f'trainX shape = {x_train.shape}')
print(f'trainY shape = {y_train.shape}')

model = Sequential()
model.add(LSTM(64, return_sequences=True,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(LSTM(50,return_sequences=False))
model.add(Dropout(.2))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
model.summary()
history = model.fit(x_train,y_train, batch_size=16, validation_split = .1, epochs=10, verbose=1)

plt.figure()
plt.plot(history.history['loss'], 'r',label='Training loss')
plt.plot(history.history['val_loss'], 'b',label='Validation loss')
plt.legend()
plt.title("Training vs Validation loss for LSMT Model in Temperature forecasting")
plt.ylabel("Loss Values")
plt.xlabel("Epochs")
plt.show()

test_data = scaled_data[training_data_len-365:,:]
x_test = []
y_test = dataset[training_data_len:,-1]

for i in range(365,len(test_data)):
    x_test.append(test_data[i-365:i,0:18])

x_test = np.array(x_test)
predictions = model.predict(x_test)
forecast_copies = np.repeat(predictions, 19, axis=-1)
predictions = scaler1.inverse_transform(forecast_copies)[:,-1]
#
train = data.iloc[:training_data_len]
test = data.iloc[training_data_len:]
test["Predictions"] = predictions

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
ax.set_title("Temperature prediction using LSTM network",fontsize=18)
ax.set_xlabel("Date",fontsize=18)
ax.set_ylabel("Temperature (degC)", fontsize=18)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['r','b','c'])
test["T(degC)"].plot(ax=ax,color='red',label="Test")
test["Predictions"].plot(ax=ax,color='black',label="Predictions")
ax.legend(["Test Set", "Predictions"], loc = "lower right", fontsize=18)
ax.grid()
plt.show()

e_forecast = test.reset_index()['T(degC)'] - test.reset_index()['Predictions']
RMSE_lstm = np.sqrt(np.square(e_forecast).mean())
MAE_lstm = np.abs(e_forecast).mean()
MAPE_lstm = np.abs(e_forecast/test['T(degC)'].values.ravel()).mean()
print(f"The RMSE of LSTM Model is: {RMSE_lstm}\nThe MAE of LSTM Model is: {MAE_lstm}\nThe MAPE of LASTM Model is: {MAPE_lstm}")

# Residual diagnostic
fitted_values = model.predict(x_train)
fitted_copies = np.repeat(fitted_values, 19, axis=-1)
fit_values = scaler1.inverse_transform(fitted_copies)[:,-1]

y_train_orig_copies = np.repeat(y_train.reshape(-1,1), 19, axis=-1)
y_train_orig = scaler1.inverse_transform(y_train_orig_copies)[:,-1]

e_residuals = y_train_orig - fit_values
np.var(e_residuals)
np.mean(e_residuals)

# Plotting the distribution of the residual error to assess bias in the model
pd.DataFrame(e_residuals).hist()
plt.title("Residual Histogram for LSTM Model")
plt.ylabel("Frequency")
plt.xlabel("Error Value")
plt.show()

# It looks normally distributed. Plotting the ACF of the curve and then performing the ljung-box test to test whiteness.
cal_autocorr(e_residuals,50,"ACF plot for residuals of LSTM Model")
plt.show()
# It's hard to say whether the residuals are white or not visually because of lag = 1 values
test_results = sm.stats.diagnostic.acorr_ljungbox(e_residuals, lags=[365])
print(test_results)
# Based on the ljung box-test we cannot reject the null hypothesis that the residuals are white. However, visually lag 1 seems uncertain.

# FINAL MODEL SELECTION -
# NOTE: This section is used to perform a comparative analysis across all models built till now. It utilizes variables across different files and might not run if performed on the same console.

# Apart from the base models, which are only used for benchmarking, the final models i'll compare the performance on this dataset are:
# 1. Holt-winter
# 2. Multiple linear regression
# 3. SARIMA model
# 4. LSTM Model
# Within these models, i will select the best model that i found to fit my data as i built multiple variations within each model type as well.

# The key criteria in selecting the final model will be the model that captures all the correlation within my target series. This is judged based on the residual diagnostic tests. I'll evaluate the residuals of each model and determine the whiteness in each model. This way, we'll find the best model that represents the underlying the data well.
# This whiteness test has been performed for various models already while building them, I'll aggregate the results in this section.

residuals_ols = model3.resid
fig, ax = plt.subplots(2,2,figsize=(16,8))
cal_autocorr(holt1.resid,60,'Residuals for Holt-Winter model',ax=ax[0,0])
cal_autocorr(residuals_ols,60,'Residuals for Multiple Linear Regression Model',ax=ax[0,1])
cal_autocorr(e2,60,"Residuals for ARIMA(3,0,0)xSARIMA(0,1,0,365)",ax=ax[1,0])
cal_autocorr(e_residuals,60,"Residuals for LSTM Model",ax=ax[1,1])
fig.suptitle("Residual Diagnostics for all Final Models",size=16)
plt.tight_layout()
plt.show()

# box-test and ljung box test has been performed separately for all the models. Accordingly, the only white residuals are present in SARIMA, and LSTM model.
# However, Sarima model residuals appear to be more white than LSTM (visually)

# Next i'll plot the performance of the model on the test set in subplots.
fig, ax = plt.subplots(2,2,figsize=(16,8))
y_test['T(degC)'].plot(ax=ax[0,0],label='Actual T(degC)')
holt1f['Holt-Winter with damping'].plot(ax=ax[0,0],label="Holt-Winter Forecast")
ax[0,0].legend(loc='lower right')
ax[0,0].set_title(f'Holt-Winter Model Forecast vs Actual T(degC)')
ax[0,0].set_xlabel('Date')
ax[0,0].set_ylabel('Temperature (degC)')
ax[0,0].grid()
predictions.plot(ax=ax[0,1],label="MLR Forecast")
y_test['T(degC)'].plot(ax=ax[0,1],label="Actual T(degC)",linestyle='--')
ax[0,1].legend(loc='lower right')
ax[0,1].grid()
ax[0,1].set_title('MLR Model Forecast vs Actual T(degC)')
ax[0,1].set_xlabel('Date')
ax[0,1].set_ylabel('Temperature (degC)')
y_test['T(degC)'].plot(ax=ax[1,0],label='Actual T(degC)')
y_testdf['Forecast'].plot(ax=ax[1,0],label="SARIMA Forecast")
ax[1,0].legend(loc='lower right')
ax[1,0].set_title(f'ARIMA(3,0,0)xSARIMA(0,1,0,365) Model Forecast vs Actual T(degC)')
ax[1,0].set_xlabel('Date')
ax[1,0].set_ylabel('Temperature (degC)')
ax[1,0].grid()
test["T(degC)"].plot(ax=ax[1,1],color='red',label="Actual T(degC)")
test["Predictions"].plot(ax=ax[1,1],color='blue',label="LSTM Forecast")
ax[1,1].legend(loc = "lower right")
ax[1,1].grid()
ax[1,1].set_title("LSTM Model Forecast vs Actual T(degC)")
ax[1,1].set_xlabel("Date")
ax[1,1].set_ylabel("Temperature (degC)")
fig.suptitle("Model Forecast vs Actual T(degC)",fontsize=18)
plt.tight_layout()
plt.show()

# Judging by these results, the MLR model seems to perform the best on the test set, followed by the LSTM model, followed by SARIMA model and then the Holt-Winter model.
# To better quantify these values, i'll create a table of model performance metric that i've stored while creating each model

performance_table = [['Model Type', 'Root-Mean-Square-Error','Mean-Absolute-Error','Mean-Absolute-Percent-Error'],
         ["Holt-Winter Model",round(RMSE_hw2,3),round(MAE_hw2,3),round(MAPE_hw2,3)],["MLR Model",round(RMSE_ols,3),round(MAE_ols,3),round(MAPE_ols,3)],["SARIMA Model",round(RMSE_sarima,3),round(MAE_sarima,3),round(MAPE_sarima,3)],["LSTM Model",round(RMSE_lstm,3),round(MAE_lstm,3),round(MAPE_lstm,3)]]
print(tabulate(performance_table,headers='firstrow', tablefmt = 'fancy_grid'))

# This table helps quantify the results we obtain visually from the forecast vs actual dataset.

# According to all results obtained the LSTM model performs the best as it has the second lowest performance errors and also has a white residual. However, for the sake of forecast function creation, I will go ahead with SARIMA model as it also has a white residual and comparable model performance to LSTM.


