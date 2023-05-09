import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
# from keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dense, LSTM, Dropout  # ,CuDNNLSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam

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
history = model.fit(x_train,y_train, batch_size=16, validation_split = .1, epochs=5, verbose=1)

plt.figure()
plt.plot(history.history['loss'], 'r',label='Training loss')
plt.plot(history.history['val_loss'], 'b',label='Validation loss')
plt.legend()
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
RMSE = np.sqrt(np.square(e_forecast).mean())
print(RMSE)