import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Dataset/WeatherJena.csv')

# Looking at all the column names
print(df.columns)

# Target variable is Temperature in degree celsius
print(df.head().to_string())
print(df.tail())
# One extra observation for a new day. Will remove this observation from the dataset
df.drop(df.tail(1).index,inplace=True)

print(df.shape)
print(df.nunique())

# I notice the presence of few duplicated dates as the number of unique dates is less than total number of rows. Dropping duplicated rows

df.drop_duplicates(inplace=True)
print(df.shape)
print(df.nunique())
# Number of date-time now matches the number of rows.

# Getting rid of the white spaces in column names
df.columns = df.columns.str.replace(' ','')

# Looking at the number of rows
print(df.shape)

# Treating DateTime column as index
df.index = pd.to_datetime(df.DateTime,dayfirst=True)
print(df.head())

# We don't require the DateTime column anymore.
df.drop(columns=['DateTime'],axis=1,inplace=True)
print(df.shape)
print(df.head())

# # Subsetting only the time stamps and target variable for plotting.
# df_new = df.iloc[:,0:3:2]
# print(df_new.head())
#
# type(df_new.DateTime)
# df_new.index = pd.to_datetime(df_new.DateTime)
# df_new = df_new.drop('DateTime',axis = 1)
#
# fig, ax = plt.subplots(figsize=(30,16))
# df['Tpot(K)'].plot()
# plt.tight_layout()
# plt.grid()
# plt.show()

# Result of some external analysis - 2 missing dates + 6 days have incomplete 10-minute interval observations.

# To count the number of rows for each day in the dataset. Expectation is to see 144 count for each days as it is a 10 - minute interval. There are 1440 minutes in a day.
j = pd.DataFrame([],columns = ['Count','Year','Month','Day'])
count = 0
for i in range(len(df)):
    if i+1 > len(df)-1:
        break
    if df.iloc[i].name.day == df.iloc[i+1].name.day:
        count = count + 1
    else:
        count = count + 1
        temp = pd.DataFrame({'Count':count,'Year':df.iloc[i].name.year,'Month':df.iloc[i].name.month,'Day':df.iloc[i].name.day},index = [0])
        j = pd.concat([j,temp],ignore_index=True,)
        count = 0

# Now checking how many days don't have exactly 144 observations as is the expectation
print(j[j['Count']!=144])
# From this analysis we see a few drawbacks in the 10-minute interval data. 9 days don't have all the 10 minute intervals covered while measuring the data. We cannot use the 10 minute interval as time series.
# To convert into a usable time-series, i will instead aggregate my data to day level. I will do this by taking the average of all temperatures within the day so as to avoid bias.

# df_1 = df[['rain(mm)']]
# df_1 = df_1.resample('D').sum()

df_2 = df[['T(degC)']]
df_2 = df_2.resample('D').mean()

# Looking at the number of days that are avilable between this time period to see if this operation has been performed successfully. Using an external tool, i've caluclated the number of days between these dates (inclusive) to be 4564.
print(len(df_2))
# The length matches.

# Now evaluating whether any day is missing from the dataset, expectation is to not see any NA's after aggregation
print(df_2[df_2.isna().values])
# From this analysis, we see that 2 days don't have any observations in the original data. Since this is a time-series, I will have to use an appropriate method to impute this data.
# The method I will be using will be drift method to impute missing values.

Vals = np.linspace(df_2.iloc[0,0],df_2.iloc[-1,0],len(df_2))
Temp = pd.Series(Vals,name='T(degC)',index = df_2.index)
Temp = pd.DataFrame(Temp)
df_2 = df_2.fillna(Temp)

# Checking for NA's againg to see if this worked
print(df_2.isna().sum())

# Checking to see the whether the two known missing days have been imputed '2016-10-26' & '2016-10-27'
print(Temp.loc['2016-10-26':'2016-10-27'])
print(df_2.loc['2016-10-26':'2016-10-27'])

# Plotting the newly aggregated data along with the drift line used for imputation
fig, ax = plt.subplots(figsize=(16,8))
df_2['T(degC)'].plot(label="Daily time-series Data")
Temp['T(degC)'].plot(label="Interpolation Line")
plt.grid()
plt.legend()
plt.title("Plotting the daily aggregated time-series along with the interpolation line used to impute missing values")
plt.xlabel("Time")
plt.ylabel("Temperature (degC)")
plt.show()

from toolbox import cal_autocorr
lags = 90
Temp_1_arr = np.array(Temp_1)
cal_autocorr(Temp_1_arr,lags,'Temperature (degC)')
plt.show()

from toolbox import ADF_Cal
ADF_Cal(Temp_1_arr)

from toolbox import kpss_test
kpss_test(Temp_1_arr)

Temp_1 = pd.Series(df_2['T(degC)'].values,index = df_2.index,
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
from statsmodels.tsa.seasonal import STL
Temp = pd.Series(df['Tlog(degC)'].values,index = df_new.index,
                 name = 'temp')
STL = STL(Temp,period=144)
res = STL.fit()
T = res.trend
S = res.seasonal
R = res.resid
fig = res.plot()
plt.show()
SOT = max(0,(1-((np.var(R))/(np.var(R+T)))))
SOT


# We see that the observations are at 10 minute intervals. Although this is fine, I would like to convert the dataframe to 1-hour intervals. To do this, I will have to aggregate the rest of the columns accordingly.
