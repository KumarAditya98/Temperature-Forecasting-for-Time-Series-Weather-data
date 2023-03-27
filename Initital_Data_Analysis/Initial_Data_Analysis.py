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
j = []
count = 0
for i in range(len(df)):
    if i+1 > len(df)-1:
        break
    if df.iloc[i].name.day == df.iloc[i+1].name.day:
        count = count + 1
    else:
        count = count + 1
        j.append(count)
        count = 0

df_1 = df[['rain(mm)']]
df_1 = df_1.resample('D').sum()

df_2 = df[['T(degC)']]
df_2 = df_2.resample('D').mean().fillna('backfill')

a = pd.Series(j,name='Day_Counts',index = df_2.index)

fig, ax = plt.subplots(figsize=(30,16))
df_2.plot()
plt.tight_layout()
plt.grid()
plt.show()

Temp_1 = pd.Series(df_2['Tpot(K)'].values,index = df_1.index,
                 name = 'temp')
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
