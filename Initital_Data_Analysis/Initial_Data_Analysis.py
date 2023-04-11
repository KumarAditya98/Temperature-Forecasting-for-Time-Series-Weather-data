import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('Dataset/WeatherJena.csv')

# Looking at all the column names
print(df.columns)

# Target variable is Temperature in degree celsius
print(df.head().to_string())
print(df.tail().to_string())
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

# ----------------------------x--------------------------------------
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

df_target = df[['T(degC)']]
df_target = df_target.resample('D').mean()

# Looking at the number of days that are avilable between this time period to see if this operation has been performed successfully. Using an external tool, i've caluclated the number of days between these dates (inclusive) to be 4564.
print(len(df_target))
# The length matches.

# Now evaluating whether any day is missing from the dataset, expectation is to not see any NA's after aggregation
print(df_target[df_target.isna().values])
# From this analysis, we see that 2 days don't have any observations in the original data. Since this is a time-series, I will have to use an appropriate method to impute this data.
# The method I will be using will be drift method to impute missing values. I will perform this imputation after the train-test split only using the train data since with a train test split of 0.2/0.25, only the training data will have missing values.


# df_target contains only my dependent variable. I need to perform a similar aggregation for my independent variables.
# Listing down my dependent variables below to see what type of aggregation will be required for each:
# p (mbar) : The pascal SI derived unit of pressure used to quantify internal pressure. Meteorological reports typically state - I can take an average here.
# T (degC) : Temperature in Celsius - Target variable, already used.
# Tpot (K) : Temperature in Kelvin - Redundant variable, will be dropped.
# Tdew (degC) : Temperature in Celsius relative to humidity. Dew Point is a measure of the absolute amount of water in the air. - Could be highly correlated to temperature and/or have the same meaning as temperature. Will have to research on this variable. After Research - The dew point is the temperature the air needs to be cooled to (at constant pressure) in order to achieve a relative humidity (RH) of 100%. This feature may be highly correlated with relative humidity. Will do a correlation test later. Taking an average of this feature will do.
# rh (%) : Relative Humidity is a measure of how saturated the air is with water vapor, the %RH determines the amount of water. We can take an average of this feature as well for a day.
# VPmax (mbar) : Saturation vapor pressure - Average
# VPact (mbar) : Vapor pressure - Average
# VPdef (mbar) : Vapor pressure deficit - Average
# sh (g/kg) : Specific humidity - Average. Might be correlated with relative humidity.
# H2OC (mmol/mol) : Water vapor concentration - Average. Might be correlated with specific humidity. Expressed in different units.
# rho (g/m**3) : Airtight - Average
# wv (m/s) : Wind speed - Average
# max. wv (m/s) : Maximum wind speed - Average
# wd (deg) : Wind direction in degrees - Average
# rain (mm) : Rain in milimetrers - Will have to take a sum of this feature to aggregate for a day.
# raining (s) : Duration of rain in seconds - Will have to take a sum of this feature to aggregate for a day.
# SWDR (W/m*2) : The definition of this features wasn't given with the dataset. After research i found this to be the Solar Radiation. We can take the Average of this as well.
# PAR (mol/m*2/s) : The definition of this features wasn't given with the dataset. After research i found this to be the Photo Active Radiation. This might be correlated with solar radiation. For now, will take an average of this.
# max PAR (mol/m*2/s) : Maximum Photo Active Radiation. Will take an average of this.
# Tlog (degC) : Log of Temperature in Degree Celsius - Redundant variable. Will have to drop this.
# CO2 (ppm) : Carbon Dioxide parts per million - Average of the CO2 content in a day.

print(df.columns)
df_features = df.drop(columns=['Tpot(K)','Tlog(degC)','T(degC)'],axis=1)

print(df_features.columns)

df_features = df_features.resample('D').agg({'p(mbar)': 'mean', 'Tdew(degC)': 'mean', 'rh(%)': 'mean', 'VPmax(mbar)': 'mean', 'VPact(mbar)': 'mean', 'VPdef(mbar)': 'mean', 'sh(g/kg)': 'mean', 'H2OC(mmol/mol)': 'mean', 'rho(g/m**3)': 'mean', 'wv(m/s)': 'mean', 'max.wv(m/s)': 'mean', 'wd(deg)': 'mean', 'rain(mm)': 'sum', 'raining(s)': 'sum', 'SWDR(W/m�)': 'mean', 'PAR(�mol/m�/s)': 'mean', 'max.PAR(�mol/m�/s)': 'mean', 'CO2(ppm)': 'mean'})

nan = df_features.isnull()
print(df_features[nan.any(axis=1)])

# This feature set also has only those 2 dates worth of missing data. Will apply the same imputation to these time-series data.

# ---------------------Code no longer in use-----------------------------------------------
# Prior to train-test split, I'll go ahead and save out these two dataframes so that they can be used for broaded analyis in other files.
#---- DON'T RUN -----------------------------------------
# df_target.to_csv('Dataset/target_series.csv')
# df_features.to_csv('Dataset/feature_series.csv')
#-------------------------------------------------------------



# # Performing the train-test split here so that data imputation can be performed prior to proceeding.
# X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, shuffle=False, test_size=0.2, random_state=6313)
#
# # Making sure there aren't any null values in the test set
# print(X_test.isnull().all())
# print(y_test.isnull().all())
#
# # No null values. Will go ahead and save out the test data now
# #-----DON'T RUN------------------------------
# # X_test.to_csv('Dataset/X_test.csv')
# # y_test.to_csv('Dataset/y_test.csv')
# #-----DON'T RUN------------------------------
#
# # Confirming the presence of NA's in target train set
# print(y_train[y_train.isna().any(axis=1)])
#
# # Data imputation for target train variable using drift method
# Vals = np.linspace(y_train.iloc[0,0],y_train.iloc[-1,0],len(y_train))
# Temp = pd.Series(Vals,name='T(degC)',index = y_train.index)
# Temp = pd.DataFrame(Temp)
# y_train = y_train.fillna(Temp)
#
# # Checking for NA's again to see if this worked
# print(y_train.isna().sum())
#
# # Checking to see the whether the two known missing days have been imputed '2016-10-26' & '2016-10-27'
# print(Temp.loc['2016-10-26':'2016-10-27'])
# print(y_train.loc['2016-10-26':'2016-10-27'])
#
# # Repeating the same for feature train set
# # Confirming the presence of NA's in feature train set
# print(X_train[X_train.isna().any(axis=1)])
#
# # Data imputation for feature train variables using drift method
# def linspace(column):
#     return np.linspace(column.iloc[0],column.iloc[-1],len(column))
#
# Temp1 = pd.DataFrame(X_train.apply(linspace, axis=0))
# X_train = X_train.fillna(Temp1)
#
# # Checking for NA's again to see if this worked
# print(X_train.isna().sum())
#
# # Checking to see the whether the two known missing days have been imputed '2016-10-26' & '2016-10-27'
# print(Temp1.loc['2016-10-26':'2016-10-27'])
# print(X_train.loc['2016-10-26':'2016-10-27'])
#
# # No null values. Will go ahead and save out the train data now
# #-----DON'T RUN------------------------------
# #X_train.to_csv('Dataset/X_train.csv')
# #y_train.to_csv('Dataset/y_train.csv')
# #-----DON'T RUN------------------------------

# The following plot and subset of data highlighted problems with drift method of data imputation
#
# # Plotting the newly aggregated target train data along with the drift line used for imputation
# fig, ax = plt.subplots(figsize=(16,8))
# y_train['T(degC)'].plot(label="Daily time-series Data")
# Temp['T(degC)'].plot(label="Interpolation Line")
# plt.grid()
# plt.legend()
# plt.title("Plotting the daily aggregated time-series along with the interpolation line used to impute missing values")
# plt.xlabel("Time")
# plt.ylabel("Temperature (degC)")
# plt.show()

# print(X_train.loc['2016-10-20':'2016-10-30'])
# print(y_train.loc['2016-10-20':'2016-10-30'])
# --------------------------- Code above not in use--------------------------------------------------
# It seems like because of the size of the train set and the end value of the temperature, we are unable to retrieve an accurate representation of the data through a drift line. Imputation using drift may not be the best choice.

# Instead I will do a simple linear interpolation for my dataset for the 2 missing dates.

df_target.interpolate(method='time',axis=0,inplace=True)
print(df_target.loc['2016-10-20':'2016-10-30'])
df_features.interpolate(method='time',axis=0,inplace=True)
print(df_features.loc['2016-10-20':'2016-10-30'].to_string())

# Saving out these clean sets of data of targets and features for analysis in later files.
df_target.to_csv('Dataset/target_series.csv')
df_features.to_csv('Dataset/feature_series.csv')

