import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
df = pd.read_csv("C:/Users/USER/Dev/cfehome/FYP/clean2.csv", parse_dates=['date'], index_col="date")
df=df.drop(columns=['created_at', 'time', 'hour', 'day_of_month', 'day_of_week', 'month', 'PM10', 'PM2_5', 'Humidity', 'Temperature', 'Methane', 'CO'], axis=1)
print(df.info(), df.index)

adfuller_result = adfuller(df['CO2'].dropna())
pvalue = adfuller_result[1]
static=adfuller_result[0]
print(adfuller_result)
if pvalue < 0.05:
    print("stationary", str(pvalue))
else:
    print("non-stationary", str(pvalue))
#differencing
adfuller_result = adfuller(df.dropna().diff().dropna())
value = adfuller_result[1]
print(adfuller_result)
if value < 0.05:
    print("stationary")
else:
    print("non-stationary")

 #40, 60 lags

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20,10))
plot_acf(df['CO2'].diff().dropna(), ax = ax1, lags = 40)
plot_pacf(df['CO2'].diff().dropna(), ax = ax2, lags = 40)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20,10))
plot_acf(df['CO2'].diff().dropna(), ax = ax1, lags = 600)
plot_pacf(df['CO2'].diff().dropna(), ax = ax2, lags = 600)
plt.show()


decompose = seasonal_decompose(df['CO2'].dropna(), period = 7).plot()
decompose.set_size_inches((20,10))
plt.show()

train, test = train_test_split(df['CO2'], test_size = 0.7, random_state = 43, shuffle = False)
p, d, q = 1, 0, 0
fcst = []
for step in range(test.shape[0]):
        arima = ARIMA(train, order = (p, d, q))
        arima_final = arima.fit()
        prediction = arima_final.forecast(steps = 1)
        fcst.append(prediction.iloc[0])
        train = train.append(pd.Series(test.iloc[step]))
    


test["fcst"] = fcst
test.plot(figsize = (20,10))
plt.show()

r2_score(test.CO2, test.fcst)
print(r2_score)


adfuller_result = adfuller(df['CO2'].dropna())
pvalue = adfuller_result[1]
static=adfuller_result[0]
print(adfuller_result)
if pvalue < 0.05:
    print("stationary", str(pvalue))
else:
    print("non-stationary", str(pvalue))
#differencing
adfuller_result = adfuller(df.dropna().diff().dropna())
value = adfuller_result[1]
print(adfuller_result)
if value < 0.05:
    print("stationary")
else:
    print("non-stationary")

decompose = seasonal_decompose(df['CO2'].dropna(), period = 7).plot()
decompose.set_size_inches((20,10))
plt.show()