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
import statsmodels.api as sm


df = pd.read_csv("C:/Users/USER/Dev/cfehome/FYP/clean2.csv", parse_dates=['created_at'], index_col='created_at')

df.index = pd.to_datetime(df.index)
df = df.sort_index()
df = df.asfreq('D')
print(df.index.freq)
df=df.drop(columns=['date', 'time', 'hour', 'month', 'day_of_month', 'day_of_week', 'PM10', 'PM2_5', 'Humidity', 'Temperature', 'Methane', 'CO'], axis=1)
print(df.info(), df.index)


df['CO2'] = df['CO2'].fillna(df['CO2'].mean()) # handle NaNs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20,10))
plot_acf(df['CO2'].diff().dropna(), ax = ax1)
plot_pacf(df['CO2'].diff().dropna(), ax = ax2)
plt.show()
 # Backward fill as a precaution
decompose = seasonal_decompose(df['CO2'].dropna(), period = 7).plot()
decompose.set_size_inches((20,10))
plt.show()



train, test = train_test_split(df['CO2'], test_size = 0.4, random_state = 12, shuffle = False)


# Fit the ARIMA model on the training set
model = ARIMA(train, order=(1, 0, 2))
modelfit = model.fit()

# Forecast for the test period
forecast_steps = len(test)
predictions = modelfit.forecast(steps=forecast_steps)

# Combine the training data and the predictions for plotting
history = train.append(test)
print(predictions)
from sklearn.metrics import mean_squared_error, r2_score
Rmse = np.sqrt(mean_squared_error(test, predictions))
print('Rmse Score:', Rmse)
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(history.index, history.values, label='Historical Values')
plt.plot(test.index, test.values, label='Actual Values', color='orange', marker='o')  # Increase linewidth and add marker)
plt.plot(test.index, predictions, label='Predicted Values', color='green')
plt.xlabel('Time')
plt.ylabel('CO2 Levels, ppm')
plt.title('CO2 Levels: History, Actual and Predicted Values')
plt.legend()
plt.show()


model2=sm.tsa.statespace.SARIMAX(train, order=(0, 0, 0), seasonal_order=(0, 1, 0, 7))
results=model2.fit()
forecast_steps = len(test)
forecast = results.forecast(steps=forecast_steps)

# Combine the training data and the predictions for plotting
history = train.append(test)
print(forecast)
from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(test, forecast))

# Print RMSE and R²
print(f'RMSE: {rmse}')

r2 = r2_score(test, forecast)
print(f'R-squared: {r2}')
#print(f'R²: {r2}')
# Plot the results

plt.figure(figsize=(10, 6))
plt.plot(history.index, history.values, label='Historical Values')
plt.plot(test.index, test.values, label='Actual Values', color='red', marker='o')
plt.plot(test.index, forecast, label='Predicted Values', color='green')
plt.xlabel('Time')
plt.ylabel('CO2 Levels, ppm')
plt.title('CO2 Levels: History, Actual and Predicted Values')
plt.legend()
plt.show()


from pmdarima.arima import auto_arima

from pmdarima.arima import auto_arima

# Assuming `train` is your training data series
mod = auto_arima(train, 
                 start_p=1, start_q=1,
                 test='adf',            # Use ADF test to find optimal `d`
                 max_p=5, max_q=5,
                 m=7,                   # Weekly seasonality
                 d=0,                   # No differencing needed since data is stationary
                 seasonal=True,         # Enable seasonal component
                 start_P=1, start_Q=1,  # Start with no seasonal AR terms
                 D=1,                   # Seasonal differencing if required
                 max_P=5, max_Q=5,      # Limit seasonal AR terms to a reasonable range
                 trace=True,
                 error_action='ignore', 
                 suppress_warnings=True, 
                 stepwise=True)

print(mod.summary())

