import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters

# Load and preprocess the data
df5 = pd.read_csv("C:/Users/USER/Dev/cfehome/FYP/clean2.csv", parse_dates=['created_at'], index_col="created_at")

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Plot data
print(df5.head())
sns.lineplot(x=df5.index, y="CO2", data=df5)
df_by_month = df5.resample('M').sum(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.lineplot(x=df_by_month.index, y=df_by_month["Temperature"], label="Temperature")
sns.lineplot(x=df_by_month.index, y=df_by_month["Humidity"], label="Humidity")
sns.lineplot(x=df_by_month.index, y=df_by_month["CO"], label="CO")
sns.lineplot(x=df_by_month.index, y=df_by_month["PM10"], label="PM10")
plt.legend()
plt.show()

# Split data into train and test sets
train_size = int(len(df5) * 0.9)
test_size = len(df5) - train_size
train, test = df5.iloc[0:train_size], df5.iloc[train_size:len(df5)]
print(len(train), len(test))

from sklearn.preprocessing import RobustScaler

# Scaling data
f_columns = ['Temperature', 'Humidity', 'PM10', 'CO']
cnt_column = ['CO2']

f_transformer = RobustScaler()
cnt_transformer = RobustScaler()

f_transformer.fit(train[f_columns].to_numpy())
cnt_transformer.fit(train[cnt_column].to_numpy())

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train.loc[:, cnt_column] = cnt_transformer.transform(train[cnt_column].to_numpy())
test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test.loc[:, cnt_column] = cnt_transformer.transform(test[cnt_column].to_numpy())

# Ensure no NaN values
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# Ensure data is float type
train[f_columns + cnt_column] = train[f_columns + cnt_column].astype(np.float32)
test[f_columns + cnt_column] = test[f_columns + cnt_column].astype(np.float32)

# Create dataset for LSTM
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X_train, y_train = create_dataset(train[f_columns + cnt_column], train['CO2'], time_steps)
X_test, y_test = create_dataset(test[f_columns + cnt_column], test['CO2'], time_steps)

print(X_train.shape, y_train.shape)

# Build the LSTM model
model = keras.Sequential()
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, shuffle=False)

# Plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions and true values
y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = cnt_transformer.inverse_transform(y_pred)

from sklearn.metrics import r2_score

# Calculate R-squared
r2 = r2_score(y_test_inv, y_pred_inv)
print(f'R-squared: {r2}')

from sklearn.metrics import mean_squared_error, accuracy_score


Mse = mean_squared_error(y_test_inv, y_pred_inv)**(1/2)

print(f'Mean Squared Error of Linear regression: {Mse}')



# Plot true values and predictions
plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('CO2')
plt.xlabel('Time Step')
plt.legend()
plt.show()

plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('CO2')
plt.xlabel('Time Step')
plt.legend()
plt.show()
