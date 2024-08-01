import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df=pd.read_csv("C:/Users/USER/Dev/cfehome/FYP/imp.csv")
print(df.info())
df.isna().sum()
df.isnull().sum()
df.duplicated().sum()
columnsToDrop=['entry_id', 'latitude', 'longitude', 'elevation', 'status']
df=df.drop(columns=columnsToDrop, axis=1)
colToFill=['field1', 'field2', 'field3', 'field4', 'field5', 'field6', 'field7']
#for col in colToFill:
df[colToFill]=df[colToFill].fillna(df[colToFill].mean())
print(df.info())

#columnsToDrop=['entry_id']
#df=df.drop(columnsToDrop, axis=1)
#df=df.drop(columns='longitude', inplace=True)
#df=df.drop(columns='elevation', inplace=True)
#df=df.drop(columns='status', inplace=True)

#print(df.head())
df['created_at'] = pd.to_datetime(df['created_at'])


#for index, row in df.iterrows():
df['date'] = df['created_at'].dt.date
df['time'] = df['created_at'].dt.time
#print(df.head())


df= df.rename(columns = {'field1':'CO'})
df= df.rename(columns = {'field2':'CO2'})
df= df.rename(columns = {'field3':'Methane'})
df= df.rename(columns = {'field4':'Temperature'})
df= df.rename(columns = {'field5':'Humidity'})
df= df.rename(columns = {'field6':'PM2_5'})
df= df.rename(columns = {'field7':'PM10'})
  
df=df.replace([np.inf, -np.inf], np.nan)

na=df.isna().sum()
print('na values in data ', str(na))
df=df.drop(na, axis=0)
#df=df.fillna(df.mean())
naa=df.isna().sum()

print('after ', str(naa))
print(df.tail())
df['date']=pd.to_datetime(df['date'])
df=df[(df['date']>='2024-04-05') & (df['date']<='2024-05-28')]
#box plotting

quantile75=df["CO2"].quantile(0.75)
quantile25=df["CO2"].quantile(0.25)
iqr=quantile75-quantile25
upper=quantile75+(iqr*1.5)
lower=quantile25-(iqr*1.5)
df=df[(df["CO2"]>lower)&(df["CO2"]<upper)]

PMquantile75=df["PM2_5"].quantile(0.75)
PMquantile25=df["PM2_5"].quantile(0.25)
PMiqr=PMquantile75-PMquantile25
PMupper=PMquantile75+(PMiqr*1.75)
PMlower=PMquantile25-(PMiqr*1.75)
df=df[(df["PM2_5"]>PMlower)&(df["PM2_5"]<PMupper)]

PM1quantile75=df["PM10"].quantile(0.75)
PM1quantile25=df["PM10"].quantile(0.25)
PM1iqr=PM1quantile75-PM1quantile25
PM1upper=PM1quantile75+(PM1iqr*0.75)
PM1lower=PM1quantile25-(PM1iqr*0.25)
df=df[(df["PM10"]>PM1lower)&(df["PM10"]<PM1upper)]

Pquantile75=df["Humidity"].quantile(0.75)
Pquantile25=df["Humidity"].quantile(0.25)
Piqr=Pquantile75-Pquantile25
Pupper=Pquantile75+(Piqr*0.75)
Plower=Pquantile25-(Piqr*0.25)
df=df[(df["Humidity"]>Plower)&(df["Humidity"]<Pupper)]
df=df[(df["Temperature"]>15)&(df["Temperature"]<50)]
#df=df.drop(minmaxTemp, axis=0)
print(df["Temperature"].min())


      


#boxplot after cleaning data
#df.xticks(rot=45))
plt.subplot(1, 3, 1)

df1=df.drop(columns=["CO2", "CO", "Methane", "created_at"], axis=1)
sns.boxplot(data=df1)
plt.subplot(1, 3, 2)
#df2=df.drop(columns=["Temperature", "Humidity", "PM2_5", "PM10", "CO2", "Methane", "created_at"])
#sns.boxplot(data=df2)



sns.boxplot(data=df, y="CO2")
plt.tight_layout()
plt.show()
#ML part
#cols=['Methane', 'PM2_5', 'PM10']
#df=df.drop(cols, axis=1)
df=df.drop(columns=["Methane", "PM2_5"], axis=1)
print(df.head())
print(df.describe())
print(df.corr())
sns.heatmap(data=df.corr(), annot=True, cmap='coolwarm', center=0)
plt.show()

#set index 
df=df.set_index(['date', 'time'])
print(df.head())

fig, ax=plt.subplots()
df.drop(columns=["CO2", "CO", "created_at"], axis=1).plot(ax=ax)
plt.ylabel('PM10(ppm), Temperature(C), Humidity(%)')
plt.xticks(rotation=5)



fig, ax=plt.subplots()
df["CO2"].plot(ax=ax)
plt.ylabel('CO2(ppm)')
plt.xticks(rotation=5)



fig, ax=plt.subplots()
df["CO"].plot(ax=ax)
plt.xticks(rotation=5)
plt.ylabel('CO(ppm)')
plt.tight_layout()
plt.show()
#applying ml

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Separate the features and the target variables
# Assuming the dataset columns are 'CO', 'CO2', 'Temperature', 'Humidity'
# We will predict 'CO2' and 'CO' levels
x = df.drop(columns=["CO2", "created_at"], axis=1)
y1 = df['CO2']


print(type(x), type(y1))
print(x.shape, y1.shape)

#data reshape by np array 

y1=y1.to_numpy().reshape(-1, 1)
x=x.to_numpy()

print(x.shape, y1.shape)
print(type(x), type(y1))


import scipy.stats as stats


#lineare regression

xtrain, xtest, ytrain, ytest=train_test_split(x, y1, test_size=0.3, random_state=42)
reg1=LinearRegression()
reg1.fit(xtrain, ytrain)
predictions=reg1.predict(xtest)

#residuals=y1-predictions


#plt.scatter(x[:, 0], y1)
#plt.plot(predictions, color="red")
#plt.ylabel("Temp")
#plt.xlabel("CO")
#plt.show()

#plt.scatter(x[:, 1], y1)
#plt.plot(predictions, color="red")
#plt.ylabel("Temp")
#plt.xlabel("CO2")
#plt.show()

plt.subplot(2, 2, 1)
sns.regplot(data=df, x="Temperature", y="CO2", line_kws={'color': 'red'})
plt.title('Linear Regression Fit: CO2 vs Temperature')
plt.xlabel('Temperatre(C)')
plt.ylabel('CO2')

plt.subplot(2, 2, 2)
sns.regplot(data=df, x="CO", y="CO2", line_kws={'color': 'red'})
plt.title('Linear Regression Fit: CO vs CO2')
plt.xlabel('CO(ppm)')
plt.ylabel('CO2(PPM)')

plt.subplot(2, 2, 3)
sns.regplot(data=df, x="Humidity", y="CO2", line_kws={'color': 'red'})
plt.title('Linear Regression Fit: Humidity vs CO2')
plt.xlabel('Humidity(%)')
plt.ylabel('CO2(PPM)')

plt.subplot(2, 2, 4)
sns.regplot(data=df, x="PM10", y="CO2", line_kws={'color': 'red'})
plt.title('Linear Regression Fit: PM10 vs CO2')
plt.xlabel('PM10(ppm)')
plt.ylabel('CO2(PPM)')


plt.tight_layout()
plt.show()





#residuals=residuals.reshape(-1)
from sklearn.metrics import r2_score
Mse = mean_squared_error(ytest, predictions)**(1/2)

print(f'Mean Squared Error of Linear regression: {Mse}')


r2 = r2_score(ytest, predictions)
print(f'R-squared: {r2}')


#plt.figure(figsize=(10, 4))

#plt.subplot(1, 2, 1)
#sns.histplot(residuals, kde=True, bins=20, color='blue', alpha=0.7)
#plt.xlabel('Residuals')
#plt.ylabel('Frequency')
#plt.title('Histogram of Residuals')

# Q-Q plot of residuals against a normal distribution
#plt.subplot(1, 2, 2)
#stats.probplot(residuals, dist="norm", plot=plt)
#plt.title('Q-Q Plot of Residuals')

#plt.tight_layout()
#plt.show()
#Random forest
print(df.columns)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Load your data

from sklearn.preprocessing import StandardScaler, RobustScaler

# Initialize the scaler

print(df.head())

# Features and target variable
X = df.drop(columns=["CO2", "CO", "created_at"], axis=1)
y = df['CO2']
SEED=1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Train your model and make predictions
model3 = RandomForestRegressor(n_estimators=420, random_state=42)
model3.fit(X_train_scaled, y_train_scaled.ravel())
y_pred_scaled = model3.predict(X_test_scaled)

msee = mean_squared_error(y_test_scaled, y_pred_scaled)**(1/2)

print(f'Mean Squared Error of RF: {msee}')

RR2 = r2_score(y_test_scaled, y_pred_scaled)
print(f'R-squared: {RR2}')


# Inverse transform the predictions
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test = scaler_y.inverse_transform(y_test_scaled)
y_train = scaler_y.inverse_transform(y_train_scaled)

# Initialize the Random Forest Regressor

# Evaluate the model


# Plotting Actual vs Predicted CO2 values

# Use datetime for creating date objects for plotting

 #Plot the actual values
plt.plot(range(len(y_test)), y_test, 'b-', label = 'actual')
# Plot the predicted values
plt.plot(range(len(y_pred)), y_pred, 'ro', label = 'prediction')
plt.xticks(rotation=60)
plt.legend()
# Graph labels
plt.xlabel('Date'); plt.ylabel('CO2(ppm)'); plt.title('Actual and Predicted Values')
plt.show()

print(df.head())


# Plotting Feature Importances
feature_importances = model3.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='b', align='center')
plt.xlabel('Relative Importance')
plt.title('Feature Importances')
plt.show()
#xgboost

#Gradient Boosting Machines




#Support Vector Machines with Non-linear Kernels

#ARIMA
