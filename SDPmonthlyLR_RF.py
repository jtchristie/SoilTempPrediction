import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = pd.read_csv('hf001-05-monthly-e.csv')

#making months for cold season/warm season
data['date'] = pd.to_datetime(data['date'])
warm_season = ['04', '05', '06', '07', '08', '09']
cold_season = ['10', '11', '12', '01', '02', '03']


#mapping months to warm/cold
data['month'] = data['date'].dt.month
data['season'] = data['month'].astype(str).str.zfill(2)
season_mapping = {1: 'cold', 2: 'cold', 3: 'cold', 4: 'warm', 5: 'warm', 6: 'warm', 7: 'warm', 8: 'warm', 9: 'warm', 10: 'cold', 11: 'cold', 12: 'cold'}
data['season'] = data['month'].map(season_mapping)

#dropping NaN values and extracting data to X and Y variables
data = data.dropna(subset=['s10t'])
X = data[['airt', 'prec', 'slrt', 'wspd']]
Y = data[['s10t']]

#segmenting data into seasons
warm_data = data[data['season'] == 'warm']
X_warm = warm_data[['airt', 'prec', 'slrt', 'wspd']]
Y_warm = warm_data[['s10t']]
Y_warm = Y_warm.values.ravel() 

X_train_warm, X_test_warm, Y_train_warm, Y_test_warm = train_test_split(X_warm, Y_warm, test_size=0.2, random_state=42)
cold_data = data[data['season'] == 'cold']
X_cold = cold_data[['airt', 'prec', 'slrt', 'wspd']]
Y_cold = cold_data[['s10t']]
Y_cold = Y_cold.values.ravel()
X_train_cold, X_test_cold, Y_train_cold, Y_test_cold = train_test_split(X_cold, Y_cold, test_size=0.2, random_state=42)

#training for warm season
LRmodel_warm = LinearRegression()
LRmodel_warm.fit(X_train_warm, Y_train_warm)
LRpredict_warm = LRmodel_warm.predict(X_test_warm)

#training for cold season
LRmodel_cold = LinearRegression()
LRmodel_cold.fit(X_train_cold, Y_train_cold)
LRpredict_cold = LRmodel_cold.predict(X_test_cold)

#added RF implementation
#warm
rf_model_warm = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_warm.fit(X_train_warm, Y_train_warm)
rf_predict_warm = rf_model_warm.predict(X_test_warm)

#cold
rf_model_cold = RandomForestRegressor(n_estimators=100, random_state=42)  
rf_model_cold.fit(X_train_cold, Y_train_cold)
rf_predict_cold = rf_model_cold.predict(X_test_cold)


#evaluate and plot
mse_warm = mean_squared_error(Y_test_warm, LRpredict_warm)
mse_cold = mean_squared_error(Y_test_cold, LRpredict_cold)

print('Linear Regression MSE Warm:', mse_warm)  
print('Linear Regression MSE Cold:', mse_cold)

# Random Forest
rf_mse_warm = mean_squared_error(Y_test_warm, rf_predict_warm)
rf_mse_cold = mean_squared_error(Y_test_cold, rf_predict_cold)
   
print('Random Forest MSE Warm:', rf_mse_warm)
print('Random Forest MSE Cold:', rf_mse_cold)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].scatter(Y_test_warm, LRpredict_warm)
axs[0].scatter(Y_test_warm, rf_predict_warm, c='r')
axs[0].legend(['LR', 'RF'])
axs[0].set_title('Warm Season')

axs[1].scatter(Y_test_cold, LRpredict_cold)
axs[1].scatter(Y_test_cold, rf_predict_cold, c='r')
axs[1].legend(['LR', 'RF'])  
axs[1].set_title('Cold Season')

plt.show()
