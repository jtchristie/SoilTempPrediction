import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Load daily data
data = pd.read_csv('daily.csv')

# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Handling NaN values by dropping individually
data = data.dropna(subset=['s10t', 'airt', 'prec', 'slrt', 'wspd'])
X_neural = data[['airt', 'prec', 'slrt', 'wspd']]
Y_neural = data[['s10t']]

# Split data into training and testing sets
X_train_neural, X_test_neural, Y_train_neural, Y_test_neural = train_test_split(
    X_neural, Y_neural, test_size=0.2, random_state=42
)

# Standardize the data
scaler_X = StandardScaler()
X_train_neural_scaled = scaler_X.fit_transform(X_train_neural)
X_test_neural_scaled = scaler_X.transform(X_test_neural)

# Separate scaler for the target variable Y
scaler_Y = StandardScaler()
Y_train_neural_scaled = scaler_Y.fit_transform(Y_train_neural)

# Build the neural network model
model = Sequential()
model.add(Dense(16, input_dim=X_train_neural_scaled.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train_neural_scaled, Y_train_neural_scaled,
    epochs=50, batch_size=32,
    verbose=1, validation_split=0.2
)

# Print training history
print("Training Loss:", history.history['loss'])
print("Validation Loss:", history.history['val_loss'])

# Make predictions on the test set
Y_pred_neural_scaled = model.predict(X_test_neural_scaled)

# Inverse transform the scaled predictions to get actual values
Y_pred_neural = scaler_Y.inverse_transform(Y_pred_neural_scaled)

# Drop NaN values from Y_test_neural and Y_pred_neural
nan_indices = np.isnan(Y_test_neural['s10t'])
Y_test_neural = Y_test_neural[~nan_indices]
Y_pred_neural = pd.DataFrame(Y_pred_neural, index=Y_test_neural.index, columns=['s10t'])

# Ensure indices are aligned
Y_test_neural = Y_test_neural.loc[Y_pred_neural.index]

# Evaluate the model
mse_neural = mean_squared_error(Y_test_neural, Y_pred_neural)
print(f'Mean Squared Error (Neural Network): {mse_neural}')

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(Y_test_neural.index, Y_test_neural['s10t'], label='Actual Temperature', marker='o')
plt.scatter(Y_pred_neural.index, Y_pred_neural['s10t'], label='Predicted Temperature', marker='x')
plt.xlabel('Data Points')
plt.ylabel('Soil Temperature')
plt.title('Actual vs Predicted Soil Temperature (Neural Network)')
plt.legend()
plt.grid(True)
plt.show()

 