# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Simulate a dataset (You can replace this with a real dataset)
data = pd.DataFrame({
    'speed_limit': np.random.randint(30, 120, 100),
    'weather_condition': np.random.choice(['clear', 'rainy', 'foggy'], 100),
    'road_type': np.random.choice(['urban', 'rural'], 100),
    'light_conditions': np.random.choice(['daylight', 'night', 'dusk'], 100),
    'num_vehicles': np.random.randint(1, 5, 100),
    'alcohol_influence': np.random.choice([0, 1], 100),  # 0: No, 1: Yes
    'accident_severity': np.random.randint(1, 4, 100)  # 1: Minor, 2: Serious, 3: Fatal
})

# Step 2: Convert categorical data to numeric using one-hot encoding
data = pd.get_dummies(data, drop_first=True)
print("Sample Data (after encoding):\n", data.head())

# Step 3: Split the data into features (X) and target (y)
X = data.drop('accident_severity', axis=1)
y = data['accident_severity']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test data and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 6: Save the trained model to a file for future use
joblib.dump(model, 'accident_severity_model.pkl')
print("Model saved successfully as 'accident_severity_model.pkl'")

# Step 7: Load the saved model and use it for prediction on new data
loaded_model = joblib.load('accident_severity_model.pkl')

# Example input for prediction (hypothetical values)
example = pd.DataFrame({
    'speed_limit': [80],
    'num_vehicles': [3],
    'alcohol_influence': [1],
    'weather_condition_rainy': [1],
    'weather_condition_foggy': [0],
    'road_type_urban': [1],
    'light_conditions_night': [0],
    'light_conditions_dusk': [0]
})

# Predict accident severity using the loaded model
predicted_severity = loaded_model.predict(example)
print(f'Predicted Accident Severity: {predicted_severity[0]:.2f}')
