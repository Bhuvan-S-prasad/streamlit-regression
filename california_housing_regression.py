import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

print("Loading California housing dataset...")
df = pd.read_csv('california_housing.csv')

print("\nBasic dataset information:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values:")
print(missing_values)

# Display correlation with target variable
print("\nCorrelation with target variable (MedHouseValue):")
correlations = df.corr()['MedHouseValue'].sort_values(ascending=False)
print(correlations)



# new features -- from EDA
df['BedroomRatio'] = df['AveBedrms'] / df['AveRooms']
df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']
df['BedroomsPerHousehold'] = df['AveBedrms'] / df['AveOccup']
df['PopulationDensity'] = df['Population'] / df['AveOccup']

# Create location-based features
df['DistanceToLA'] = np.sqrt(((df['Latitude'] - 34.05) ** 2) + ((df['Longitude'] - (-118.25)) ** 2))
df['DistanceToSF'] = np.sqrt(((df['Latitude'] - 37.77) ** 2) + ((df['Longitude'] - (-122.42)) ** 2))
df['CoastalProximity'] = np.abs(df['Longitude'] + 122)  # West coast is around -122 longitude


# Prepare data for modeling
X = df.drop('MedHouseValue', axis=1)
y = df['MedHouseValue']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Linear Regression model
print("\nTraining Linear Regression model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Display metrics
print("\nModel Performance:")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Model 'Accuracy' on test data: {test_r2 * 100:.2f}%")

# Display model coefficients
print("\nModel Coefficients:")
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coefficients.sort_values('Coefficient', ascending=False))

# Function to make predictions
def predict_house_value(features_dict):
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([features_dict])
    
    # Apply the same preprocessing as during training
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return prediction

# Test the function with a sample from the test set
sample_idx = 0
sample_features = X_test.iloc[sample_idx].to_dict()
sample_actual = y_test.iloc[sample_idx]

sample_prediction = predict_house_value(sample_features)

print("\nPrediction Test:")
print(f"Actual house value: {sample_actual}")
print(f"Predicted house value: {sample_prediction}")
print(f"Prediction error: {abs(sample_actual - sample_prediction)}")
print(f"Prediction error percentage: {abs(sample_actual - sample_prediction) / sample_actual * 100:.2f}%")

# Save the model and other components
print("\nSaving the model and components...")
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Model and components saved successfully!")

# Creating a simple plot for visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Housing Values')
plt.savefig('prediction_plot.png')
plt.close()

print("\nPrediction plot saved as 'prediction_plot.png'")
print("\nDone! The model is now trained and saved for future use.") 