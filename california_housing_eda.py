"""
California Housing Dataset: Exploratory Data Analysis (EDA)
This script performs comprehensive EDA on the California housing dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set visualization styles
plt.style.use('seaborn')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)

print("California Housing Dataset: Exploratory Data Analysis (EDA)")
print("="*70)

# Create output directory for plots
if not os.path.exists('eda_plots'):
    os.makedirs('eda_plots')

# 1. Data Loading and Initial Inspection
print("\n1. Data Loading and Initial Inspection")
print("-"*50)

# Load the dataset
df = pd.read_csv('california_housing.csv')

# View the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Dataset basic information
print("\nDataset shape:", df.shape)
print("\nData types:")
df.info()

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# 2. Descriptive Statistics
print("\n2. Descriptive Statistics")
print("-"*50)

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Analyze the skewness of numerical features
skewness = df.skew().sort_values(ascending=False)
print("\nSkewness of features:")
print(skewness)

plt.figure(figsize=(10, 6))
sns.barplot(x=skewness.index, y=skewness.values)
plt.title('Skewness of Numerical Features')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('eda_plots/skewness.png')
plt.close()

# 3. Target Variable Analysis
print("\n3. Target Variable Analysis")
print("-"*50)

# Distribution of the target variable (MedHouseValue)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['MedHouseValue'], kde=True)
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value ($100,000s)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['MedHouseValue'])
plt.title('Boxplot of Median House Value')
plt.ylabel('Median House Value ($100,000s)')

plt.tight_layout()
plt.savefig('eda_plots/target_variable.png')
plt.close()

# Check for potential ceiling or floor effects in the target variable
max_value = df['MedHouseValue'].max()
min_value = df['MedHouseValue'].min()

print(f"\nMinimum value: {min_value}")
print(f"Maximum value: {max_value}")

# Count of houses at or near max value (potential ceiling effect)
ceiling_threshold = max_value * 0.95  # 95% of max value
ceiling_count = df[df['MedHouseValue'] >= ceiling_threshold].shape[0]
ceiling_percentage = ceiling_count / len(df) * 100

print(f"Number of houses at or above 95% of maximum value: {ceiling_count} ({ceiling_percentage:.2f}%)")

# 4. Correlation Analysis
print("\n4. Correlation Analysis")
print("-"*50)

# Calculate correlation matrix
corr_matrix = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.savefig('eda_plots/correlation_matrix.png')
plt.close()

# Correlation with the target variable (MedHouseValue)
target_corr = corr_matrix['MedHouseValue'].sort_values(ascending=False)
print("\nCorrelation with target variable (MedHouseValue):")
print(target_corr)

plt.figure(figsize=(10, 6))
sns.barplot(x=target_corr.index, y=target_corr.values)
plt.title('Correlation with Median House Value')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('eda_plots/target_correlation.png')
plt.close()

# 5. Feature Engineering
print("\n5. Feature Engineering")
print("-"*50)

# Create new features
df_features = df.copy()

# Bedroom ratio (bedrooms per room)
df_features['BedroomRatio'] = df['AveBedrms'] / df['AveRooms']

# Rooms per household
df_features['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']

# Bedrooms per household
df_features['BedroomsPerHousehold'] = df['AveBedrms'] / df['AveOccup']

# Population density within households
df_features['PopulationDensity'] = df['Population'] / df['AveOccup']

# Location-based features
df_features['DistanceToLA'] = np.sqrt(((df['Latitude'] - 34.05) ** 2) + ((df['Longitude'] - (-118.25)) ** 2))
df_features['DistanceToSF'] = np.sqrt(((df['Latitude'] - 37.77) ** 2) + ((df['Longitude'] - (-122.42)) ** 2))
df_features['CoastalProximity'] = np.abs(df['Longitude'] + 122)  # West coast is around -122 longitude

print("\nNew features created:")
for feature in ['BedroomRatio', 'RoomsPerHousehold', 'BedroomsPerHousehold', 
                'PopulationDensity', 'DistanceToLA', 'DistanceToSF', 'CoastalProximity']:
    print(f"- {feature}")

# Correlation of new features with the target variable
new_corr = df_features.corr()['MedHouseValue'].sort_values(ascending=False)
print("\nCorrelation with target after feature engineering (top 10):")
print(new_corr.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(x=new_corr.head(15).index, y=new_corr.head(15).values)
plt.title('Top 15 Correlations with Median House Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_plots/new_features_correlation.png')
plt.close()

# 6. Geographic Analysis
print("\n6. Geographic Analysis")
print("-"*50)

# Create a scatter plot of housing locations colored by median house value
plt.figure(figsize=(12, 10))
scatter = plt.scatter(df['Longitude'], df['Latitude'], 
                      c=df['MedHouseValue'], 
                      cmap='viridis', 
                      alpha=0.5,
                      s=10)

plt.colorbar(scatter, label='Median House Value')
plt.title('Geographic Distribution of California Housing Prices')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Mark major California cities
cities = {
    'San Francisco': (-122.42, 37.77),
    'Los Angeles': (-118.25, 34.05),
    'San Diego': (-117.16, 32.72),
    'Sacramento': (-121.49, 38.58)
}

for city, (lon, lat) in cities.items():
    plt.plot(lon, lat, 'ro')
    plt.text(lon, lat, city, fontsize=12, ha='right')

plt.grid(True)
plt.tight_layout()
plt.savefig('eda_plots/geographic_distribution.png')
plt.close()

print("\nGeographic analysis completed. See plot in 'eda_plots/geographic_distribution.png'")

# 7. Bivariate Analysis
print("\n7. Bivariate Analysis")
print("-"*50)

# Analyze relationship between top features and house value
top_features = ['MedInc', 'AveRooms', 'HouseAge', 'Latitude', 'Longitude']

plt.figure(figsize=(15, 12))
for i, feature in enumerate(top_features):
    plt.subplot(2, 3, i+1)
    plt.scatter(df[feature], df['MedHouseValue'], alpha=0.5)
    plt.title(f'{feature} vs MedHouseValue')
    plt.xlabel(feature)
    plt.ylabel('Median House Value')
    
    # Add trend line
    z = np.polyfit(df[feature], df['MedHouseValue'], 1)
    p = np.poly1d(z)
    plt.plot(df[feature], p(df[feature]), "r--")
    
    # Calculate correlation
    corr = df[feature].corr(df['MedHouseValue'])
    plt.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')

plt.tight_layout()
plt.savefig('eda_plots/bivariate_analysis.png')
plt.close()

print("\nBivariate analysis completed. See plot in 'eda_plots/bivariate_analysis.png'")

# 8. Feature Distributions
print("\n8. Feature Distributions")
print("-"*50)

# Distribution of all numerical features
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Calculate rows and columns for subplots
n_cols = 3
n_rows = (len(num_cols) + n_cols - 1) // n_cols

plt.figure(figsize=(15, n_rows * 4))
for i, col in enumerate(num_cols):
    plt.subplot(n_rows, n_cols, i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()

plt.savefig('eda_plots/feature_distributions.png')
plt.close()

print("\nFeature distribution analysis completed. See plot in 'eda_plots/feature_distributions.png'")

# 9. Outlier Analysis
print("\n9. Outlier Analysis")
print("-"*50)

# Box plots for outlier detection
plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
plt.savefig('eda_plots/outlier_analysis.png')
plt.close()

# Calculate percentage of outliers in each feature
def get_outliers_percentage(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    percentage = len(outliers) / len(df) * 100
    
    return percentage

outlier_percentages = {col: get_outliers_percentage(df, col) for col in num_cols}
print("\nPercentage of outliers in each feature:")
for col, percentage in sorted(outlier_percentages.items(), key=lambda x: x[1], reverse=True):
    print(f"{col}: {percentage:.2f}%")

print("\nOutlier analysis completed. See plot in 'eda_plots/outlier_analysis.png'")

# 10. Summary of Findings
print("\n10. Summary of Findings")
print("-"*50)
print("\nKey findings from the exploratory data analysis:")
print("\n1. Target Variable:")
print("   - Median house value shows some right-skewness")
print(f"   - {ceiling_percentage:.2f}% of houses are at or above 95% of the maximum value")

print("\n2. Important Features (correlation with target):")
for feature, corr_value in target_corr.head(5).items():
    print(f"   - {feature}: {corr_value:.4f}")

print("\n3. Feature Engineering:")
print("   - Several engineered features show strong correlation with the target")
print("   - Top engineered features by correlation:")
for feature, corr_value in new_corr.head(10).items():
    if feature != 'MedHouseValue' and feature not in target_corr.head(5).index:
        print(f"   - {feature}: {corr_value:.4f}")

print("\n4. Geographic Patterns:")
print("   - Clear geographic clusters of higher house values")
print("   - Coastal areas and regions near major cities have higher values")

print("\n5. Outliers:")
print("   - Several features contain outliers that may need treatment")
print("   - Features with highest outlier percentages:")
top_outliers = sorted(outlier_percentages.items(), key=lambda x: x[1], reverse=True)[:3]
for col, percentage in top_outliers:
    print(f"   - {col}: {percentage:.2f}%")

print("\nEDA completed successfully! All plots saved in the 'eda_plots' directory.")
print("\nRecommendations for modeling:")
print("1. Use engineered features that showed strong correlation with the target")
print("2. Consider handling outliers before modeling")
print("3. Geographic features are important - include location-based variables")
print("4. Median income is the strongest predictor - ensure it's prominently used in models") 