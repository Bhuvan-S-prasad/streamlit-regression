import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("California Housing Price Predictor")
st.markdown("""
This application predicts the median house value in California based on various features.
Use the sliders and inputs in the sidebar to specify house characteristics, and the model will predict the house value.
""")

@st.cache_resource
def load_model_components():
    """Load the saved model, scaler, and feature names."""
    model_path = 'models/linear_regression_model.pkl'
    scaler_path = 'models/scaler.pkl'
    features_path = 'models/feature_names.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
        st.error("Model files not found. Please run california_housing_regression.py first.")
        return None, None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, scaler, feature_names

def predict_house_value(input_data, model, scaler, feature_names):
    """Make a prediction using the saved model."""
    # Ensure the input data has all required features
    for feature in feature_names:
        if feature not in input_data:
            # For location-based features, calculate them if latitude and longitude are provided
            if feature == 'DistanceToLA' and 'Latitude' in input_data and 'Longitude' in input_data:
                input_data['DistanceToLA'] = np.sqrt(((input_data['Latitude'] - 34.05) ** 2) + 
                                                    ((input_data['Longitude'] - (-118.25)) ** 2))
            elif feature == 'DistanceToSF' and 'Latitude' in input_data and 'Longitude' in input_data:
                input_data['DistanceToSF'] = np.sqrt(((input_data['Latitude'] - 37.77) ** 2) + 
                                                    ((input_data['Longitude'] - (-122.42)) ** 2))
            elif feature == 'CoastalProximity' and 'Longitude' in input_data:
                input_data['CoastalProximity'] = np.abs(input_data['Longitude'] + 122)
            elif feature == 'BedroomRatio' and 'AveBedrms' in input_data and 'AveRooms' in input_data:
                input_data['BedroomRatio'] = input_data['AveBedrms'] / input_data['AveRooms']
            elif feature == 'RoomsPerHousehold' and 'AveRooms' in input_data and 'AveOccup' in input_data:
                input_data['RoomsPerHousehold'] = input_data['AveRooms'] / input_data['AveOccup']
            elif feature == 'BedroomsPerHousehold' and 'AveBedrms' in input_data and 'AveOccup' in input_data:
                input_data['BedroomsPerHousehold'] = input_data['AveBedrms'] / input_data['AveOccup']
            elif feature == 'PopulationDensity' and 'Population' in input_data and 'AveOccup' in input_data:
                input_data['PopulationDensity'] = input_data['Population'] / input_data['AveOccup']
            else:
                st.error(f"Missing feature: {feature}")
                return None
    
    # Convert to DataFrame and ensure correct order of features
    input_df = pd.DataFrame([input_data])[feature_names]
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return prediction

def load_california_map():
    # If prediction_plot.png exists, load it
    if os.path.exists('prediction_plot.png'):
        return Image.open('prediction_plot.png')
    return None

def main():
    # Load model components
    model, scaler, feature_names = load_model_components()
    
    if model is None:
        st.error("Failed to load model components. Please ensure the model is trained.")
        return
    
    # Create sidebar for inputs
    st.sidebar.header("House Features")
    
    # Add input widgets
    input_data = {}
    
    # Median Income (slider)
    input_data['MedInc'] = st.sidebar.slider(
        "Median Income (tens of thousands $)",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.1,
        help="Median income in block group in tens of thousands of US Dollars"
    )
    
    # House Age (slider)
    input_data['HouseAge'] = st.sidebar.slider(
        "House Age (years)",
        min_value=1,
        max_value=52,
        value=30,
        step=1,
        help="Median house age in block group"
    )
    
    # Average Rooms (slider)
    input_data['AveRooms'] = st.sidebar.slider(
        "Average Rooms per Household",
        min_value=1.0,
        max_value=10.0,
        value=6.0,
        step=0.1,
        help="Average number of rooms per household"
    )
    
    # Average Bedrooms (slider)
    input_data['AveBedrms'] = st.sidebar.slider(
        "Average Bedrooms per Household",
        min_value=0.5,
        max_value=5.0,
        value=1.2,
        step=0.1,
        help="Average number of bedrooms per household"
    )
    
    # Population (slider)
    input_data['Population'] = st.sidebar.slider(
        "Population",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Population in block group"
    )
    
    # Average Occupancy (slider)
    input_data['AveOccup'] = st.sidebar.slider(
        "Average Occupancy",
        min_value=1.0,
        max_value=6.0,
        value=3.0,
        step=0.1,
        help="Average number of household members"
    )
    
    # Location inputs - use a map for California
    st.sidebar.subheader("Location")
    
    # Default location (San Francisco Bay Area)
    default_lat = 37.85
    default_long = -122.25
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        input_data['Latitude'] = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=default_lat,
            step=0.01,
            help="Latitude (between 32.5 and 42 for California)"
        )
    
    with col2:
        input_data['Longitude'] = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=default_long,
            step=0.01,
            help="Longitude (between -124 and -114 for California)"
        )
    
    # Display a small map of California with markers
    st.sidebar.markdown("### California Map Reference")
    st.sidebar.markdown("Major cities reference:")
    st.sidebar.markdown("- San Francisco: 37.77, -122.42")
    st.sidebar.markdown("- Los Angeles: 34.05, -118.25")
    st.sidebar.markdown("- San Diego: 32.72, -117.16")
    st.sidebar.markdown("- Sacramento: 38.58, -121.49")
    
    # Prediction button
    predict_button = st.sidebar.button("Predict House Value", key="predict")
    
    # Main content area
    if predict_button:
        with st.spinner("Predicting house value..."):
            # Compute derived features
            input_data['BedroomRatio'] = input_data['AveBedrms'] / input_data['AveRooms']
            input_data['RoomsPerHousehold'] = input_data['AveRooms'] / input_data['AveOccup']
            input_data['BedroomsPerHousehold'] = input_data['AveBedrms'] / input_data['AveOccup']
            input_data['PopulationDensity'] = input_data['Population'] / input_data['AveOccup']
            input_data['DistanceToLA'] = np.sqrt(((input_data['Latitude'] - 34.05) ** 2) + 
                                                ((input_data['Longitude'] - (-118.25)) ** 2))
            input_data['DistanceToSF'] = np.sqrt(((input_data['Latitude'] - 37.77) ** 2) + 
                                                ((input_data['Longitude'] - (-122.42)) ** 2))
            input_data['CoastalProximity'] = np.abs(input_data['Longitude'] + 122)
            
            # Make prediction
            prediction = predict_house_value(input_data, model, scaler, feature_names)
            
            if prediction is not None:
                # Display prediction
                st.subheader("Prediction Results")
                
                # Format prediction in dollars
                prediction_dollars = prediction * 100000  # Convert to dollars (values in dataset are in $100,000s)
                
                # Display in large text
                st.markdown(f"<h2 style='text-align: center; color: #0066cc;'>Predicted House Value: ${prediction_dollars:,.2f}</h2>", 
                            unsafe_allow_html=True)
                
                # Display input data summary
                st.subheader("Input Summary")
                
                # Create three columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Median Income", f"${input_data['MedInc'] * 10000:,.2f}")
                    st.metric("House Age", f"{input_data['HouseAge']} years")
                    st.metric("Average Rooms", f"{input_data['AveRooms']:.1f}")
                
                with col2:
                    st.metric("Average Bedrooms", f"{input_data['AveBedrms']:.1f}")
                    st.metric("Population", f"{input_data['Population']}")
                    st.metric("Average Occupancy", f"{input_data['AveOccup']:.1f}")
                
                with col3:
                    st.metric("Bedroom Ratio", f"{input_data['BedroomRatio']:.2f}")
                    st.metric("Distance to SF", f"{input_data['DistanceToSF']:.2f} degrees")
                    st.metric("Distance to LA", f"{input_data['DistanceToLA']:.2f} degrees")
                
                # Visualizations
                st.subheader("Visualizations")
                
                # Location on map
                st.markdown("### Selected Location")
                
                # Create a dataframe for the map
                map_data = pd.DataFrame({
                    'latitude': [input_data['Latitude'], 37.77, 34.05],
                    'longitude': [input_data['Longitude'], -122.42, -118.25],
                    'label': ['Your Location', 'San Francisco', 'Los Angeles']
                })
                
                # Display the map
                st.map(map_data)
                
                # Feature importance plot
                st.markdown("### Feature Importance")
                
                # Get coefficients from the model
                coefficients = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', ascending=False)
                
                # Create horizontal bar chart
                fig = px.bar(
                    coefficients,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title='Model Coefficients (Feature Importance)',
                    color='Coefficient',
                    color_continuous_scale='RdBu_r'
                )
                
                # Update layout
                fig.update_layout(
                    height=500,
                    xaxis_title='Coefficient Value',
                    yaxis_title='Feature'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # If prediction_plot.png exists, show it
                prediction_plot = load_california_map()
                if prediction_plot:
                    st.markdown("### Model Prediction Performance")
                    st.image(prediction_plot, caption="Actual vs Predicted Values from Test Data")
    
    else:
        # Show welcome/instruction message when app first loads
        st.write("👈 Please adjust the input values in the sidebar and click 'Predict House Value' to see the prediction.")
        
        # Show an image or description of the app features
        st.subheader("How to use this app:")
        st.markdown("""
        1. **Adjust input values**: Use the sliders in the sidebar to set house features
        2. **Set location**: Enter latitude and longitude values for the house location in California
        3. **Click predict**: Press the 'Predict House Value' button to see the prediction
        4. **View results**: The app will display the predicted house value and visualizations
        """)
        
        # Show feature description
        st.subheader("Feature Description:")
        feature_desc = pd.DataFrame({
            'Feature': [
                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                'Population', 'AveOccup', 'Latitude', 'Longitude'
            ],
            'Description': [
                'Median income in block group (tens of thousands of US Dollars)',
                'Median house age in block group (years)',
                'Average number of rooms per household',
                'Average number of bedrooms per household',
                'Block group population',
                'Average number of household members',
                'Block group latitude',
                'Block group longitude'
            ]
        })
        
        st.table(feature_desc)
        
        # Display model accuracy information
        st.subheader("Model Information:")
        st.markdown("""
        This application uses a Linear Regression model trained on the California Housing dataset.
        - **Model performance**: ~65% accuracy (R² score)
        - **Primary factors**: Income, location, and number of rooms have the strongest influence on price
        """)
        
        # If prediction_plot.png exists, show it
        prediction_plot = load_california_map()
        if prediction_plot:
            st.image(prediction_plot, caption="Model Performance: Actual vs Predicted Values from Test Data")

if __name__ == "__main__":
    main() 