import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('delivery_delay.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the features below to predict if a delivery will be delayed.')

# Define the input features as they were used in training
# This order and names must match the X.columns from the training notebook
feature_order = [
    'Delivery_Distance',
    'Traffic_Congestion',
    'Weather_Condition',
    'Delivery_Slot',
    'Driver_Experience',
    'Num_Stops',
    'Vehicle_Age',
    'Road_Condition_Score',
    'Package_Weight',
    'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

# Create input widgets for each feature
# Using reasonable default values and ranges based on the dataset
delivery_distance = st.number_input('Delivery Distance (km)', min_value=1.0, max_value=50.0, value=25.0, step=0.1)
traffic_congestion = st.slider('Traffic Congestion (1=Low, 5=High)', min_value=1, max_value=5, value=3)
weather_condition = st.slider('Weather Condition (1=Good, 3=Bad)', min_value=1, max_value=3, value=2)
delivery_slot = st.slider('Delivery Slot (1=Morning, 2=Afternoon, 3=Evening)', min_value=1, max_value=3, value=2)
driver_experience = st.number_input('Driver Experience (years)', min_value=0, max_value=20, value=5)
num_stops = st.number_input('Number of Stops', min_value=1, max_value=10, value=3)
vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, max_value=15, value=5)
road_condition_score = st.slider('Road Condition Score (1=Poor, 5=Excellent)', min_value=1, max_value=5, value=3)
package_weight = st.number_input('Package Weight (kg)', min_value=0.1, max_value=50.0, value=5.0, step=0.1)
fuel_efficiency = st.number_input('Fuel Efficiency (km/l)', min_value=5.0, max_value=25.0, value=15.0, step=0.1)
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=10, max_value=100, value=45)

# Create a dictionary of input features
input_data = {
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}

# Create a button for prediction
if st.button('Predict Delivery Delay'):
    # Convert input data to a Pandas DataFrame, ensuring the order of columns
    features_df = pd.DataFrame([input_data], columns=feature_order)
    
    # Make prediction
    prediction = model.predict(features_df)
    prediction_proba = model.predict_proba(features_df)
    
    # Display the prediction
    if prediction[0] == 1:
        st.error(f'Prediction: Delayed (Probability: {prediction_proba[0][1]*100:.2f}%)')
    else:
        st.success(f'Prediction: Not Delayed (Probability: {prediction_proba[0][0]*100:.2f}%)')
