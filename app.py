"""Launch locally with: streamlit run app.py"""
import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from schema import FEATURES, validate_features

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='ABC Delivery Predictor', page_icon='📦', layout='centered')

@st.cache_resource
def load_model():
    with (ROOT/'delivery_delay_model.sav').open('rb') as f:
        model = pickle.load(f)
    metadata = json.loads((ROOT/'model_metadata.json').read_text())
    return model, metadata

try:
    model, metadata = load_model()
except FileNotFoundError:
    st.error('Missing model files. Upload delivery_delay_model.sav and model_metadata.json alongside app.py.')
    st.stop()

st.title('ABC Delivery Predictor')
st.write('Enter the delivery details known before dispatch to estimate the chance of a delay.')
st.caption('Classroom model. Estimates need validation on future operational deliveries.')

with st.form('delivery_form'):
    left, right = st.columns(2)
    with left:
        distance = st.number_input('Delivery distance (km)', min_value=0.0, value=19.35, step=0.1)
        traffic = st.selectbox('Traffic congestion', [1,2,3,4,5], index=3,
                               format_func=lambda v: f'{v}'+(' - low' if v==1 else ' - high' if v==5 else ''))
        weather = st.selectbox('Weather', [1,2,3], index=2,
                               format_func=lambda v: {1:'Clear',2:'Rainy',3:'Stormy'}[v])
        slot = st.selectbox('Delivery slot', [1,2,3], index=1,
                            format_func=lambda v: {1:'Morning',2:'Afternoon',3:'Evening'}[v])
        experience = st.number_input('Driver experience (years)', min_value=0.0, value=16.0, step=1.0)
        stops = st.number_input('Stops before final delivery', min_value=0, value=6, step=1)
    with right:
        age = st.number_input('Vehicle age (years)', min_value=0.0, value=9.0, step=1.0)
        road = st.selectbox('Road quality', [1,2,3,4,5], index=2,
                            format_func=lambda v: f'{v}'+(' - poor' if v==1 else ' - excellent' if v==5 else ''))
        weight = st.number_input('Package weight (kg)', min_value=0.0, value=33.62, step=0.1)
        fuel = st.number_input('Fuel efficiency (km/l)', min_value=0.1, value=13.02, step=0.1)
        warehouse = st.number_input('Warehouse processing time (minutes)', min_value=0.0, value=58.0, step=1.0)
    submitted = st.form_submit_button('Predict delivery', type='primary')

if submitted:
    row = dict(zip(FEATURES, [distance,traffic,weather,slot,experience,stops,age,road,weight,fuel,warehouse]))
    x = validate_features(pd.DataFrame([row]))
    probability = float(model.predict_proba(x)[0,1])
    st.metric('Estimated probability of delay', f'{probability:.1%}')
    if probability >= 0.5:
        st.warning('Predicted: delayed. Review this delivery before dispatch.')
    else:
        st.success('Predicted: on time.')
    st.caption('A probability of 50% or higher is classified as delayed. This is an estimate, not a guarantee.')
    outside = [c.replace('_',' ') for c,v in row.items()
               if not metadata['training_ranges'][c]['min'] <= v <= metadata['training_ranges'][c]['max']]
    if outside:
        st.info('Outside the training data range: '+', '.join(outside)+'. Interpret this estimate cautiously.')

with st.expander('How the model works'):
    st.write('Gradient boosting combines 100 small decision trees to predict delayed (1) or on time (0).')
    st.write('Models were compared using five-fold cross-validation on 800 training deliveries. The selected model correctly classified all 200 held-out test deliveries in this teaching dataset.')
    st.write('Warehouse processing time and vehicle age were the strongest predictors. Predictive importance does not establish a root cause.')
    st.write('The app loads the trained model. It does not retrain when you enter new values.')
