import streamlit as st
import joblib

def app():
    rf_model = joblib.load('rf_hyper.pkl')
    
    features_names = ['CO', 'Toluene', 'NO2', 'NOx', 'Benzene', 'Xylene', 'AQI', 'PM2.5']
    input_features = []
    
    for features_name in features_names:
        input_feature = st.number_input(features_name, step=0.01)
        input_features.append(input_feature)

    if st.button('buat_prediksi'):
        input_array = [input_features]
        prediksi = rf_model.predict(input_array)[0]
        
        kategori_aqi = {
            0: 'Good',
            1: 'Moderate',
            2: 'Poor',
            3: 'Satisfactory',
            4: 'Severe',
            5: 'Very Poor'
        }
        
        prediksi_kategori = kategori_aqi[prediksi]
        
        st.write(f'Prediksi Kualitas Udara : {prediksi_kategori}')
