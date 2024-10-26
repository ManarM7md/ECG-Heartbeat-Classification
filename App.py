import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io

def load_model_from_url(url):
    """Load a model file from a specified URL using joblib."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        file_content = io.BytesIO(response.content)
        return joblib.load(file_content)
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load the file from {url}: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the PCA and classifier models
pca_1 = load_model_from_url('https://github.com/ManarM7md/ECG-Heartbeat-Classification/raw/main/pca_1_transform.pkl')
pca_2 = load_model_from_url('https://github.com/ManarM7md/ECG-Heartbeat-Classification/raw/main/pca_2_transform.pkl')
scaler_1 = load_model_from_url('https://github.com/ManarM7md/ECG-Heartbeat-Classification/raw/main/scaler_1.pkl')
scaler_2 = load_model_from_url('https://github.com/ManarM7md/ECG-Heartbeat-Classification/raw/main/scaler_2.pkl')
classifier = load_model_from_url('https://github.com/ManarM7md/ECG-Heartbeat-Classification/raw/main/Binary classifier_random_forest_model.pkl')
sub_classifier = load_model_from_url('https://github.com/ManarM7md/ECG-Heartbeat-Classification/raw/main/sub-classifier_random_forest_model.pkl')

# Class mapping
class_mapping = {
    0: "Normal 'N'",
    1: "Supra-ventricular premature 'S'",
    2: "Ventricular escape 'V'",
    3: "Fusion of ventricular and normal 'F'",
    4: "Fusion of paced and normal 'Q'"
}

# Function to predict ECG class from file
def predict_ecg_class_from_file(ecg_data):
    if ecg_data.shape[0] != 188:
        return "Error: The file should contain exactly 188 features."

    ecg_data = ecg_data.reshape(1, -1)  # Reshape for PCA
    ecg_data = scaler_1.transform(ecg_data)
    ecg_data_pca = pca_1.transform(ecg_data)
    binary_prediction = classifier.predict(ecg_data_pca)

    if binary_prediction[0] != 0:
        ecg_data_pca = scaler_2.transform(ecg_data_pca)
        ecg_data_pca = pca_2.transform(ecg_data)
        sub_prediction = sub_classifier.predict(ecg_data_pca)
        return f"Predicted Class: {class_mapping[sub_prediction[0]]}"
    else:
        return f"Predicted Class: {class_mapping[binary_prediction[0]]}"

# Streamlit application
st.title("ECG Signal Classifier")
st.write("Upload a .csv file containing the 188 features of the ECG signal to predict its class.")

# File uploader
uploaded_file = st.file_uploader("Upload ECG .csv File", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the ECG data from the uploaded file
        ecg_data = pd.read_csv(uploaded_file)

        # Check if the uploaded data has 188 columns
        if ecg_data.shape[1] != 188:
            st.error("The uploaded file must contain exactly 188 features.")
        else:
            # Convert DataFrame to numpy array
            ecg_data_array = ecg_data.values[:, :-1]  # Exclude the target column if present
            
            # Predict the class
            prediction = predict_ecg_class_from_file(ecg_data_array)
            st.success(prediction)

    except Exception as e:
        st.error(f"Error processing file: {e}")
