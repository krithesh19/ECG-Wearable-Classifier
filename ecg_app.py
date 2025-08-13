import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob

st.set_page_config(page_title="Smartwatch ECG Predictor", layout="wide")
st.title("💓 Heart Condition Prediction from Wearable ECG")

# Load your trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('ecg_cnn_model.keras')
model = load_model()

st.write("Upload a single-beat ECG CSV file (140 values, one column, no header):")
uploaded_file = st.file_uploader("Choose an ECG CSV file", type="csv")

# ----------- Manual Upload Section -----------
if uploaded_file is not None:
    ecg_beat = pd.read_csv(uploaded_file, header=None).values.flatten()[:140]
    ecg = (ecg_beat - np.mean(ecg_beat)) / (np.std(ecg_beat) + 1e-8)   # normalize
    ecg = ecg.reshape(1, 140, 1)
    prob = model.predict(ecg)[0][0]
    label = "Abnormal" if prob > 0.5 else "Normal"

    st.subheader("Raw ECG Signal")
    st.line_chart(ecg_beat)

    st.subheader("Prediction Result")
    if label == "Normal":
        st.success(f"✅ Your ECG segment appears normal. (Probability of abnormal: {prob:.2f})")
    else:
        st.error(f"⚠️ This ECG segment may show abnormal patterns! (Probability of abnormal: {prob:.2f})")
        st.warning("Please discuss these results with a healthcare professional.")

else:
    st.info("Please upload your ECG data to get a prediction.")

st.caption("This app is for educational/demo purposes. It does not provide a medical diagnosis.")

# ----------- Automatic Cloud "Watch" Upload Section -----------
st.subheader("📡 Or get the latest ECG automatically from the wearable (cloud upload demo)")

if st.button("Predict using latest ECG from watch"):
    watch_dir = "ecg_data_from_watch"
    files = glob.glob(os.path.join(watch_dir, "*.csv"))
    if not files:
        st.warning("No ECG files found in the watch directory.")
    else:
        latest_file = max(files, key=os.path.getctime)
        st.info(f"Using file: {os.path.basename(latest_file)}")
        ecg_beat = pd.read_csv(latest_file, header=None).values.flatten()[:140]
        ecg = (ecg_beat - np.mean(ecg_beat)) / (np.std(ecg_beat) + 1e-8)
        ecg = ecg.reshape(1, 140, 1)
        prob = model.predict(ecg)[0][0]
        label = "Abnormal" if prob > 0.5 else "Normal"
        st.subheader("Raw ECG Signal")
        st.line_chart(ecg_beat)
        st.success(f"Prediction: {label} (probability: {prob:.2f})")
