# 💓 ECG Wearable Classifier

## 📌 Overview
This project demonstrates how to classify ECG (Electrocardiogram) heartbeats collected from wearable devices using a **Convolutional Neural Network (CNN)** and deploy it as an **interactive web app** using **Streamlit**.

The application allows users to upload a CSV file containing ECG data and receive predictions on whether the heartbeat is **Normal** or **Abnormal** in real-time.

---

## 🚀 Features
- **Real-time Prediction**: Upload an ECG beat in CSV format and get instant classification.
- **Pre-trained CNN Model**: The heavy lifting is done by a trained model (`ecg_cnn_model.h5` / `.keras`).
- **Sample Data**: Includes example ECG data for testing.
- **Interactive UI**: Powered by Streamlit, easy to use in any web browser.
- **Lightweight Deployment**: Works locally or on cloud platforms (Streamlit Cloud, Heroku, etc.).

---

## 📂 Project Structure
ECG-Wearable-Classifier/
│── ecg_app.py # Streamlit application
│── ecg_cnn_model.h5 # Pre-trained CNN model (HDF5 format)
│── ecg_cnn_model.keras # Pre-trained CNN model (Keras format)
│── ecg_data_from_watch/ # Folder containing sample ECG CSV files
│ ├── abnormal_beat.csv
│ ├── normal_beat.csv
│ └── user_beat_001.csv
│── requirements.txt # Project dependencies
│── README.md # This file

yaml
Copy
Edit

---

## 📊 Dataset
The `ecg_data_from_watch` folder contains sample ECG beats:
- **normal_beat.csv** → Example of a normal heartbeat
- **abnormal_beat.csv** → Example of an abnormal heartbeat
- **user_beat_001.csv** → Example user ECG input

Each file contains **140 sequential ECG amplitude values** representing one heartbeat.

---

## 🧠 Model Details
- **Type**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow / Keras
- **Input Shape**: (140, 1) normalized amplitude values
- **Output**: Binary classification → `Normal` (0) or `Abnormal` (1)

---

## ⚙️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ECG-Wearable-Classifier.git
cd ECG-Wearable-Classifier

2️⃣ Create a Virtual Environment (Optional but Recommended)

python -m venv venv
# Activate it:
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate

3️⃣ Install Dependencies

pip install -r requirements.txt
🖥️ Running the App
Run the Streamlit app with:


streamlit run ecg_app.py
You will see a link like:

Local URL: http://localhost:8501
Click it to open the app in your browser.

☁️ Deployment
You can deploy this app to:

Streamlit Cloud – free hosting for Streamlit apps

Heroku

AWS / GCP / Azure

📌 Future Improvements
Add multi-class classification for different heart arrhythmias.

Integrate live data streaming from wearable devices.

Improve UI/UX with real-time signal plotting.

👨‍💻 Authors
Your Name – krithesh

📝 License
This project is licensed under the MIT License – see the LICENSE file for details.
That will give it a more **open-source project look**.  

Do you want me to make that upgraded, more stylish README version?
