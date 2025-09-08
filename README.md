# 💓 ECG Wearable Classifier

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-7C3AED?style=flat-square)

---

## 📑 Table of Contents
- [Overview](#-overview)  
- [Features](#-features)  
- [Project Structure](#-project-structure)  
- [Dataset](#-dataset)  
- [Model Details](#-model-details)  
- [Installation & Setup](#️-installation--setup)  
- [Running the App](#️-running-the-app)  
- [Deployment](#-deployment)  
- [Future Improvements](#-future-improvements)  
- [Authors](#-authors)  
- [License](#-license)  

---

## 📌 Overview
This project demonstrates how to classify **ECG (Electrocardiogram) heartbeats** collected from wearable devices using a **Convolutional Neural Network (CNN)**.  

It includes an **interactive Streamlit web app** that lets users upload a CSV file containing ECG data and instantly predicts whether the heartbeat is **Normal** or **Abnormal**.

---

## 🚀 Features
- ⚡ **Real-time Prediction** – Upload a CSV file and get instant results  
- 🧠 **Pre-trained CNN Model** – Provided in `.h5` and `.keras` formats  
- 📂 **Sample Data** – Ready-to-use ECG CSV files included  
- 🌐 **Interactive Web App** – Simple Streamlit interface  
- ☁️ **Lightweight Deployment** – Run locally or deploy on Streamlit Cloud, Heroku, or AWS  

---

## 📂 Project Structure
```
ECG-Wearable-Classifier/
│── ecg_app.py              # Streamlit application
│── ecg_cnn_model.h5        # Pre-trained CNN model (HDF5 format)
│── ecg_cnn_model.keras     # Pre-trained CNN model (Keras format)
│── ecg_data_from_watch/    # Sample ECG CSV files
│   ├── abnormal_beat.csv
│   ├── normal_beat.csv
│   └── user_beat_001.csv
│── requirements.txt        # Project dependencies
│── README.md               # Project documentation
```

---

## 📊 Dataset
The **`ecg_data_from_watch/`** folder contains sample beats:  
- 🟢 `normal_beat.csv` → Example of a normal heartbeat  
- 🔴 `abnormal_beat.csv` → Example of an abnormal heartbeat  
- 👤 `user_beat_001.csv` → Example of a user input  

Each file contains **140 sequential ECG amplitude values** representing one heartbeat.  

---

## 🧠 Model Details
- **Architecture**: Convolutional Neural Network (CNN)  
- **Framework**: TensorFlow / Keras  
- **Input Shape**: `(140, 1)` normalized amplitude values  
- **Output**: Binary classification → `Normal (0)` or `Abnormal (1)`  

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/krithesh19/ECG-Wearable-Classifier.git
cd ECG-Wearable-Classifier
```

### 2️⃣ Create a Virtual Environment (optional)
```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on Mac/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the App
Run the Streamlit app:
```bash
streamlit run ecg_app.py
```

Then open in your browser:  
👉 [http://localhost:8501](http://localhost:8501)

---

## ☁️ Deployment
You can deploy this app on:
- 🌐 **Streamlit Cloud** – free & easy hosting  
- 🚀 **Heroku** – scalable cloud app platform  
- ☁️ **AWS / GCP / Azure** – enterprise deployment  

---

## 📌 Future Improvements
- 🔍 Add **multi-class classification** for different arrhythmias  
- ⌚ Integrate **live data streaming** from wearable devices  
- 📈 Enhance UI with **real-time ECG signal plotting**  

---

## 👨‍💻 Authors
- **Kritheshvar K (Krithesh)**  

---

## 📝 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  
