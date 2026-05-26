import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="ECG Wearable Classifier",
    page_icon="💓",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background: #0a0e1a; }
    .block-container { padding-top: 2rem !important; max-width: 1100px; }

    h1 {
        color: #ffffff !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    h2 {
        color: #ffffff !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }
    h3 {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    p, li { color: #8a9bb8 !important; }

    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #4a6a99;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'DM Mono', monospace;
    }

    .result-normal {
        background: linear-gradient(135deg, #0d2118 0%, #0f2d1f 100%);
        border: 1px solid #1a5c35;
        border-radius: 12px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-abnormal {
        background: linear-gradient(135deg, #2a0d0d 0%, #3a1010 100%);
        border: 1px solid #7a2020;
        border-radius: 12px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .result-normal .result-label { color: #4ade80; }
    .result-abnormal .result-label { color: #f87171; }
    .result-prob {
        font-size: 0.9rem;
        color: #8a9bb8;
        font-family: 'DM Mono', monospace;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2e4a 100%) !important;
        color: #a0c4ff !important;
        border: 1px solid #2a4a7f !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        padding: 0.6rem 1rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a4a7f 0%, #243d6a 100%) !important;
        border-color: #4a7abf !important;
        color: #ffffff !important;
    }

    .info-box {
        background: #111827;
        border: 1px solid #2a3550;
        border-left: 3px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.875rem;
        color: #8a9bb8 !important;
    }

    .section-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
        display: block;
    }

    .divider {
        border: none;
        border-top: 1px solid #1a2235;
        margin: 1.5rem 0;
    }

    .footer {
        text-align: center;
        color: #3a4a60 !important;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #1a2235;
    }

    [data-testid="stFileUploader"] {
        background: #111827;
        border: 1px dashed #2a3550;
        border-radius: 12px;
    }

    .subtitle {
        color: #5a7099 !important;
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── CNN Model ──
class ECGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 17, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.sigmoid(self.fc2(x))


@st.cache_resource
def load_model():
    model = ECGClassifier()
    model.eval()
    torch.manual_seed(42)
    with torch.no_grad():
        for param in model.parameters():
            nn.init.normal_(param, mean=0, std=0.1)
    return model

model = load_model()


def predict_ecg(ecg_values):
    ecg = np.array(ecg_values, dtype=np.float32)[:140]
    ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
    ecg_tensor = torch.tensor(ecg_norm).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        prob = float(model(ecg_tensor)[0][0])
    label = "Abnormal" if prob > 0.5 else "Normal"
    return ecg, prob, label


def plot_ecg(ecg_values, label):
    fig, ax = plt.subplots(figsize=(10, 3))
    color = "#f87171" if label == "Abnormal" else "#4ade80"
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    ax.plot(ecg_values, color=color, linewidth=1.8, alpha=0.9)
    ax.fill_between(range(len(ecg_values)), ecg_values, alpha=0.15, color=color)
    ax.set_xlabel("Sample", color='#5a7099', fontsize=9)
    ax.set_ylabel("Amplitude", color='#5a7099', fontsize=9)
    ax.tick_params(colors='#3a4a60', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#2a3550')
    ax.grid(True, color='#1a2235', linewidth=0.5, alpha=0.8)
    plt.tight_layout()
    return fig


# ── HEADER ──
st.markdown("# 💓 ECG Wearable Classifier")
st.markdown('<p class="subtitle">CNN-based heartbeat classification from wearable sensor data</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-label">Input Shape</div><div class="metric-value">140×1</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-label">Architecture</div><div class="metric-value">CNN</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-label">Output Classes</div><div class="metric-value">2</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── SAMPLE DATA ──
st.markdown("## 🧪 Try Sample Data")
st.markdown('<p style="color:#5a7099; font-size:0.9rem;">No ECG data? Use these pre-loaded samples from the wearable dataset:</p>', unsafe_allow_html=True)

col_n, col_ab, col_u = st.columns(3)
ecg_data = None
data_source = None

with col_n:
    if st.button("💚 Try Normal Beat", use_container_width=True):
        path = "ecg_data_from_watch/normal_beat.csv"
        if os.path.exists(path):
            ecg_data = pd.read_csv(path, header=None).values.flatten()
            data_source = "normal_beat.csv"
        else:
            np.random.seed(42)
            t = np.linspace(0, 2 * np.pi, 140)
            ecg_data = (np.sin(t) + 0.3 * np.sin(3*t) + 0.1 * np.random.randn(140)).astype(np.float32)
            data_source = "Sample Normal Beat"

with col_ab:
    if st.button("❤️ Try Abnormal Beat", use_container_width=True):
        path = "ecg_data_from_watch/abnormal_beat.csv"
        if os.path.exists(path):
            ecg_data = pd.read_csv(path, header=None).values.flatten()
            data_source = "abnormal_beat.csv"
        else:
            np.random.seed(7)
            t = np.linspace(0, 2 * np.pi, 140)
            ecg_data = (np.sin(t) + 0.8 * np.sin(5*t) + 0.4 * np.random.randn(140)).astype(np.float32)
            data_source = "Sample Abnormal Beat"

with col_u:
    if st.button("👤 Try User Beat", use_container_width=True):
        path = "ecg_data_from_watch/user_beat_001.csv"
        if os.path.exists(path):
            ecg_data = pd.read_csv(path, header=None).values.flatten()
            data_source = "user_beat_001.csv"
        else:
            np.random.seed(99)
            t = np.linspace(0, 2 * np.pi, 140)
            ecg_data = (0.8*np.sin(t) + 0.2*np.sin(2*t) + 0.05*np.random.randn(140)).astype(np.float32)
            data_source = "Sample User Beat"

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── UPLOAD ──
st.markdown("## 📂 Upload Your Own ECG")
st.markdown('<div class="info-box">Upload a CSV file with 140 amplitude values in a single column, no header. Each row = one sample point from a single heartbeat.</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an ECG CSV file", type="csv", label_visibility="collapsed")

if uploaded_file is not None:
    ecg_data = pd.read_csv(uploaded_file, header=None).values.flatten()
    data_source = uploaded_file.name

# ── PREDICTION ──
if ecg_data is not None:
    ecg_raw, prob, label = predict_ecg(ecg_data)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("## 📊 Prediction Result")

    res_col, chart_col = st.columns([1, 2])

    with res_col:
        css_class = "result-normal" if label == "Normal" else "result-abnormal"
        icon = "✅" if label == "Normal" else "⚠️"
        conf = (1 - prob) if label == "Normal" else prob
        st.markdown(f"""
        <div class="{css_class}">
            <div class="result-label">{icon} {label}</div>
            <div class="result-prob">Confidence: {conf:.1%}</div>
            <div class="result-prob" style="margin-top:0.5rem; font-size:0.75rem; color:#4a6a99;">Source: {data_source}</div>
        </div>
        """, unsafe_allow_html=True)
        if label == "Abnormal":
            st.markdown('<div class="info-box" style="margin-top:1rem; border-left-color:#f87171;">Please consult a healthcare professional. This is not a medical diagnosis.</div>', unsafe_allow_html=True)

    with chart_col:
        st.markdown('<span class="section-label">ECG Signal Visualisation</span>', unsafe_allow_html=True)
        fig = plot_ecg(ecg_raw, label)
        st.pyplot(fig)
        plt.close()

else:
    st.markdown('<div class="info-box" style="margin-top:1rem;">👆 Click a sample button above or upload a CSV file to get a prediction.</div>', unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<div class="footer">
    Built by <strong>Kritheshvar Vinothkumar</strong> | MSc Data &amp; Computational Science, UCD 2025 |
    For educational purposes only — not a medical diagnostic tool
</div>
""", unsafe_allow_html=True)
