
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os

# Page configuration
st.set_page_config(
    page_title="PPID | Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# Neural network architecture
class IDSNeuralNetwork(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


# Load model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

@st.cache_resource
def load_model():

    with open(os.path.join(ARTIFACTS, "metadata.json")) as f:
        metadata = json.load(f)

    preprocessor = joblib.load(
        os.path.join(ARTIFACTS, "preprocessor.pkl")
    )

    scaler = joblib.load(
        os.path.join(ARTIFACTS, "scaler.pkl")
    )

    model = IDSNeuralNetwork(metadata["input_features"])

    state = torch.load(
        os.path.join(ARTIFACTS, "ppid_model.pth"),
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    return model, preprocessor, scaler, metadata


# Header
st.title("🛡️ PPID - Privacy Protection and Intrusion Detection")
st.caption(
    "Machine Learning Approach to Network Vulnerability Mitigation "
    "Using Intrusion Detection"
)

st.divider()

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Intrusion Detection",
        "About Project"
    ]
)

# Load resources
try:
    model, preprocessor, scaler, metadata = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# Dashboard
if page == "Dashboard":

    st.subheader("System Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Organizations", "5")
    col2.metric("Communication Rounds", "5")
    col3.metric("Model", "Federated NN")

    st.info(
        "Upload network traffic data through the Intrusion Detection "
        "module to identify suspicious traffic."
    )

    st.subheader("Model Information")

    st.write("Model:", metadata["model_type"])
    st.write("Input Features:", metadata["input_features"])
    st.write("Detection Threshold:", metadata["threshold"])

    st.subheader("Detection Workflow")

    st.markdown("""
    1. Upload network traffic CSV.
    2. Preprocess network features.
    3. Predict normal or malicious traffic.
    4. Generate intrusion alerts.
    5. Recommend mitigation actions.
    """)


# Intrusion detection
elif page == "Intrusion Detection":

    st.subheader("Network Traffic Analysis")

    uploaded_file = st.file_uploader(
        "Upload UNSW-NB15 formatted CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            data = pd.read_csv(uploaded_file)

            st.write("Uploaded Dataset Preview")
            st.dataframe(data.head())

            required_columns = metadata["feature_columns"]

            missing = [
                col for col in required_columns
                if col not in data.columns
            ]

            if missing:
                st.error(
                    "Missing required feature columns: "
                    + ", ".join(missing)
                )

            elif st.button("Detect Intrusions", type="primary"):

                X = data[required_columns].copy()

                processed = preprocessor.transform(X)

                if hasattr(processed, "toarray"):
                    processed = processed.toarray()

                scaled = scaler.transform(processed)

                tensor = torch.tensor(
                    scaled,
                    dtype=torch.float32
                )

                with torch.no_grad():
                    outputs = model(tensor)
                    probabilities = torch.sigmoid(
                        outputs
                    ).numpy().flatten()

                predictions = (
                    probabilities >= metadata["threshold"]
                ).astype(int)

                confidence = np.where(
                    predictions == 1,
                    probabilities,
                    1 - probabilities
                )

                results = pd.DataFrame({
                    "Prediction": [
                        "Attack" if p == 1 else "Normal"
                        for p in predictions
                    ],
                    "Confidence (%)": np.round(
                        confidence * 100, 2
                    ),
                    "Detection Status": [
                        "ATTACK DETECTED" if p == 1
                        else "NORMAL TRAFFIC"
                        for p in predictions
                    ],
                    "Recommended Action": [
                        "BLOCK / ISOLATE" if p == 1
                        else "ALLOW"
                        for p in predictions
                    ]
                })

                attacks = int(predictions.sum())
                normal = len(predictions) - attacks

                st.success("Traffic analysis completed!")

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Records", len(data))
                col2.metric("Attacks Detected", attacks)
                col3.metric("Normal Traffic", normal)

                st.subheader("Detection Results")
                st.dataframe(results, use_container_width=True)

                st.subheader("Detection Summary")

                st.bar_chart(
                    pd.DataFrame({
                        "Records": [normal, attacks]
                    }, index=["Normal", "Attack"])
                )

                csv = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Detection Report",
                    csv,
                    "PPID_detection_report.csv",
                    "text/csv"
                )

                st.warning(
                    "Mitigation actions are recommendations only. "
                    "This application does not modify firewall rules "
                    "or block real network connections."
                )

        except Exception as e:
            st.error(f"Analysis failed: {e}")


# About project
elif page == "About Project":

    st.subheader("About PPID")

    st.write("""
    PPID is a machine-learning-based intrusion detection project
    designed to identify malicious network traffic.

    The project simulates five organizations collaborating through
    Federated Learning without sharing their raw training records.

    A neural network is trained using federated aggregation.
    A noisy-gradient training experiment explores the effect of
    gradient clipping and Gaussian noise on model performance.
    """)

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - PyTorch
    - Scikit-learn
    - Pandas and NumPy
    - Federated Averaging
    - Streamlit
    """)

st.divider()

st.caption(
    "PPID | Academic Project | Network Intrusion Detection"
)
