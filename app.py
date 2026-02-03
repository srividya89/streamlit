import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Machine Learning Classification Dashboard")


# =====================================================
# Load Trained Models and Stored Metrics
# =====================================================

with open("models/saved_models.pkl", "rb") as f:
    models = pickle.load(f)

results_df = pd.read_csv("models/model_results.csv", index_col=0)


# =====================================================
# Sidebar Options
# =====================================================

st.sidebar.header("Options")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Preprocessed Test Dataset (CSV)",
    type=["csv"]
)


# =====================================================
# Display Evaluation Metrics from Training Phase
# =====================================================

st.subheader(f"Evaluation Metrics: {model_name}")

st.dataframe(results_df.loc[[model_name]])


# =====================================================
# Prediction Section
# =====================================================

if uploaded_file:

    test_data = pd.read_csv(uploaded_file)

    st.write("Uploaded Dataset Preview")
    st.dataframe(test_data.head())

    model = models[model_name]

    # -------------------------------------------------
    # Validate Required Columns
    # -------------------------------------------------

    required_columns = [
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
        "Month",
        "VisitorType",
        "Weekend"
    ]

    missing = [col for col in required_columns if col not in test_data.columns]

    if missing:
        st.error(f"Missing required columns in uploaded file: {missing}")
        st.stop()

    # -------------------------------------------------
    # Separate Actual column if present
    # -------------------------------------------------

    if "Actual" in test_data.columns:
        y_actual = test_data["Actual"]
        input_data = test_data.drop("Actual", axis=1)
    else:
        y_actual = None
        input_data = test_data

    # -------------------------------------------------
    # Make Predictions
    # -------------------------------------------------

    try:
        preds = model.predict(input_data)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

    # -------------------------------------------------
    # Probability Handling
    # -------------------------------------------------

    try:
        prob_values = model.predict_proba(input_data)

        if prob_values.shape[1] == 2:
            probs = prob_values[:, 1]
        else:
            probs = prob_values.max(axis=1)

    except:
        probs = ["N/A"] * len(preds)

    # -------------------------------------------------
    # Show Prediction Output
    # -------------------------------------------------

    st.subheader("Predictions Preview")

    output = test_data.copy()
    output["Prediction"] = preds
    output["Probability"] = probs

    st.dataframe(output.head())

    # -------------------------------------------------
    # Prediction Distribution
    # -------------------------------------------------

    st.subheader("Prediction Distribution")

    st.write(output["Prediction"].value_counts())

    # -------------------------------------------------
    # CONFUSION
