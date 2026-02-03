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
# DISPLAY OVERALL COMPARISON TABLE  (NEW FEATURE)
# =====================================================

st.subheader("Overall Model Performance Comparison")

st.dataframe(results_df)

# =====================================================
# Display Stored Evaluation Metrics
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
    # Automatically separate features and actual column
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
    # CONFUSION MATRIX SECTION
    # -------------------------------------------------

    st.subheader("Confusion Matrix")

    if y_actual is not None:

        cm = confusion_matrix(y_actual, preds)

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )

        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("Actual Labels")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

        # -------------------------------------------------
        # Accuracy and Classification Report
        # -------------------------------------------------

        acc = accuracy_score(y_actual, preds)

        st.subheader("Model Accuracy")
        st.write(acc)

        st.subheader("Classification Report")

        report = classification_report(y_actual, preds)

        st.text(report)

    else:
        st.warning(
            "To generate confusion matrix, your uploaded CSV must contain an 'Actual' column."
        )

else:
    st.info("Please upload a CSV file to test predictions.")
