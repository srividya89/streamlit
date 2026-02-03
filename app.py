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

st.title(" Machine Learning Classification Dashboard")


# =====================================================
# Load Trained Models
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
    "Upload Test Dataset (CSV)",
    type=["csv"]
)


# =====================================================
# Display Evaluation Metrics
# =====================================================

st.subheader(f"Evaluation Metrics: {model_name}")

st.dataframe(results_df.loc[[model_name]])


# =====================================================
# Prediction and Confusion Matrix Section
# =====================================================

if uploaded_file:

    test_data = pd.read_csv(uploaded_file)

    model = models[model_name]

    # Make Predictions
    preds = model.predict(test_data)

    # Handle probability safely (binary or multi-class)
    try:
        prob_values = model.predict_proba(test_data)

        if prob_values.shape[1] == 2:
            probs = prob_values[:, 1]
        else:
            probs = prob_values.max(axis=1)

    except:
        probs = ["N/A"] * len(preds)


    # -------------------------------------------------
    # Show Prediction Output
    # -------------------------------------------------

    st.subheader(" Predictions Preview")

    output = test_data.copy()
    output["Prediction"] = preds
    output["Probability"] = probs

    st.dataframe(output.head())


    # -------------------------------------------------
    # Prediction Distribution
    # -------------------------------------------------

    st.subheader(" Prediction Distribution")

    st.write(output["Prediction"].value_counts())


    # -------------------------------------------------
    # CONFUSION MATRIX SECTION  (NEWLY ADDED)
    # -------------------------------------------------

    st.subheader(" Confusion Matrix")

    if "Actual" in test_data.columns:

        y_actual = test_data["Actual"]
        y_pred = preds

        cm = confusion_matrix(y_actual, y_pred)

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

        acc = accuracy_score(y_actual, y_pred)

        st.subheader(" Model Accuracy")
        st.write(acc)

        st.subheader("📄 Classification Report")

        report = classification_report(y_actual, y_pred)

        st.text(report)

    else:
        st.warning(
            "To generate confusion matrix, your uploaded CSV must contain an 'Actual' column."
        )

else:
    st.info("Please upload a CSV file to test predictions.")
