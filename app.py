# app.py

import streamlit as st
import pandas as pd

from src.data_generator import generate_employee_data
from src.preprocessing import get_processed_data
from src.model import build_model, train_model
from src.explainability import get_feature_importance

st.set_page_config(page_title="Employee Performance Predictor", layout="wide")

st.title("💼 Employee Performance Predictor")
st.markdown("Predict employee performance and get HR insights using AI")

# -----------------------------
# MODE SELECTION
# -----------------------------
mode = st.sidebar.radio(
    "Select Mode",
    ["Single Employee", "Bulk Prediction"]
)

# -----------------------------
# LOAD MODEL (CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    df = generate_employee_data()
    X_train, X_test, y_train, y_test, preprocessor = get_processed_data()

    model = build_model(preprocessor, y_train)
    model = train_model(model, X_train, y_train)

    top_features = get_feature_importance(model, X_test, y_test)

    return model, top_features

model, top_features = load_model()

# ============================================================
# 🔹 SINGLE EMPLOYEE MODE
# ============================================================
if mode == "Single Employee":

    st.sidebar.header("🧾 Employee Details")

    def user_input():
        age = st.sidebar.slider("Age", 22, 60, 30)
        experience_years = st.sidebar.slider("Experience", 0, 35, 5)
        job_level = st.sidebar.slider("Job Level", 1, 5, 2)

        on_time_delivery_rate = st.sidebar.slider("On-Time Delivery Rate", 0.4, 1.0, 0.7)
        bug_count = st.sidebar.slider("Bug Count", 0, 50, 10)
        code_review_score = st.sidebar.slider("Code Review Score", 1.0, 5.0, 3.0)
        manager_score = st.sidebar.slider("Manager Score", 1.0, 5.0, 3.0)
        training_hours = st.sidebar.slider("Training Hours", 0, 100, 20)

        data = {
            "age": age,
            "education_level": "Bachelor",
            "experience_years": experience_years,
            "department": "IT",
            "job_level": job_level,
            "manager_tenure": 5,
            "projects_completed": 5,
            "story_points_completed": 100,
            "billable_hours_ratio": 0.7,
            "avg_task_delay_days": 5,
            "on_time_delivery_rate": on_time_delivery_rate,
            "bug_count": bug_count,
            "code_review_score": code_review_score,
            "defect_density": 2,
            "training_hours": training_hours,
            "certifications_count": 2,
            "sick_days": 2,
            "unplanned_leaves": 2,
            "peer_feedback_score": 3,
            "manager_score": manager_score,
            "kudos_count": 5,
            "promotions_last_2_years": 0,
            "salary_percentile": 50
        }

        return pd.DataFrame([data])

    input_df = user_input()

    # -----------------------------
    # CLEAN UI
    # -----------------------------
    st.subheader("📋 Employee Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Age", input_df["age"][0])
        st.metric("Experience", input_df["experience_years"][0])
        st.metric("Job Level", input_df["job_level"][0])

    with col2:
        st.metric("Delivery Rate", round(input_df["on_time_delivery_rate"][0], 2))
        st.metric("Bug Count", input_df["bug_count"][0])
        st.metric("Code Score", round(input_df["code_review_score"][0], 2))

    with col3:
        st.metric("Manager Score", round(input_df["manager_score"][0], 2))
        st.metric("Training Hours", input_df["training_hours"][0])

    st.success("Showing analysis for a single employee")

    # -----------------------------
    # EXPLANATION FUNCTION
    # -----------------------------
    def explain_employee_ui(input_df, top_features):
        explanations = []

        for feature in top_features.index[:5]:
            value = input_df[feature].values[0]

            if feature == "on_time_delivery_rate":
                explanations.append(f"On-time delivery rate: {value:.2f}")
            elif feature == "bug_count":
                explanations.append(f"Bug count: {value}")
            elif feature == "manager_score":
                explanations.append(f"Manager score: {value:.2f}")
            elif feature == "training_hours":
                explanations.append(f"Training hours: {value}")
            elif feature == "code_review_score":
                explanations.append(f"Code review score: {value:.2f}")

        return explanations

    # -----------------------------
    # PREDICTION
    # -----------------------------
    if st.button("🔍 Predict Performance"):

        prediction = model.predict(input_df)[0]

        st.subheader("🎯 Prediction Result")

        if prediction == "High":
            st.success(f"Predicted Performance: {prediction}")
        elif prediction == "Medium":
            st.warning(f"Predicted Performance: {prediction}")
        else:
            st.error(f"Predicted Performance: {prediction}")

        # Key Factors
        st.subheader("🧠 Key Factors")

        reasons = explain_employee_ui(input_df, top_features)

        for r in reasons:
            st.write(f"✔ {r}")

        # Recommendations
        st.subheader("📌 HR Recommendations")

        for r in reasons:
            if "delivery" in r:
                st.write("👉 Improve time management and task planning")
            elif "bug" in r:
                st.write("👉 Focus on code quality and testing practices")
            elif "manager" in r:
                st.write("👉 Improve communication with manager")
            elif "training" in r:
                st.write("👉 Enroll in training programs")
            elif "code review" in r:
                st.write("👉 Improve code review practices")

# ============================================================
# 🔹 BULK PREDICTION MODE
# ============================================================
elif mode == "Bulk Prediction":

    st.subheader("📂 Upload Employee Dataset (CSV)")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Uploaded Data Preview")
        st.write(df.head())

        if st.button("🚀 Predict for All Employees"):

            predictions = model.predict(df)
            df["Predicted_Performance"] = predictions

            # ==============================
            # 📊 SUMMARY INSIGHTS
            # ==============================
            st.subheader("📊 Performance Summary")

            summary = df["Predicted_Performance"].value_counts()

            col1, col2, col3 = st.columns(3)
            col1.metric("Low", summary.get("Low", 0))
            col2.metric("Medium", summary.get("Medium", 0))
            col3.metric("High", summary.get("High", 0))

            # ==============================
            # 🧠 KEY DRIVERS
            # ==============================
            st.subheader("🧠 Key Drivers (Overall)")

            for f in top_features.index[:5]:
                st.write(f"✔ {f}")

            # ==============================
            # 📌 HR RECOMMENDATIONS
            # ==============================
            st.subheader("📌 HR Recommendations (Overall)")

            if "on_time_delivery_rate" in top_features.index[:5]:
                st.write("👉 Improve delivery timelines across teams")

            if "bug_count" in top_features.index[:5]:
                st.write("👉 Focus on code quality training")

            if "training_hours" in top_features.index[:5]:
                st.write("👉 Increase employee training programs")

            if "manager_score" in top_features.index[:5]:
                st.write("👉 Improve manager feedback systems")

            if "code_review_score" in top_features.index[:5]:
                st.write("👉 Strengthen code review practices")

            # ==============================
            # 📈 RESULTS TABLE
            # ==============================
            st.subheader("📈 Prediction Results")
            st.write(df)

            # Download
            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="⬇ Download Results",
                data=csv,
                file_name="employee_predictions.csv",
                mime="text/csv"
            )