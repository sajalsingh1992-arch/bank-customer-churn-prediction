#%%
#
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


#%%
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("Bank Customer Churn Prediction")
st.write("Predict customer churn using Machine Learning.")


#%%
# Load trained model and preprocessing objects

xgb_model = joblib.load("xgboost_churn_model.pkl")
sc = joblib.load("standard_scaler.pkl")
le = joblib.load("gender_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")


#%%
# Model performance

st.header("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "86.55%")
col2.metric("Precision", "75.20%")
col3.metric("Recall", "47.07%")
col4.metric("ROC-AUC", "87.28%")


#%%
# Compare ROC-AUC scores

results = pd.DataFrame({
    "Model": [
        "XGBoost",
        "Random Forest",
        "ANN",
        "Naive Bayes",
        "Logistic Regression",
        "KNN Classifier"
    ],
    "ROC-AUC": [
        0.872787,
        0.860000,
        0.859665,
        0.799815,
        0.779524,
        0.762344
    ]
})

st.header("Model Performance Comparison")

fig, ax = plt.subplots(figsize=(11, 6))

colors = sns.color_palette("viridis", len(results))

bars = ax.bar(
    results["Model"],
    results["ROC-AUC"],
    color=colors,
    width=0.65,
    edgecolor="black",
    linewidth=0.7
)

ax.set_ylim(0, 1)
ax.set_ylabel("ROC-AUC Score", fontsize=13, fontweight="bold")
ax.set_xlabel("Machine Learning Models", fontsize=13, fontweight="bold")

ax.set_title(
    "ROC-AUC Score Comparison",
    fontsize=17,
    fontweight="bold",
    pad=20
)

ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=11)

plt.xticks(rotation=15, fontweight="bold")
plt.yticks(fontweight="bold")

ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

for bar, score in zip(bars, results["ROC-AUC"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{score:.3f}",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

sns.despine()
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)


#%%
# Customer details

st.header("Customer Churn Prediction")

with st.form("customer_details"):

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", 350, 850, 650)
        gender = st.selectbox("Gender", le.classes_.tolist())
        age = st.number_input("Age", 18, 100, 40)
        tenure = st.number_input("Tenure", 0, 10, 5)
        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )

    with col2:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=75000.0
        )

        products = st.selectbox("Number of Products", [1, 2, 3, 4])

        credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        active_member = st.selectbox("Active Member?", ["Yes", "No"])

        salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=100000.0
        )

    predict = st.form_submit_button("Predict Churn")


#%%
if predict:

    gender_encoded = le.transform([gender])[0]

    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Gender": gender_encoded,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": 1 if credit_card == "Yes" else 0,
        "IsActiveMember": 1 if active_member == "Yes" else 0,
        "EstimatedSalary": salary,
        "Balance_to_salary": balance / salary if salary > 0 else 0,
        "tenure_by_age": tenure / age,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0
    }])

    customer = customer[feature_names]
    customer_scaled = sc.transform(customer)

    prediction = xgb_model.predict(customer_scaled)[0]
    probability = xgb_model.predict_proba(customer_scaled)[0][1]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    col1.metric("Churn Probability", f"{probability * 100:.2f}%")
    col2.metric(
        "Prediction",
        "Churn" if prediction == 1 else "No Churn"
    )

    st.progress(float(probability))

    if prediction == 1:
        st.error("Customer is likely to leave the bank.")
    else:
        st.success("Customer is likely to stay with the bank.")


#%%
st.divider()

st.caption("Bank Customer Churn Prediction | Machine Learning Project")
