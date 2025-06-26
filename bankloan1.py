
import streamlit as st
import pandas as pd
import joblib
import os

# Function to load data safely
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"❌ File not found: {file_path}")
        st.stop()

# Function to load model safely
def load_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()

# Load dataset and model
df = load_data("cl_train_loan_eligibility.csv")
model = load_model("rf_model.pkl")

# Set Streamlit app settings
st.set_page_config(page_title="Loan Eligibility App", layout="centered")
st.title("🏦 Loan Prediction Analysis")

# Select Loan_ID
loan_ids = df['Loan_ID'].unique().tolist()
selected_id = st.selectbox("🔍 Select Loan_ID to prefill data", options=loan_ids)

# Get selected row
selected_row = df[df['Loan_ID'] == selected_id].iloc[0]

# Update session_state on Loan_ID change
if "last_selected_id" not in st.session_state or st.session_state.last_selected_id != selected_id:
    st.session_state.last_selected_id = selected_id
    st.session_state.gender = selected_row['Gender']
    st.session_state.married = selected_row['Married']
    st.session_state.dependents = str(selected_row['Dependents'])
    st.session_state.education = selected_row['Education']
    st.session_state.self_employed = selected_row['Self_Employed']
    st.session_state.property_area = selected_row['Property_Area']
    st.session_state.credit_history = "Yes" if selected_row['Credit_History'] == 1 else "No"
    st.session_state.applicant_income = int(selected_row['ApplicantIncome'])
    st.session_state.coapplicant_income = int(selected_row['CoapplicantIncome'])
    st.session_state.loan_amount = int(selected_row['LoanAmount'])
    st.session_state.loan_amount_term = int(selected_row['Loan_Amount_Term'])

# Applicant Information Form
st.header("👤 Applicant Information")
gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.gender == "Male" else 1)
married = st.selectbox("Married", ["Yes", "No"], index=0 if st.session_state.married == "Yes" else 1)
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=["0", "1", "2", "3+"].index(st.session_state.dependents))
education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0 if st.session_state.education == "Graduate" else 1)
self_employed = st.selectbox("Self Employed", ["Yes", "No"], index=0 if st.session_state.self_employed == "Yes" else 1)
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=["Urban", "Semiurban", "Rural"].index(st.session_state.property_area))
credit_history = st.selectbox("Credit History", ["Yes", "No"], index=0 if st.session_state.credit_history == "Yes" else 1)

# Income Details
st.subheader("💼 Applicant Income Details")
applicant_income = st.number_input("Applicant Income", min_value=0, value=st.session_state.applicant_income)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=st.session_state.coapplicant_income)
loan_amount = st.number_input("Loan Amount", min_value=0, value=st.session_state.loan_amount)
loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0, value=st.session_state.loan_amount_term)

# Encode categorical values for model
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1.0 if credit_history == "Yes" else 0.0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
dependents = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
total_income = applicant_income + coapplicant_income

# Prepare input dataframe
input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area],
    'TotalIncome': [total_income]
})

# Prediction Button
if st.button("🚀 Predict Loan Eligibility"):
    prediction = model.predict(input_data)
    result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"
    st.success(f"Loan Status Prediction for {selected_id}: {result}")
