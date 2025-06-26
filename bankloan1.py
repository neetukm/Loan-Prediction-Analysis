import streamlit as st
import pandas as pd
import joblib
import pickle
# # Load data and model
# df = pd.read_csv("cl_train_loan_eligibility.csv")  # Your dataset
# model = joblib.load("rf_model.pkl")



# # Page Config (Must be First Streamlit Command)
# st.set_page_config(page_title="Loan Eligibility App", layout="wide")

# # Load Data
# @st.cache_data
# def load_data():
#     return pd.read_csv('cl_train_loan_eligibility.csv')  # Update path if local testing

# # Load Pre-trained Model
# @st.cache_resource
# def load_model():
#     with open('rf_model.pkl', 'rb') as f:   # Update path if local testing
#         return pickle.load(f)

# data = load_data()
# model = load_model()

# # st.set_page_config(page_title="Loan Eligibility App", layout="centered")
# # st.title("🏦 Loan Prediction Analysis")

# # Step 1: Filter by Loan_ID
# loan_ids = df['Loan_ID'].unique().tolist()
# selected_id = st.selectbox("🔍 Select Loan_ID to prefill data", options=loan_ids)

# # Step 2: Auto-fill details for selected Loan_ID
# selected_row = df[df['Loan_ID'] == selected_id].iloc[0]

# # Step 3: UI Inputs (auto-filled)
# st.header("👤 Applicant Information")

# gender = st.selectbox("Gender", ["Male", "Female"], index=0 if selected_row['Gender'] == "Male" else 1)
# married = st.selectbox("Married", ["Yes", "No"], index=0 if selected_row['Married'] == "Yes" else 1)
# dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=["0", "1", "2", "3+"].index(str(selected_row['Dependents'])))
# education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0 if selected_row['Education'] == "Graduate" else 1)
# self_employed = st.selectbox("Self Employed", ["Yes", "No"], index=0 if selected_row['Self_Employed'] == "Yes" else 1)
# property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=["Urban", "Semiurban", "Rural"].index(selected_row['Property_Area']))
# credit_history = st.selectbox("Credit History", ["Yes", "No"], index=0 if selected_row['Credit_History'] == 1 else 1)

# st.subheader("💼 Applicant Income Details")
# applicant_income = st.number_input("Applicant Income", min_value=0, value=int(selected_row['ApplicantIncome']))
# coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=int(selected_row['CoapplicantIncome']))
# loan_amount = st.number_input("Loan Amount", min_value=0, value=int(selected_row['LoanAmount']))
# loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0, value=int(selected_row['Loan_Amount_Term']))

# # Encode values
# gender = 1 if gender == "Male" else 0
# married = 1 if married == "Yes" else 0
# education = 1 if education == "Graduate" else 0
# self_employed = 1 if self_employed == "Yes" else 0
# credit_history = 1.0 if credit_history == "Yes" else 0.0
# property_area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
# property_area = property_area_map[property_area]
# dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
# dependents = dependents_map[dependents]

# # Total Income
# total_income = applicant_income + coapplicant_income

# # Input Data
# input_data = pd.DataFrame({
#     'Gender': [gender],
#     'Married': [married],
#     'Dependents': [dependents],
#     'Education': [education],
#     'Self_Employed': [self_employed],
#     'ApplicantIncome': [applicant_income],
#     'CoapplicantIncome': [coapplicant_income],
#     'LoanAmount': [loan_amount],
#     'Loan_Amount_Term': [loan_amount_term],
#     'Credit_History': [credit_history],
#     'Property_Area': [property_area],
#     'TotalIncome': [total_income]
# })

# # Predict
# if st.button("🚀 Predict Loan Eligibility"):
#     prediction = model.predict(input_data)
#     result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"
#     st.success(f"Loan Status Prediction for {selected_id}: {result}")



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

# Configure Streamlit page
st.set_page_config(page_title="Loan Eligibility App", layout="centered")
st.title("🏦 Loan Prediction Analysis")

# Select Loan_ID and store in session_state
loan_ids = df['Loan_ID'].unique().tolist()
selected_id = st.selectbox("🔍 Select Loan_ID to prefill data", options=loan_ids)

# Get selected row based on Loan_ID
selected_row = df[df['Loan_ID'] == selected_id].iloc[0]

# Initialize session_state to store values
if 'form_updated' not in st.session_state or st.session_state.selected_id != selected_id:
    st.session_state.selected_id = selected_id
    st.session_state.gender = "Male" if selected_row['Gender'] == "Male" else "Female"
    st.session_state.married = "Yes" if selected_row['Married'] == "Yes" else "No"
    st.session_state.dependents = str(selected_row['Dependents'])
    st.session_state.education = "Graduate" if selected_row['Education'] == "Graduate" else "Not Graduate"
    st.session_state.self_employed = "Yes" if selected_row['Self_Employed'] == "Yes" else "No"
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

# Encode categorical values for prediction
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1.0 if credit_history == "Yes" else 0.0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
dependents = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
total_income = applicant_income + coapplicant_income

# Prepare input data for model
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
