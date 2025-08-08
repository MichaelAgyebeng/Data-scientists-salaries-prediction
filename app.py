import streamlit as st
import joblib
import pandas as pd

# --- Load the trained model ---
try:
    model = joblib.load('best_XGboosting_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run the model training and saving steps first.")
    st.stop()

# --- Load the fitted preprocessor ---
try:
    preprocessor = joblib.load('preprocessor.pkl')
except FileNotFoundError:
    st.error("Preprocessor file not found. Please train and save it first.")
    st.stop()

# --- Streamlit UI ---
st.title('Salary Prediction App')

st.write("""
This application predicts data science salaries based on factors such as experience level, job title, location, and more.
""")

categorical_features = [
    'work_year', 'experience_level', 'job_title', 'employee_residence',
    'employment_type', 'company_location', 'company_size'
]
numerical_features = ['remote_ratio']

# Input fields
work_year = st.selectbox('Work Year', [2020, 2021, 2022])

col1, col2 = st.columns(2)
with col1:
    experience_level = st.selectbox('Experience Level', ['MI', 'SE', 'EN', 'EX'])
with col2:
    employment_type = st.selectbox('Employment Type', ['FT', 'CT', 'PT', 'FL'])

job_title = st.selectbox('Job Title', [
    'Data Scientist', 'Machine Learning Scientist', 'Big Data Engineer', 'Product Data Analyst',
    'Machine Learning Engineer', 'Data Analyst', 'Lead Data Scientist', 'Business Data Analyst',
    'Lead Data Engineer', 'Lead Data Analyst', 'Data Engineer', 'Data Science Consultant',
    'BI Data Analyst', 'Research Scientist', 'Machine Learning Manager', 'Data Engineering Manager',
    'Machine Learning Infrastructure Engineer', 'ML Engineer', 'AI Scientist', 'Computer Vision Engineer',
    'Principal Data Scientist', 'Data Science Manager', 'Head of Data', '3D Computer Vision Researcher',
    'Data Analytics Engineer', 'Applied Data Scientist', 'Director of Data Science', 'Marketing Data Analyst',
    'Cloud Data Engineer', 'Computer Vision Software Engineer', 'Director of Data Engineering',
    'Data Science Engineer', 'Principal Data Engineer', 'Machine Learning Developer',
    'Applied Machine Learning Scientist', 'Data Analytics Manager', 'Head of Data Science',
    'Data Specialist', 'Data Architect', 'Finance Data Analyst', 'Principal Data Analyst',
    'Big Data Architect', 'Staff Data Scientist', 'Analytics Engineer', 'ETL Developer',
    'Head of Machine Learning', 'NLP Engineer', 'Lead Machine Learning Engineer', 'Financial Data Analyst'
])

employee_residence = st.selectbox('Employee Residence', [
    'DE', 'JP', 'GB', 'HN', 'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'CN', 'GR', 'AE', 'NL',
    'MX', 'CA', 'RU', 'ES', 'DZ', 'NG', 'MY', 'TR', 'AU', 'IQ', 'HR', 'IL', 'UA', 'LB',
    'SG', 'SI', 'AT', 'PR', 'RS', 'IE', 'KE', 'SA', 'SK', 'BD', 'CZ', 'JE', 'CH', 'CL',
    'LT', 'MK', 'BO', 'PH', 'KR', 'EE', 'IR', 'CO', 'IT', 'CY'
])

remote_ratio = st.selectbox('Remote Ratio', [0, 50, 100])

col3, col4 = st.columns(2)
with col3:
    company_location = st.selectbox('Company Location', [
        'DE', 'JP', 'GB', 'HN', 'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'CN', 'GR', 'AE', 'NL',
        'MX', 'CA', 'RU', 'ES', 'DZ', 'NG', 'MY', 'TR', 'AU', 'IQ', 'HR', 'IL', 'UA', 'LB',
        'SG', 'SI', 'AT', 'PR', 'RS', 'IE', 'KE', 'SA', 'SK', 'BD', 'CZ', 'JE', 'CH', 'CL',
        'LT', 'MK', 'BO', 'PH', 'KR', 'EE', 'IR', 'CO', 'IT', 'CY'
    ])
with col4:
    company_size = st.selectbox('Company Size', ['S', 'M', 'L'])

# Predict button
if st.button('Predict Salary'):
    # Prepare DataFrame
    input_data = pd.DataFrame([[
        work_year, experience_level, employment_type, job_title,
        employee_residence, remote_ratio, company_location, company_size
    ]], columns=[
        'work_year', 'experience_level', 'employment_type', 'job_title',
        'employee_residence', 'remote_ratio', 'company_location', 'company_size'
    ])

    # Validate inputs
    if input_data.isnull().values.any():
        st.warning("Please fill in all inputs correctly.")
        st.stop()

    # Preprocess input using loaded preprocessor
    input_processed = preprocessor.transform(input_data)

    # Predict
    predicted_salary = model.predict(input_processed)

    st.success(f"Estimated annual salary: ${predicted_salary[0]:,.2f}")
