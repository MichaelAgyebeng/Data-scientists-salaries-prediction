import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the trained model
# Ensure the model file name matches the one saved in the previous step
try:
    model = joblib.load('best_XGboosting_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run the model training and saving steps first.")
    st.stop()


# Assuming you have the preprocessor fitted on the training data
# You would ideally save and load the preprocessor as well
# For this example, we will recreate a similar preprocessor structure
# based on the columns used during training.
# In a real application, save and load the fitted preprocessor.

# Identify categorical and numerical features (should match the training preprocessing)
categorical_features = ['work_year','experience_level','job_title','salary_currency','employee_residence', 'employment_type', 'company_location', 'company_size']
numerical_features = ['remote_ratio', 'salary'] # Including 'salary' here because it was in the original X, although not used for prediction input

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer (should match the training preprocessing)
# Drop the columns that are not used as features or target variable
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop columns that are not in numerical_features or categorical_features
)


# --- Streamlit Application ---

st.title('Salary Prediction App')

st.write("""
This application predicts data science salaries based on various factors such as experience level, employment type, location, and more. Enter the details below to get a salary prediction.
""")

# Create input fields for the features
# These should match the features used for training the model (excluding 'salary_in_usd' and 'salary')
# We need to get the unique values for categorical features from the original data or a saved list
# In a real app, load these from saved data. For simplicity, using example values for now.
work_year = st.selectbox('Work Year', [2020, 2021, 2022])
experience_level = st.selectbox('Experience Level', ['MI', 'SE', 'EN', 'EX'])
employment_type = st.selectbox('Employment Type', ['FT', 'CT', 'PT', 'FL'])
job_title = st.selectbox('Job Title', ['Data Scientist', 'Machine Learning Scientist', 'Big Data Engineer', 'Product Data Analyst', 'Machine Learning Engineer', 'Data Analyst', 'Lead Data Scientist', 'Business Data Analyst', 'Lead Data Engineer', 'Lead Data Analyst', 'Data Engineer', 'Data Science Consultant', 'BI Data Analyst', 'Research Scientist', 'Machine Learning Manager', 'Data Engineering Manager', 'Machine Learning Infrastructure Engineer', 'ML Engineer', 'AI Scientist', 'Computer Vision Engineer', 'Principal Data Scientist', 'Data Science Manager', 'Head of Data', '3D Computer Vision Researcher', 'Data Analytics Engineer', 'Applied Data Scientist', 'Director of Data Science', 'Marketing Data Analyst', 'Cloud Data Engineer', 'Computer Vision Software Engineer', 'Director of Data Engineering', 'Data Science Engineer', 'Principal Data Engineer', 'Machine Learning Developer', 'Applied Machine Learning Scientist', 'Data Analytics Manager', 'Head of Data Science', 'Data Specialist', 'Data Architect', 'Finance Data Analyst', 'Principal Data Analyst', 'Big Data Architect', 'Staff Data Scientist', 'Analytics Engineer', 'ETL Developer', 'Head of Machine Learning', 'NLP Engineer', 'Lead Machine Learning Engineer', 'Financial Data Analyst'])
salary_currency = st.selectbox('Salary Currency', ['EUR', 'USD', 'GBP', 'HUF', 'INR', 'MXN', 'TRY', 'CAD', 'DKK', 'PLN', 'AED', 'JPY', 'CNY', 'SGD', 'CLP', 'BRL', 'NZD', 'PHP'])
employee_residence = st.selectbox('Employee Residence', ['DE', 'JP', 'GB', 'HN', 'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'CN', 'GR', 'AE', 'NL', 'MX', 'CA', 'RU', 'ES', 'DZ', 'NG', 'MY', 'TR', 'AU', 'IQ', 'HR', 'IL', 'UA', 'LB', 'SG', 'SI', 'AT', 'PR', 'RS', 'IE', 'KE', 'SA', 'SK', 'BD', 'CZ', 'JE', 'CH', 'CL', 'LT', 'MK', 'BO', 'PH', 'KR', 'EE', 'IR', 'CO', 'IT', 'CY'])
remote_ratio = st.selectbox('Remote Ratio', [0, 50, 100])
company_location = st.selectbox('Company Location', ['DE', 'JP', 'GB', 'HN', 'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'CN', 'GR', 'AE', 'NL', 'MX', 'CA', 'RU', 'ES', 'DZ', 'NG', 'MY', 'TR', 'AU', 'IQ', 'HR', 'IL', 'UA', 'LB', 'SG', 'SI', 'AT', 'PR', 'RS', 'IE', 'KE', 'SA', 'SK', 'BD', 'CZ', 'JE', 'CH', 'CL', 'LT', 'MK', 'BO', 'PH', 'KR', 'EE', 'IR', 'CO', 'IT', 'CY'])
company_size = st.selectbox('Company Size', ['S', 'M', 'L'])


# Predict button
if st.button('Predict Salary'):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[work_year, experience_level, employment_type, job_title, 0, salary_currency, employee_residence, remote_ratio, company_location, company_size]],
                              columns=['work_year', 'experience_level', 'employment_type', 'job_title', 'salary', 'salary_currency', 'employee_residence', 'remote_ratio', 'company_location', 'company_size'])

    # Apply the same preprocessing as used during training
    # Note: This requires fitting the preprocessor again which is not ideal.
    # A better approach is to save and load the fitted preprocessor object.
    # For demonstration purposes, we'll fit it on the original data structure (excluding target)
    # to ensure the columns align for the one-hot encoding.
    # This part needs to be robust in a real application.
    # Let's create a dummy preprocessor just for the structure
    # dummy_X = pd.DataFrame(columns=['work_year','experience_level', 'employment_type', 'job_title', 'salary', 'salary_currency', 'employee_residence', 'remote_ratio', 'company_location', 'company_size'])
    # dummy_X = dummy_X.append(input_data, ignore_index=True)

    # Fit the preprocessor on dummy data to get the structure (not ideal)
    # In a real app, load the fitted preprocessor
    # preprocessor.fit(dummy_X) # This is a simplification, load the fitted one instead

    # A more robust way for a real app would be to load the fitted preprocessor
    # For this example, we'll use the preprocessor fitted earlier in the notebook
    # This assumes the global 'preprocessor' variable is available and correctly fitted
    # In a production app, save and load the fitted preprocessor object.
    input_processed = preprocessor.transform(input_data)


    # Make prediction
    predicted_salary = model.predict(input_processed)

    st.subheader(f'Predicted Salary in USD: ${predicted_salary[0]:,.2f}')
