# Load preprocessor and model
import joblib
import pandas as pd
import streamlit as st

preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

# These must match the order used when training
categorical_features = ['work_year', 'experience_level', 'employment_type',
                        'job_title', 'employee_residence', 'company_location',
                        'company_size']
numeric_features = ['remote_ratio']

# --- Helper: check missing fields ---
def check_missing_fields(input_df):
    missing = [col for col in input_df.columns if pd.isnull(input_df[col].iloc[0]) or input_df[col].iloc[0] == ""]
    return missing

# --- Helper: check unseen categories ---
def check_unseen_categories(input_df):
    unseen_messages = []
    cat_transformer = preprocessor.named_transformers_['cat']  # The categorical pipeline
    ohe = cat_transformer.named_steps['onehot']                # The OneHotEncoder
    
    for col, categories in zip(categorical_features, ohe.categories_):
        value = input_df[col].iloc[0]
        if value not in categories:
            unseen_messages.append(f"**{col}** → '{value}' (not seen during training)")
    return unseen_messages

# --- Streamlit UI ---
st.title("Data Scientist Salary Prediction")

# Collect user inputs
user_inputs = {
    'work_year': st.selectbox('Work Year', ['2020', '2021', '2022', '2023']),
    'experience_level': st.selectbox('Experience Level', ['EN', 'MI', 'SE', 'EX']),
    'employment_type': st.selectbox('Employment Type', ['FT', 'PT', 'CT', 'FL']),
    'job_title': st.text_input('Job Title'),
    'employee_residence': st.text_input('Employee Residence (e.g., US, GB, IN)'),
    'remote_ratio': st.number_input('Remote Ratio (%)', min_value=0, max_value=100, step=1),
    'company_location': st.text_input('Company Location (e.g., US, GB, IN)'),
    'company_size': st.selectbox('Company Size', ['S', 'M', 'L'])
}

input_data = pd.DataFrame([user_inputs])

if st.button("Predict Salary"):
    # 1. Check for missing fields
    missing = check_missing_fields(input_data)
    if missing:
        st.warning("The following fields are missing:\n" + "\n".join(missing))
    else:
        # 2. Check for unseen categories
        unseen = check_unseen_categories(input_data)
        if unseen:
            st.warning("The following inputs were not seen during training:\n" + "\n".join(unseen))
        
        # 3. Transform & predict
        input_processed = preprocessor.transform(input_data)
        prediction = model.predict(input_processed)
        st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
