import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
@st.cache_resource
def load_model():
  model = xgb.XGBRegressor()
  model.load_model('xgboost_model.json') # Load model from the saved JSON
  return model

model = load_model()

# Streamlit interface
st.title("Hospital Length of Stay Prediction")

# Sidebar user input sliders and checkboxes for features
st.sidebar.header('User Input Parameters')

def user_input_features():
  ADMISSION_DEPOSIT = st.sidebar.slider('Admission Deposit', 1000, 10000, 5000)
  AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL = st.sidebar.slider('Available Extra Rooms', 0, 10, 3)
  VISITORS_WITH_PATIENT = st.sidebar.slider('Visitors With Patient', 0, 10, 3)
  
  WARD_TYPE_Q = st.sidebar.selectbox('Ward Type (Q)', ['Yes', 'No']) == 'Yes'
  WARD_TYPE_P = st.sidebar.selectbox('Ward Type (P)', ['Yes', 'No']) == 'Yes'
  WARD_TYPE_S = st.sidebar.selectbox('Ward Type (S)', ['Yes', 'No']) == 'Yes'
  
  AGE_41_50 = st.sidebar.selectbox('Age Group (41-50)', ['Yes', 'No']) == 'Yes'
  AGE_61_70 = st.sidebar.selectbox('Age Group (61-70)', ['Yes', 'No']) == 'Yes'
  
  SEVERITY_OF_ILLNESS_Extreme = st.sidebar.selectbox('Severity of Illness (Extreme)', ['Yes', 'No']) == 'Yes'
  SEVERITY_OF_ILLNESS_Minor = st.sidebar.selectbox('Severity of Illness (Minor)', ['Yes', 'No']) == 'Yes'
  SEVERITY_OF_ILLNESS_Moderate = st.sidebar.selectbox('Severity of Illness (Moderate)', ['Yes', 'No']) == 'Yes'
  
  TYPE_OF_ADMISSION_Trauma = st.sidebar.selectbox('Type of Admission (Trauma)', ['Yes', 'No']) == 'Yes'
  TYPE_OF_ADMISSION_Emergency = st.sidebar.selectbox('Type of Admission (Emergency)', ['Yes', 'No']) == 'Yes'
  
  CITY_CODE_HOSPITAL_7 = st.sidebar.selectbox('City Code Hospital 7', ['Yes', 'No']) == 'Yes'
  CITY_CODE_HOSPITAL_2 = st.sidebar.selectbox('City Code Hospital 2', ['Yes', 'No']) == 'Yes'
  
  BED_GRADE_2 = st.sidebar.selectbox('Bed Grade (2)', ['Yes', 'No']) == 'Yes'
  CITY_CODE_PATIENT_10 = st.sidebar.selectbox('City Code Patient (10)', ['Yes', 'No']) == 'Yes'
  
  # Create a dictionary for the user inputs and default values for missing features
  data = {
  'ADMISSION_DEPOSIT': ADMISSION_DEPOSIT,
  'AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL': AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL,
  'VISITORS_WITH_PATIENT': VISITORS_WITH_PATIENT,
  'WARD_TYPE_Q': 1 if WARD_TYPE_Q else 0,
  'WARD_TYPE_P': 1 if WARD_TYPE_P else 0,
  'WARD_TYPE_S': 1 if WARD_TYPE_S else 0,
  'AGE_41-50': 1 if AGE_41_50 else 0,
  'AGE_61-70': 1 if AGE_61_70 else 0,
  'SEVERITY_OF_ILLNESS_Extreme': 1 if SEVERITY_OF_ILLNESS_Extreme else 0,
  'SEVERITY_OF_ILLNESS_Minor': 1 if SEVERITY_OF_ILLNESS_Minor else 0,
  'SEVERITY_OF_ILLNESS_Moderate': 1 if SEVERITY_OF_ILLNESS_Moderate else 0,
  'TYPE_OF_ADMISSION_Trauma': 1 if TYPE_OF_ADMISSION_Trauma else 0,
  'TYPE_OF_ADMISSION_Emergency': 1 if TYPE_OF_ADMISSION_Emergency else 0,
  'CITY_CODE_HOSPITAL_7': 1 if CITY_CODE_HOSPITAL_7 else 0,
  'CITY_CODE_HOSPITAL_2': 1 if CITY_CODE_HOSPITAL_2 else 0,
  'BED_GRADE_2.0': 1 if BED_GRADE_2 else 0,
  'CITY_CODE_PATIENT_10.0': 1 if CITY_CODE_PATIENT_10 else 0,
  'AGE_21-30': 0, # Default values for columns not used by user
  'AGE_31-40': 0,
  'AGE_71-80': 0
  }
  
  # Convert the dictionary to a DataFrame
  features = pd.DataFrame(data, index=[0])
  return features
  
# Collect input data
input_df = user_input_features()

# Display input data
st.subheader('Input Data')
st.write(input_df)
