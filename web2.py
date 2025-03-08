import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title='Prediction of Disease Outbreaks',
                   layout='wide',
                   page_icon="🧑‍⚕️")

diabetes_model = pickle.load(open(r"C:\Users\mahis\Desktop\predictions\training_models\diabetes_model.sav", 'rb'))
heart_disease_model = pickle.load(open(r"C:\Users\mahis\Desktop\predictions\training_models\heart_model.sav", 'rb'))
parkinsons_model = pickle.load(open(r"C:\Users\mahis\Desktop\predictions\training_models\parkinsons_model.sav", 'rb'))

with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreak System',
                          ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Disease Prediction'],
                          menu_icon='hospital-fill', icons=['activity', 'heart', 'person'], default_index=0)

if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI Value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    with col2:
        Age = st.text_input('Age of the Person')
    
    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        try:
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin),
                          float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            diab_prediction = diabetes_model.predict([user_input])
            diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
            st.success(diab_diagnosis)
        except ValueError:
            st.error("Please enter valid numerical values for all input fields.")

if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age of the Person')
    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'])
    with col3:
        cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholesterol in mg/dl')
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
    with col1:
        restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    with col1:
        oldpeak = st.text_input('ST Depression Induced by Exercise')
    with col2:
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Up Sloping', 'Flat', 'Down Sloping'])
    with col3:
        ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', ['0', '1', '2', '3'])
    with col1:
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    
    heart_disease_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(age), 1 if sex == 'Male' else 0, {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}[cp],
                          float(trestbps), float(chol), 1 if fbs == 'Yes' else 0, {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}[restecg],
                          float(thalach), 1 if exang == 'Yes' else 0, float(oldpeak), {'Up Sloping': 0, 'Flat': 1, 'Down Sloping': 2}[slope], int(ca), {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}[thal]]
            heart_disease_prediction = heart_disease_model.predict([user_input])
            heart_disease_diagnosis = 'The person is likely to have heart disease.' if heart_disease_prediction[0] == 1 else 'The person is not likely to have heart disease.'
            st.success(heart_disease_diagnosis)
        except ValueError:
            st.error("Please enter valid numerical values for all input fields.")

if selected == 'Parkinsons Disease Prediction':
    st.title('Parkinson’s Disease Prediction using ML')
    input_fields = [st.text_input(label) for label in ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'Status', 'RPDE', 'DFA', 'Spread1', 'Spread2', 'D2', 'PPE']]
    
    parkinsons_diagnosis = ''
    if st.button('Parkinson’s Test Result'):
        try:
            user_input = [float(value) for value in input_fields]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            parkinsons_diagnosis = 'The person has Parkinson’s disease.' if parkinsons_prediction[0] == 1 else 'The person does not have Parkinson’s disease.'
            st.success(parkinsons_diagnosis)
        except ValueError:
            st.error("Please enter valid numerical values for all input fields.")
