import os
import pickle # pre trained model loading
import streamlit as st    # web app
from streamlit_option_menu import option_menu

st.set_page_config(page_title='Prediction of Disease Outbreaks',
                   layout='wide',
                   page_icon="🧑‍⚕")
diabetes_model= pickle.load(open(r"training_models\diabetes_model.sav",'rb'))
heart_disease_model=pickle.load(open(r"training_models\heart_model.sav",'rb'))
parkinsons_model= pickle.load(open(r"training_models\parkinsons_model.sav",'rb'))

with st.sidebar:
    selected= option_menu('Prediction of disease outbreak system',
                          ['Diabetes Prediction','Heart Disease Prediction','Parkinsons prediction'],
                          menu_icon='hospital-fill',icons=['activity','heart','person'],default_index=0)

if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using Ml')
    col1,col2,col3 = st.columns(3)
    with col1:
        Pregnancies= st.text_input('Number of Pregnancies')
    with col2:
        Glucose= st.text_input('Glucose level')
    with col3:
        Bloodpressure= st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin= st.text_input('Insulin level')
    with col3:
        BMI = st.text_input('BMI  value')
    with col1:
        DiabetesPedigreeFunction= st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age= st.text_input('Age of the person')

diab_diagnosis = ''
if st.button('Diabetes Test Result'):
    user_input=[Pregnancies, Glucose, Bloodpressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
    user_input= [float(x) for x in user_input]
    diab_prediction= diabetes_model.predict([user_input])
    if diab_prediction[0]==1:
        diab_diagnosis= 'The person is diabetic'
    else:
        diab_diagnosis= 'The person is not diabetic'
st.success(diab_diagnosis)

if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    
    # Creating three columns for the input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age of the person')
    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'])  # Assuming male=1, female=0
    with col3:
        cp = st.selectbox('Chest Pain type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])  # You can map these to integers later
    with col1:
        trestbps = st.text_input('Resting Blood Pressure (trestbps)')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])  # Map Yes to 1, No to 0
    with col1:
        restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])  # You can map these later
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])  # Map Yes to 1, No to 0
    with col1:
        oldpeak = st.text_input('Depression Induced by Exercise Relative to Rest')
    with col2:
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Up Sloping', 'Flat', 'Down Sloping'])  # You can map these later
    with col3:
        ca = st.selectbox('Number of Major Vessels Colored by Flourosopy', ['0', '1', '2', '3'])
    with col1:
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])  # You can map these later
    
    # Placeholder for prediction result
    heart_disease_diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        # Convert the user input to the correct format
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        
        # Data preprocessing for conversion to numeric values
        user_input = [float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x for x in user_input]
        
        # Convert categorical variables to numeric
        user_input[1] = 1 if user_input[1] == 'Male' else 0  # Convert Sex to 1/0 (Male/Female)
        user_input[2] = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}.get(user_input[2], 0)  # Map chest pain type
        user_input[5] = 1 if user_input[5] == 'Yes' else 0  # Fasting blood sugar (Yes=1, No=0)
        user_input[6] = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}.get(user_input[6], 0)  # Map restecg
        user_input[8] = 1 if user_input[8] == 'Yes' else 0  # Exercise induced angina (Yes=1, No=0)
        user_input[10] = {'Up Sloping': 0, 'Flat': 1, 'Down Sloping': 2}.get(user_input[10], 0)  # Map slope
        user_input[12] = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}.get(user_input[12], 0)  # Map thalassemia

        # Make the prediction
        heart_disease_prediction = heart_disease_model.predict([user_input])
        
        if heart_disease_prediction[0] == 1:
            heart_disease_diagnosis = 'The person is likely to have heart disease.'
        else:
            heart_disease_diagnosis = 'The person is not likely to have heart disease.'
        
        st.success(heart_disease_diagnosis)



# Load the trained model
with open("training_models/parkinsons_model.sav", "rb") as file:
    parkinsons_model = pickle.load(file)
if selected == 'Parkinsons prediction':
    st.title("Parkinson’s Disease Prediction using ML")

    # Ensure all variables are always defined
    MDVP_Fo = MDVP_Fhi = MDVP_Flo = MDVP_Jitter_percent = MDVP_Jitter_Abs = ""
    MDVP_RAP = MDVP_PPQ = Jitter_DDP = MDVP_Shim = MDVP_Shim_dB = ""
    Shimmer_APQ3 = Shimmer_APQ5 = MDVP_APQ = Shimmer_DDA = ""
    NHR = HNR = status = RPDE = DFA = spread1 = spread2 = D2 = PPE = ""

    # Check if user selected Parkinson's Prediction
    selected = st.sidebar.selectbox("Choose Prediction Type", ["Parkinsons prediction", "Other"])
    if selected == "Parkinsons prediction":
        # Creating three columns for input fields
        col1, col2, col3 = st.columns(3)

        with col1:
            name = st.text_input('Name of the person')
        with col2:
            MDVP_Fo = st.text_input('MDVP:Fo(Hz)')
        with col3:
            MDVP_Fhi = st.text_input('MDVP:Fhi(Hz)')
        with col1:
            MDVP_Flo = st.text_input('MDVP:Flo(Hz)')
        with col2:
            MDVP_Jitter_percent = st.text_input('MDVP:Jitter(%)')
        with col3:
            MDVP_Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        with col1:
            MDVP_RAP = st.text_input('MDVP:RAP')
        with col2:
            MDVP_PPQ = st.text_input('MDVP:PPQ')
        with col3:
            Jitter_DDP = st.text_input('Jitter:DDP')
        with col1:
            MDVP_Shim = st.text_input('MDVP:Shimmer')
        with col2:
            MDVP_Shim_dB = st.text_input('MDVP:Shimmer(dB)')
        with col3:
            Shimmer_APQ3 = st.text_input('Shimmer:APQ3')
        with col1:
            Shimmer_APQ5 = st.text_input('Shimmer:APQ5')
        with col2:
            MDVP_APQ = st.text_input('MDVP:APQ')
        with col3:
            Shimmer_DDA = st.text_input('Shimmer:DDA')
        with col1:
            NHR = st.text_input('NHR')
        with col2:
            HNR = st.text_input('HNR')
        with col3:
            status = st.text_input('Status')
        with col1:
            RPDE = st.text_input('RPDE')
        with col2:
            DFA = st.text_input('DFA')
        with col3:
            spread1 = st.text_input('spread1')
        with col1:
            spread2 = st.text_input('spread2')
        with col2:
            D2 = st.text_input('D2')
        with col3:
            PPE = st.text_input('PPE')

        # Placeholder for the prediction result
        parkinsons_diagnosis = ""

        if st.button("Parkinson’s Test Result"):
            try:
                # Convert inputs to float safely
                user_input = [
                    float(MDVP_Fo), float(MDVP_Fhi), float(MDVP_Flo), float(MDVP_Jitter_percent),
                    float(MDVP_Jitter_Abs), float(MDVP_RAP), float(MDVP_PPQ), float(Jitter_DDP),
                    float(MDVP_Shim), float(MDVP_Shim_dB), float(Shimmer_APQ3), float(Shimmer_APQ5),
                    float(MDVP_APQ), float(Shimmer_DDA), float(NHR), float(HNR), float(status),
                    float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)
                ]
                
                # Predict the result using the trained model
                parkinsons_prediction = parkinsons_model.predict([user_input])

                if parkinsons_prediction[0] == 1:
                    parkinsons_diagnosis = "The person has Parkinson’s disease."
                else:
                    parkinsons_diagnosis = "The person does not have Parkinson’s disease."

                st.success(parkinsons_diagnosis)

            except ValueError:
                st.error("Please enter valid numerical values for all input fields.")
