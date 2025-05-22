import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model and scaler
@st.cache_resource
def load_model():
    model = pickle.load(open('xgboost_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_model()

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #2a9d8f !important;
        margin-bottom: 20px !important;
    }
    .subheader {
        font-size: 20px !important;
        color: #264653 !important;
        margin-bottom: 15px !important;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2a9d8f;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #21867a;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0px;
        background-color: #f0f2f6;
    }
    .high-risk {
        color: #e63946;
        font-weight: bold;
    }
    .low-risk {
        color: #2a9d8f;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921224.png", width=100)
    st.markdown("## About")
    st.markdown("""
    This app predicts the risk of diabetes based on health metrics and lifestyle factors.
    The model was trained using XGBoost on clinical data.
    """)
    st.markdown("---")
    st.markdown("### How to use:")
    st.markdown("1. Fill in your health information")
    st.markdown("2. Click 'Predict Diabetes Risk'")
    st.markdown("3. View your results")
    st.markdown("---")
    st.markdown("Made with ❤️ by HealthAI")

# Main content
st.markdown('<div class="header">Diabetes Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Assess your risk of developing diabetes</div>', unsafe_allow_html=True)

# Create form for user input
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Health Metrics")
        pregnancies = st.slider("Number of Pregnancies", 0, 17, 0)
        glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 100)
        blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 122, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)
        insulin = st.slider("Insulin Level (μU/mL)", 0, 846, 80)
        bmi = st.slider("BMI", 0.0, 67.1, 25.0)
        diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.slider("Age", 21, 81, 30)
        
    with col2:
        st.markdown("### Lifestyle Factors")
        age_group = st.selectbox("Age Group", ["20-29", "30-39", "40-49", "50-59", "60+"])
        bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
        glucose_status = st.selectbox("Glucose Status", ["Normal", "Prediabetic", "Diabetic"])
        
        st.markdown("---")
        st.markdown("### Additional Information")
        activity_level = st.selectbox("Physical Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
        family_history = st.selectbox("Family History of Diabetes", ["No", "Yes (Parents)", "Yes (Siblings)", "Yes (Both)"])
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    
    submitted = st.form_submit_button("Predict Diabetes Risk")

# Process the form when submitted
if submitted:
    # Create DataFrame from inputs
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age,
    }
    
    # Add one-hot encoded features
    age_groups = ["20-29", "30-39", "40-49", "50-59", "60+"]
    bmi_categories = ["Underweight", "Normal", "Overweight", "Obese"]
    glucose_statuses = ["Normal", "Prediabetic", "Diabetic"]
    
    for group in age_groups:
        input_data[f'AgeGroup_{group}'] = 1 if age_group == group else 0
        
    for category in bmi_categories:
        input_data[f'BMI_Category_{category}'] = 1 if bmi_category == category else 0
        
    for status in glucose_statuses:
        input_data[f'Glucose_Status_{status}'] = 1 if glucose_status == status else 0
    
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Reorder columns to match training data
    feature_order = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'AgeGroup_20-29', 'AgeGroup_30-39',
        'AgeGroup_40-49', 'AgeGroup_50-59', 'AgeGroup_60+', 'BMI_Category_Underweight',
        'BMI_Category_Normal', 'BMI_Category_Overweight', 'BMI_Category_Obese',
        'Glucose_Status_Normal', 'Glucose_Status_Prediabetic', 'Glucose_Status_Diabetic'
    ]
    
    # Ensure all columns are present (fill missing with 0)
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_order]
    
    # Scale the data
    scaled_data = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)
    
    # Display results
    st.markdown("---")
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("### Prediction Results")
    
    risk_percentage = prediction_proba[0][1] * 100
    
    if prediction[0] == 1:
        st.markdown(f'<p style="font-size:24px">Risk of Diabetes: <span class="high-risk">HIGH RISK ({risk_percentage:.1f}%)</span></p>', 
                    unsafe_allow_html=True)
        
        st.warning("Our model indicates you may be at higher risk for diabetes. Please consult with a healthcare professional for further evaluation and guidance.")
        
        st.markdown("### Recommendations:")
        st.markdown("- Schedule a check-up with your doctor")
        st.markdown("- Monitor your blood sugar levels regularly")
        st.markdown("- Consider dietary changes to reduce sugar intake")
        st.markdown("- Increase physical activity")
        st.markdown("- Maintain a healthy weight")
    else:
        st.markdown(f'<p style="font-size:24px">Risk of Diabetes: <span class="low-risk">LOW RISK ({risk_percentage:.1f}%)</span></p>', 
                    unsafe_allow_html=True)
        
        st.success("Our model indicates you have a lower risk of diabetes. Continue maintaining healthy habits!")
        
        st.markdown("### Prevention Tips:")
        st.markdown("- Maintain a balanced diet with plenty of vegetables")
        st.markdown("- Engage in regular physical activity")
        st.markdown("- Get regular health check-ups")
        st.markdown("- Avoid excessive sugar consumption")
        st.markdown("- Manage stress effectively")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show feature importance (optional)
    st.markdown("### Key Factors in Your Assessment:")
    important_features = {
        'Glucose Level': glucose,
        'BMI': bmi,
        'Age': age,
        'Diabetes Pedigree Function': diabetes_pedigree,
        'Blood Pressure': blood_pressure
    }
    
    st.table(pd.DataFrame.from_dict(important_features, orient='index', columns=['Your Value']))
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <small>**Disclaimer:** This prediction is for informational purposes only and should not be considered medical advice. 
    Always consult with a qualified healthcare professional for medical concerns.</small>
    """, unsafe_allow_html=True)
