import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import hashlib
import base64
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Maternal Health Risk Prediction",
    page_icon="👩‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498db;
        margin-bottom: 1rem;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 5px;
        background-color: rgba(231, 76, 60, 0.1);
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 5px;
        background-color: rgba(243, 156, 18, 0.1);
    }
    .risk-low {
        color: #2ecc71;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 5px;
        background-color: rgba(46, 204, 113, 0.1);
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .logout-btn {
        position: absolute;
        top: 0.5rem;
        right: 1rem;
        z-index: 999;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #7f8c8d;
        font-size: 0.8rem;
    }
    .landing-page {
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }
    .landing-title {
        font-size: 3rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .landing-subtitle {
        font-size: 1.5rem;
        color: #3498db;
        margin-bottom: 2rem;
    }
    .team-section {
        margin: 2rem 0;
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .team-member {
        margin-bottom: 0.5rem;
    }
    .college-section {
        margin: 2rem 0;
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .enter-app-btn {
        margin-top: 2rem;
    }
    .enter-app-btn button {
        padding: 0.75rem 2rem !important;
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Create directory for models if it doesn't exist
os.makedirs("models", exist_ok=True)

# Function to load model files
@st.cache_resource
def load_model_files():
    # For demonstration, we'll create dummy model files if they don't exist
    if not os.path.exists("models/random_forest_model.pkl"):
        # Create a dummy Random Forest model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        feature_names = [
            'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate',
            'Hemoglobin_Level', 'Thyroid_Condition', 'BP_Ratio', 'Risk_Score',
            'Age_BP_Interaction', 'BS_HeartRate_Interaction',
            'Hemoglobin_Group_Low', 'Hemoglobin_Group_Medium', 
            'Hemoglobin_Group_High', 'Hemoglobin_Group_Very High'
        ]
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array([0, 1, 2])  # Low Risk, Mid Risk, High Risk
        
        # Save dummy model files
        with open("models/random_forest_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("models/feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)
        with open("models/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
    
    # Load the model files
    with open("models/random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    return model, feature_names, label_encoder

# Load model files
model, feature_names, label_encoder = load_model_files()

# Create or load database
def load_database():
    if os.path.exists("database.csv"):
        return pd.read_csv("database.csv")
    else:
        # Create empty database with columns
        columns = ['patient_id', 'timestamp'] + feature_names + ['risk_level', 'doctor_notes']
        df = pd.DataFrame(columns=columns)
        df.to_csv("database.csv", index=False)
        return df

# Save prediction to database
def save_prediction(patient_id, inputs, risk_level, doctor_notes=""):
    df = load_database()
    
    # Create new row
    new_row = {
        'patient_id': patient_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'risk_level': risk_level,
        'doctor_notes': doctor_notes
    }
    
    # Add input features to the row
    for feature, value in inputs.items():
        new_row[feature] = value
    
    # Append to dataframe
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save to CSV
    df.to_csv("database.csv", index=False)

# User authentication functions
def get_users():
    if not os.path.exists("users.csv"):
        # Create default users (doctor and patient)
        users_df = pd.DataFrame({
            'username': ['doctor', 'patient'],
            'password': [
                hashlib.sha256('doctor123'.encode()).hexdigest(),
                hashlib.sha256('patient123'.encode()).hexdigest()
            ],
            'role': ['doctor', 'patient'],
            'patient_id': ['N/A', 'P001']
        })
        users_df.to_csv("users.csv", index=False)
    
    return pd.read_csv("users.csv")

def authenticate(username, password):
    users_df = get_users()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    user = users_df[(users_df['username'] == username) & 
                    (users_df['password'] == hashed_password)]
    
    if not user.empty:
        return {
            'username': user['username'].values[0],
            'role': user['role'].values[0],
            'patient_id': user['patient_id'].values[0]
        }
    return None

def add_user(username, password, role, patient_id=None):
    users_df = get_users()
    
    # Check if username already exists
    if username in users_df['username'].values:
        return False
    
    # Generate patient ID if not provided
    if patient_id is None and role == 'patient':
        existing_ids = users_df[users_df['role'] == 'patient']['patient_id'].tolist()
        if existing_ids:
            max_id = max([int(pid.replace('P', '')) for pid in existing_ids if pid.startswith('P')])
            patient_id = f"P{max_id + 1:03d}"
        else:
            patient_id = "P001"
    elif role == 'doctor':
        patient_id = 'N/A'
    
    # Add new user
    new_user = pd.DataFrame({
        'username': [username],
        'password': [hashlib.sha256(password.encode()).hexdigest()],
        'role': [role],
        'patient_id': [patient_id]
    })
    
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv("users.csv", index=False)
    return True

# Function to calculate derived features
def calculate_derived_features(inputs):
    # Calculate BP_Ratio
    inputs['BP_Ratio'] = inputs['SystolicBP'] / inputs['DiastolicBP']
    
    # Calculate Risk_Score
    inputs['Risk_Score'] = (inputs['BS'] * 0.3) + (inputs['Age'] * 0.2) + (inputs['SystolicBP'] * 0.15)
    
    # Calculate Age_BP_Interaction
    inputs['Age_BP_Interaction'] = inputs['Age'] * (inputs['SystolicBP'] + inputs['DiastolicBP'])
    
    # Calculate BS_HeartRate_Interaction
    inputs['BS_HeartRate_Interaction'] = inputs['BS'] * inputs['HeartRate']
    
    # Determine Hemoglobin_Group
    hemoglobin = inputs['Hemoglobin_Level']
    # These thresholds are approximations - adjust based on your actual data distribution
    if hemoglobin < 9:
        inputs['Hemoglobin_Group_Low'] = 1
        inputs['Hemoglobin_Group_Medium'] = 0
        inputs['Hemoglobin_Group_High'] = 0
        inputs['Hemoglobin_Group_Very High'] = 0
    elif hemoglobin < 11:
        inputs['Hemoglobin_Group_Low'] = 0
        inputs['Hemoglobin_Group_Medium'] = 1
        inputs['Hemoglobin_Group_High'] = 0
        inputs['Hemoglobin_Group_Very High'] = 0
    elif hemoglobin < 13:
        inputs['Hemoglobin_Group_Low'] = 0
        inputs['Hemoglobin_Group_Medium'] = 0
        inputs['Hemoglobin_Group_High'] = 1
        inputs['Hemoglobin_Group_Very High'] = 0
    else:
        inputs['Hemoglobin_Group_Low'] = 0
        inputs['Hemoglobin_Group_Medium'] = 0
        inputs['Hemoglobin_Group_High'] = 0
        inputs['Hemoglobin_Group_Very High'] = 1
    
    return inputs

# Prediction function
def predict_risk(inputs):
    # Calculate derived features
    inputs = calculate_derived_features(inputs)
    
    # Convert inputs to numpy array in the correct order
    features = np.array([[inputs[feature] for feature in feature_names]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Decode prediction
    risk_level_map = {0: "low risk", 1: "mid risk", 2: "high risk"}
    risk_level = risk_level_map.get(prediction, "unknown risk")
    
    return risk_level

# UI Components
def landing_page():
    st.markdown("<div class='landing-page'>", unsafe_allow_html=True)
    st.markdown("<h1 class='landing-title'>Maternal Health Risk Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='landing-subtitle'>An AI-powered system to predict maternal health risks</p>", unsafe_allow_html=True)
    
    # College Information
    st.markdown("<div class='college-section'>", unsafe_allow_html=True)
    st.markdown("<h2>College Information</h2>", unsafe_allow_html=True)
    st.markdown("<p><strong>College Name:</strong> [Your College Name]</p>", unsafe_allow_html=True)
    st.markdown("<p><strong>Department:</strong> [Your Department]</p>", unsafe_allow_html=True)
    st.markdown("<p><strong>Course:</strong> [Your Course]</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Team Information
    st.markdown("<div class='team-section'>", unsafe_allow_html=True)
    st.markdown("<h2>Team Members</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='team-member'>", unsafe_allow_html=True)
        st.markdown("<p><strong>Name:</strong> [Team Member 1]</p>", unsafe_allow_html=True)
        st.markdown("<p><strong>ID:</strong> [ID Number]</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='team-member'>", unsafe_allow_html=True)
        st.markdown("<p><strong>Name:</strong> [Team Member 2]</p>", unsafe_allow_html=True)
        st.markdown("<p><strong>ID:</strong> [ID Number]</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='team-member'>", unsafe_allow_html=True)
        st.markdown("<p><strong>Name:</strong> [Team Member 3]</p>", unsafe_allow_html=True)
        st.markdown("<p><strong>ID:</strong> [ID Number]</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='team-member'>", unsafe_allow_html=True)
        st.markdown("<p><strong>Name:</strong> [Team Member 4]</p>", unsafe_allow_html=True)
        st.markdown("<p><strong>ID:</strong> [ID Number]</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Project Description
    st.markdown("<div class='college-section'>", unsafe_allow_html=True)
    st.markdown("<h2>Project Overview</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p>This Maternal Health Risk Prediction System uses machine learning to assess potential health risks for expectant mothers. 
    The system analyzes various health parameters including blood pressure, blood sugar, hemoglobin levels, and more to classify 
    patients into low, medium, or high-risk categories.</p>
    
    <p>Key features include:</p>
    <ul>
        <li>Role-based access for doctors and patients</li>
        <li>Real-time risk prediction using a Random Forest model</li>
        <li>Comprehensive analytics dashboard for healthcare providers</li>
        <li>Patient history tracking and visualization</li>
        <li>Personalized recommendations based on risk assessment</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Enter Application Button
    st.markdown("<div class='enter-app-btn'>", unsafe_allow_html=True)
    if st.button("Enter Application", key="enter_app"):
        st.session_state['show_landing'] = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='footer'>© 2023 Maternal Health Risk Prediction System</div>", unsafe_allow_html=True)

def login_page():
    st.markdown("<h1 class='main-header'>Maternal Health Risk Prediction System</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Login</h2>", unsafe_allow_html=True)
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            user = authenticate(username, password)
            if user:
                st.session_state['user'] = user
                st.rerun()
            else:
                st.error("Invalid username or password")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3>New User Registration</h3>", unsafe_allow_html=True)
        
        with st.expander("Register"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            role = st.selectbox("Role", ["patient", "doctor"], index=0)
            
            if st.button("Register"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif not new_username or not new_password:
                    st.error("Username and password cannot be empty")
                else:
                    if add_user(new_username, new_password, role):
                        st.success("Registration successful! You can now login.")
                    else:
                        st.error("Username already exists")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='footer'>© 2023 Maternal Health Risk Prediction System</div>", unsafe_allow_html=True)

def patient_portal(user):
    st.markdown("<h1 class='main-header'>Maternal Health Risk Assessment</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        st.markdown(f"<h3>Welcome, {user['username']}!</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>Patient ID: {user['patient_id']}</p>", unsafe_allow_html=True)
        
        if st.button("Logout"):
            st.session_state.pop('user')
            st.rerun()
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h4>About Risk Levels:</h4>", unsafe_allow_html=True)
        st.markdown("""
        - 🔴 **High Risk**: Immediate medical attention required
        - 🟡 **Medium Risk**: Regular monitoring recommended
        - 🟢 **Low Risk**: Continue routine check-ups
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Enter Your Health Information</h2>", unsafe_allow_html=True)
        
        # Input form
        with st.form("health_form"):
            # Basic vital signs
            age = st.number_input("Age (years)", min_value=15, max_value=60, value=25)
            
            col_bp1, col_bp2 = st.columns(2)
            with col_bp1:
                systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=200, value=120)
            with col_bp2:
                diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
            
            blood_sugar = st.number_input("Blood Sugar (mmol/L)", min_value=1.0, max_value=20.0, value=6.0, step=0.1)
            body_temp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=105.0, value=98.6, step=0.1)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)
            
            # Additional parameters from the model
            hemoglobin = st.number_input("Hemoglobin Level (g/dL)", min_value=5.0, max_value=20.0, value=12.0, step=0.1)
            thyroid_condition = st.radio("Thyroid Condition", ["No", "Yes"])
            
            submit_button = st.form_submit_button("Predict Risk")
        
        if submit_button:
            # Prepare inputs
            inputs = {
                'Age': age,
                'SystolicBP': systolic_bp,
                'DiastolicBP': diastolic_bp,
                'BS': blood_sugar,
                'BodyTemp': body_temp,
                'HeartRate': heart_rate,
                'Hemoglobin_Level': hemoglobin,
                'Thyroid_Condition': 1 if thyroid_condition == "Yes" else 0
            }
            
            # Make prediction
            risk_level = predict_risk(inputs)
            
            # Calculate derived features for database storage
            full_inputs = calculate_derived_features(inputs.copy())
            
            # Save prediction to database
            save_prediction(user['patient_id'], full_inputs, risk_level)
            
            # Display result
            st.markdown("<h3>Risk Assessment Result:</h3>", unsafe_allow_html=True)
            
            if risk_level == "high risk":
                st.markdown("<div class='risk-high'>🔴 High Risk: Please consult with a healthcare provider immediately.</div>", unsafe_allow_html=True)
            elif risk_level == "mid risk":
                st.markdown("<div class='risk-medium'>🟡 Medium Risk: Regular monitoring is recommended.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='risk-low'>🟢 Low Risk: Continue with routine check-ups.</div>", unsafe_allow_html=True)
            
            # Display recommendations
            st.markdown("<h3>Recommendations:</h3>", unsafe_allow_html=True)
            if risk_level == "high risk":
                st.markdown("""
                - Seek immediate medical attention
                - Contact your healthcare provider right away
                - Avoid strenuous activities
                - Monitor your symptoms closely
                """)
            elif risk_level == "mid risk":
                st.markdown("""
                - Schedule a follow-up appointment with your doctor
                - Monitor your blood pressure regularly
                - Maintain a balanced diet
                - Get adequate rest
                """)
            else:
                st.markdown("""
                - Continue with regular prenatal check-ups
                - Maintain a healthy lifestyle
                - Stay hydrated and eat nutritious foods
                - Exercise moderately as recommended by your doctor
                """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Your History</h2>", unsafe_allow_html=True)
        
        # Load patient history
        df = load_database()
        patient_history = df[df['patient_id'] == user['patient_id']].sort_values('timestamp', ascending=False)
        
        if not patient_history.empty:
            for _, row in patient_history.iterrows():
                risk_class = "risk-high" if row['risk_level'] == "high risk" else "risk-medium" if row['risk_level'] == "mid risk" else "risk-low"
                risk_icon = "🔴" if row['risk_level'] == "high risk" else "🟡" if row['risk_level'] == "mid risk" else "🟢"
                
                st.markdown(f"<p><strong>{row['timestamp']}</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<div class='{risk_class}'>{risk_icon} {row['risk_level'].title()}</div>", unsafe_allow_html=True)
                
                with st.expander("View Details"):
                    # Display basic vital signs
                    st.markdown("**Basic Vital Signs:**")
                    st.markdown(f"- Age: {row['Age']} years")
                    st.markdown(f"- Blood Pressure: {row['SystolicBP']}/{row['DiastolicBP']} mmHg")
                    st.markdown(f"- Blood Sugar: {row['BS']} mmol/L")
                    st.markdown(f"- Body Temperature: {row['BodyTemp']} °F")
                    st.markdown(f"- Heart Rate: {row['HeartRate']} bpm")
                    
                    # Display additional parameters
                    st.markdown("**Additional Parameters:**")
                    st.markdown(f"- Hemoglobin Level: {row['Hemoglobin_Level']} g/dL")
                    st.markdown(f"- Thyroid Condition: {'Yes' if row['Thyroid_Condition'] == 1 else 'No'}")
                    
                    # Display calculated metrics
                    st.markdown("**Calculated Metrics:**")
                    st.markdown(f"- BP Ratio: {row['BP_Ratio']:.2f}")
                    st.markdown(f"- Risk Score: {row['Risk_Score']:.2f}")
                
                st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.info("No history available. Complete your first assessment to see results here.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='footer'>© 2023 Maternal Health Risk Prediction System</div>", unsafe_allow_html=True)

def doctor_dashboard(user):
    st.markdown("<h1 class='main-header'>Doctor Dashboard</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        st.markdown(f"<h3>Welcome, Dr. {user['username']}!</h3>", unsafe_allow_html=True)
        
        if st.button("Logout"):
            st.session_state.pop('user')
            st.rerun()
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h4>Navigation</h4>", unsafe_allow_html=True)
        
        page = st.radio("", ["Patient Records", "Analytics Dashboard", "Add New Patient"])
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Load database
    df = load_database()
    
    if page == "Patient Records":
        st.markdown("<h2 class='sub-header'>Patient Records</h2>", unsafe_allow_html=True)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not df.empty and 'patient_id' in df.columns:
                patient_ids = ['All'] + sorted(df['patient_id'].unique().tolist())
                selected_patient = st.selectbox("Patient ID", patient_ids)
            else:
                selected_patient = "All"
                st.info("No patient records available")
        
        with col2:
            if not df.empty and 'risk_level' in df.columns:
                risk_levels = ['All'] + sorted(df['risk_level'].unique().tolist())
                selected_risk = st.selectbox("Risk Level", risk_levels)
            else:
                selected_risk = "All"
        
        with col3:
            if not df.empty and 'timestamp' in df.columns:
                date_range = st.date_input(
                    "Date Range",
                    value=(
                        pd.to_datetime(df['timestamp']).min().date() if not df.empty else datetime.now().date(),
                        datetime.now().date()
                    ),
                    max_value=datetime.now().date()
                )
            else:
                date_range = (datetime.now().date(), datetime.now().date())
        
        # Filter data
        filtered_df = df.copy()
        
        if not df.empty:
            if selected_patient != "All":
                filtered_df = filtered_df[filtered_df['patient_id'] == selected_patient]
            
            if selected_risk != "All":
                filtered_df = filtered_df[filtered_df['risk_level'] == selected_risk]
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df['date'] = pd.to_datetime(filtered_df['timestamp']).dt.date
                filtered_df = filtered_df[
                    (filtered_df['date'] >= start_date) & 
                    (filtered_df['date'] <= end_date)
                ]
                filtered_df = filtered_df.drop(columns=['date'])
        
        # Display filtered data
        if not filtered_df.empty:
            st.markdown(f"<p>Showing {len(filtered_df)} records</p>", unsafe_allow_html=True)
            
            
            
            for _, row in filtered_df.iterrows():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"<h3>Patient ID: {row['patient_id']}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Date:</strong> {row['timestamp']}</p>", unsafe_allow_html=True)
                    
                    risk_class = "risk-high" if row['risk_level'] == "high risk" else "risk-medium" if row['risk_level'] == "mid risk" else "risk-low"
                    risk_icon = "🔴" if row['risk_level'] == "high risk" else "🟡" if row['risk_level'] == "mid risk" else "🟢"
                    
                    st.markdown(f"<div class='{risk_class}'>{risk_icon} {row['risk_level'].title()}</div>", unsafe_allow_html=True)
                    
                    with st.expander("View Health Parameters"):
                        # Basic vital signs
                        st.markdown("**Basic Vital Signs:**")
                        col_basic1, col_basic2 = st.columns(2)
                        
                        with col_basic1:
                            st.markdown(f"- Age: {row['Age']} years")
                            st.markdown(f"- Blood Pressure: {row['SystolicBP']}/{row['DiastolicBP']} mmHg")
                            st.markdown(f"- Blood Sugar: {row['BS']} mmol/L")
                        
                        with col_basic2:
                            st.markdown(f"- Body Temperature: {row['BodyTemp']} °F")
                            st.markdown(f"- Heart Rate: {row['HeartRate']} bpm")
                            st.markdown(f"- Hemoglobin Level: {row['Hemoglobin_Level']} g/dL")
                        
                        # Additional parameters
                        st.markdown("**Additional Parameters:**")
                        col_add1, col_add2 = st.columns(2)
                        
                        with col_add1:
                            st.markdown(f"- Thyroid Condition: {'Yes' if row['Thyroid_Condition'] == 1 else 'No'}")
                            st.markdown(f"- BP Ratio: {row['BP_Ratio']:.2f}")
                        
                        with col_add2:
                            st.markdown(f"- Risk Score: {row['Risk_Score']:.2f}")
                            
                            # Determine hemoglobin group
                            hemoglobin_group = "Unknown"
                            if row['Hemoglobin_Group_Low'] == 1:
                                hemoglobin_group = "Low"
                            elif row['Hemoglobin_Group_Medium'] == 1:
                                hemoglobin_group = "Medium"
                            elif row['Hemoglobin_Group_High'] == 1:
                                hemoglobin_group = "High"
                            elif row['Hemoglobin_Group_Very High'] == 1:
                                hemoglobin_group = "Very High"
                            
                            st.markdown(f"- Hemoglobin Group: {hemoglobin_group}")
                    
                    # Doctor notes
                    notes = row['doctor_notes'] if 'doctor_notes' in row and not pd.isna(row['doctor_notes']) else ""
                    new_notes = st.text_area("Doctor's Notes", value=notes, key=f"notes_{_}")
                    
                    if new_notes != notes:
                        # Update notes in database
                        df.loc[df.index == _, 'doctor_notes'] = new_notes
                        df.to_csv("database.csv", index=False)
                        st.success("Notes updated successfully")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No records found matching the selected filters")
    
    elif page == "Analytics Dashboard":
        st.markdown("<h2 class='sub-header'>Analytics Dashboard</h2>", unsafe_allow_html=True)
        
        if not df.empty:
            # Create analytics dashboard
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3>Risk Level Distribution</h3>", unsafe_allow_html=True)
                
                risk_counts = df['risk_level'].value_counts().reset_index()
                risk_counts.columns = ['Risk Level', 'Count']
                
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#2ecc71', '#f39c12', '#e74c3c']
                
                # Sort risk levels in order: low, mid, high
                risk_order = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
                risk_counts['order'] = risk_counts['Risk Level'].map(risk_order)
                risk_counts = risk_counts.sort_values('order')
                risk_counts = risk_counts.drop(columns=['order'])
                
                sns.barplot(x='Risk Level', y='Count', data=risk_counts, palette=colors, ax=ax)
                ax.set_title('Distribution of Risk Levels')
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Number of Patients')
                
                # Customize x-axis labels
                ax.set_xticklabels([level.title() for level in risk_counts['Risk Level']])
                
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3>Age vs. Risk Level</h3>", unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Create violin plot
                sns.violinplot(x='risk_level', y='Age', data=df, palette=colors, ax=ax)
                ax.set_title('Age Distribution by Risk Level')
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Age (years)')
                
                # Customize x-axis labels
                ax.set_xticklabels([level.title() for level in sorted(df['risk_level'].unique(), key=lambda x: risk_order.get(x, 0))])
                
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Second row of charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3>Blood Pressure Analysis</h3>", unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Create scatter plot
                scatter = ax.scatter(
                    df['SystolicBP'], 
                    df['DiastolicBP'],
                    c=[risk_order.get(r, 0) for r in df['risk_level']],
                    cmap='RdYlGn_r',
                    alpha=0.7,
                    s=50
                )
                
                ax.set_title('Systolic vs. Diastolic Blood Pressure')
                ax.set_xlabel('Systolic BP (mmHg)')
                ax.set_ylabel('Diastolic BP (mmHg)')
                
                # Add color bar
                cbar = plt.colorbar(scatter)
                cbar.set_ticks([0, 1, 2])
                cbar.set_ticklabels(['Low Risk', 'Medium Risk', 'High Risk'])
                
                # Add reference lines for normal BP
                ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=120, color='gray', linestyle='--', alpha=0.5)
                
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3>Hemoglobin Analysis</h3>", unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Create boxplot for hemoglobin levels by risk
                sns.boxplot(x='risk_level', y='Hemoglobin_Level', data=df, palette=colors, ax=ax)
                ax.set_title('Hemoglobin Levels by Risk Category')
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Hemoglobin Level (g/dL)')
                
                # Customize x-axis labels
                ax.set_xticklabels([level.title() for level in sorted(df['risk_level'].unique(), key=lambda x: risk_order.get(x, 0))])
                
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Key Insights</h3>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_patients = len(df['patient_id'].unique())
                st.metric("Total Patients", total_patients)
            
            with col2:
                high_risk_count = len(df[df['risk_level'] == 'high risk']['patient_id'].unique())
                high_risk_percent = (high_risk_count / total_patients * 100) if total_patients > 0 else 0
                st.metric("High Risk Patients", f"{high_risk_count} ({high_risk_percent:.1f}%)")
            
            with col3:
                avg_age = df['Age'].mean()
                st.metric("Average Age", f"{avg_age:.1f} years")
            
            with col4:
                thyroid_percent = (df['Thyroid_Condition'].mean() * 100)
                st.metric("Thyroid Condition", f"{thyroid_percent:.1f}%")
            
            # Feature importance chart
            st.markdown("<h3>Risk Factors Analysis</h3>", unsafe_allow_html=True)
            
            # This is a simplified feature importance visualization
            # In a real application, you would calculate this from the model
            feature_importance = pd.DataFrame({
                'Feature': ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate', 'Hemoglobin_Level', 'Thyroid_Condition'],
                'Importance': [0.18, 0.22, 0.15, 0.20, 0.10, 0.12, 0.03]
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis', ax=ax)
            ax.set_title('Feature Importance for Risk Prediction')
            ax.set_xlabel('Relative Importance')
            
            st.pyplot(fig)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No data available for analysis. Add patient records to see analytics.")
    
    elif page == "Add New Patient":
        st.markdown("<h2 class='sub-header'>Add New Patient</h2>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        with st.form("add_patient_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            submit_button = st.form_submit_button("Add Patient")
        
        if submit_button:
            if not username or not password:
                st.error("Username and password cannot be empty")
            elif password != confirm_password:
                st.error("Passwords do not match")
            else:
                if add_user(username, password, 'patient'):
                    st.success(f"Patient {username} added successfully")
                else:
                    st.error("Username already exists")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='footer'>© 2023 Maternal Health Risk Prediction System</div>", unsafe_allow_html=True)

# Main app
def main():
    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state['user'] = None
    
    if 'show_landing' not in st.session_state:
        st.session_state['show_landing'] = True
    
    # Check if landing page should be shown
    if st.session_state['show_landing']:
        landing_page()
    else:
        # Check if user is logged in
        if st.session_state['user'] is None:
            login_page()
        else:
            # Route to appropriate page based on user role
            if st.session_state['user']['role'] == 'doctor':
                doctor_dashboard(st.session_state['user'])
            else:
                patient_portal(st.session_state['user'])

if __name__ == "__main__":
    main()