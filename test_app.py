import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
import os
warnings.filterwarnings('ignore')

# For PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# For SHAP explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class ChronicCareRiskPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = ['age', 'sex', 'bmi', 'blood_pressure', 'cholesterol', 
                             'glucose', 'heart_rate', 'exercise_tolerance', 'family_history']
        self.is_trained = False
        self.shap_explainer = None
        
    def load_and_prepare_data(self):
        """Load and combine multiple chronic disease datasets"""
        try:
            # Load datasets
            heart_df = pd.read_csv('heart_disease.csv')
            diabetes_df = pd.read_csv('diabetes.csv')
            
            # Process heart disease data
            heart_processed = pd.DataFrame()
            
            # Map heart disease columns to our standard format
            heart_processed['age'] = heart_df['Age'] if 'Age' in heart_df.columns else heart_df.get('age', 50)
            
            # Handle sex mapping
            if 'Sex' in heart_df.columns:
                heart_processed['sex'] = heart_df['Sex'].map({'M': 1, 'F': 0})
            else:
                heart_processed['sex'] = np.random.binomial(1, 0.5, len(heart_df))
            
            # Generate BMI (not in heart disease dataset)
            heart_processed['bmi'] = np.random.normal(26, 4, len(heart_df))
            heart_processed['bmi'] = np.clip(heart_processed['bmi'], 15, 45)
            
            # Blood pressure
            heart_processed['blood_pressure'] = heart_df.get('RestingBP', 120)
            
            # Cholesterol
            heart_processed['cholesterol'] = heart_df.get('Cholesterol', 200)
            # Replace 0 cholesterol values with mean
            heart_processed['cholesterol'] = heart_processed['cholesterol'].replace(0, heart_processed['cholesterol'][heart_processed['cholesterol'] > 0].mean())
            
            # Glucose estimation from FastingBS
            if 'FastingBS' in heart_df.columns:
                heart_processed['glucose'] = heart_df['FastingBS'] * 50 + 100  # Convert binary to glucose level
            else:
                heart_processed['glucose'] = np.random.normal(100, 15, len(heart_df))
            
            # Heart rate
            heart_processed['heart_rate'] = heart_df.get('MaxHR', 150)
            
            # Exercise tolerance (inverse of ExerciseAngina)
            if 'ExerciseAngina' in heart_df.columns:
                heart_processed['exercise_tolerance'] = heart_df['ExerciseAngina'].map({'Y': 0, 'N': 1}).fillna(1)
            else:
                heart_processed['exercise_tolerance'] = np.random.binomial(1, 0.7, len(heart_df))
            
            # Family history (synthetic)
            heart_processed['family_history'] = np.random.binomial(1, 0.3, len(heart_df))
            
            # Target variable
            heart_processed['target'] = heart_df.get('HeartDisease', 0)
            
            # Process diabetes data
            diabetes_processed = pd.DataFrame()
            
            # Age
            diabetes_processed['age'] = diabetes_df['Age'] if 'Age' in diabetes_df.columns else 50
            
            # Sex (synthetic for diabetes dataset)
            diabetes_processed['sex'] = np.random.binomial(1, 0.5, len(diabetes_df))
            
            # BMI
            diabetes_processed['bmi'] = diabetes_df['BMI'] if 'BMI' in diabetes_df.columns else 25
            
            # Blood pressure
            diabetes_processed['blood_pressure'] = diabetes_df.get('BloodPressure', 80)
            # Replace 0 values with median
            diabetes_processed['blood_pressure'] = diabetes_processed['blood_pressure'].replace(0, diabetes_processed['blood_pressure'][diabetes_processed['blood_pressure'] > 0].median())
            
            # Cholesterol (synthetic)
            diabetes_processed['cholesterol'] = np.random.normal(200, 40, len(diabetes_df))
            diabetes_processed['cholesterol'] = np.clip(diabetes_processed['cholesterol'], 120, 350)
            
            # Glucose
            diabetes_processed['glucose'] = diabetes_df['Glucose'] if 'Glucose' in diabetes_df.columns else 100
            
            # Heart rate (synthetic)
            diabetes_processed['heart_rate'] = np.random.normal(72, 12, len(diabetes_df))
            diabetes_processed['heart_rate'] = np.clip(diabetes_processed['heart_rate'], 50, 120)
            
            # Exercise tolerance (synthetic, correlated with age and BMI)
            diabetes_processed['exercise_tolerance'] = np.where(
                (diabetes_processed['age'] > 60) | (diabetes_processed['bmi'] > 30), 
                np.random.binomial(1, 0.4, len(diabetes_df)),
                np.random.binomial(1, 0.8, len(diabetes_df))
            )
            
            # Family history from DiabetesPedigreeFunction
            if 'DiabetesPedigreeFunction' in diabetes_df.columns:
                diabetes_processed['family_history'] = (diabetes_df['DiabetesPedigreeFunction'] > 0.5).astype(int)
            else:
                diabetes_processed['family_history'] = np.random.binomial(1, 0.4, len(diabetes_df))
            
            # Target variable
            diabetes_processed['target'] = diabetes_df.get('Outcome', 0)
            
            # Combine datasets
            combined_df = pd.concat([heart_processed, diabetes_processed], ignore_index=True)
            
            # Clean data
            combined_df = combined_df.fillna(combined_df.median())
            combined_df = combined_df[(combined_df['age'] > 0) & (combined_df['age'] < 120)]
            combined_df = combined_df[combined_df['blood_pressure'] > 0]
            combined_df = combined_df[combined_df['cholesterol'] > 0]
            combined_df = combined_df[combined_df['glucose'] > 0]
            
            # Ensure all features are numeric
            for feature in self.feature_names:
                combined_df[feature] = pd.to_numeric(combined_df[feature], errors='coerce')
            
            combined_df = combined_df.dropna()
            
            print(f"Combined dataset shape: {combined_df.shape}")
            print(f"Target distribution: {combined_df['target'].value_counts()}")
            
            return combined_df
            
        except FileNotFoundError as e:
            st.error(f"Dataset file not found: {e}")
            st.error("Please ensure heart_disease.csv and diabetes.csv are in the same directory as this script.")
            return None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            print(f"Detailed error: {str(e)}")
            return None
    
    def train_models(self):
        """Train ensemble of models for risk prediction"""
        data = self.load_and_prepare_data()
        if data is None:
            return False
        
        X = data[self.feature_names]
        y = data['target']
        
        print(f"Training data shape: {X.shape}")
        print(f"Feature names: {self.feature_names}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train multiple models
        self.models['rf'] = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.models['gb'] = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6)
        self.models['lr'] = LogisticRegression(random_state=42, max_iter=1000)
        
        # Fit models
        self.models['rf'].fit(X_train, y_train)
        self.models['gb'].fit(X_train, y_train)
        self.models['lr'].fit(X_train_scaled, y_train)
        
        # Evaluate models
        performance = {}
        for name, model in self.models.items():
            try:
                if name == 'lr':
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                performance[name] = auc
                print(f"{name} AUC: {auc:.3f}")
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                performance[name] = 0.5
        
        # Setup SHAP explainer if available
        if SHAP_AVAILABLE:
            try:
                self.shap_explainer = shap.Explainer(self.models['rf'], X_train)
            except Exception as e:
                print(f"SHAP explainer setup failed: {e}")
                self.shap_explainer = None
        
        # Save models
        try:
            joblib.dump(self.models, 'chronic_care_models.pkl')
            joblib.dump(self.scalers, 'chronic_care_scalers.pkl')
            joblib.dump(self.feature_names, 'feature_names.pkl')
            print("Models saved successfully")
        except Exception as e:
            print(f"Error saving models: {e}")
        
        self.is_trained = True
        return performance
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            if (os.path.exists('chronic_care_models.pkl') and 
                os.path.exists('chronic_care_scalers.pkl') and 
                os.path.exists('feature_names.pkl')):
                
                self.models = joblib.load('chronic_care_models.pkl')
                self.scalers = joblib.load('chronic_care_scalers.pkl')
                self.feature_names = joblib.load('feature_names.pkl')
                self.is_trained = True
                print("Models loaded successfully")
                return True
            else:
                print("Model files not found")
                return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF medical record"""
        if not PDF_AVAILABLE:
            st.error("PDF processing not available. Please install PyPDF2: pip install PyPDF2")
            return None
        
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None
    
    def parse_medical_text(self, text):
        """Parse medical information from text using regex patterns"""
        features = {
            'age': 50, 'sex': 0, 'bmi': 25, 'blood_pressure': 120,
            'cholesterol': 200, 'glucose': 100, 'heart_rate': 72,
            'exercise_tolerance': 1, 'family_history': 0
        }
        
        text = text.lower()
        
        # Age extraction
        age_patterns = [r'age[:\s]*(\d+)', r'(\d+)\s*years?\s*old', r'(\d+)\s*y\.?o\.?']
        for pattern in age_patterns:
            match = re.search(pattern, text)
            if match:
                features['age'] = int(match.group(1))
                break
        
        # Gender extraction
        if any(word in text for word in ['female', 'woman', 'she', 'her']):
            features['sex'] = 0
        elif any(word in text for word in ['male', 'man', 'he', 'his']):
            features['sex'] = 1
        
        # BMI extraction
        bmi_patterns = [r'bmi[:\s]*(\d+\.?\d*)', r'body mass index[:\s]*(\d+\.?\d*)']
        for pattern in bmi_patterns:
            match = re.search(pattern, text)
            if match:
                features['bmi'] = float(match.group(1))
                break
        
        # Blood pressure extraction
        bp_patterns = [r'bp[:\s]*(\d+)[/\s]*(\d+)?', r'blood pressure[:\s]*(\d+)[/\s]*(\d+)?']
        for pattern in bp_patterns:
            match = re.search(pattern, text)
            if match:
                features['blood_pressure'] = int(match.group(1))
                break
        
        # Cholesterol extraction
        chol_patterns = [r'cholesterol[:\s]*(\d+)', r'total cholesterol[:\s]*(\d+)']
        for pattern in chol_patterns:
            match = re.search(pattern, text)
            if match:
                features['cholesterol'] = int(match.group(1))
                break
        
        # Glucose extraction
        glucose_patterns = [r'glucose[:\s]*(\d+)', r'blood sugar[:\s]*(\d+)', r'hba1c[:\s]*(\d+\.?\d*)']
        for pattern in glucose_patterns:
            match = re.search(pattern, text)
            if match:
                if 'hba1c' in pattern:
                    features['glucose'] = int(float(match.group(1)) * 28.7 - 46.7)  # Convert HbA1c to glucose
                else:
                    features['glucose'] = int(match.group(1))
                break
        
        # Heart rate extraction
        hr_patterns = [r'heart rate[:\s]*(\d+)', r'hr[:\s]*(\d+)', r'pulse[:\s]*(\d+)']
        for pattern in hr_patterns:
            match = re.search(pattern, text)
            if match:
                features['heart_rate'] = int(match.group(1))
                break
        
        # Family history
        if any(phrase in text for phrase in ['family history', 'familial', 'hereditary', 'genetic']):
            features['family_history'] = 1
        
        # Exercise tolerance
        if any(phrase in text for phrase in ['exercise intolerance', 'shortness of breath', 'fatigue']):
            features['exercise_tolerance'] = 0
        
        return features
    
    def predict_risk(self, features):
        """Predict deterioration risk using ensemble of models"""
        if not self.is_trained:
            if not self.load_models():
                st.error("No trained models available. Please train models first.")
                return None
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Ensure all features are present and in correct order
        for feature in self.feature_names:
            if feature not in feature_df.columns:
                feature_df[feature] = features.get(feature, 0)
        
        feature_df = feature_df[self.feature_names]  # Ensure correct order
        
        # Make predictions with each model
        predictions = {}
        
        try:
            # Random Forest and Gradient Boosting
            for name in ['rf', 'gb']:
                pred_proba = self.models[name].predict_proba(feature_df)[:, 1]
                predictions[name] = pred_proba[0]
            
            # Logistic Regression (needs scaling)
            feature_scaled = self.scalers['standard'].transform(feature_df)
            pred_proba = self.models['lr'].predict_proba(feature_scaled)[:, 1]
            predictions['lr'] = pred_proba[0]
            
            # Ensemble prediction (weighted average)
            ensemble_pred = (predictions['rf'] * 0.4 + 
                            predictions['gb'] * 0.4 + 
                            predictions['lr'] * 0.2)
            
            return {
                'ensemble_risk': ensemble_pred,
                'individual_predictions': predictions,
                'risk_category': self.categorize_risk(ensemble_pred),
                'confidence': self.calculate_confidence(predictions)
            }
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None
    
    def categorize_risk(self, risk_score):
        """Categorize risk into clinical categories"""
        if risk_score < 0.25:
            return "Low Risk"
        elif risk_score < 0.5:
            return "Moderate Risk"
        elif risk_score < 0.75:
            return "High Risk"
        else:
            return "Critical Risk"
    
    def calculate_confidence(self, predictions):
        """Calculate prediction confidence based on model agreement"""
        pred_values = list(predictions.values())
        std_dev = np.std(pred_values)
        # Higher agreement (lower std) = higher confidence
        confidence = max(0, 1 - (std_dev * 4))
        return confidence
    
    def get_explanations(self, features):
        """Get SHAP explanations for the prediction"""
        if not self.is_trained or self.shap_explainer is None:
            # Return simple feature importance based on values
            importance = {}
            # Normalize features and calculate simple importance
            if features['age'] > 65:
                importance['age'] = 0.8
            elif features['age'] > 45:
                importance['age'] = 0.5
            else:
                importance['age'] = 0.2
                
            if features['blood_pressure'] > 140:
                importance['blood_pressure'] = 0.9
            elif features['blood_pressure'] > 120:
                importance['blood_pressure'] = 0.6
            else:
                importance['blood_pressure'] = 0.3
                
            if features['cholesterol'] > 240:
                importance['cholesterol'] = 0.7
            elif features['cholesterol'] > 200:
                importance['cholesterol'] = 0.4
            else:
                importance['cholesterol'] = 0.2
                
            if features['glucose'] > 140:
                importance['glucose'] = 0.8
            elif features['glucose'] > 100:
                importance['glucose'] = 0.5
            else:
                importance['glucose'] = 0.2
                
            if features['bmi'] > 30:
                importance['bmi'] = 0.6
            elif features['bmi'] > 25:
                importance['bmi'] = 0.4
            else:
                importance['bmi'] = 0.2
                
            importance['heart_rate'] = 0.3
            importance['sex'] = 0.2
            importance['exercise_tolerance'] = 0.4 if features['exercise_tolerance'] == 0 else 0.1
            importance['family_history'] = 0.5 if features['family_history'] == 1 else 0.1
            
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'feature_importance': sorted_importance,
                'shap_values': None
            }
        
        try:
            feature_df = pd.DataFrame([features])
            feature_df = feature_df[self.feature_names]
            shap_values = self.shap_explainer(feature_df)
            
            # Get feature importance
            importance = {}
            for i, feature in enumerate(self.feature_names):
                importance[feature] = abs(shap_values.values[0][i])
            
            # Sort by importance
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'feature_importance': sorted_importance,
                'shap_values': shap_values
            }
        except Exception as e:
            print(f"SHAP explanation error: {e}")
            return None

# Streamlit App
def main():
    st.set_page_config(
        page_title="Chronic Care Risk Predictor",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 Chronic Care Risk Prediction Engine")
    st.markdown("### AI-Powered 90-Day Deterioration Risk Assessment")
    
    # Initialize the predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = ChronicCareRiskPredictor()
    
    # Check if models are trained
    models_available = st.session_state.predictor.load_models()
    
    if not models_available:
        st.warning("⚠️ **Models not trained yet!** Please go to the Model Training page first to train the models with your datasets.")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Model Training", "Risk Prediction", "Dataset Info", "About"])
    
    if page == "Risk Prediction":
        risk_prediction_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Dataset Info":
        dataset_info_page()
    elif page == "About":
        about_page()

def model_training_page():
    st.header("Model Training & Evaluation")
    
    st.info("📋 **Required Datasets**: Ensure heart_disease.csv and diabetes.csv are in the same directory.")
    
    # Check if datasets exist
    heart_exists = os.path.exists('heart_disease.csv')
    diabetes_exists = os.path.exists('diabetes.csv')
    
    col1, col2 = st.columns(2)
    with col1:
        if heart_exists:
            st.success("✅ heart_disease.csv found")
        else:
            st.error("❌ heart_disease.csv not found")
    
    with col2:
        if diabetes_exists:
            st.success("✅ diabetes.csv found")
        else:
            st.error("❌ diabetes.csv not found")
    
    if heart_exists and diabetes_exists:
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                performance = st.session_state.predictor.train_models()
                
                if performance:
                    st.success("✅ Models trained successfully!")
                    
                    st.subheader("Model Performance")
                    perf_df = pd.DataFrame(list(performance.items()), 
                                         columns=['Model', 'AUC Score'])
                    perf_df['Model'] = perf_df['Model'].map({
                        'rf': 'Random Forest', 
                        'gb': 'Gradient Boosting', 
                        'lr': 'Logistic Regression'
                    })
                    
                    fig = px.bar(perf_df, x='Model', y='AUC Score', 
                               title="Model Performance (AUC Scores)")
                    fig.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(perf_df)
                else:
                    st.error("❌ Model training failed. Please check the datasets.")
    else:
        st.error("Cannot train models without required datasets.")

def risk_prediction_page():
    st.header("Patient Risk Assessment")
    
    # Check if models are available
    if not st.session_state.predictor.is_trained and not st.session_state.predictor.load_models():
        st.error("❌ **No trained models available.** Please go to the Model Training page and train the models first.")
        st.stop()
    
    # Input method selection
    input_method = st.radio("Choose input method:", 
                           ["Manual Entry", "Text Input", "Upload PDF"])
    
    features = {}
    
    if input_method == "Manual Entry":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographics")
            features['age'] = st.slider("Age", 18, 100, 50)
            features['sex'] = st.selectbox("Sex", ["Female", "Male"])
            features['sex'] = 0 if features['sex'] == "Female" else 1
            
            st.subheader("Vitals")
            features['blood_pressure'] = st.slider("Systolic BP (mmHg)", 80, 200, 120)
            features['heart_rate'] = st.slider("Heart Rate (bpm)", 50, 120, 72)
            features['bmi'] = st.slider("BMI", 15.0, 45.0, 25.0)
        
        with col2:
            st.subheader("Lab Results")
            features['cholesterol'] = st.slider("Total Cholesterol (mg/dL)", 100, 400, 200)
            features['glucose'] = st.slider("Glucose Level (mg/dL)", 70, 300, 100)
            
            st.subheader("Clinical History")
            features['exercise_tolerance'] = st.selectbox("Exercise Tolerance", 
                                                        ["Good", "Poor"])
            features['exercise_tolerance'] = 1 if features['exercise_tolerance'] == "Good" else 0
            features['family_history'] = st.selectbox("Family History of Chronic Disease", 
                                                     ["No", "Yes"])
            features['family_history'] = 1 if features['family_history'] == "Yes" else 0
    
    elif input_method == "Text Input":
        medical_text = st.text_area("Enter medical history or clinical notes:", 
                                   height=200,
                                   placeholder="Patient is a 58-year-old male with BMI 28.5, blood pressure 145/92, total cholesterol 245 mg/dL, glucose 165 mg/dL, heart rate 78 bpm. Family history of diabetes and heart disease. Reports exercise intolerance and shortness of breath on exertion.",
                                   value="Patient is a 58-year-old male with BMI 28.5, blood pressure 145/92, total cholesterol 245 mg/dL, glucose 165 mg/dL, heart rate 78 bpm. Family history of diabetes and heart disease. Reports exercise intolerance and shortness of breath on exertion.")
        if medical_text:
            features = st.session_state.predictor.parse_medical_text(medical_text)
            st.success("Medical text processed successfully!")
            
            # Show extracted features in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Age", features['age'])
                st.metric("BMI", f"{features['bmi']:.1f}")
                st.metric("Blood Pressure", features['blood_pressure'])
            with col2:
                st.metric("Cholesterol", features['cholesterol'])
                st.metric("Glucose", features['glucose'])
                st.metric("Heart Rate", features['heart_rate'])
            with col3:
                st.metric("Sex", "Female" if features['sex'] == 0 else "Male")
                st.metric("Exercise Tolerance", "Good" if features['exercise_tolerance'] == 1 else "Poor")
                st.metric("Family History", "Yes" if features['family_history'] == 1 else "No")
    
    elif input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Upload Medical Record (PDF)", type=['pdf'])
        if uploaded_file is not None:
            if PDF_AVAILABLE:
                text = st.session_state.predictor.extract_text_from_pdf(uploaded_file)
                if text:
                    st.text_area("Extracted Text (Preview)", text[:500] + "...", height=150)
                    features = st.session_state.predictor.parse_medical_text(text)
                    st.success("Medical record processed successfully!")
                    
                    # Show extracted features
                    st.subheader("Extracted Information")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Age", features['age'])
                        st.metric("BMI", f"{features['bmi']:.1f}")
                        st.metric("Blood Pressure", features['blood_pressure'])
                    with col2:
                        st.metric("Cholesterol", features['cholesterol'])
                        st.metric("Glucose", features['glucose'])
                        st.metric("Heart Rate", features['heart_rate'])
                    with col3:
                        st.metric("Sex", "Female" if features['sex'] == 0 else "Male")
                        st.metric("Exercise Tolerance", "Good" if features['exercise_tolerance'] == 1 else "Poor")
                        st.metric("Family History", "Yes" if features['family_history'] == 1 else "No")
            else:
                st.error("PDF processing not available. Please install PyPDF2.")
    
    # Predict button
    if st.button("🔮 Predict Risk", type="primary") and features:
        with st.spinner("Analyzing patient data..."):
            result = st.session_state.predictor.predict_risk(features)
            
            if result:
                # Main risk display
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    risk_score = result['ensemble_risk']
                    risk_category = result['risk_category']
                    
                    # Risk gauge chart
                    gauge_color = "darkred" if risk_score > 0.75 else "orange" if risk_score > 0.5 else "yellow" if risk_score > 0.25 else "green"
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = risk_score * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "90-Day Deterioration Risk (%)"},
                        delta = {'reference': 25},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': gauge_color},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgreen"},
                                {'range': [25, 50], 'color': "lightyellow"},
                                {'range': [50, 75], 'color': "lightcoral"},
                                {'range': [75, 100], 'color': "lightpink"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90}}))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Risk Category", risk_category)
                    st.metric("Confidence Score", f"{result['confidence']:.1%}")
                    st.metric("Risk Probability", f"{risk_score:.1%}")
                
                # Individual model predictions
                st.subheader("Model Breakdown")
                model_cols = st.columns(3)
                model_names = {"rf": "Random Forest", "gb": "Gradient Boosting", "lr": "Logistic Regression"}
                
                for i, (model_name, pred) in enumerate(result['individual_predictions'].items()):
                    with model_cols[i]:
                        st.metric(model_names[model_name], f"{pred:.1%}")
                
                # Feature importance and recommendations
                explanations = st.session_state.predictor.get_explanations(features)
                if explanations:
                    st.subheader("Key Risk Factors")
                    
                    # Feature importance chart
                    importance_df = pd.DataFrame(list(explanations['feature_importance'].items()), 
                                               columns=['Feature', 'Importance'])
                    importance_df['Feature'] = importance_df['Feature'].map({
                        'age': 'Age', 'sex': 'Sex', 'bmi': 'BMI', 
                        'blood_pressure': 'Blood Pressure', 'cholesterol': 'Cholesterol',
                        'glucose': 'Glucose', 'heart_rate': 'Heart Rate',
                        'exercise_tolerance': 'Exercise Tolerance', 'family_history': 'Family History'
                    })
                    
                    fig = px.bar(importance_df.head(5), x='Importance', y='Feature', orientation='h',
                                title="Top 5 Risk Factors", color='Importance', 
                                color_continuous_scale='Reds')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("Recommended Actions")
                recommendations = generate_recommendations(features, risk_score)
                for rec in recommendations:
                    st.write(f"• {rec}")
                
                # Risk factors breakdown
                st.subheader("Risk Factors Analysis")
                risk_factors = analyze_risk_factors(features)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**High Risk Factors:**")
                    for factor in risk_factors['high']:
                        st.write(f"🔴 {factor}")
                
                with col2:
                    st.write("**Moderate Risk Factors:**")
                    for factor in risk_factors['moderate']:
                        st.write(f"🟡 {factor}")

def dataset_info_page():
    st.header("Dataset Information")
    
    st.subheader("📊 Dataset Status")
    
    # Check dataset availability
    heart_exists = os.path.exists('heart_disease.csv')
    diabetes_exists = os.path.exists('diabetes.csv')
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Heart Disease Dataset")
        if heart_exists:
            st.success("✅ heart_disease.csv found")
            try:
                heart_df = pd.read_csv('heart_disease.csv')
                st.write(f"**Shape:** {heart_df.shape}")
                st.write(f"**Columns:** {', '.join(heart_df.columns[:5])}...")
                if 'HeartDisease' in heart_df.columns:
                    st.write(f"**Target distribution:** {heart_df['HeartDisease'].value_counts().to_dict()}")
            except Exception as e:
                st.error(f"Error reading heart_disease.csv: {e}")
        else:
            st.error("❌ heart_disease.csv not found")
    
    with col2:
        st.markdown("### Diabetes Dataset")
        if diabetes_exists:
            st.success("✅ diabetes.csv found")
            try:
                diabetes_df = pd.read_csv('diabetes.csv')
                st.write(f"**Shape:** {diabetes_df.shape}")
                st.write(f"**Columns:** {', '.join(diabetes_df.columns[:5])}...")
                if 'Outcome' in diabetes_df.columns:
                    st.write(f"**Target distribution:** {diabetes_df['Outcome'].value_counts().to_dict()}")
            except Exception as e:
                st.error(f"Error reading diabetes.csv: {e}")
        else:
            st.error("❌ diabetes.csv not found")
    
    st.subheader("📋 Required Dataset Information")
    st.markdown("""
    ### 1. Heart Disease Prediction Dataset
    - **File name**: `heart_disease.csv`
    - **Expected columns**: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease
    
    ### 2. Pima Indians Diabetes Database
    - **File name**: `diabetes.csv`
    - **Expected columns**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
    """)
    
    st.subheader("🔄 Data Processing Pipeline")
    st.markdown("""
    The system automatically:
    - **Combines** both datasets into a unified format
    - **Standardizes** column names and formats
    - **Handles** missing values and outliers
    - **Creates** synthetic features where needed (e.g., BMI for heart disease data)
    - **Normalizes** feature ranges for model training
    - **Splits** data for training and validation
    """)
    
    if heart_exists and diabetes_exists:
        st.subheader("📈 Dataset Preview")
        if st.button("Load Dataset Preview"):
            try:
                predictor_temp = ChronicCareRiskPredictor()
                combined_data = predictor_temp.load_and_prepare_data()
                if combined_data is not None:
                    st.write("**Combined Dataset Shape:**", combined_data.shape)
                    st.write("**Feature Summary:**")
                    st.dataframe(combined_data.describe())
                    
                    # Show correlation matrix
                    feature_cols = ['age', 'sex', 'bmi', 'blood_pressure', 'cholesterol', 
                                   'glucose', 'heart_rate', 'exercise_tolerance', 'family_history']
                    corr_matrix = combined_data[feature_cols + ['target']].corr()
                    
                    fig = px.imshow(corr_matrix, 
                                   title="Feature Correlation Matrix",
                                   color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading dataset preview: {e}")

def about_page():
    st.header("About the Chronic Care Risk Predictor")
    
    st.markdown("""
    ### 🎯 Purpose
    This AI-powered system predicts the 90-day deterioration risk for chronic care patients using machine learning ensemble models.
    
    ### 🔬 Methodology
    - **Ensemble Learning**: Combines Random Forest, Gradient Boosting, and Logistic Regression
    - **Feature Engineering**: Processes clinical data, vitals, and lab results
    - **Risk Stratification**: Categorizes patients into Low, Moderate, High, and Critical risk
    - **Text Processing**: Extracts medical information from clinical notes
    
    ### 📊 Model Performance
    - **Random Forest**: Tree-based ensemble for handling non-linear relationships
    - **Gradient Boosting**: Sequential learning for complex pattern recognition
    - **Logistic Regression**: Linear baseline with interpretable coefficients
    - **Ensemble Weighting**: 40% RF, 40% GB, 20% LR for optimal performance
    
    ### 🎨 Features
    - **Multiple Input Methods**: Manual entry, text parsing, PDF processing
    - **Real-time Predictions**: Instant risk assessment
    - **Visual Analytics**: Interactive charts and risk gauges
    - **Clinical Recommendations**: Actionable insights for care teams
    - **Model Explainability**: Feature importance and risk factor analysis
    
    ### 🏥 Clinical Applications
    - **Risk Stratification**: Identify high-risk patients
    - **Resource Allocation**: Prioritize care interventions
    - **Preventive Care**: Early intervention opportunities
    - **Care Coordination**: Support clinical decision-making
    
    ### ⚠️ Important Disclaimers
    - **Educational Purpose**: This is a prototype for research and educational use
    - **Not for Clinical Use**: Not intended for actual medical decision-making
    - **Physician Oversight**: Always consult healthcare professionals
    - **Data Privacy**: Ensure patient data protection in real implementations
    
    ### 🛠️ Technical Stack
    - **ML Framework**: Scikit-learn
    - **Interface**: Streamlit
    - **Visualization**: Plotly
    - **Data Processing**: Pandas, NumPy
    - **Text Processing**: Regular Expressions
    - **PDF Processing**: PyPDF2 (optional)
    """)
    
    st.subheader("🚀 Getting Started")
    st.markdown("""
    1. **Prepare Datasets**: Place `heart_disease.csv` and `diabetes.csv` in the same directory
    2. **Train Models**: Go to Model Training page and click "Train Models"
    3. **Make Predictions**: Use the Risk Prediction page to assess patient risk
    4. **Interpret Results**: Review risk scores, recommendations, and explanations
    """)

def generate_recommendations(features, risk_score):
    """Generate personalized recommendations based on risk factors"""
    recommendations = []
    
    # Risk level recommendations
    if risk_score > 0.75:
        recommendations.append("🚨 **Critical Risk**: Immediate medical evaluation recommended")
        recommendations.append("📞 Contact physician within 24 hours")
        recommendations.append("🏥 Consider emergency department evaluation if symptoms worsen")
    elif risk_score > 0.5:
        recommendations.append("⚠️ **High Risk**: Schedule follow-up within 1-2 weeks")
        recommendations.append("📋 Comprehensive medical evaluation recommended")
    elif risk_score > 0.25:
        recommendations.append("📋 **Moderate Risk**: Routine monitoring recommended")
        recommendations.append("👨‍⚕️ Schedule regular check-ups every 3-6 months")
    else:
        recommendations.append("✅ **Low Risk**: Continue current care plan")
        recommendations.append("🔄 Annual health maintenance visits")
    
    # Specific recommendations based on features
    if features['blood_pressure'] > 140:
        recommendations.append("🩺 **Hypertension Management**: Blood pressure control needed")
        recommendations.append("💊 Consider antihypertensive medication review")
        recommendations.append("🧂 Reduce sodium intake (<2300mg/day)")
    
    if features['glucose'] > 140:
        recommendations.append("🍎 **Diabetes Management**: Glucose control and dietary consultation")
        recommendations.append("📊 Monitor HbA1c levels quarterly")
        recommendations.append("🥗 Diabetes-friendly diet consultation")
    
    if features['cholesterol'] > 240:
        recommendations.append("💊 **Lipid Management**: Cholesterol evaluation needed")
        recommendations.append("🍳 Low-saturated fat diet")
        recommendations.append("📈 Lipid panel monitoring every 3-6 months")
    
    if features['bmi'] > 30:
        recommendations.append("🏃‍♀️ **Weight Management**: Obesity intervention recommended")
        recommendations.append("🥗 Nutritionist consultation")
        recommendations.append("💪 Structured exercise program")
    elif features['bmi'] > 25:
        recommendations.append("⚖️ **Weight Monitoring**: Maintain healthy weight")
    
    if features['exercise_tolerance'] == 0:
        recommendations.append("💪 **Exercise Tolerance**: Cardiac rehabilitation assessment")
        recommendations.append("🏃‍♂️ Gradual exercise program initiation")
        recommendations.append("💓 Cardiac stress test consideration")
    
    if features['family_history'] == 1:
        recommendations.append("🧬 **Genetic Risk**: Enhanced screening due to family history")
        recommendations.append("📅 Earlier and more frequent preventive screenings")
    
    if features['heart_rate'] > 100:
        recommendations.append("💓 **Tachycardia**: Heart rate evaluation recommended")
    elif features['heart_rate'] < 60:
        recommendations.append("💓 **Bradycardia**: Heart rate monitoring needed")
    
    # Age-specific recommendations
    if features['age'] > 65:
        recommendations.append("👴 **Senior Care**: Age-specific health monitoring")
        recommendations.append("💉 Vaccination status review (flu, pneumonia)")
    
    return recommendations

def analyze_risk_factors(features):
    """Analyze individual risk factors and categorize them"""
    risk_factors = {
        'high': [],
        'moderate': [],
        'low': []
    }
    
    # Age analysis
    if features['age'] > 70:
        risk_factors['high'].append(f"Advanced age ({features['age']} years)")
    elif features['age'] > 50:
        risk_factors['moderate'].append(f"Middle age ({features['age']} years)")
    else:
        risk_factors['low'].append(f"Younger age ({features['age']} years)")
    
    # Blood pressure analysis
    if features['blood_pressure'] > 160:
        risk_factors['high'].append(f"Severe hypertension ({features['blood_pressure']} mmHg)")
    elif features['blood_pressure'] > 140:
        risk_factors['moderate'].append(f"Stage 2 hypertension ({features['blood_pressure']} mmHg)")
    elif features['blood_pressure'] > 130:
        risk_factors['moderate'].append(f"Stage 1 hypertension ({features['blood_pressure']} mmHg)")
    
    # Cholesterol analysis
    if features['cholesterol'] > 260:
        risk_factors['high'].append(f"Very high cholesterol ({features['cholesterol']} mg/dL)")
    elif features['cholesterol'] > 240:
        risk_factors['moderate'].append(f"High cholesterol ({features['cholesterol']} mg/dL)")
    elif features['cholesterol'] > 200:
        risk_factors['moderate'].append(f"Borderline high cholesterol ({features['cholesterol']} mg/dL)")
    
    # Glucose analysis
    if features['glucose'] > 180:
        risk_factors['high'].append(f"Very high glucose ({features['glucose']} mg/dL)")
    elif features['glucose'] > 140:
        risk_factors['moderate'].append(f"High glucose ({features['glucose']} mg/dL)")
    elif features['glucose'] > 100:
        risk_factors['moderate'].append(f"Impaired fasting glucose ({features['glucose']} mg/dL)")
    
    # BMI analysis
    if features['bmi'] > 35:
        risk_factors['high'].append(f"Class II obesity (BMI {features['bmi']:.1f})")
    elif features['bmi'] > 30:
        risk_factors['moderate'].append(f"Obesity (BMI {features['bmi']:.1f})")
    elif features['bmi'] > 25:
        risk_factors['moderate'].append(f"Overweight (BMI {features['bmi']:.1f})")
    
    # Exercise tolerance
    if features['exercise_tolerance'] == 0:
        risk_factors['high'].append("Poor exercise tolerance")
    
    # Family history
    if features['family_history'] == 1:
        risk_factors['moderate'].append("Positive family history")
    
    # Heart rate analysis
    if features['heart_rate'] > 100:
        risk_factors['moderate'].append(f"Tachycardia ({features['heart_rate']} bpm)")
    elif features['heart_rate'] < 60:
        risk_factors['moderate'].append(f"Bradycardia ({features['heart_rate']} bpm)")
    
    return risk_factors

if __name__ == "__main__":
    main()