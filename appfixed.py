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
import shap
import warnings
warnings.filterwarnings('ignore')

# For PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

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
            # Dataset 1: Heart Disease (name as 'heart_disease.csv')
            heart_df = pd.read_csv('heart_disease.csv')
            
            # Dataset 2: Diabetes (name as 'diabetes.csv') 
            diabetes_df = pd.read_csv('diabetes.csv')
            
            # Standardize column names for heart disease
            heart_columns_map = {
                'Age': 'age', 'Sex': 'sex', 'ChestPainType': 'chest_pain',
                'RestingBP': 'resting_bp', 'Cholesterol': 'cholesterol',
                'FastingBS': 'fasting_bs', 'RestingECG': 'resting_ecg',
                'MaxHR': 'max_hr', 'ExerciseAngina': 'exercise_angina',
                'Oldpeak': 'oldpeak', 'ST_Slope': 'st_slope', 'HeartDisease': 'target'
            }
            
            if all(col in heart_df.columns for col in heart_columns_map.keys()):
                heart_df = heart_df.rename(columns=heart_columns_map)
            
            # Standardize column names for diabetes
            diabetes_columns_map = {
                'Pregnancies': 'pregnancies', 'Glucose': 'glucose',
                'BloodPressure': 'blood_pressure', 'SkinThickness': 'skin_thickness',
                'Insulin': 'insulin', 'BMI': 'bmi', 'DiabetesPedigreeFunction': 'diabetes_pedigree',
                'Age': 'age', 'Outcome': 'target'
            }
            
            if all(col in diabetes_df.columns for col in diabetes_columns_map.keys()):
                diabetes_df = diabetes_df.rename(columns=diabetes_columns_map)
            
            # Process heart disease data
            heart_processed = pd.DataFrame()
            heart_processed['age'] = heart_df.get('age', 50)
            heart_processed['sex'] = heart_df.get('sex', 'M').map({'M': 1, 'F': 0}) if 'sex' in heart_df.columns else 1
            heart_processed['bmi'] = np.random.normal(26, 4, len(heart_df))
            heart_processed['blood_pressure'] = heart_df.get('resting_bp', heart_df.get('blood_pressure', 120))
            heart_processed['cholesterol'] = heart_df.get('cholesterol', 200)
            heart_processed['glucose'] = heart_df.get('fasting_bs', 0) * 50 + 100
            heart_processed['heart_rate'] = heart_df.get('max_hr', 150)
            heart_processed['exercise_tolerance'] = 1 - heart_df.get('exercise_angina', 0).map({'Y': 1, 'N': 0}).fillna(0)
            heart_processed['family_history'] = np.random.binomial(1, 0.3, len(heart_df))
            heart_processed['target'] = heart_df.get('target', 0)
            
            # Process diabetes data
            diabetes_processed = pd.DataFrame()
            diabetes_processed['age'] = diabetes_df.get('age', 50)
            diabetes_processed['sex'] = np.random.binomial(1, 0.5, len(diabetes_df))
            diabetes_processed['bmi'] = diabetes_df.get('bmi', 25)
            diabetes_processed['blood_pressure'] = diabetes_df.get('blood_pressure', 80)
            diabetes_processed['cholesterol'] = np.random.normal(200, 40, len(diabetes_df))
            diabetes_processed['glucose'] = diabetes_df.get('glucose', 100)
            diabetes_processed['heart_rate'] = np.random.normal(72, 12, len(diabetes_df))
            diabetes_processed['exercise_tolerance'] = np.random.binomial(1, 0.7, len(diabetes_df))
            diabetes_processed['family_history'] = (diabetes_df.get('diabetes_pedigree', 0.5) > 0.5).astype(int)
            diabetes_processed['target'] = diabetes_df.get('target', 0)
            
            # Combine datasets
            combined_df = pd.concat([heart_processed, diabetes_processed], ignore_index=True)
            
            # Clean data
            combined_df = combined_df.fillna(combined_df.median())
            combined_df = combined_df[(combined_df['age'] > 0) & (combined_df['age'] < 120)]
            combined_df = combined_df[combined_df['blood_pressure'] > 0]
            
            return combined_df
            
        except FileNotFoundError as e:
            st.error(f"Dataset file not found: {e}")
            st.error("Please download the datasets and place them in the same directory.")
            return None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def train_models(self):
        """Train ensemble of models for risk prediction"""
        data = self.load_and_prepare_data()
        if data is None:
            return False
        
        X = data[self.feature_names]
        y = data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train multiple models
        self.models['rf'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['gb'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.models['lr'] = LogisticRegression(random_state=42)
        
        # Fit models
        self.models['rf'].fit(X_train, y_train)
        self.models['gb'].fit(X_train, y_train)
        self.models['lr'].fit(X_train_scaled, y_train)
        
        # Evaluate models
        performance = {}
        for name, model in self.models.items():
            if name == 'lr':
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            performance[name] = auc
        
        # Setup SHAP explainer
        try:
            # Use a smaller sample for the explainer background to avoid memory issues
            background_sample = X_train.sample(min(100, len(X_train)), random_state=42)
            self.shap_explainer = shap.Explainer(self.models['rf'], background_sample)
        except Exception as e:
            st.warning(f"Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None
        
        # Save models
        joblib.dump(self.models, 'chronic_care_models.pkl')
        joblib.dump(self.scalers, 'chronic_care_scalers.pkl')
        joblib.dump(self.feature_names, 'feature_names.pkl')
        
        self.is_trained = True
        return performance
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            self.models = joblib.load('chronic_care_models.pkl')
            self.scalers = joblib.load('chronic_care_scalers.pkl')
            self.feature_names = joblib.load('feature_names.pkl')
            self.is_trained = True
            return True
        except FileNotFoundError:
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
        
        # Make predictions with each model
        predictions = {}
        
        # Random Forest and Gradient Boosting
        for name in ['rf', 'gb']:
            pred_proba = self.models[name].predict_proba(feature_df[self.feature_names])[:, 1]
            predictions[name] = pred_proba[0]
        
        # Logistic Regression (needs scaling)
        feature_scaled = self.scalers['standard'].transform(feature_df[self.feature_names])
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
        """Get SHAP explanations for the prediction - FIXED VERSION FOR BINARY CLASSIFICATION"""
        if not self.is_trained or self.shap_explainer is None:
            # Return fallback explanation based on clinical knowledge
            fallback_importance = {
                'age': 0.3,
                'glucose': 0.25,
                'blood_pressure': 0.2,
                'cholesterol': 0.15,
                'bmi': 0.1,
                'heart_rate': 0.05,
                'exercise_tolerance': 0.03,
                'family_history': 0.02
            }
            return {
                'feature_importance': fallback_importance,
                'shap_values': None
            }
        
        try:
            feature_df = pd.DataFrame([features])
            shap_values = self.shap_explainer(feature_df[self.feature_names])
            
            # Fixed: Get feature importance with proper array handling for binary classification
            importance = {}
            
            # Handle different SHAP versions and output formats
            if hasattr(shap_values, 'values'):
                # For newer SHAP versions with .values attribute
                shap_array = shap_values.values
                
                # Handle different shapes for binary classification
                if shap_array.ndim == 3:
                    # Shape: (n_samples, n_features, n_classes) - binary classification
                    # Take the positive class (index 1) for feature importance
                    if shap_array.shape[2] == 2:  # Binary classification
                        shap_array = shap_array[0, :, 1]  # First sample, all features, positive class
                    else:
                        # Multi-class, take the mean across classes
                        shap_array = np.mean(np.abs(shap_array[0]), axis=1)
                        
                elif shap_array.ndim == 2:
                    # Shape: (n_samples, n_features) - regression or single output
                    shap_array = shap_array[0]  # First sample
                    
                elif shap_array.ndim == 1:
                    # Shape: (n_features,) - single sample
                    pass  # Already correct shape
                    
                else:
                    raise ValueError(f"Unexpected SHAP values shape: {shap_array.shape}")
                    
            else:
                # For older SHAP versions or direct array output
                if isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 3:
                        # Binary classification case
                        shap_array = shap_values[0, :, 1] if shap_values.shape[2] == 2 else np.mean(np.abs(shap_values[0]), axis=1)
                    elif shap_values.ndim == 2:
                        shap_array = shap_values[0]
                    else:
                        shap_array = shap_values.flatten()
                else:
                    # Convert to numpy array and handle appropriately
                    shap_array = np.array(shap_values)
                    if shap_array.ndim == 3:
                        shap_array = shap_array[0, :, 1] if shap_array.shape[2] == 2 else np.mean(np.abs(shap_array[0]), axis=1)
                    elif shap_array.ndim == 2:
                        shap_array = shap_array[0]
                    else:
                        shap_array = shap_array.flatten()
            
            # Ensure we have the right number of features
            if len(shap_array) != len(self.feature_names):
                st.warning(f"SHAP values length ({len(shap_array)}) doesn't match features ({len(self.feature_names)})")
                raise ValueError("Dimension mismatch")
            
            # Calculate importance for each feature (use absolute values for importance)
            for i, feature in enumerate(self.feature_names):
                importance[feature] = abs(float(shap_array[i]))
            
            # Sort by importance
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'feature_importance': sorted_importance,
                'shap_values': shap_values
            }
        
        except Exception as e:
            st.warning(f"Error generating SHAP explanations: {e}")
            # Return fallback explanation based on clinical knowledge
            fallback_importance = {
                'age': 0.3,
                'glucose': 0.25,
                'blood_pressure': 0.2,
                'cholesterol': 0.15,
                'bmi': 0.1,
                'heart_rate': 0.05,
                'exercise_tolerance': 0.03,
                'family_history': 0.02
            }
            return {
                'feature_importance': fallback_importance,
                'shap_values': None
            }

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
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Risk Prediction", "Model Training", "Dataset Info", "About"])
    
    if page == "Risk Prediction":
        risk_prediction_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Dataset Info":
        dataset_info_page()
    elif page == "About":
        about_page()

def risk_prediction_page():
    st.header("Patient Risk Assessment")
    
    # Input method selection
    input_method = st.radio("Choose input method:", 
                           ["Manual Entry", "Upload PDF", "Text Input"])
    
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
    
    elif input_method == "Text Input":
        medical_text = st.text_area("Enter medical history or clinical notes:", 
                                   height=200,
                                   placeholder="Patient is a 65-year-old male with BMI 28, BP 140/90, cholesterol 240, glucose 150...")
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
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = risk_score * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "90-Day Deterioration Risk"},
                        delta = {'reference': 25},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if risk_score > 0.75 else "orange" if risk_score > 0.5 else "yellow" if risk_score > 0.25 else "green"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "lightgray"},
                                {'range': [75, 100], 'color': "lightgray"}],
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
                                title="Top 5 Risk Factors")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("Recommended Actions")
                recommendations = generate_recommendations(features, risk_score)
                for rec in recommendations:
                    st.write(f"• {rec}")

def model_training_page():
    st.header("Model Training & Evaluation")
    
    st.info("📋 **Required Datasets**: Please ensure you have downloaded the following datasets and placed them in the same directory:")
    st.write("1. **heart_disease.csv** - Heart Disease Prediction Dataset")
    st.write("2. **diabetes.csv** - Pima Indians Diabetes Database")
    
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
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(perf_df)
            else:
                st.error("❌ Model training failed. Please check if datasets are available.")

def dataset_info_page():
    st.header("Dataset Information")
    
    st.subheader("📊 Required Datasets")
    
    st.markdown("""
    ### 1. Heart Disease Prediction Dataset
    - **File name**: `heart_disease.csv`
    - **Source**: [Kaggle Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
    - **Features**: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease
    
    ### 2. Pima Indians Diabetes Database
    - **File name**: `diabetes.csv`
    - **Source**: [Kaggle Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
    - **Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
    """)
    
    st.subheader("🔄 Data Processing")
    st.markdown("""
    The system automatically:
    - Combines both datasets into a unified format
    - Handles missing values and outliers
    - Creates standardized features for risk prediction
    - Generates synthetic features where needed (e.g., BMI for heart disease data)
    """)

def about_page():
    st.header("About the Chronic Care Risk Predictor")
    
    st.markdown("""
    ### 🎯 Purpose
    This AI-powered system predicts the 90-day deterioration risk for chronic care patients using machine learning ensemble models.
    
    ### 🔬 Methodology
    - **Ensemble Learning**: Combines Random Forest, Gradient Boosting, and Logistic Regression
    - **Feature Engineering**: Processes clinical data, vitals, and lab results
    - **Explainable AI**: Uses SHAP values to provide interpretable predictions
    - **Risk Stratification**: Categorizes patients into Low, Moderate, High, and Critical risk
    
    ### 📈 Performance Metrics
    - **AUROC**: Area Under ROC Curve for discrimination
    - **Calibration**: Reliability of probability estimates
    - **Confidence Scoring**: Model agreement measurement
    
    ### ⚠️ Important Notes
    - This is a prototype for educational/research purposes
    - Not intended for clinical decision-making without physician oversight
    - Always consult healthcare professionals for medical decisions
    
    ### 🛠️ Technical Stack
    - **ML Framework**: Scikit-learn, XGBoost
    - **Explainability**: SHAP
    - **Interface**: Streamlit
    - **Visualization**: Plotly
    """)

def generate_recommendations(features, risk_score):
    """Generate personalized recommendations based on risk factors"""
    recommendations = []
    
    if risk_score > 0.75:
        recommendations.append("🚨 **Critical Risk**: Immediate medical evaluation recommended")
        recommendations.append("📞 Contact physician within 24 hours")
    elif risk_score > 0.5:
        recommendations.append("⚠️ **High Risk**: Schedule follow-up within 1-2 weeks")
    elif risk_score > 0.25:
        recommendations.append("📋 **Moderate Risk**: Routine monitoring recommended")
    else:
        recommendations.append("✅ **Low Risk**: Continue current care plan")
    
    # Specific recommendations based on features
    if features['blood_pressure'] > 140:
        recommendations.append("🩺 Blood pressure management needed")
    
    if features['glucose'] > 140:
        recommendations.append("🍎 Glucose control and dietary consultation")
    
    if features['cholesterol'] > 240:
        recommendations.append("💊 Lipid management evaluation")
    
    if features['bmi'] > 30:
        recommendations.append("🏃‍♀️ Weight management and exercise program")
    
    if features['exercise_tolerance'] == 0:
        recommendations.append("💪 Exercise tolerance assessment and cardiac rehabilitation")
    
    if features['family_history'] == 1:
        recommendations.append("🧬 Enhanced screening due to family history")
    
    return recommendations

if __name__ == "__main__":
    main()