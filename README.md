# Chronic Care Risk Predictor

This project is a Streamlit-based web application for predicting the risk of chronic diseases such as heart disease and diabetes. It provides functionalities for data loading, model training, risk prediction, and personalized recommendations.

## Features

- Load and preprocess datasets for heart disease and diabetes.
- Train machine learning models (Random Forest, Gradient Boosting, Logistic Regression) to predict disease risk.
- Interactive Streamlit interface for users to input their health data and get risk predictions.
- Visualization of risk factors and model explanations using SHAP.
- Ability to upload and process PDF reports (if PyPDF2 is installed).
- Detailed dataset information and model training pages.
- Personalized health recommendations based on risk factors.

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run appfixed.py
```

## Files

- `appfixed.py`: Main application code including model training, prediction, and Streamlit UI.
- `test_app.py`: Test script for the application.
- `heart_disease.csv`: Dataset for heart disease.
- `diabetes.csv`: Dataset for diabetes.
- `requirements.txt`: Python dependencies.

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- plotly
- shap
- joblib
- PyPDF2 (optional, for PDF processing)

## Notes

- Ensure datasets `heart_disease.csv` and `diabetes.csv` are present in the working directory.
- SHAP explanations require the `shap` package.
- PDF report processing is optional and requires `PyPDF2`.

## License

This project is licensed under the MIT License.

---

Feel free to contribute or raise issues for improvements!
