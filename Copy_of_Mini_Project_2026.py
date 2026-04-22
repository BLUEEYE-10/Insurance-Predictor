import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

print("Loading insurance.csv...")
insurance_dataset = pd.read_csv('insurance.csv')
insurance_dataset = insurance_dataset.dropna()

print("Data Analysis")
print(insurance_dataset.describe())

X = insurance_dataset.drop(columns='charges')
Y = insurance_dataset['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

print("Training the XGBoost Pipeline...")
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

pipeline.fit(X_train, Y_train)
print('Accuracy on training set: {:.2f}'.format(pipeline.score(X_train, Y_train)))
print('Accuracy on testing set: {:.2f}'.format(pipeline.score(X_test, Y_test)))

joblib.dump(pipeline, "insurance_model.pkl")
print("Model saved to insurance_model.pkl")

# --- Flask App ---
print("Starting Flask Backend...")
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

CSV_FILE = 'collected_data.csv'

@app.route('/')
def home():
    # Serve the HTML frontend
    return send_from_directory('.', 'insure.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Ensure correct types are parsed from JSON
        df_data = {
            'age': [int(data.get('age', 0))],
            'sex': [str(data.get('sex', 'male'))],
            'bmi': [float(data.get('bmi', 0))],
            'children': [int(data.get('children', 0))],
            'smoker': [str(data.get('smoker', 'no'))],
            'region': [str(data.get('region', 'southwest'))]
        }
        input_df = pd.DataFrame(df_data)

        # Log prediction query to CSV
        if not os.path.isfile(CSV_FILE):
            input_df.to_csv(CSV_FILE, index=False)
        else:
            input_df.to_csv(CSV_FILE, mode='a', header=False, index=False)

        prediction = pipeline.predict(input_df)
        return jsonify({'estimated_charges': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Server running locally on http://127.0.0.1:5000")
    print("Press Ctrl+C to exit.")
    app.run(port=5000, debug=False)
