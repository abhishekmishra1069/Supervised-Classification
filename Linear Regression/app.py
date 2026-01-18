"""
Linear Regression Classification Web Application
This Flask application provides a web interface for predicting user purchasing behavior
using a trained Linear Regression model.
"""

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import os

# ============================================================================
# INITIALIZE FLASK APPLICATION
# ============================================================================
app = Flask(__name__)

# ============================================================================
# GLOBAL VARIABLES FOR MODEL AND SCALER
# ============================================================================
# Initialize model and scaler as None (will be loaded on startup)
regressor = None
sc = None
model_path = 'linear_regression_classifier.pkl'
scaler_path = 'scaler.pkl'


# ============================================================================
# FUNCTION: LOAD MODEL
# ============================================================================
def load_model():
    """
    Load a pre-trained model and scaler from pickle files.
    Returns the regressor and scaler objects.
    """
    global regressor, sc
    
    try:
        # Check if pickle files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        # Load existing model and scaler from pickle files
        with open(model_path, 'rb') as f:
            regressor = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            sc = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
        print(f"Scaler loaded successfully from {scaler_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please ensure pickle files are generated from linear_regression_classification.ipynb")
        return None, None
    except Exception as e:
        print(f"Error during model loading: {str(e)}")
        return None, None
    
    return regressor, sc


# ============================================================================
# ROUTE: HOME PAGE
# ============================================================================
@app.route('/')
def home():
    """
    Serve the home page with prediction form.
    """
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Linear Regression Classification</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 50px;
                background-color: #f4f4f4;
            }
            .container {
                max-width: 500px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            form {
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            label {
                font-weight: bold;
                color: #555;
            }
            input[type="number"] {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }
            button {
                padding: 10px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
            }
            button:hover {
                background-color: #0b7dda;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 4px;
                text-align: center;
                display: none;
                font-size: 18px;
                font-weight: bold;
            }
            .purchased {
                background-color: #d4edda;
                color: #155724;
            }
            .not-purchased {
                background-color: #f8d7da;
                color: #721c24;
            }
            .probability {
                margin-top: 10px;
                font-size: 14px;
                font-weight: normal;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Linear Regression Classification</h1>
            <p style="text-align: center; color: #666;">Predict if a user will purchase based on age and salary</p>
            <form onsubmit="predict(event)">
                <label for="age">Age:</label>
                <input type="number" id="age" name="age" required min="0" max="150">
                
                <label for="salary">Estimated Salary ($):</label>
                <input type="number" id="salary" name="salary" required min="0">
                
                <button type="submit">Predict</button>
            </form>
            <div id="result"></div>
        </div>
        
        <script>
            // Function to handle form submission and make prediction
            function predict(event) {
                event.preventDefault();
                
                // Get input values
                const age = parseFloat(document.getElementById('age').value);
                const salary = parseFloat(document.getElementById('salary').value);
                
                // Make API request to prediction endpoint
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({age: age, salary: salary})
                })
                .then(response => response.json())
                .then(data => {
                    // Display result
                    const resultDiv = document.getElementById('result');
                    if (data.prediction === 1) {
                        resultDiv.innerHTML = 'Prediction: User will <strong>PURCHASE</strong> ✓<div class="probability">Confidence: ' + (data.probability * 100).toFixed(2) + '%</div>';
                        resultDiv.className = 'purchased';
                    } else {
                        resultDiv.innerHTML = 'Prediction: User will <strong>NOT PURCHASE</strong> ✗<div class="probability">Confidence: ' + ((1 - data.probability) * 100).toFixed(2) + '%</div>';
                        resultDiv.className = 'not-purchased';
                    }
                    resultDiv.style.display = 'block';
                })
                .catch(error => {
                    alert('Error: ' + error);
                });
            }
        </script>
    </body>
    </html>
    '''


# ============================================================================
# ROUTE: PREDICTION ENDPOINT (API)
# ============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to make predictions on new data.
    Expects JSON with 'age' and 'salary' fields.
    Returns JSON with prediction result and probability.
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        age = data.get('age')
        salary = data.get('salary')
        
        # Validate input
        if age is None or salary is None:
            return jsonify({'error': 'Missing age or salary'}), 400
        
        # Scale the input using the trained scaler
        input_scaled = sc.transform([[age, salary]])
        
        # Make prediction (continuous value from 0 to 1)
        prediction_prob = regressor.predict(input_scaled)[0]
        # Convert to binary classification using 0.5 threshold
        prediction = 1 if prediction_prob >= 0.5 else 0
        
        # Return prediction result
        return jsonify({
            'prediction': int(prediction),
            'probability': float(prediction_prob),
            'age': age,
            'salary': salary,
            'message': f'User will {"purchase" if prediction == 1 else "not purchase"}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ROUTE: MODEL STATUS ENDPOINT
# ============================================================================
@app.route('/status', methods=['GET'])
def status():
    """
    API endpoint to check model status and get model information.
    """
    try:
        # Import required modules for evaluation
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
        
        # Load dataset
        dataset = pd.read_csv('Social_Network_Ads.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        # Split data same way as training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        X_test_scaled = sc.transform(X_test)
        
        # Get predictions (continuous)
        y_pred_prob = regressor.predict(X_test_scaled)
        # Convert to binary using 0.5 threshold
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred_prob)
        
        return jsonify({
            'status': 'Model loaded successfully',
            'accuracy': float(accuracy),
            'mse': float(mse),
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    # Load the pre-trained model and scaler on startup
    print("Loading model and scaler from pickle files...")
    regressor, sc = load_model()
    
    # Check if model was successfully loaded
    if regressor is None or sc is None:
        print("Failed to load model. Exiting.")
        exit(1)
    
    # Run Flask application
    print("Starting Flask application on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
