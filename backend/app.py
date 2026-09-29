import os
from dotenv import load_dotenv

load_dotenv()
"""
REST API for Anomaly Detection
Flask backend that serves predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Load the trained model and features
try:
    model = joblib.load('ml/model.pkl')
    features = joblib.load('ml/features.pkl')
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("ERROR: Model files not found!")
    print("Please run: python ml/train_model.py")
    model = None
    features = None

# Root route
@app.route('/', methods=['GET'])
def home():
    """Home page"""
    return jsonify({
        'message': 'Anomaly Detection API',
        'version': '1.0',
        'status': 'running'
    })

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if a transaction is normal or anomalous
    
    Expected JSON:
    {
        "amount": 5000,
        "hour": 14,
        "login_attempts": 1,
        "response_time": 150,
        "transaction_frequency": 2,
        "location_change": 0
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['amount', 'hour', 'login_attempts', 'response_time', 
                          'transaction_frequency', 'location_change']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing field: {field}',
                    'required_fields': required_fields
                }), 400
        
        # Create DataFrame with the data
        transaction = pd.DataFrame([data])
        
        # Select only required features in correct order
        X = transaction[features].values
        
        # Make prediction
        prediction = model.predict(X)[0]
        anomaly_score = model.score_samples(X)[0]
        
        # Determine result
        result = 'NORMAL' if prediction == 1 else 'ANOMALY'
        
        # Return response
        return jsonify({
            'transaction': data,
            'prediction': int(prediction),
            'anomaly_score': float(anomaly_score),
            'result': result,
            'confidence': float(abs(anomaly_score)) * 100
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# Health check route
@app.route('/health', methods=['GET'])
def health():
    """Check if API is working"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'features': features
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Route not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=False, port=port)