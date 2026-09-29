"""

Detect Anomalies Using Trained Model

Tests the model with sample transactions

"""

import pandas as pd

import numpy as np

import joblib

def load_model():

    """Load the trained model"""

    try:

        model = joblib.load('ml/model.pkl')

        features = joblib.load('ml/features.pkl')

        print("✅ Model loaded successfully!")

        return model, features

    except FileNotFoundError:

        print("ERROR: Model not found!")

        print("Please run: python ml/train_model.py")

        return None, None

def predict_transaction(model, features, transaction_data):

    """Predict if a transaction is normal or anomalous"""

    # Create a DataFrame from the transaction

    df = pd.DataFrame([transaction_data])

    

    # Select only the required features in the correct order

    X = df[features].values

    

    # Make prediction (-1 = anomaly, 1 = normal)

    prediction = model.predict(X)[0]

    

    # Get anomaly score (lower = more anomalous)

    anomaly_score = model.score_samples(X)[0]

    

    return prediction, anomaly_score

def print_result(test_num, transaction_data, prediction, anomaly_score):

    """Pretty print the result"""

    result = "NORMAL ✓" if prediction == 1 else "ANOMALY ⚠️"

    

    print(f"\n{'='*60}")

    print(f"TEST CASE {test_num} - {result}")

    print(f"{'='*60}")

    print(f"Transaction Details:")

    print(f"  Amount: ₹{transaction_data['amount']:,.2f}")

    print(f"  Hour: {transaction_data['hour']}:00")

    print(f"  Login Attempts: {transaction_data['login_attempts']}")

    print(f"  Response Time: {transaction_data['response_time']}ms")

    print(f"  Transaction Frequency: {transaction_data['transaction_frequency']}")

    print(f"  Location Change: {transaction_data['location_change']}")

    print(f"\nModel Prediction: {prediction}")

    print(f"Anomaly Score: {anomaly_score:.4f}")

    print(f"Result: {result}")

    print(f"{'='*60}")

def main():

    """Main anomaly detection pipeline"""

    print("="*60)

    print("ANOMALY DETECTION - Testing Trained Model")

    print("="*60)

    

    # Load model

    model, features = load_model()

    if model is None:

        return

    

    print(f"\nFeatures used by model: {features}\n")

    

    # Test Case 1: NORMAL Transaction

    print("\n📌 TEST CASE 1: Expected - NORMAL")

    test1 = {

        'amount': 2500,

        'hour': 14,

        'login_attempts': 1,

        'response_time': 150,

        'transaction_frequency': 2,

        'location_change': 0

    }

    pred1, score1 = predict_transaction(model, features, test1)

    print_result(1, test1, pred1, score1)

    

    # Test Case 2: ANOMALY Transaction

    print("\n📌 TEST CASE 2: Expected - ANOMALY")

    test2 = {

        'amount': 95000,

        'hour': 3,

        'login_attempts': 8,

        'response_time': 1000,

        'transaction_frequency': 20,

        'location_change': 1

    }

    pred2, score2 = predict_transaction(model, features, test2)

    print_result(2, test2, pred2, score2)

    

    # Summary

    print("\n" + "="*60)

    print("📊 PREDICTION SUMMARY")

    print("="*60)

    print(f"Test 1 (NORMAL): {pred1} → {'✓ CORRECT' if pred1 == 1 else '✗ INCORRECT'}")

    print(f"Test 2 (ANOMALY): {pred2} → {'✓ CORRECT' if pred2 == -1 else '✗ INCORRECT'}")

    print("="*60)

    

    print("\n💡 Understanding Predictions:")

    print("  1  = NORMAL transaction")

    print(" -1  = ANOMALOUS transaction")

    print("\n💡 Understanding Anomaly Scores:")

    print("  Higher score → More likely to be NORMAL")

    print("  Lower score  → More likely to be ANOMALOUS")

    print("\n" + "="*60)

if __name__ == "__main__":

    main()
