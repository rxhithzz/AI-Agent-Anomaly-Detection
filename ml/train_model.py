"""
Train Isolation Forest Model
Trains on normal transactions only
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def load_dataset():
    """Load the generated dataset"""
    try:
        df = pd.read_csv('data/transactions.csv')
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("ERROR: data/transactions.csv not found!")
        print("Please run: python ml/generate_data.py")
        return None

def prepare_features(df):
    """Prepare features for training"""
    # Select only normal transactions for training
    normal_df = df[df['label'] == 0].copy()
    
    # Features to use for anomaly detection
    features = ['amount', 'hour', 'login_attempts', 'response_time', 
                'transaction_frequency', 'location_change']
    
    X = normal_df[features].values
    
    return X, features, normal_df

def train_model(X, features):
    """Train Isolation Forest model"""
    print("\nTraining Isolation Forest model...")
    
    # Initialize and train the model
    model = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=100
    )
    
    # Fit the model
    model.fit(X)
    
    print("Model trained successfully!")
    return model

def save_model(model, features):
    """Save the trained model"""
    # Create ml directory if it doesn't exist
    os.makedirs('ml', exist_ok=True)
    
    # Save model using joblib
    joblib.dump(model, 'ml/model.pkl')
    
    # Save features list for reference
    joblib.dump(features, 'ml/features.pkl')
    
    print("Model saved as: ml/model.pkl")
    print("Features saved as: ml/features.pkl")

def main():
    """Main training pipeline"""
    print("="*50)
    print("TRAINING PHASE - Isolation Forest")
    print("="*50)
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return
    
    # Prepare features
    X, features, normal_df = prepare_features(df)
    
    print(f"\nTraining records (normal transactions): {len(X)}")
    print(f"Features used: {features}")
    
    # Train model
    model = train_model(X, features)
    
    # Save model
    save_model(model, features)
    
    print("\n" + "="*50)
    print("✅ Training Complete!")
    print("="*50)
    print(f"Model trained on {len(X)} normal transactions")
    print("="*50)

if __name__ == "__main__":
    main()