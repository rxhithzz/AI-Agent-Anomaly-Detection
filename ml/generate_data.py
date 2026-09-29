"""

Generate Synthetic Transaction Data

Creates 5000 normal transactions and 250 anomalous transactions

"""

import pandas as pd

import numpy as np

import os

# Set random seed for reproducibility

np.random.seed(42)

def generate_normal_transactions(count=5000):

    """Generate normal transaction data"""

    data = {

        'amount': np.random.uniform(100, 50000, count),

        'hour': np.random.randint(6, 23, count),

        'login_attempts': np.random.randint(1, 3, count),

        'response_time': np.random.uniform(100, 500, count),

        'transaction_frequency': np.random.randint(1, 10, count),

        'location_change': np.random.choice([0, 0, 0, 1], count),

        'label': 0

    }

    return pd.DataFrame(data)

def generate_anomaly_transactions(count=250):

    """Generate anomalous transaction data"""

    data = {

        'amount': np.random.uniform(50000, 500000, count),

        'hour': np.random.choice([0, 1, 2, 3, 4, 5], count),

        'login_attempts': np.random.randint(5, 15, count),

        'response_time': np.random.uniform(800, 2000, count),

        'transaction_frequency': np.random.randint(15, 30, count),

        'location_change': np.random.choice([1, 1, 1, 0], count),

        'label': 1

    }

    return pd.DataFrame(data)

def create_dataset():

    """Create and save the complete dataset"""

    # Generate data

    print("Generating dataset...")

    normal_data = generate_normal_transactions(5000)

    anomaly_data = generate_anomaly_transactions(250)

    

    # Combine datasets

    dataset = pd.concat([normal_data, anomaly_data], ignore_index=True)

    

    # Shuffle the dataset

    dataset = dataset.sample(frac=1).reset_index(drop=True)

    

    # Create data directory if it doesn't exist

    os.makedirs('data', exist_ok=True)

    

    # Save to CSV

    dataset.to_csv('data/transactions.csv', index=False)

    

    # Print statistics

    print("\n" + "="*50)

    print("Dataset created successfully!")

    print("="*50)

    print(f"Total records: {len(dataset)}")

    print(f"Normal records: {len(normal_data)}")

    print(f"Anomaly records: {len(anomaly_data)}")

    print("="*50)

    print("\nDataset saved at: data/transactions.csv")

if __name__ == "__main__":

    create_dataset()
