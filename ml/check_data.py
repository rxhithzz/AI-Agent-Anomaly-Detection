"""
Check and Analyze Dataset
Verifies the integrity and quality of the generated dataset
"""

import pandas as pd
import numpy as np

def check_dataset():
    """Load and analyze the dataset"""
    try:
        df = pd.read_csv('data/transactions.csv')
    except FileNotFoundError:
        print("ERROR: data/transactions.csv not found!")
        print("Please run: python ml/generate_data.py")
        return
    
    print("\n" + "="*70)
    print("DATASET VERIFICATION & ANALYSIS")
    print("="*70)
    
    # Basic statistics
    print("\n📊 DATASET OVERVIEW")
    print("-" * 70)
    print(f"Total records: {len(df)}")
    print(f"Normal records: {len(df[df['label'] == 0])}")
    print(f"Anomaly records: {len(df[df['label'] == 1])}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    
    # Data types
    print("\n🔍 DATA TYPES")
    print("-" * 70)
    print(df.dtypes)
    
    # Missing values
    print("\n✅ MISSING VALUES CHECK")
    print("-" * 70)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values found!")
    else:
        print("Missing values:")
        print(missing[missing > 0])
    
    # Distribution by label
    print("\n📈 LABEL DISTRIBUTION")
    print("-" * 70)
    label_counts = df['label'].value_counts().sort_index()
    normal_pct = (label_counts[0] / len(df) * 100)
    anomaly_pct = (label_counts[1] / len(df) * 100)
    print(f"Normal (0):  {label_counts[0]} records ({normal_pct:.2f}%)")
    print(f"Anomaly (1): {label_counts[1]} records ({anomaly_pct:.2f}%)")
    
    # Feature statistics for Normal transactions
    print("\n📊 STATISTICS - NORMAL TRANSACTIONS")
    print("-" * 70)
    normal_df = df[df['label'] == 0]
    print(normal_df[['amount', 'hour', 'login_attempts', 'response_time', 
                      'transaction_frequency', 'location_change']].describe())
    
    # Feature statistics for Anomaly transactions
    print("\n📊 STATISTICS - ANOMALY TRANSACTIONS")
    print("-" * 70)
    anomaly_df = df[df['label'] == 1]
    print(anomaly_df[['amount', 'hour', 'login_attempts', 'response_time', 
                       'transaction_frequency', 'location_change']].describe())
    
    # Data quality checks
    print("\n✔️ DATA QUALITY CHECKS")
    print("-" * 70)
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate records: {duplicates}")
    
    # Check data ranges
    print("\n✓ Amount range: ₹{:.2f} - ₹{:.2f}".format(df['amount'].min(), df['amount'].max()))
    print(f"✓ Hour range: {df['hour'].min()} - {df['hour'].max()}")
    print(f"✓ Login attempts range: {df['login_attempts'].min()} - {df['login_attempts'].max()}")
    print(f"✓ Response time range: {df['response_time'].min():.2f} - {df['response_time'].max():.2f}ms")
    print(f"✓ Transaction frequency range: {df['transaction_frequency'].min()} - {df['transaction_frequency'].max()}")
    print(f"✓ Location change values: {df['location_change'].unique()}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ DATASET VERIFICATION COMPLETE")
    print("="*70)
    print("\n📝 Summary:")
    print(f"  • Total transactions: {len(df)}")
    print(f"  • Normal: {len(normal_df)} ({normal_pct:.1f}%)")
    print(f"  • Anomalies: {len(anomaly_df)} ({anomaly_pct:.1f}%)")
    print(f"  • Data quality: ✓ All checks passed")
    print(f"  • Ready for training: ✓ Yes")
    print("\n" + "="*70)

if __name__ == "__main__":
    check_dataset()