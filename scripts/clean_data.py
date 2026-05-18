import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data(filepath):
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def fix_total_charges(df):
    logger.info("Fixing TotalCharges column...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    empty_rows = df['TotalCharges'].isnull().sum()
    logger.info(f"Found {empty_rows} empty TotalCharges rows — filling with 0")
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df


def fix_column_names(df):
    logger.info("Standardizing column names to lowercase...")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def encode_binary_columns(df):
    logger.info("Encoding Yes/No columns to 1/0...")
    binary_cols = ['partner', 'dependents', 'phoneservice',
                   'paperlessbilling', 'churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    df['seniorcitizen'] = df['seniorcitizen'].astype(int)
    return df


def encode_categorical_columns(df):
    logger.info("Encoding categorical columns...")
    df['gender_encoded'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['contract_encoded'] = df['contract'].map({
        'Month-to-month': 0, 'One year': 1, 'Two year': 2
    })
    df['internetservice_encoded'] = df['internetservice'].map({
        'No': 0, 'DSL': 1, 'Fiber optic': 2
    })
    df['paymentmethod_encoded'] = df['paymentmethod'].map({
        'Electronic check': 0, 'Mailed check': 1,
        'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
    })
    return df


def add_derived_features(df):
    logger.info("Adding derived features...")
    df['total_value'] = df['tenure'] * df['monthlycharges']
    df['is_high_value'] = (df['total_value'] > 1000).astype(int)
    df['is_high_risk'] = (
        (df['tenure'] < 6) & (df['monthlycharges'] > 70)
    ).astype(int)
    df['avg_monthly_total'] = df.apply(
        lambda row: row['totalcharges'] / row['tenure']
        if row['tenure'] > 0 else 0, axis=1
    )
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df)} duplicate rows")
    return df


def save_clean_data(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Clean data saved to {filepath}")
    logger.info(f"Final shape: {df.shape}")


def run_cleaning_pipeline():
    logger.info("=" * 50)
    logger.info("NEXAIQ DATA CLEANING PIPELINE STARTED")
    logger.info("=" * 50)

    df = load_raw_data('data/raw/telco_churn.csv')
    df = remove_duplicates(df)
    df = fix_total_charges(df)
    df = fix_column_names(df)
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)
    df = add_derived_features(df)
    save_clean_data(df, 'data/processed/telco_churn_clean.csv')

    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 50)
    return df


if __name__ == "__main__":
    df_clean = run_cleaning_pipeline()
    print("\nSample of clean data:")
    print(df_clean[['customerid', 'tenure', 'monthlycharges',
                     'totalcharges', 'contract_encoded',
                     'is_high_risk', 'churn']].head(10))
    print("\nFinal shape:", df_clean.shape)