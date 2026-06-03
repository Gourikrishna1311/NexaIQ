import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_best_model():
    logger.info("Loading and preparing data...")
    df = pd.read_csv('data/processed/telco_churn_clean.csv')

    feature_cols = [
        'tenure', 'monthlycharges', 'totalcharges',
        'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'paperlessbilling',
        'gender_encoded', 'contract_encoded',
        'internetservice_encoded', 'paymentmethod_encoded',
        'total_value', 'is_high_value', 'avg_monthly_total'
    ]

    X = df[feature_cols]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)

    logger.info("Training final XGBoost model...")
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train_scaled, y_train_balanced)

    os.makedirs('models', exist_ok=True)

    with open('models/churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    logger.info("Model saved: models/churn_model.pkl")

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved: models/scaler.pkl")

    with open('models/feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    logger.info("Feature columns saved: models/feature_cols.pkl")

    print("\nModel files saved:")
    print("  models/churn_model.pkl")
    print("  models/scaler.pkl")
    print("  models/feature_cols.pkl")
    print("\nThese files will be used by the AI assistant in Phase 5")


if __name__ == "__main__":
    save_best_model()