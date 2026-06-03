import pandas as pd
import numpy as np
import shap
import pickle
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_all_shap_values():
    logger.info("Loading model and data...")

    with open('models/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('models/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)

    df = pd.read_csv('data/processed/telco_churn_clean.csv')

    X = df[feature_cols]
    X_scaled = scaler.transform(X)

    logger.info("Computing SHAP values for all 7043 customers...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.columns = ['shap_' + col for col in feature_cols]

    shap_df['customerid'] = df['customerid'].values
    shap_df['churn_probability'] = model.predict_proba(X_scaled)[:, 1]
    shap_df['predicted_churn'] = model.predict(X_scaled)
    shap_df['actual_churn'] = df['churn'].values

    shap_df['top_churn_reason'] = shap_df[
        ['shap_' + col for col in feature_cols]
    ].apply(lambda row: feature_cols[row.values.argmax()], axis=1)

    shap_df.to_csv('data/processed/shap_values.csv', index=False)
    logger.info(f"SHAP values saved for {len(shap_df)} customers")

    print("\nSample SHAP output:")
    print(shap_df[['customerid', 'churn_probability',
                   'predicted_churn', 'actual_churn',
                   'top_churn_reason']].head(10).to_string())

    print(f"\nTop churn reasons distribution:")
    print(shap_df['top_churn_reason'].value_counts().head(5))


if __name__ == "__main__":
    save_all_shap_values()