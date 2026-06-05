import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_churn_model():

    with open('models/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('models/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)

    return model, scaler, feature_cols


def load_lead_model():

    with open('models/lead_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/lead_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('models/lead_features.pkl', 'rb') as f:
        feature_cols = pickle.load(f)

    return model, scaler, feature_cols


def explain_churn_model():

    logger.info("Explaining churn model with SHAP...")

    model, scaler, feature_cols = load_churn_model()

    df = pd.read_csv(
        'data/processed/telco_churn_clean.csv'
    )

    X = df[feature_cols].head(500)

    X_scaled = scaler.transform(X)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_scaled)

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=feature_cols,
        show=False,
        plot_type='dot'
    )

    plt.title(
        'SHAP — Churn Model Feature Impact'
    )

    plt.tight_layout()

    os.makedirs('outputs', exist_ok=True)

    plt.savefig(
        'outputs/shap_churn_detailed.png',
        bbox_inches='tight'
    )

    plt.close()

    logger.info("Churn SHAP plot saved")

    mean_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_shap': mean_shap
    }).sort_values(
        'mean_shap',
        ascending=False
    )

    print("\nTop 5 churn drivers:")

    print(
        feature_importance.head(5).to_string(index=False)
    )

    return shap_values, feature_cols


def explain_lead_model():

    logger.info("Explaining lead model with SHAP...")

    model, scaler, feature_cols = load_lead_model()

    df = pd.read_csv(
        'data/processed/lead_scores.csv'
    )

    X = df[feature_cols].head(500)

    X_scaled = scaler.transform(X)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=feature_cols,
        show=False,
        plot_type='bar'
    )

    plt.title(
        'SHAP — Lead Scoring Model Feature Impact'
    )

    plt.tight_layout()

    plt.savefig(
        'outputs/shap_lead_detailed.png',
        bbox_inches='tight'
    )

    plt.close()

    logger.info("Lead SHAP plot saved")

    mean_shap = np.abs(shap_values).mean(axis=0)

    if len(mean_shap.shape) > 1:
        mean_shap = mean_shap[:, 0]

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_shap': mean_shap
    }).sort_values(
        'mean_shap',
        ascending=False
    )

    print("\nTop 5 lead conversion drivers:")

    print(
        feature_importance.head(5).to_string(index=False)
    )


def generate_customer_explanation(
    customer_index=0
):

    logger.info(
        f"Generating plain English explanation for customer {customer_index}"
    )

    model, scaler, feature_cols = load_churn_model()

    df = pd.read_csv(
        'data/processed/telco_churn_clean.csv'
    )

    customer = df[feature_cols].iloc[
        customer_index:customer_index+1
    ]

    customer_scaled = scaler.transform(customer)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(customer_scaled)

    probability = model.predict_proba(
        customer_scaled
    )[0][1]

    prediction = model.predict(
        customer_scaled
    )[0]

    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'value': customer.values[0],
        'shap': shap_values[0]
    }).sort_values(
        'shap',
        key=abs,
        ascending=False
    )

    print(f"\n{'='*60}")

    print(
        f"CUSTOMER {customer_index} — PLAIN ENGLISH EXPLANATION"
    )

    print(f"{'='*60}")

    print(
        f"Prediction: {'WILL CHURN' if prediction == 1 else 'WILL STAY'}"
    )

    print(
        f"Churn probability: {probability*100:.1f}%"
    )

    print(f"\nWhy this prediction was made:")

    explanations = []

    for _, row in shap_df.head(5).iterrows():

        if row['shap'] > 0:
            direction = "increases churn risk"
        else:
            direction = "reduces churn risk"

        feature_map = {

            'contract_encoded':
                f"Contract type ({int(row['value'])})",

            'tenure':
                f"Tenure ({int(row['value'])} months)",

            'monthlycharges':
                f"Monthly charges (${row['value']:.2f})",

            'internetservice_encoded':
                f"Internet service type ({int(row['value'])})",

            'total_value':
                f"Total value (${row['value']:.2f})",

            'paymentmethod_encoded':
                f"Payment method",

            'seniorcitizen':
                f"Senior citizen status",

            'is_high_risk':
                f"High risk flag"
        }

        feature_name = feature_map.get(
            row['feature'],
            row['feature']
        )

        explanations.append(
            f"  • {feature_name} {direction} (impact: {abs(row['shap']):.3f})"
        )

    for exp in explanations:
        print(exp)

    return probability, explanations


def run_shap_dashboard():

    logger.info("=" * 60)

    logger.info(
        "NEXAIQ SHAP DASHBOARD STARTED"
    )

    logger.info("=" * 60)

    explain_churn_model()

    explain_lead_model()

    for i in [0, 5, 10]:
        generate_customer_explanation(i)

    logger.info("=" * 60)

    logger.info(
        "SHAP DASHBOARD COMPLETE"
    )

    logger.info("=" * 60)


if __name__ == "__main__":

    run_shap_dashboard()