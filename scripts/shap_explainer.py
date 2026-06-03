import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_data():
    logger.info("Loading saved model and data...")

    with open('models/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('models/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)

    df = pd.read_csv('data/processed/telco_churn_clean.csv')

    logger.info("Model and data loaded successfully")
    return model, scaler, feature_cols, df


def prepare_test_data(df, feature_cols, scaler):
    from sklearn.model_selection import train_test_split
    X = df[feature_cols]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, X_test, y_test


def compute_shap_values(model, X_test_scaled, feature_cols):
    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    logger.info(f"SHAP values computed for {len(X_test_scaled)} customers")
    return explainer, shap_values


def plot_shap_summary(shap_values, X_test_scaled, feature_cols):
    logger.info("Generating SHAP summary plot...")
    os.makedirs('outputs', exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_scaled,
                      feature_names=feature_cols,
                      show=False)
    plt.title('SHAP Summary — Feature Impact on Churn Prediction')
    plt.tight_layout()
    plt.savefig('outputs/shap_summary.png', bbox_inches='tight')
    plt.show()
    logger.info("SHAP summary plot saved to outputs/shap_summary.png")


def plot_shap_bar(shap_values, feature_cols):
    logger.info("Generating SHAP bar chart...")
    mean_shap = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_shap)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_cols)),
            mean_shap[indices], color='steelblue')
    plt.xticks(range(len(feature_cols)),
               [feature_cols[i] for i in indices],
               rotation=45, ha='right')
    plt.title('Mean SHAP Values — Average Feature Impact on Churn')
    plt.ylabel('Mean |SHAP Value|')
    plt.tight_layout()
    plt.savefig('outputs/shap_bar.png')
    plt.show()
    logger.info("SHAP bar chart saved to outputs/shap_bar.png")


def explain_single_customer(model, scaler, feature_cols, df, customer_index=0):
    logger.info(f"Explaining prediction for customer index {customer_index}")

    customer = df[feature_cols].iloc[customer_index:customer_index+1]
    customer_scaled = scaler.transform(customer)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(customer_scaled)

    prediction = model.predict(customer_scaled)[0]
    probability = model.predict_proba(customer_scaled)[0][1]

    print(f"\n{'='*60}")
    print(f"CUSTOMER EXPLANATION — Index {customer_index}")
    print(f"{'='*60}")
    print(f"Prediction:    {'WILL CHURN' if prediction == 1 else 'WILL STAY'}")
    print(f"Churn Probability: {probability*100:.1f}%")
    print(f"\nTop factors driving this prediction:")
    print(f"{'Feature':<30} {'Customer Value':<20} {'SHAP Impact':<15}")
    print("-" * 65)

    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'value': customer.values[0],
        'shap': shap_values[0]
    }).sort_values('shap', key=abs, ascending=False)

    for _, row in shap_df.head(8).iterrows():
        direction = "pushes TOWARD churn" if row['shap'] > 0 else "pushes AWAY from churn"
        print(f"{row['feature']:<30} {str(round(row['value'], 2)):<20} {direction}")

    return shap_values, probability


def explain_high_risk_customers(model, scaler, feature_cols, df):
    logger.info("Explaining high risk customers...")

    high_risk = df[df['is_high_risk'] == 1].head(5)

    print(f"\n{'='*60}")
    print("HIGH RISK CUSTOMER EXPLANATIONS")
    print(f"{'='*60}")

    for idx in high_risk.index[:3]:
        customer = df[feature_cols].iloc[idx:idx+1]
        customer_scaled = scaler.transform(customer)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(customer_scaled)

        probability = model.predict_proba(customer_scaled)[0][1]
        top_feature_idx = np.argmax(np.abs(shap_values[0]))
        top_feature = feature_cols[top_feature_idx]

        print(f"\nCustomer ID: {df.iloc[idx]['customerid']}")
        print(f"Churn Probability: {probability*100:.1f}%")
        print(f"Main reason: {top_feature} = {df.iloc[idx][top_feature]}")


def run_shap_analysis():
    logger.info("="*60)
    logger.info("NEXAIQ SHAP EXPLAINABILITY ANALYSIS STARTED")
    logger.info("="*60)

    model, scaler, feature_cols, df = load_model_and_data()
    X_test_scaled, X_test, y_test = prepare_test_data(df, feature_cols, scaler)
    explainer, shap_values = compute_shap_values(model, X_test_scaled, feature_cols)

    plot_shap_summary(shap_values, X_test_scaled, feature_cols)
    plot_shap_bar(shap_values, feature_cols)

    explain_single_customer(model, scaler, feature_cols, df, customer_index=1)
    explain_high_risk_customers(model, scaler, feature_cols, df)

    logger.info("="*60)
    logger.info("SHAP ANALYSIS COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    run_shap_analysis()