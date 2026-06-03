import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    logger.info("Loading clean data...")
    df = pd.read_csv('data/processed/telco_churn_clean.csv')
    logger.info(f"Loaded {len(df)} rows")
    return df


def prepare_features(df):
    logger.info("Preparing features...")
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
    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    return X, y, feature_cols


def split_and_scale(X, y):
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train size: {len(X_train)} rows")
    logger.info(f"Test size: {len(X_test)} rows")

    logger.info("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {y_train_balanced.value_counts().to_dict()}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUC Score: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Retained', 'Churned']))
    return accuracy, auc


def train_all_models(X_train, X_test, y_train, y_test):
    results = {}

    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    acc, auc = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    results['Logistic Regression'] = {'model': lr, 'accuracy': acc, 'auc': auc}

    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc, auc = evaluate_model(rf, X_test, y_test, "Random Forest")
    results['Random Forest'] = {'model': rf, 'accuracy': acc, 'auc': auc}

    logger.info("Training XGBoost...")
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    acc, auc = evaluate_model(xgb, X_test, y_test, "XGBoost")
    results['XGBoost'] = {'model': xgb, 'accuracy': acc, 'auc': auc}

    return results


def plot_feature_importance(model, feature_cols):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_cols)),
            importance[indices], color='steelblue')
    plt.xticks(range(len(feature_cols)),
               [feature_cols[i] for i in indices],
               rotation=45, ha='right')
    plt.title('Feature Importance — XGBoost Churn Model')
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/feature_importance.png')
    plt.show()
    logger.info("Feature importance chart saved to outputs/feature_importance.png")


def plot_model_comparison(results):
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in models]
    aucs = [results[m]['auc'] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(models, accuracies, color=['steelblue', 'tomato', 'green'])
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylabel('Accuracy %')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')

    axes[1].bar(models, aucs, color=['steelblue', 'tomato', 'green'])
    axes[1].set_title('Model AUC Score Comparison')
    axes[1].set_ylabel('AUC Score')
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(aucs):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png')
    plt.show()
    logger.info("Model comparison chart saved to outputs/model_comparison.png")


def run_churn_model():
    logger.info("=" * 60)
    logger.info("NEXAIQ CHURN PREDICTION MODEL STARTED")
    logger.info("=" * 60)

    df = load_data()
    X, y, feature_cols = prepare_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    results = train_all_models(X_train, X_test, y_train, y_test)

    plot_model_comparison(results)

    best_model_name = max(results, key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']

    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Best AUC:   {results[best_model_name]['auc']:.4f}")
    print(f"Accuracy:   {results[best_model_name]['accuracy']*100:.2f}%")
    print(f"{'='*50}")

    plot_feature_importance(best_model, feature_cols)

    logger.info("=" * 60)
    logger.info("CHURN MODEL PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_churn_model()