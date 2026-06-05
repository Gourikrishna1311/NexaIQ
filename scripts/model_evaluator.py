import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_churn_model():
    with open('models/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols


def get_churn_test_data(feature_cols, scaler):
    df = pd.read_csv('data/processed/telco_churn_clean.csv')
    X = df[feature_cols]
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, y_test


def plot_roc_curve(model, X_test, y_test, model_name, ax, color):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{model_name} (AUC = {roc_auc:.3f})')
    return roc_auc


def plot_confusion_matrix(model, X_test, y_test, model_name, ax):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix\n{model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Retained', 'Churned'])
    ax.set_yticklabels(['Retained', 'Churned'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=14, fontweight='bold')


def generate_combined_report():
    logger.info("Loading all models...")
    churn_model, churn_scaler, churn_features = load_churn_model()
    X_churn_test, y_churn_test = get_churn_test_data(churn_features, churn_scaler)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    roc_auc = plot_roc_curve(churn_model, X_churn_test,
                              y_churn_test, 'Churn XGBoost', ax, 'steelblue')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — NexaIQ Models')
    ax.legend(loc='lower right')

    plot_confusion_matrix(churn_model, X_churn_test,
                          y_churn_test, 'Churn Model', axes[1])

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/model_evaluation.png')
    plt.show()
    logger.info("Model evaluation chart saved")

    print(f"\n{'='*60}")
    print("NEXAIQ — COMPLETE MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"Model 1: Churn Prediction (XGBoost)")
    print(f"         AUC: {roc_auc:.4f} — predicts customer churn")
    print(f"\nModel 2: Sales Forecasting (GradientBoosting)")
    print(f"         Saved at models/sales_forecast_model.pkl")
    print(f"\nModel 3: Lead Scoring (RandomForest)")
    print(f"         Saved at models/lead_model.pkl")
    print(f"\nAll 3 models ready for AI assistant integration in Phase 4")
    print(f"{'='*60}")


def run_evaluation():
    logger.info("="*60)
    logger.info("NEXAIQ MODEL EVALUATION STARTED")
    logger.info("="*60)

    generate_combined_report()

    logger.info("="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    run_evaluation()