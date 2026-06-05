import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_lead_data():
    logger.info("Creating synthetic lead scoring dataset...")
    np.random.seed(42)
    n = 2000

    df = pd.DataFrame({
        'lead_age_days': np.random.randint(1, 90, n),
        'email_opens': np.random.randint(0, 20, n),
        'website_visits': np.random.randint(0, 50, n),
        'demo_requested': np.random.binomial(1, 0.3, n),
        'pricing_page_visits': np.random.randint(0, 10, n),
        'company_size': np.random.choice([1, 2, 3, 4], n),
        'industry': np.random.choice([0, 1, 2, 3, 4], n),
        'lead_source': np.random.choice([0, 1, 2, 3], n),
        'response_time_hours': np.random.randint(1, 72, n),
        'num_interactions': np.random.randint(1, 30, n),
        'budget_indicated': np.random.binomial(1, 0.4, n),
        'decision_maker': np.random.binomial(1, 0.5, n),
    })

    score = (
        df['email_opens'] * 2 +
        df['website_visits'] * 1.5 +
        df['demo_requested'] * 20 +
        df['pricing_page_visits'] * 5 +
        df['budget_indicated'] * 15 +
        df['decision_maker'] * 10 +
        df['num_interactions'] * 1.2 -
        df['response_time_hours'] * 0.3 -
        df['lead_age_days'] * 0.2
    )

    threshold = score.quantile(0.65)
    df['converted'] = (score > threshold).astype(int)

    logger.info(f"Created {len(df)} leads")
    logger.info(f"Conversion rate: {df['converted'].mean()*100:.1f}%")
    return df


def train_lead_model(df):
    feature_cols = [
        'lead_age_days', 'email_opens', 'website_visits',
        'demo_requested', 'pricing_page_visits', 'company_size',
        'industry', 'lead_source', 'response_time_hours',
        'num_interactions', 'budget_indicated', 'decision_maker'
    ]

    X = df[feature_cols]
    y = df['converted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print("LEAD SCORING MODEL RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"AUC Score: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Not Converted', 'Converted']))

    return model, scaler, feature_cols


def score_new_leads(model, scaler, feature_cols, df):
    X = df[feature_cols]
    X_scaled = scaler.transform(X)

    df['conversion_probability'] = model.predict_proba(X_scaled)[:, 1]
    df['lead_score'] = (df['conversion_probability'] * 100).round(1)
    df['lead_grade'] = pd.cut(df['lead_score'],
                               bins=[0, 25, 50, 75, 100],
                               labels=['Cold', 'Warm', 'Hot', 'Very Hot'])

    print(f"\nLead Grade Distribution:")
    print(df['lead_grade'].value_counts())

    print(f"\nTop 10 Hottest Leads:")
    print(df.nlargest(10, 'lead_score')[
        ['lead_score', 'lead_grade', 'demo_requested',
         'budget_indicated', 'decision_maker']
    ].to_string())

    return df


def save_lead_model(model, scaler, feature_cols):
    os.makedirs('models', exist_ok=True)

    with open('models/lead_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('models/lead_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('models/lead_features.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    logger.info("Lead scoring model saved to models folder")


def run_lead_scoring():
    logger.info("="*60)
    logger.info("NEXAIQ LEAD SCORING MODEL STARTED")
    logger.info("="*60)

    df = create_lead_data()
    model, scaler, feature_cols = train_lead_model(df)
    df = score_new_leads(model, scaler, feature_cols, df)
    save_lead_model(model, scaler, feature_cols)

    df.to_csv('data/processed/lead_scores.csv', index=False)
    logger.info("Lead scores saved to data/processed/lead_scores.csv")

    logger.info("="*60)
    logger.info("LEAD SCORING PIPELINE COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    run_lead_scoring()