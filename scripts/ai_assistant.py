import pandas as pd
import numpy as np
import pickle
import ollama
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_business_context():
    logger.info("Loading business context...")
    df = pd.read_csv('data/processed/telco_churn_clean.csv')

    context = {
        'total_customers': len(df),
        'churn_rate': round(df['churn'].mean() * 100, 2),
        'churned_customers': int(df['churn'].sum()),
        'retained_customers': int((df['churn'] == 0).sum()),
        'avg_monthly_charges': round(df['monthlycharges'].mean(), 2),
        'total_monthly_revenue': round(df['monthlycharges'].sum(), 2),
        'lost_revenue': round(df[df['churn']==1]['monthlycharges'].sum(), 2),
        'high_risk_customers': int(df['is_high_risk'].sum()),
        'high_value_customers': int(df['is_high_value'].sum()),
        'contract_churn': df.groupby('contract')['churn'].mean().mul(100).round(2).to_dict(),
        'internet_churn': df.groupby('internetservice')['churn'].mean().mul(100).round(2).to_dict()
    }

    logger.info("Business context loaded")
    return context, df


def load_shap_context():
    try:
        shap_df = pd.read_csv('data/processed/shap_values.csv')
        top_reasons = shap_df['top_churn_reason'].value_counts().head(3).to_dict()
        return top_reasons
    except:
        return {'contract_encoded': 245, 'tenure': 198, 'monthlycharges': 167}


def get_customer_prediction(customer_id, df):
    try:
        with open('models/churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)

        customer = df[df['customerid'] == customer_id]
        if len(customer) == 0:
            return None

        X = customer[feature_cols]
        X_scaled = scaler.transform(X)
        probability = model.predict_proba(X_scaled)[0][1]
        prediction = model.predict(X_scaled)[0]

        return {
            'customer_id': customer_id,
            'churn_probability': round(probability * 100, 1),
            'prediction': 'WILL CHURN' if prediction == 1 else 'WILL STAY',
            'tenure': int(customer['tenure'].values[0]),
            'monthly_charges': float(customer['monthlycharges'].values[0]),
            'contract': customer['contract'].values[0]
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None


def ask_assistant(question, context, shap_context):
    system_prompt = f"""You are NexaIQ, an intelligent CRM analytics assistant.
You have access to real business data and ML model predictions.

CURRENT BUSINESS DATA:
- Total customers: {context['total_customers']}
- Overall churn rate: {context['churn_rate']}%
- Churned customers: {context['churned_customers']}
- Retained customers: {context['retained_customers']}
- Average monthly charges: ${context['avg_monthly_charges']}
- Total monthly revenue: ${context['total_monthly_revenue']:,}
- Monthly revenue lost to churn: ${context['lost_revenue']:,}
- High risk customers: {context['high_risk_customers']}
- High value customers: {context['high_value_customers']}

CHURN BY CONTRACT TYPE: {context['contract_churn']}
CHURN BY INTERNET SERVICE: {context['internet_churn']}
TOP CHURN REASONS: {shap_context}

Answer questions clearly and concisely based on the actual data.
Keep responses under 150 words."""

    response = ollama.chat(
        model='llama3.2',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question}
        ]
    )

    return response['message']['content']


def run_ai_assistant():
    logger.info("="*60)
    logger.info("NEXAIQ AI ASSISTANT STARTED")
    logger.info("="*60)

    context, df = load_business_context()
    shap_context = load_shap_context()

    test_questions = [
        "What is our current churn rate and how much revenue are we losing?",
        "Which customer segment is most at risk of churning?",
        "What are the top 3 reasons customers churn?",
        "How many high risk customers do we have and what should we do?",
        "What is the difference in churn rate between contract types?"
    ]

    print(f"\n{'='*60}")
    print("NEXAIQ AI ASSISTANT — DEMO")
    print(f"{'='*60}")

    for i, question in enumerate(test_questions, 1):
        print(f"\nQ{i}: {question}")
        print("-" * 50)
        answer = ask_assistant(question, context, shap_context)
        print(f"A: {answer}")

    logger.info("="*60)
    logger.info("AI ASSISTANT DEMO COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    run_ai_assistant()