import pandas as pd
import numpy as np
import pickle
import shap
import ollama
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_all_models():
    models = {}
    try:
        with open('models/churn_model.pkl', 'rb') as f:
            models['churn_model'] = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            models['churn_scaler'] = pickle.load(f)
        with open('models/feature_cols.pkl', 'rb') as f:
            models['churn_features'] = pickle.load(f)
        logger.info("All models loaded")
    except Exception as e:
        logger.error(f"Model loading error: {e}")
    return models


def explain_customer_in_plain_english(customer_id, df, models):
    customer = df[df['customerid'] == customer_id]
    if len(customer) == 0:
        return f"Customer {customer_id} not found."

    feature_cols = models['churn_features']
    scaler = models['churn_scaler']
    model = models['churn_model']

    X = customer[feature_cols]
    X_scaled = scaler.transform(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    probability = model.predict_proba(X_scaled)[0][1]

    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'value': X.values[0],
        'shap': shap_values[0]
    }).sort_values('shap', key=abs, ascending=False)

    factors = []
    for _, row in shap_df.head(4).iterrows():
        direction = "increases" if row['shap'] > 0 else "decreases"
        factors.append(f"{row['feature']} = {round(row['value'], 2)} ({direction} churn risk)")

    shap_summary = "\n".join(factors)

    prompt = f"""Customer {customer_id} analysis:
Churn Probability: {probability*100:.1f}%
Tenure: {int(customer['tenure'].values[0])} months
Monthly Charges: ${customer['monthlycharges'].values[0]:.2f}
Contract: {customer['contract'].values[0]}

Top factors:
{shap_summary}

Write 3 sentences explaining why this customer is at risk and what action to take."""

    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content']


def interactive_chat(df, models):
    system_content = """You are NexaIQ, an intelligent CRM analytics AI assistant.
Key metrics: 7043 customers, 26.54% churn rate, $456,117 monthly revenue, $139,131 lost revenue, 521 high risk customers.
Answer concisely and give actionable recommendations."""

    conversation_history = []

    print(f"\n{'='*60}")
    print("NEXAIQ INTERACTIVE AI ASSISTANT")
    print("Type your question. Type 'quit' to exit.")
    print(f"{'='*60}\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("NexaIQ: Goodbye!")
            break

        if not user_input:
            continue

        if 'explain customer' in user_input.lower():
            words = user_input.upper().split()
            customer_id = None
            for word in words:
                if '-' in word and len(word) > 5:
                    customer_id = word
                    break
            if customer_id:
                print(f"NexaIQ: Analyzing {customer_id}...")
                explanation = explain_customer_in_plain_english(customer_id, df, models)
                print(f"NexaIQ: {explanation}\n")
                continue

        conversation_history.append({
            'role': 'user',
            'content': user_input
        })

        messages = [{'role': 'system', 'content': system_content}] + conversation_history

        response = ollama.chat(
            model='llama3.2',
            messages=messages
        )

        assistant_response = response['message']['content']
        conversation_history.append({
            'role': 'assistant',
            'content': assistant_response
        })

        print(f"NexaIQ: {assistant_response}\n")


def run_nexaiq_chat():
    logger.info("="*60)
    logger.info("NEXAIQ INTERACTIVE CHAT STARTED")
    logger.info("="*60)

    df = pd.read_csv('data/processed/telco_churn_clean.csv')
    models = load_all_models()
    interactive_chat(df, models)


if __name__ == "__main__":
    run_nexaiq_chat()