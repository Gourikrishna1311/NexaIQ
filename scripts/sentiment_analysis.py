import pandas as pd
import numpy as np
import ollama
import matplotlib.pyplot as plt
import logging
import os
import random

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_support_tickets():
    logger.info("Generating synthetic support tickets...")
    random.seed(42)
    np.random.seed(42)

    positive_messages = [
        "The service has been excellent, very happy with everything",
        "Support team was incredibly helpful and resolved my issue quickly",
        "Great experience overall, will definitely continue using the service",
        "Very satisfied with the product quality and customer service",
        "Amazing service, the best I have used in years",
        "Fast response time and professional support, highly recommend",
        "Everything works perfectly, no complaints at all",
        "The team went above and beyond to help me"
    ]

    negative_messages = [
        "Very disappointed with the service, considering cancelling",
        "The product stopped working and nobody is helping me",
        "Terrible customer support, waited 3 days with no response",
        "I am extremely frustrated with the billing issues",
        "This is unacceptable, I want a refund immediately",
        "The service keeps going down and affecting my business",
        "Nobody responds to my tickets, worst experience ever",
        "I have been overcharged multiple times and nobody fixes it"
    ]

    neutral_messages = [
        "I need to update my billing information",
        "Can you tell me when the next maintenance window is",
        "How do I reset my password",
        "I would like to upgrade my plan",
        "Please send me my invoice for last month",
        "What are the system requirements for the new update",
        "I need to add a new user to my account",
        "Can you confirm my subscription renewal date"
    ]

    tickets = []
    for i in range(150):
        sentiment_type = random.choice(['positive', 'negative', 'neutral'])
        if sentiment_type == 'positive':
            message = random.choice(positive_messages)
            label = 'positive'
        elif sentiment_type == 'negative':
            message = random.choice(negative_messages)
            label = 'negative'
        else:
            message = random.choice(neutral_messages)
            label = 'neutral'

        tickets.append({
            'ticket_id': f'TKT_{i+1:04d}',
            'customer_message': message,
            'actual_sentiment': label,
            'priority': random.choice(['low', 'medium', 'high']),
            'category': random.choice(['billing', 'technical',
                                       'general', 'cancellation'])
        })

    df = pd.DataFrame(tickets)
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/support_tickets.csv', index=False)
    logger.info(f"Generated {len(df)} support tickets")
    return df


def analyze_sentiment_single(message):
    prompt = f"""Classify the sentiment of this customer message.
Reply with ONLY one word: positive, negative, or neutral.

Message: {message}"""

    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}]
    )

    result = response['message']['content'].strip().lower()
    if 'positive' in result:
        return 'positive'
    elif 'negative' in result:
        return 'negative'
    else:
        return 'neutral'


def run_sentiment_analysis(df):
    logger.info("Running sentiment analysis on 50 tickets...")
    df_sample = df.head(50).copy()

    sentiments = []
    for i, row in df_sample.iterrows():
        sentiment = analyze_sentiment_single(row['customer_message'])
        sentiments.append(sentiment)
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/50 tickets")

    df_sample['predicted_sentiment'] = sentiments

    correct = (df_sample['predicted_sentiment'] == df_sample['actual_sentiment']).sum()
    accuracy = correct / len(df_sample) * 100

    print(f"\n{'='*50}")
    print("SENTIMENT ANALYSIS RESULTS")
    print(f"{'='*50}")
    print(f"Tickets analyzed: {len(df_sample)}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"\nSentiment Distribution:")
    print(df_sample['predicted_sentiment'].value_counts())

    df_sample.to_csv('data/processed/sentiment_results.csv', index=False)
    logger.info("Sentiment results saved")
    return df_sample


def plot_sentiment_results(df_sample):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sentiment_counts = df_sample['predicted_sentiment'].value_counts()
    colors = ['#3BB273', '#E84855', '#F9C74F']
    axes[0].pie(sentiment_counts.values,
                labels=sentiment_counts.index,
                autopct='%1.1f%%',
                colors=colors[:len(sentiment_counts)],
                startangle=90)
    axes[0].set_title('Customer Sentiment Distribution')

    category_sentiment = pd.crosstab(
        df_sample['category'],
        df_sample['predicted_sentiment']
    )
    category_sentiment.plot(kind='bar', ax=axes[1],
                            color=colors[:len(category_sentiment.columns)],
                            edgecolor='white')
    axes[1].set_title('Sentiment by Ticket Category')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/sentiment_analysis.png')
    plt.show()
    logger.info("Sentiment chart saved")


def generate_automated_report(df_sample):
    logger.info("Generating automated executive report...")

    negative_pct = (df_sample['predicted_sentiment'] == 'negative').mean() * 100

    prompt = f"""Generate a concise executive business report for NexaIQ CRM platform.

KEY METRICS:
- Total customers: 7043
- Churn rate: 26.54%
- Monthly revenue lost: $139,131
- High risk customers: 521
- Negative sentiment tickets: {negative_pct:.1f}%

Write a professional 3-paragraph executive summary covering:
1. Current business health and churn situation
2. Key risks and customer sentiment findings
3. Recommended immediate actions

Keep it under 200 words. Be specific with numbers."""

    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}]
    )

    report = response['message']['content']

    with open('outputs/executive_report.txt', 'w') as f:
        f.write("NEXAIQ EXECUTIVE REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(report)

    print(f"\n{'='*60}")
    print("AUTOMATED EXECUTIVE REPORT")
    print(f"{'='*60}")
    print(report)
    logger.info("Executive report saved to outputs/executive_report.txt")


def run_sentiment_pipeline():
    logger.info("="*60)
    logger.info("NEXAIQ SENTIMENT ANALYSIS STARTED")
    logger.info("="*60)

    df = generate_support_tickets()
    df_sample = run_sentiment_analysis(df)
    plot_sentiment_results(df_sample)
    generate_automated_report(df_sample)

    logger.info("="*60)
    logger.info("SENTIMENT PIPELINE COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    run_sentiment_pipeline()