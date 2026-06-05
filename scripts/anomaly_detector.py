import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
import os
import pickle
from datetime import datetime, timedelta
import random

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_access_logs():
    logger.info("Generating synthetic system access logs...")
    np.random.seed(42)
    random.seed(42)

    n_normal = 1000
    n_anomaly = 50

    normal_logs = []
    base_time = datetime(2026, 1, 1, 9, 0, 0)

    for i in range(n_normal):
        hour = random.randint(9, 17)
        log = {
            'user_id': f'USER_{random.randint(1, 20):03d}',
            'timestamp': base_time + timedelta(
                days=random.randint(0, 90),
                hours=hour,
                minutes=random.randint(0, 59)
            ),
            'query_count': random.randint(1, 20),
            'data_accessed_mb': random.uniform(0.1, 10),
            'failed_attempts': random.randint(0, 2),
            'hour_of_day': hour,
            'is_weekend': 0,
            'unique_tables': random.randint(1, 5),
            'session_duration_min': random.randint(5, 120),
            'is_anomaly': 0
        }
        normal_logs.append(log)

    anomaly_logs = []
    for i in range(n_anomaly):
        anomaly_type = random.choice(['off_hours', 'bulk_download',
                                       'brute_force', 'unusual_tables'])

        if anomaly_type == 'off_hours':
            hour = random.choice([0, 1, 2, 3, 4, 22, 23])
            log = {
                'user_id': f'USER_{random.randint(1, 20):03d}',
                'timestamp': base_time + timedelta(
                    days=random.randint(0, 90),
                    hours=hour
                ),
                'query_count': random.randint(1, 10),
                'data_accessed_mb': random.uniform(0.1, 5),
                'failed_attempts': random.randint(0, 1),
                'hour_of_day': hour,
                'is_weekend': 1,
                'unique_tables': random.randint(1, 3),
                'session_duration_min': random.randint(5, 30),
                'is_anomaly': 1
            }
        elif anomaly_type == 'bulk_download':
            log = {
                'user_id': f'USER_{random.randint(1, 20):03d}',
                'timestamp': base_time + timedelta(days=random.randint(0, 90)),
                'query_count': random.randint(100, 500),
                'data_accessed_mb': random.uniform(500, 2000),
                'failed_attempts': 0,
                'hour_of_day': random.randint(9, 17),
                'is_weekend': 0,
                'unique_tables': random.randint(10, 20),
                'session_duration_min': random.randint(1, 5),
                'is_anomaly': 1
            }
        elif anomaly_type == 'brute_force':
            log = {
                'user_id': f'USER_{random.randint(1, 20):03d}',
                'timestamp': base_time + timedelta(days=random.randint(0, 90)),
                'query_count': random.randint(1, 5),
                'data_accessed_mb': random.uniform(0.1, 1),
                'failed_attempts': random.randint(10, 50),
                'hour_of_day': random.randint(0, 23),
                'is_weekend': random.randint(0, 1),
                'unique_tables': 1,
                'session_duration_min': random.randint(1, 10),
                'is_anomaly': 1
            }
        else:
            log = {
                'user_id': f'USER_{random.randint(1, 20):03d}',
                'timestamp': base_time + timedelta(days=random.randint(0, 90)),
                'query_count': random.randint(50, 200),
                'data_accessed_mb': random.uniform(50, 200),
                'failed_attempts': random.randint(0, 3),
                'hour_of_day': random.randint(9, 17),
                'is_weekend': 0,
                'unique_tables': random.randint(15, 25),
                'session_duration_min': random.randint(60, 300),
                'is_anomaly': 1
            }
        anomaly_logs.append(log)

    df = pd.DataFrame(normal_logs + anomaly_logs)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/access_logs.csv', index=False)
    logger.info(f"Generated {len(df)} access logs ({n_anomaly} anomalies)")
    return df


def detect_security_anomalies(df):
    logger.info("Running Isolation Forest for security anomaly detection...")

    feature_cols = [
        'query_count', 'data_accessed_mb', 'failed_attempts',
        'hour_of_day', 'is_weekend', 'unique_tables',
        'session_duration_min'
    ]

    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=100
    )
    iso_forest.fit(X_scaled)

    df['anomaly_score'] = iso_forest.decision_function(X_scaled)
    df['is_detected_anomaly'] = (iso_forest.predict(X_scaled) == -1).astype(int)

    detected = df[df['is_detected_anomaly'] == 1]
    actual_anomalies = df[df['is_anomaly'] == 1]

    true_positives = len(df[(df['is_detected_anomaly'] == 1) & (df['is_anomaly'] == 1)])
    precision = true_positives / len(detected) if len(detected) > 0 else 0
    recall = true_positives / len(actual_anomalies) if len(actual_anomalies) > 0 else 0

    print(f"\n{'='*50}")
    print("SECURITY ANOMALY DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"Total logs analyzed:     {len(df)}")
    print(f"Actual anomalies:        {len(actual_anomalies)}")
    print(f"Detected anomalies:      {len(detected)}")
    print(f"True positives:          {true_positives}")
    print(f"Precision:               {precision:.2%}")
    print(f"Recall:                  {recall:.2%}")

    with open('models/anomaly_detector.pkl', 'wb') as f:
        pickle.dump(iso_forest, f)
    with open('models/anomaly_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    logger.info("Anomaly detection model saved")
    return df, detected


def detect_business_anomalies():
    logger.info("Running business anomaly detection on CRM data...")
    df = pd.read_csv('data/processed/telco_churn_clean.csv')

    feature_cols = ['monthlycharges', 'totalcharges', 'tenure', 'total_value']
    X = df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(contamination=0.03, random_state=42)
    iso_forest.fit(X_scaled)

    df['business_anomaly_score'] = iso_forest.decision_function(X_scaled)
    df['is_business_anomaly'] = (iso_forest.predict(X_scaled) == -1).astype(int)

    anomalies = df[df['is_business_anomaly'] == 1]

    print(f"\n{'='*50}")
    print("BUSINESS ANOMALY DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"Total customers analyzed: {len(df)}")
    print(f"Business anomalies found: {len(anomalies)}")
    print(f"\nSample anomalous customers:")
    print(anomalies[['customerid', 'monthlycharges',
                      'totalcharges', 'tenure',
                      'churn']].head(10).to_string(index=False))

    anomalies.to_csv('outputs/business_anomalies.csv', index=False)
    logger.info("Business anomalies saved to outputs/business_anomalies.csv")

    return df


def plot_anomaly_results(df_security, detected):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['tomato' if x == 1 else 'steelblue'
              for x in df_security['is_detected_anomaly']]
    axes[0].scatter(df_security['query_count'],
                    df_security['data_accessed_mb'],
                    c=colors, alpha=0.6, s=20)
    axes[0].set_xlabel('Query Count')
    axes[0].set_ylabel('Data Accessed (MB)')
    axes[0].set_title('Security Anomaly Detection\nRed = Detected Anomaly')

    axes[1].hist(df_security['anomaly_score'], bins=30,
                 color='steelblue', edgecolor='white')
    axes[1].axvline(x=0, color='tomato', linestyle='--',
                    linewidth=2, label='Anomaly threshold')
    axes[1].set_xlabel('Anomaly Score')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Anomaly Score Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('outputs/anomaly_detection.png')
    plt.show()
    logger.info("Anomaly detection chart saved")


def run_anomaly_detection():
    logger.info("="*60)
    logger.info("NEXAIQ ANOMALY DETECTION ENGINE STARTED")
    logger.info("="*60)

    df_logs = generate_access_logs()
    df_security, detected = detect_security_anomalies(df_logs)
    detect_business_anomalies()
    plot_anomaly_results(df_security, detected)

    logger.info("="*60)
    logger.info("ANOMALY DETECTION COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    run_anomaly_detection()