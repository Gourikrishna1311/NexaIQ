import pandas as pd
import numpy as np
import pickle
import ollama
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'nexaiq_db',
    'user': 'postgres',
    'password': 'nexaiq123'
}


def get_engine():
    conn_string = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(conn_string)


def fetch_live_kpis():
    logger.info("Fetching live KPIs from PostgreSQL...")
    engine = get_engine()

    query = """
        SELECT
            COUNT(*) as total_customers,
            SUM(churn) as churned,
            ROUND(AVG(churn::numeric * 100), 2) as churn_rate,
            ROUND(SUM(monthlycharges)::numeric, 2) as monthly_revenue,
            ROUND(SUM(CASE WHEN churn=1
                THEN monthlycharges ELSE 0 END)::numeric, 2) as lost_revenue,
            SUM(is_high_risk) as high_risk_count,
            SUM(is_high_value) as high_value_count,
            ROUND(AVG(monthlycharges)::numeric, 2) as avg_monthly_charge,
            ROUND(AVG(tenure)::numeric, 2) as avg_tenure
        FROM customers
    """

    df = pd.read_sql(query, engine)
    kpis = df.iloc[0].to_dict()
    logger.info("KPIs fetched successfully")
    return kpis


def check_kpi_alerts(kpis):
    alerts = []

    if kpis['churn_rate'] > 25:
        alerts.append(f"HIGH CHURN: {kpis['churn_rate']}% churn rate above 25% threshold")
    if kpis['high_risk_count'] > 500:
        alerts.append(f"RISK ALERT: {int(kpis['high_risk_count'])} high risk customers")
    if kpis['lost_revenue'] > 100000:
        alerts.append(f"REVENUE ALERT: ${kpis['lost_revenue']:,} monthly revenue lost")

    return alerts


def generate_kpi_dashboard(kpis):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('NexaIQ — Live KPI Dashboard',
                 fontsize=16, fontweight='bold')

    metrics = [
        ('Total Customers', f"{int(kpis['total_customers']):,}",
         axes[0, 0], '#2E86AB'),
        ('Churn Rate', f"{kpis['churn_rate']}%",
         axes[0, 1], '#E84855'),
        ('Monthly Revenue', f"${kpis['monthly_revenue']:,.0f}",
         axes[0, 2], '#3BB273'),
        ('Lost Revenue', f"${kpis['lost_revenue']:,.0f}",
         axes[1, 0], '#E84855'),
        ('High Risk', f"{int(kpis['high_risk_count']):,}",
         axes[1, 1], '#F9C74F'),
        ('Avg Tenure', f"{kpis['avg_tenure']} mo",
         axes[1, 2], '#2E86AB'),
    ]

    for label, value, ax, color in metrics:
        ax.text(0.5, 0.6, value, ha='center', va='center',
                fontsize=28, fontweight='bold', color=color,
                transform=ax.transAxes)
        ax.text(0.5, 0.25, label, ha='center', va='center',
                fontsize=12, color='gray', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/live_kpi_dashboard.png', dpi=150)
    plt.show()
    logger.info("KPI dashboard saved")


def ai_kpi_commentary(kpis, alerts):
    prompt = f"""Analyze these live business KPIs and give a 3-sentence commentary.

KPIs:
- Churn rate: {kpis['churn_rate']}%
- Monthly revenue: ${kpis['monthly_revenue']:,}
- Lost revenue: ${kpis['lost_revenue']:,}
- High risk customers: {int(kpis['high_risk_count'])}
- Average tenure: {kpis['avg_tenure']} months

Alerts: {', '.join(alerts) if alerts else 'None'}

Give one immediate action recommendation."""

    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content']


def run_platform_test():
    tests = {
        'ETL Pipeline': False,
        'Database Connection': False,
        'ML Model': False,
        'SHAP Values': False,
        'Ollama AI': False,
        'KPI Monitor': False
    }

    try:
        df = pd.read_csv('data/processed/telco_churn_clean.csv')
        assert len(df) == 7043
        tests['ETL Pipeline'] = True
    except:
        pass

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        tests['Database Connection'] = True
    except:
        pass

    try:
        with open('models/churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        tests['ML Model'] = True
    except:
        pass

    try:
        shap_df = pd.read_csv('data/processed/shap_values.csv')
        assert len(shap_df) > 0
        tests['SHAP Values'] = True
    except:
        pass

    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': 'Hello'}]
        )
        tests['Ollama AI'] = True
    except:
        pass

    try:
        kpis = fetch_live_kpis()
        assert kpis['total_customers'] > 0
        tests['KPI Monitor'] = True
    except:
        pass

    print(f"\n{'='*50}")
    print("NEXAIQ PLATFORM — END TO END TEST")
    print(f"{'='*50}")
    for test, passed in tests.items():
        icon = "✓" if passed else "✗"
        status = "PASS" if passed else "FAIL"
        print(f"  {icon} {test:<25} {status}")

    all_passed = all(tests.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


def run_kpi_monitor():
    logger.info("="*60)
    logger.info("NEXAIQ KPI MONITOR STARTED")
    logger.info("="*60)

    print("\n--- RUNNING PLATFORM TESTS ---")
    run_platform_test()

    print("\n--- FETCHING LIVE KPIS ---")
    kpis = fetch_live_kpis()

    print(f"\n{'='*50}")
    print("LIVE KPI SUMMARY")
    print(f"{'='*50}")
    for key, value in kpis.items():
        print(f"  {key}: {value}")

    alerts = check_kpi_alerts(kpis)
    if alerts:
        print(f"\nACTIVE ALERTS:")
        for alert in alerts:
            print(f"  ! {alert}")

    generate_kpi_dashboard(kpis)

    print(f"\n{'='*50}")
    print("AI COMMENTARY")
    print(f"{'='*50}")
    commentary = ai_kpi_commentary(kpis, alerts)
    print(commentary)

    with open('outputs/kpi_report.txt', 'w') as f:
        f.write(f"NEXAIQ KPI REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        for key, value in kpis.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nAI COMMENTARY:\n{commentary}\n")

    logger.info("KPI report saved to outputs/kpi_report.txt")
    logger.info("="*60)
    logger.info("KPI MONITOR COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    run_kpi_monitor()