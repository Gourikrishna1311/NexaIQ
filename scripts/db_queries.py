import pandas as pd
from sqlalchemy import create_engine, text
import logging

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


def get_churned_customers():
    engine = get_engine()
    query = "SELECT * FROM customers WHERE churn = 1"
    df = pd.read_sql(query, engine)
    logger.info(f"Churned customers fetched: {len(df)}")
    return df


def get_high_risk_customers():
    engine = get_engine()
    query = """
        SELECT customerid, tenure, monthlycharges,
               total_value, contract, churn
        FROM customers
        WHERE is_high_risk = 1
        ORDER BY total_value DESC
    """
    df = pd.read_sql(query, engine)
    logger.info(f"High risk customers fetched: {len(df)}")
    return df


def get_revenue_by_contract():
    engine = get_engine()
    query = """
        SELECT contract,
               COUNT(*) as customer_count,
               ROUND(AVG(monthlycharges)::numeric, 2) as avg_monthly,
               ROUND(SUM(monthlycharges)::numeric, 2) as total_monthly,
               ROUND(AVG(churn::float * 100)::numeric, 2) as churn_rate_pct
        FROM customers
        GROUP BY contract
        ORDER BY total_monthly DESC
    """
    df = pd.read_sql(query, engine)
    logger.info("Revenue by contract fetched")
    return df


def get_churn_summary():
    engine = get_engine()
    query = """
        SELECT
            COUNT(*) as total_customers,
            SUM(churn) as churned,
            COUNT(*) - SUM(churn) as retained,
            ROUND(AVG(churn::float * 100)::numeric, 2) as churn_rate_pct,
            ROUND(SUM(monthlycharges)::numeric, 2) as total_monthly_revenue,
            ROUND(SUM(CASE WHEN churn=1 THEN monthlycharges ELSE 0 END)::numeric, 2)
                as lost_revenue
        FROM customers
    """
    df = pd.read_sql(query, engine)
    logger.info("Churn summary fetched")
    return df


print("Script is running")

print("\n--- CHURN SUMMARY ---")
print(get_churn_summary().to_string())

print("\n--- REVENUE BY CONTRACT ---")
print(get_revenue_by_contract().to_string())

print("\n--- TOP 10 HIGH RISK CUSTOMERS ---")
print(get_high_risk_customers().head(10).to_string())