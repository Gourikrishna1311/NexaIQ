import pandas as pd
from sqlalchemy import create_engine
import logging
import os

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


def get_customer_segments():
    engine = get_engine()
    query = """
        SELECT 
            CASE 
                WHEN tenure < 6 THEN 'New'
                WHEN tenure < 24 THEN 'Growing'
                ELSE 'Established'
            END as segment,
            COUNT(*) as total,
            SUM(churn) as churned,
            ROUND(AVG(churn::numeric * 100), 2) as churn_rate,
            ROUND(SUM(monthlycharges)::numeric, 2) as total_revenue
        FROM customers
        GROUP BY segment
        ORDER BY churn_rate DESC
    """
    df = pd.read_sql(query, engine)
    logger.info("Customer segments fetched")
    return df


def get_retention_targets():
    engine = get_engine()
    query = """
        SELECT customerid, tenure, monthlycharges,
               total_value, contract, internetservice
        FROM customers
        WHERE is_high_risk = 1
          AND churn = 0
          AND total_value > 400
        ORDER BY total_value DESC
        LIMIT 20
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Retention targets fetched: {len(df)}")
    return df


def get_revenue_by_segment():
    engine = get_engine()
    query = """
        SELECT paymentmethod,
               COUNT(*) as customers,
               ROUND(AVG(churn::numeric * 100), 2) as churn_rate,
               ROUND(SUM(monthlycharges)::numeric, 2) as total_revenue
        FROM customers
        GROUP BY paymentmethod
        ORDER BY churn_rate DESC
    """
    df = pd.read_sql(query, engine)
    logger.info("Revenue by segment fetched")
    return df


def export_all_to_csv():
    os.makedirs('outputs', exist_ok=True)

    segments = get_customer_segments()
    segments.to_csv('outputs/customer_segments.csv', index=False)
    logger.info("Saved: customer_segments.csv")

    targets = get_retention_targets()
    targets.to_csv('outputs/retention_targets.csv', index=False)
    logger.info("Saved: retention_targets.csv")

    revenue = get_revenue_by_segment()
    revenue.to_csv('outputs/revenue_by_segment.csv', index=False)
    logger.info("Saved: revenue_by_segment.csv")

    print("\n--- CUSTOMER SEGMENTS ---")
    print(segments.to_string())

    print("\n--- TOP RETENTION TARGETS ---")
    print(targets.to_string())

    print("\n--- REVENUE BY PAYMENT METHOD ---")
    print(revenue.to_string())


export_all_to_csv()