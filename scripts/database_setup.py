import pandas as pd
from sqlalchemy import create_engine, text
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
    engine = create_engine(conn_string)
    logger.info("Database engine created successfully")
    return engine


def test_connection(engine):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        version = result.fetchone()[0]
        logger.info(f"Connected to PostgreSQL: {version[:50]}")
    return True


def load_customers_table(engine):
    logger.info("Loading customers table...")
    df = pd.read_csv('data/processed/telco_churn_clean.csv')

    customer_cols = [
        'customerid', 'gender', 'seniorcitizen', 'partner',
        'dependents', 'tenure', 'phoneservice', 'internetservice',
        'contract', 'paperlessbilling', 'paymentmethod',
        'monthlycharges', 'totalcharges', 'churn',
        'gender_encoded', 'contract_encoded',
        'internetservice_encoded', 'paymentmethod_encoded',
        'total_value', 'is_high_value', 'is_high_risk',
        'avg_monthly_total'
    ]

    df_customers = df[customer_cols].copy()
    df_customers.to_sql('customers', engine,
                        if_exists='replace',
                        index=False)
    logger.info(f"Customers table loaded: {len(df_customers)} rows")
    return len(df_customers)


def verify_tables(engine):
    logger.info("Verifying tables...")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """))
        tables = [row[0] for row in result]
        logger.info(f"Tables in database: {tables}")

        result = conn.execute(text("SELECT COUNT(*) FROM customers"))
        count = result.fetchone()[0]
        logger.info(f"Rows in customers table: {count}")

        result = conn.execute(text("""
            SELECT churn, COUNT(*) as count,
                   ROUND(AVG(monthlycharges)::numeric, 2) as avg_charges
            FROM customers
            GROUP BY churn
            ORDER BY churn
        """))
        print("\nDatabase verification query result:")
        print("Churn | Count | Avg Charges")
        print("-" * 35)
        for row in result:
            print(f"  {row[0]}   | {row[1]:5d} | ${row[2]}")


def run_setup():
    logger.info("=" * 50)
    logger.info("NEXAIQ DATABASE SETUP STARTED")
    logger.info("=" * 50)

    engine = get_engine()
    test_connection(engine)
    load_customers_table(engine)
    verify_tables(engine)

    logger.info("=" * 50)
    logger.info("DATABASE SETUP COMPLETE")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_setup()