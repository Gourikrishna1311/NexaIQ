import pandas as pd
import logging
import os
import schedule
import time
from datetime import datetime
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/etl_pipeline.log'),
        logging.StreamHandler()
    ]
)
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


def extract():
    logger.info("EXTRACT: Loading raw data...")
    df = pd.read_csv('data/raw/telco_churn.csv')
    logger.info(f"EXTRACT: {len(df)} rows extracted from CSV")
    return df


def transform(df):
    logger.info("TRANSFORM: Starting data transformation...")

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    logger.info("TRANSFORM: Fixed TotalCharges column")

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    binary_cols = ['partner', 'dependents', 'phoneservice',
                   'paperlessbilling', 'churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    df['gender_encoded'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['contract_encoded'] = df['contract'].map({
        'Month-to-month': 0, 'One year': 1, 'Two year': 2
    })
    df['internetservice_encoded'] = df['internetservice'].map({
        'No': 0, 'DSL': 1, 'Fiber optic': 2
    })
    df['paymentmethod_encoded'] = df['paymentmethod'].map({
        'Electronic check': 0, 'Mailed check': 1,
        'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
    })

    df['total_value'] = df['tenure'] * df['monthlycharges']
    df['is_high_value'] = (df['total_value'] > 1000).astype(int)
    df['is_high_risk'] = (
        (df['tenure'] < 6) & (df['monthlycharges'] > 70)
    ).astype(int)
    df['avg_monthly_total'] = df.apply(
        lambda row: row['totalcharges'] / row['tenure']
        if row['tenure'] > 0 else 0, axis=1
    )
    df['etl_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    df = df.drop_duplicates()
    logger.info(f"TRANSFORM: {len(df)} rows after transformation")
    return df


def load(df):
    logger.info("LOAD: Loading data into PostgreSQL...")
    engine = get_engine()

    customer_cols = [
        'customerid', 'gender', 'seniorcitizen', 'partner',
        'dependents', 'tenure', 'phoneservice', 'internetservice',
        'contract', 'paperlessbilling', 'paymentmethod',
        'monthlycharges', 'totalcharges', 'churn',
        'gender_encoded', 'contract_encoded',
        'internetservice_encoded', 'paymentmethod_encoded',
        'total_value', 'is_high_value', 'is_high_risk',
        'avg_monthly_total', 'etl_timestamp'
    ]

    df[customer_cols].to_sql('customers', engine,
                              if_exists='replace',
                              index=False)
    logger.info(f"LOAD: {len(df)} rows loaded into customers table")

    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM customers"))
        count = result.fetchone()[0]
        logger.info(f"LOAD: Verified {count} rows in database")


def run_etl():
    logger.info("=" * 60)
    logger.info("NEXAIQ ETL PIPELINE STARTED")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    try:
        df_raw = extract()
        df_clean = transform(df_raw)
        load(df_clean)

        logger.info("=" * 60)
        logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"ETL PIPELINE FAILED: {str(e)}")
        raise


def run_once():
    run_etl()


def run_scheduled():
    logger.info("Scheduling ETL pipeline to run every 60 seconds...")
    schedule.every(60).seconds.do(run_etl)
    run_etl()
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    run_once()