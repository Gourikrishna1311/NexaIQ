import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import logging
import os
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_and_prepare_data():

    logger.info("Loading Superstore Sales data...")

    df = pd.read_csv(
        'data/raw/superstore_sales.csv',
        encoding='latin-1'
    )

    logger.info(f"Loaded {len(df)} rows")

    # FIX DATE FORMAT
    df['Order Date'] = pd.to_datetime(
        df['Order Date'],
        dayfirst=True
    )

    df['Ship Date'] = pd.to_datetime(
        df['Ship Date'],
        dayfirst=True
    )

    # DATE FEATURES
    df['year'] = df['Order Date'].dt.year
    df['month'] = df['Order Date'].dt.month
    df['quarter'] = df['Order Date'].dt.quarter
    df['day_of_week'] = df['Order Date'].dt.dayofweek
    df['day_of_month'] = df['Order Date'].dt.day

    df['days_to_ship'] = (
        df['Ship Date'] - df['Order Date']
    ).dt.days

    logger.info("Date features extracted")

    return df


def create_monthly_features(df):

    logger.info("Creating monthly aggregated features...")

    monthly = df.groupby(['year', 'month']).agg(
        total_sales=('Sales', 'sum'),
        total_orders=('Order ID', 'nunique')
    ).reset_index()

    monthly = monthly.sort_values(
        ['year', 'month']
    )

    monthly['quarter'] = (
        (monthly['month'] - 1) // 3 + 1
    )

    monthly['prev_month_sales'] = (
        monthly['total_sales'].shift(1)
    )

    monthly['prev_2month_sales'] = (
        monthly['total_sales'].shift(2)
    )

    monthly['prev_3month_sales'] = (
        monthly['total_sales'].shift(3)
    )

    monthly['rolling_3month_avg'] = (
        monthly['total_sales'].rolling(3).mean()
    )

    monthly['rolling_6month_avg'] = (
        monthly['total_sales'].rolling(6).mean()
    )

    monthly = monthly.dropna()

    logger.info(
        f"Monthly features created: {len(monthly)} months"
    )

    return monthly


def train_forecast_model(monthly):

    logger.info("Training sales forecast model...")

    feature_cols = [
        'year',
        'month',
        'quarter',
        'prev_month_sales',
        'prev_2month_sales',
        'prev_3month_sales',
        'rolling_3month_avg',
        'rolling_6month_avg',
        'total_orders'
    ]

    X = monthly[feature_cols]

    y = monthly['total_sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=False
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    print(f"\n{'='*50}")
    print("SALES FORECAST MODEL RESULTS")
    print(f"{'='*50}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2:   {r2:.4f} ({r2*100:.2f}%)")
    print(f"{'='*50}")

    return (
        model,
        scaler,
        feature_cols,
        X_test,
        y_test,
        y_pred
    )


def plot_forecast_results(
    y_test,
    y_pred
):

    plt.figure(figsize=(12, 5))

    plt.plot(
        range(len(y_test)),
        y_test.values,
        label='Actual Sales',
        color='steelblue',
        linewidth=2,
        marker='o'
    )

    plt.plot(
        range(len(y_pred)),
        y_pred,
        label='Predicted Sales',
        color='tomato',
        linewidth=2,
        marker='s',
        linestyle='--'
    )

    plt.title(
        'Sales Forecast — Actual vs Predicted'
    )

    plt.xlabel('Month')

    plt.ylabel('Sales ($)')

    plt.legend()

    plt.tight_layout()

    os.makedirs('outputs', exist_ok=True)

    plt.savefig(
        'outputs/sales_forecast.png'
    )

    plt.show()

    logger.info(
        "Forecast chart saved to outputs/sales_forecast.png"
    )


def plot_category_analysis(df):

    category_sales = (
        df.groupby('Category')['Sales']
        .sum()
        .sort_values(ascending=False)
    )

    region_sales = (
        df.groupby('Region')['Sales']
        .sum()
        .sort_values(ascending=False)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].bar(
        category_sales.index,
        category_sales.values,
        color='steelblue'
    )

    axes[0].set_title(
        'Total Sales by Category'
    )

    axes[0].set_ylabel('Sales ($)')

    axes[1].bar(
        region_sales.index,
        region_sales.values,
        color='tomato'
    )

    axes[1].set_title(
        'Total Sales by Region'
    )

    axes[1].set_ylabel('Sales ($)')

    plt.tight_layout()

    os.makedirs('outputs', exist_ok=True)

    plt.savefig(
        'outputs/sales_category_analysis.png'
    )

    plt.show()

    logger.info(
        "Category analysis chart saved"
    )


def save_forecast_model(
    model,
    scaler,
    feature_cols
):

    os.makedirs(
        'models',
        exist_ok=True
    )

    with open(
        'models/sales_forecast_model.pkl',
        'wb'
    ) as f:

        pickle.dump(model, f)

    with open(
        'models/sales_forecast_scaler.pkl',
        'wb'
    ) as f:

        pickle.dump(scaler, f)

    with open(
        'models/sales_forecast_features.pkl',
        'wb'
    ) as f:

        pickle.dump(feature_cols, f)

    logger.info(
        "Sales forecast model saved to models folder"
    )


def run_sales_forecast():

    logger.info("=" * 60)

    logger.info(
        "NEXAIQ SALES FORECASTING MODEL STARTED"
    )

    logger.info("=" * 60)

    df = load_and_prepare_data()

    monthly = create_monthly_features(df)

    (
        model,
        scaler,
        feature_cols,
        X_test,
        y_test,
        y_pred

    ) = train_forecast_model(monthly)

    plot_forecast_results(
        y_test,
        y_pred
    )

    plot_category_analysis(df)

    save_forecast_model(
        model,
        scaler,
        feature_cols
    )

    logger.info("=" * 60)

    logger.info(
        "SALES FORECAST PIPELINE COMPLETE"
    )

    logger.info("=" * 60)


if __name__ == "__main__":

    run_sales_forecast()