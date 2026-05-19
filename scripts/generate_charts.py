import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLORS = {
    'primary': '#2E86AB',
    'danger': '#E84855',
    'success': '#3BB273',
    'warning': '#F9C74F',
    'neutral': '#6C757D',
    'light': '#F8F9FA'
}


def load_data():
    logger.info("Loading clean data...")
    df = pd.read_csv('data/processed/telco_churn_clean.csv')
    df['tenure_group'] = pd.cut(df['tenure'],
        bins=[0, 6, 12, 24, 48, 72],
        labels=['0-6m', '6-12m', '12-24m', '24-48m', '48-72m'])
    logger.info(f"Loaded {len(df)} rows")
    return df


def chart1_churn_overview(df, ax):
    counts = df['churn'].value_counts()
    labels = ['Retained', 'Churned']
    values = [counts[0], counts[1]]
    colors = [COLORS['success'], COLORS['danger']]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}\n({val/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('Customer Churn Overview', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Customers')
    ax.set_ylim(0, max(values) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    logger.info("Chart 1 done: Churn overview")


def chart2_churn_by_contract(df, ax):
    contract_map = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
    df['contract_label'] = df['contract_encoded'].map(contract_map)
    churn_rate = df.groupby('contract_label')['churn'].mean() * 100
    order = ['Month-to-month', 'One year', 'Two year']
    churn_rate = churn_rate.reindex(order)
    colors = [COLORS['danger'] if x > 30 else COLORS['warning']
              if x > 10 else COLORS['success'] for x in churn_rate.values]
    bars = ax.bar(churn_rate.index, churn_rate.values,
                  color=colors, width=0.5, edgecolor='white')
    for bar, val in zip(bars, churn_rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_title('Churn Rate by Contract Type', fontsize=13,
                 fontweight='bold', pad=15)
    ax.set_ylabel('Churn Rate (%)')
    ax.set_ylim(0, 55)
    ax.tick_params(axis='x', rotation=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    logger.info("Chart 2 done: Churn by contract")


def chart3_churn_by_tenure(df, ax):
    tenure_churn = df.groupby('tenure_group',
                              observed=True)['churn'].mean() * 100
    ax.plot(tenure_churn.index, tenure_churn.values,
            color=COLORS['primary'], linewidth=2.5,
            marker='o', markersize=8, markerfacecolor=COLORS['danger'])
    for i, (x, y) in enumerate(zip(tenure_churn.index, tenure_churn.values)):
        ax.annotate(f'{y:.1f}%', (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Churn Rate by Customer Tenure',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Churn Rate (%)')
    ax.set_xlabel('Tenure Group')
    ax.set_ylim(0, 65)
    ax.fill_between(range(len(tenure_churn)),
                    tenure_churn.values, alpha=0.1,
                    color=COLORS['primary'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    logger.info("Chart 3 done: Churn by tenure")


def chart4_revenue_analysis(df, ax):
    segments = ['Total Revenue', 'At Risk Revenue', 'Lost Revenue']
    values = [
        df['monthlycharges'].sum(),
        df[df['is_high_risk'] == 1]['monthlycharges'].sum(),
        df[df['churn'] == 1]['monthlycharges'].sum()
    ]
    colors = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    bars = ax.barh(segments, values, color=colors,
                   height=0.5, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                f'${val:,.0f}', va='center', fontsize=10, fontweight='bold')
    ax.set_title('Monthly Revenue Analysis',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Monthly Revenue ($)')
    ax.set_xlim(0, max(values) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    logger.info("Chart 4 done: Revenue analysis")


def chart5_churn_by_internet(df, ax):
    internet_map = {0: 'No Internet', 1: 'DSL', 2: 'Fiber Optic'}
    df['internet_label'] = df['internetservice_encoded'].map(internet_map)
    churn_rate = df.groupby('internet_label')['churn'].mean() * 100
    colors = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    wedges, texts, autotexts = ax.pie(
        churn_rate.values,
        labels=churn_rate.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for text in autotexts:
        text.set_fontweight('bold')
    ax.set_title('Churn Rate by Internet Service',
                 fontsize=13, fontweight='bold', pad=15)
    logger.info("Chart 5 done: Churn by internet service")


def chart6_high_risk_profile(df, ax):
    risk_segments = {
        'High Risk\nChurned': len(df[(df['is_high_risk']==1) & (df['churn']==1)]),
        'High Risk\nRetained': len(df[(df['is_high_risk']==1) & (df['churn']==0)]),
        'Low Risk\nChurned': len(df[(df['is_high_risk']==0) & (df['churn']==1)]),
        'Low Risk\nRetained': len(df[(df['is_high_risk']==0) & (df['churn']==0)])
    }
    colors = [COLORS['danger'], COLORS['warning'],
              COLORS['neutral'], COLORS['success']]
    bars = ax.bar(risk_segments.keys(), risk_segments.values(),
                  color=colors, width=0.6, edgecolor='white')
    for bar, val in zip(bars, risk_segments.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val:,}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_title('Risk Profile vs Churn Status',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Customers')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    logger.info("Chart 6 done: Risk profile")


def generate_all_charts():
    logger.info("=" * 50)
    logger.info("NEXAIQ CHART GENERATION STARTED")
    logger.info("=" * 50)

    df = load_data()
    os.makedirs('outputs', exist_ok=True)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('NexaIQ — CRM Analytics Dashboard',
                 fontsize=16, fontweight='bold', y=1.02)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    chart1_churn_overview(df, fig.add_subplot(gs[0, 0]))
    chart2_churn_by_contract(df, fig.add_subplot(gs[0, 1]))
    chart3_churn_by_tenure(df, fig.add_subplot(gs[0, 2]))
    chart4_revenue_analysis(df, fig.add_subplot(gs[1, 0]))
    chart5_churn_by_internet(df, fig.add_subplot(gs[1, 1]))
    chart6_high_risk_profile(df, fig.add_subplot(gs[1, 2]))

    plt.savefig('outputs/nexaiq_dashboard.png',
                bbox_inches='tight', dpi=150)
    plt.show()

    logger.info("Dashboard saved: outputs/nexaiq_dashboard.png")
    logger.info("=" * 50)
    logger.info("ALL CHARTS GENERATED SUCCESSFULLY")
    logger.info("=" * 50)


if __name__ == "__main__":
    generate_all_charts()