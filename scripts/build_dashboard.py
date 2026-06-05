import pandas as pd
import numpy as np
import json
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dashboard_data():
    df = pd.read_csv('data/processed/telco_churn_clean.csv')

    churn_overview = {
        'total': int(len(df)),
        'churned': int(df['churn'].sum()),
        'retained': int((df['churn'] == 0).sum()),
        'churn_rate': round(float(df['churn'].mean() * 100), 2)
    }

    revenue = {
        'total': round(float(df['monthlycharges'].sum()), 2),
        'lost': round(float(df[df['churn']==1]['monthlycharges'].sum()), 2),
        'at_risk': round(float(df[df['is_high_risk']==1]['monthlycharges'].sum()), 2),
        'avg': round(float(df['monthlycharges'].mean()), 2)
    }

    contract_churn = df.groupby('contract')['churn'].agg(
        ['mean', 'sum', 'count']
    ).reset_index()
    contract_churn['churn_rate'] = (contract_churn['mean'] * 100).round(2)
    contract_data = contract_churn[['contract', 'churn_rate', 'count']].to_dict('records')

    df['tenure_group'] = pd.cut(df['tenure'],
        bins=[0, 6, 12, 24, 48, 72],
        labels=['0-6m', '6-12m', '12-24m', '24-48m', '48-72m'])
    tenure_churn = df.groupby('tenure_group', observed=True)['churn'].mean().mul(100).round(2)
    tenure_data = [{'group': str(k), 'rate': float(v)}
                   for k, v in tenure_churn.items()]

    internet_data = df.groupby('internetservice')['churn'].mean().mul(100).round(2).to_dict()

    high_risk = df[df['is_high_risk']==1].nlargest(10, 'total_value')[
        ['customerid', 'tenure', 'monthlycharges', 'total_value', 'contract', 'churn']
    ].to_dict('records')

    return {
        'churn_overview': churn_overview,
        'revenue': revenue,
        'contract_data': contract_data,
        'tenure_data': tenure_data,
        'internet_data': internet_data,
        'high_risk': high_risk,
        'high_risk_count': int(df['is_high_risk'].sum()),
        'high_value_count': int(df['is_high_value'].sum())
    }


def build_html_dashboard(data):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NexaIQ — CRM Analytics Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e0e0e0; }}
  .header {{ background: #1a1d27; padding: 20px 30px;
             border-bottom: 2px solid #2E86AB; }}
  .header h1 {{ font-size: 24px; color: #2E86AB; font-weight: 600; }}
  .header p {{ color: #888; font-size: 13px; margin-top: 4px; }}
  .container {{ padding: 24px 30px; max-width: 1400px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(6, 1fr);
               gap: 14px; margin-bottom: 24px; }}
  .kpi-card {{ background: #1a1d27; border-radius: 10px;
               padding: 16px; border: 1px solid #2a2d3a;
               text-align: center; }}
  .kpi-value {{ font-size: 26px; font-weight: 700; margin-bottom: 4px; }}
  .kpi-label {{ font-size: 11px; color: #888; text-transform: uppercase;
                letter-spacing: 0.5px; }}
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr;
                  gap: 18px; margin-bottom: 24px; }}
  .chart-card {{ background: #1a1d27; border-radius: 10px;
                 padding: 18px; border: 1px solid #2a2d3a; }}
  .chart-title {{ font-size: 13px; font-weight: 600; color: #ccc;
                  margin-bottom: 14px; text-transform: uppercase;
                  letter-spacing: 0.5px; }}
  .chart-container {{ position: relative; height: 200px; }}
  .table-card {{ background: #1a1d27; border-radius: 10px;
                 padding: 18px; border: 1px solid #2a2d3a; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ padding: 10px 12px; text-align: left; background: #0f1117;
        color: #888; font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.5px; }}
  td {{ padding: 10px 12px; border-top: 1px solid #2a2d3a; }}
  .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 11px;
            font-weight: 600; }}
  .badge-danger {{ background: #E8485522; color: #E84855; }}
  .badge-success {{ background: #3BB27322; color: #3BB273; }}
  .badge-warning {{ background: #F9C74F22; color: #F9C74F; }}
  .blue {{ color: #2E86AB; }}
  .red {{ color: #E84855; }}
  .green {{ color: #3BB273; }}
  .yellow {{ color: #F9C74F; }}
</style>
</head>
<body>

<div class="header">
  <h1>NexaIQ — CRM Analytics Dashboard</h1>
  <p>Real-time business intelligence powered by XGBoost ML + SHAP Explainability</p>
</div>

<div class="container">

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-value blue">{data['churn_overview']['total']:,}</div>
      <div class="kpi-label">Total Customers</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value red">{data['churn_overview']['churn_rate']}%</div>
      <div class="kpi-label">Churn Rate</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value green">${data['revenue']['total']:,.0f}</div>
      <div class="kpi-label">Monthly Revenue</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value red">${data['revenue']['lost']:,.0f}</div>
      <div class="kpi-label">Revenue Lost</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value yellow">{data['high_risk_count']:,}</div>
      <div class="kpi-label">High Risk</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value green">{data['high_value_count']:,}</div>
      <div class="kpi-label">High Value</div>
    </div>
  </div>

  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-title">Churn by Contract Type</div>
      <div class="chart-container">
        <canvas id="contractChart"></canvas>
      </div>
    </div>
    <div class="chart-card">
      <div class="chart-title">Churn by Tenure Group</div>
      <div class="chart-container">
        <canvas id="tenureChart"></canvas>
      </div>
    </div>
    <div class="chart-card">
      <div class="chart-title">Revenue Breakdown</div>
      <div class="chart-container">
        <canvas id="revenueChart"></canvas>
      </div>
    </div>
  </div>

  <div class="table-card">
    <div class="chart-title" style="margin-bottom:14px;">
      Top High Risk Customers — Retention Targets
    </div>
    <table>
      <thead>
        <tr>
          <th>Customer ID</th>
          <th>Tenure (months)</th>
          <th>Monthly Charges</th>
          <th>Total Value</th>
          <th>Contract</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {''.join([f"""
        <tr>
          <td>{r['customerid']}</td>
          <td>{r['tenure']}</td>
          <td>${r['monthlycharges']:.2f}</td>
          <td>${r['total_value']:.2f}</td>
          <td>{r['contract']}</td>
          <td><span class="badge {'badge-danger' if r['churn'] == 1 else 'badge-warning'}">
            {'Churned' if r['churn'] == 1 else 'At Risk'}
          </span></td>
        </tr>""" for r in data['high_risk']])}
      </tbody>
    </table>
  </div>

</div>

<script>
const contractLabels = {json.dumps([r['contract'] for r in data['contract_data']])};
const contractRates = {json.dumps([r['churn_rate'] for r in data['contract_data']])};

new Chart(document.getElementById('contractChart'), {{
  type: 'bar',
  data: {{
    labels: contractLabels,
    datasets: [{{
      label: 'Churn Rate %',
      data: contractRates,
      backgroundColor: ['#E84855', '#F9C74F', '#3BB273'],
      borderRadius: 6
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ grid: {{ color: '#2a2d3a' }}, ticks: {{ color: '#888' }} }},
      x: {{ grid: {{ display: false }}, ticks: {{ color: '#888' }} }}
    }}
  }}
}});

const tenureLabels = {json.dumps([r['group'] for r in data['tenure_data']])};
const tenureRates = {json.dumps([r['rate'] for r in data['tenure_data']])};

new Chart(document.getElementById('tenureChart'), {{
  type: 'line',
  data: {{
    labels: tenureLabels,
    datasets: [{{
      label: 'Churn Rate %',
      data: tenureRates,
      borderColor: '#2E86AB',
      backgroundColor: '#2E86AB22',
      tension: 0.4,
      fill: true,
      pointBackgroundColor: '#E84855',
      pointRadius: 5
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ grid: {{ color: '#2a2d3a' }}, ticks: {{ color: '#888' }} }},
      x: {{ grid: {{ display: false }}, ticks: {{ color: '#888' }} }}
    }}
  }}
}});

new Chart(document.getElementById('revenueChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Retained Revenue', 'Lost Revenue', 'At Risk Revenue'],
    datasets: [{{
      data: [
        {data['revenue']['total'] - data['revenue']['lost'] - data['revenue']['at_risk']:.0f},
        {data['revenue']['lost']:.0f},
        {data['revenue']['at_risk']:.0f}
      ],
      backgroundColor: ['#3BB273', '#E84855', '#F9C74F'],
      borderWidth: 0
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{
        position: 'bottom',
        labels: {{ color: '#888', font: {{ size: 11 }} }}
      }}
    }}
  }}
}});
</script>

</body>
</html>"""

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/nexaiq_dashboard.html', 'w') as f:
        f.write(html)

    logger.info("Dashboard saved to outputs/nexaiq_dashboard.html")
    return html


def run_dashboard_builder():
    logger.info("="*60)
    logger.info("NEXAIQ DASHBOARD BUILDER STARTED")
    logger.info("="*60)

    data = load_dashboard_data()
    build_html_dashboard(data)

    print(f"\n{'='*50}")
    print("NEXAIQ DASHBOARD BUILT SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"File: outputs/nexaiq_dashboard.html")
    print(f"Open this file in any browser to see your dashboard")
    print(f"\nKPI Summary:")
    print(f"  Total customers: {data['churn_overview']['total']:,}")
    print(f"  Churn rate:      {data['churn_overview']['churn_rate']}%")
    print(f"  Monthly revenue: ${data['revenue']['total']:,.2f}")
    print(f"  Lost revenue:    ${data['revenue']['lost']:,.2f}")
    print(f"  High risk:       {data['high_risk_count']:,}")

    logger.info("="*60)
    logger.info("DASHBOARD BUILDER COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    run_dashboard_builder()