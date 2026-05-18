import pandas as pd
import os

df = pd.read_csv('data/processed/telco_churn_clean.csv')

df['tenure_group'] = pd.cut(df['tenure'],
    bins=[0, 6, 12, 24, 48, 72],
    labels=['0-6 months', '6-12 months', '12-24 months',
            '24-48 months', '48-72 months'])

os.makedirs('outputs', exist_ok=True)

contract_churn = df.groupby('contract')['churn'].agg(['mean', 'sum', 'count'])
contract_churn['churn_rate_%'] = (contract_churn['mean'] * 100).round(2)
contract_churn.to_csv('outputs/contract_churn_analysis.csv')
print("Saved: contract_churn_analysis.csv")

high_risk = df[df['is_high_risk'] == 1].sort_values('total_value', ascending=False)
high_risk[['customerid', 'tenure', 'monthlycharges',
           'total_value', 'contract', 'churn']].to_csv(
    'outputs/high_risk_customers.csv', index=False)
print("Saved: high_risk_customers.csv")

tenure_churn = df.groupby('tenure_group', observed=True)['churn'].mean().mul(100).round(2)
tenure_churn.to_csv('outputs/tenure_churn_analysis.csv')
print("Saved: tenure_churn_analysis.csv")

print("\nAll analysis files saved to outputs folder")