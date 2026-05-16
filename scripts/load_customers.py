import pandas as pd
import sys
sys.path.append('.')
from scripts.customer import Customer

df = pd.read_csv('data/raw/telco_churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

customers = []

for index, row in df.iterrows():
    c = Customer(
        customer_id=row['customerID'],
        name="Customer_" + str(index),
        tenure=row['tenure'],
        monthly_charges=row['MonthlyCharges'],
        churn_status=True if row['Churn'] == 'Yes' else False
    )
    customers.append(c)

print("Total Customer objects created:", len(customers))
print()

high_risk = [c for c in customers if c.get_risk_level() == "High Risk"]
medium_risk = [c for c in customers if c.get_risk_level() == "Medium Risk"]
low_risk = [c for c in customers if c.get_risk_level() == "Low Risk"]

print("High risk customers:", len(high_risk))
print("Medium risk customers:", len(medium_risk))
print("Low risk customers:", len(low_risk))
print()

high_value = [c for c in customers if c.is_high_value()]
print("High value customers:", len(high_value))

churned_high_risk = [c for c in customers if c.churn_status == True
                     and c.get_risk_level() == "High Risk"]
print("Already churned high risk customers:", len(churned_high_risk))