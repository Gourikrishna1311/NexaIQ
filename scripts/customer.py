class Customer:

    def __init__(self, customer_id, name, tenure, monthly_charges, churn_status):
        self.customer_id = customer_id
        self.name = name
        self.tenure = tenure
        self.monthly_charges = monthly_charges
        self.churn_status = churn_status

    def get_total_value(self):
        return self.tenure * self.monthly_charges

    def is_high_value(self):
        return self.get_total_value() > 1000

    def is_at_risk(self):
        return self.tenure < 6 and self.monthly_charges > 60

    def get_summary(self):
        return {
            "id": self.customer_id,
            "name": self.name,
            "tenure_months": self.tenure,
            "monthly_charges": self.monthly_charges,
            "total_value": self.get_total_value(),
            "high_value": self.is_high_value(),
            "at_risk": self.is_at_risk(),
            "churned": self.churn_status
        }


customers = [
    Customer("C001", "Arjun Mehta", 24, 75.50, False),
    Customer("C002", "Priya Nair", 3, 85.00, True),
    Customer("C003", "Rahul Sharma", 48, 45.00, False),
    Customer("C004", "Sneha Pillai", 5, 90.00, True),
    Customer("C005", "Vivek Rajan", 12, 55.00, False),
]

for c in customers:
    print(c.get_summary())
    print("---")