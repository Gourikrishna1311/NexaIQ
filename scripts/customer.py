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

    def get_risk_level(self):
        if self.tenure < 6 and self.monthly_charges > 70:
            return "High Risk"
        elif self.tenure < 12 and self.monthly_charges > 50:
            return "Medium Risk"
        else:
            return "Low Risk"

    def get_contract_recommendation(self):
        if self.churn_status == True:
            return "Offer discount to retain"
        elif self.is_at_risk():
            return "Upgrade to annual contract"
        else:
            return "No action needed"

    def get_summary(self):
        return {
            "id": self.customer_id,
            "name": self.name,
            "tenure_months": self.tenure,
            "monthly_charges": self.monthly_charges,
            "total_value": self.get_total_value(),
            "high_value": self.is_high_value(),
            "at_risk": self.is_at_risk(),
            "risk_level": self.get_risk_level(),
            "recommendation": self.get_contract_recommendation(),
            "churned": self.churn_status
        }


class PremiumCustomer(Customer):

    def __init__(self, customer_id, name, tenure,
                 monthly_charges, churn_status, loyalty_points):
        super().__init__(customer_id, name, tenure,
                        monthly_charges, churn_status)
        self.loyalty_points = loyalty_points

    def get_discount(self):
        if self.loyalty_points > 500:
            return 20
        elif self.loyalty_points > 200:
            return 10
        else:
            return 5

    def get_summary(self):
        base = super().get_summary()
        base["loyalty_points"] = self.loyalty_points
        base["discount_percent"] = self.get_discount()
        return base


print("REGULAR CUSTOMERS:")
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

print("\nPREMIUM CUSTOMERS:")
premium_customers = [
    PremiumCustomer("P001", "Meera Nambiar", 36, 95.00, False, 650),
    PremiumCustomer("P002", "Kiran Pillai", 4, 110.00, True, 100),
]

for p in premium_customers:
    print(p.get_summary())
    print("---")