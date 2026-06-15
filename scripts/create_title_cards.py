import matplotlib.pyplot as plt
import os


def create_card(title, subtitle, filename,
                bg='#0f1117', title_color='#2E86AB',
                sub_color='#888888'):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis('off')

    ax.text(0.5, 0.58, title,
            ha='center', va='center',
            fontsize=40, fontweight='bold',
            color=title_color,
            transform=ax.transAxes)

    ax.text(0.5, 0.40, subtitle,
            ha='center', va='center',
            fontsize=20, color=sub_color,
            transform=ax.transAxes)

    os.makedirs('outputs/cards', exist_ok=True)
    plt.savefig(f'outputs/cards/{filename}.png',
                dpi=150, bbox_inches='tight',
                facecolor=bg)
    plt.close()
    print(f"Card saved: {filename}.png")


cards = [
    (
        "NexaIQ",
        "Explainable AI-Driven CRM Analytics Platform",
        "01_title"
    ),
    (
        "The Problem",
        "26.54% of customers are churning\n$139,131 monthly revenue lost every month",
        "02_problem"
    ),
    (
        "Step 1 — ETL Pipeline",
        "Raw CSV data → Clean → PostgreSQL\n7,043 customers processed automatically",
        "03_etl"
    ),
    (
        "Step 2 — Machine Learning",
        "XGBoost Churn Prediction — AUC 0.815\nSales Forecasting + Lead Scoring",
        "04_ml"
    ),
    (
        "Step 3 — SHAP Explainability",
        "Every prediction explained in plain English\nAddresses black-box ML research gap",
        "05_shap"
    ),
    (
        "Step 4 — Anomaly Detection",
        "Dual-purpose Isolation Forest engine\nBusiness anomalies + Security threats",
        "06_anomaly"
    ),
    (
        "Step 5 — AI Assistant",
        "LLaMA 3.2 powered business Q&A\nConverts SHAP values to plain English",
        "07_ai"
    ),
    (
        "Step 6 — Security Layer",
        "JWT + bcrypt + OWASP 83% compliance\nAI governance + activity logging",
        "08_security"
    ),
    (
        "Live Dashboard",
        "github.io/NexaIQ — open in any browser\nReal-time KPIs + Charts + Risk table",
        "09_dashboard"
    ),
    (
        "Results",
        "3 ML models | 521 high risk customers identified\n$139K revenue at risk quantified | 4 research gaps addressed",
        "10_results"
    ),
    (
        "NexaIQ",
        "Built in 30 days\ngithub.com/Gourikrishna1311/NexaIQ",
        "11_end"
    )
]

for title, subtitle, filename in cards:
    create_card(title, subtitle, filename)

print("\nAll title cards saved to outputs/cards/")
print("Ready for screen recording")