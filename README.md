NexaIQ — Explainable AI-Driven CRM Analytics with Integrated Threat Intelligence


Overview
NexaIQ is an end-to-end CRM analytics and business intelligence platform that bridges Data Science and Information Security. It combines explainable AI-powered predictive analytics, anomaly-based threat detection, and a conversational AI layer into one unified system.
Built as a pre-MTech research project addressing documented gaps in current CRM-AI literature.

Problem Statement
Existing CRM platforms have four critical gaps:

Black-box ML — predictions are accurate but unexplainable to business users
No AI governance — AI agents have no activity logging or access controls
Data fragmentation — customer, sales, and interaction data live in silos
No conversational XAI — no natural language interface for model explanations

NexaIQ addresses all four gaps in a single integrated platform.

Tech Stack
LayerTechnologiesLanguagePython 3.13, SQLDatabasePostgreSQL 17, SQLiteML Modelsscikit-learn, XGBoost, SHAPAnomaly DetectionIsolation Forest, AutoencoderAI AssistantClaude API (Anthropic)VisualizationPower BI, matplotlib, seabornSecurityOWASP, JWT, bcrypt, AES encryptionDeploymentAWS Free Tier / RenderVersion ControlGit, GitHub

Project Architecture
Raw CRM Data
↓
ETL Pipeline (clean_data.py + etl_pipeline.py)
↓
PostgreSQL Database (nexaiq_db)
↓
ML Models + SHAP Explainability
↓
AI Business Assistant (Claude API)
↓
Security Layer + AI Governance Module
↓
Live Dashboard (Power BI)

Dataset

IBM Telco Customer Churn — 7,043 customers, 21 features
Superstore Sales — sales forecasting data
B2B Sales Leads — lead scoring data
Synthetic Access Logs — security anomaly detection


Current Results
ModelAccuracyAUC ScoreLogistic Regression73.17%0.8114Random Forest75.23%0.8095XGBoost (Best)75.23%0.8150
Key Business Insights Found:

26.54% overall churn rate — 1 in 4 customers leaving
Month-to-month customers churn at 42.71% — highest risk segment
First 6 months are most critical — 53.3% churn rate
$139,131 monthly revenue already lost from churned customers
521 high risk customers identified for retention campaigns


Project Structure
NexaIQ/
├── data/
│   ├── raw/                    # Original IBM Telco dataset
│   └── processed/              # Cleaned dataset (29 columns)
├── scripts/
│   ├── customer.py             # OOP customer model
│   ├── clean_data.py           # Automated cleaning pipeline
│   ├── etl_pipeline.py         # Full ETL automation
│   ├── database_setup.py       # PostgreSQL setup
│   ├── db_queries.py           # SQL query module
│   ├── advanced_queries.py     # CTEs, window functions
│   ├── generate_charts.py      # 6-chart dashboard module
│   ├── business_analysis.py    # 10 business questions
│   ├── churn_model.py          # ML model training
│   └── save_model.py           # Model persistence
├── models/
│   ├── churn_model.pkl         # Trained XGBoost model
│   ├── scaler.pkl              # Feature scaler
│   └── feature_cols.pkl        # Feature column names
├── notebooks/
│   └── day05_numpy_analysis    # NumPy statistical analysis
├── outputs/
│   ├── nexaiq_dashboard.png    # 6-chart dashboard
│   ├── feature_importance.png  # XGBoost feature importance
│   ├── model_comparison.png    # Model comparison chart
│   └── *.csv                   # Analysis exports
└── requirements.txt

Phases
PhaseStatusDescriptionPhase 1CompletePython foundations, data cleaning, ETL, SQL, PostgreSQLPhase 2CompleteML models — churn prediction, XGBoostPhase 3In ProgressSHAP explainability layerPhase 4PlannedAnomaly detection enginePhase 5PlannedClaude API conversational assistantPhase 6PlannedSecurity hardening and AI governancePhase 7PlannedCloud deployment

Key Features Completed

Automated ETL pipeline — raw CSV to PostgreSQL in one command
7043 customers loaded and categorized by risk level
XGBoost churn prediction model — 81.5% AUC
6-chart professional dashboard
Advanced SQL analysis — CTEs, window functions, retention targeting
521 high risk customers identified with retention recommendations


Literature Gaps Addressed
This project directly addresses four documented gaps in CRM-AI research:

Explainability — SHAP values on every prediction (in progress)
AI Governance — Claude API activity logging (planned)
Data Unification — single ETL pipeline across all sources (complete)
Conversational XAI — NLP interface for model explanations (planned)


Author
Gourikrishna — BTech CSE Graduate

Status
Actively being built — updated daily
