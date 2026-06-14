from flask import Flask, jsonify, request, send_file
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__)


def load_models():
    models = {}
    try:
        with open('models/churn_model.pkl', 'rb') as f:
            models['churn_model'] = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            models['churn_scaler'] = pickle.load(f)
        with open('models/feature_cols.pkl', 'rb') as f:
            models['churn_features'] = pickle.load(f)
        print("Models loaded successfully")
    except Exception as e:
        print(f"Model loading error: {e}")
    return models


models = load_models()


@app.route('/')
def home():
    return jsonify({
        'platform': 'NexaIQ',
        'version': '1.0',
        'status': 'running',
        'description': 'Explainable AI-Driven CRM Analytics Platform',
        'author': 'Gourikrishna',
        'github': 'github.com/Gourikrishna1311/NexaIQ',
        'endpoints': [
            'GET  /api/health',
            'GET  /api/kpis',
            'GET  /api/high-risk',
            'GET  /api/churn-by-contract',
            'GET  /api/churn-by-tenure',
            'POST /api/predict',
            'GET  /api/dashboard'
        ],
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/health')
def health():
    try:
        df = pd.read_csv('data/processed/telco_churn_clean.csv')
        db_status = 'connected'
        customer_count = len(df)
    except:
        db_status = 'error'
        customer_count = 0

    return jsonify({
        'status': 'healthy',
        'database': db_status,
        'model': 'loaded' if models else 'not loaded',
        'total_customers': customer_count,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/kpis')
def get_kpis():
    try:
        df = pd.read_csv('data/processed/telco_churn_clean.csv')
        kpis = {
            'total_customers': int(len(df)),
            'churn_rate': round(float(df['churn'].mean() * 100), 2),
            'churned_customers': int(df['churn'].sum()),
            'retained_customers': int((df['churn'] == 0).sum()),
            'total_monthly_revenue': round(float(df['monthlycharges'].sum()), 2),
            'lost_revenue': round(float(
                df[df['churn']==1]['monthlycharges'].sum()), 2),
            'high_risk_customers': int(df['is_high_risk'].sum()),
            'high_value_customers': int(df['is_high_value'].sum()),
            'avg_monthly_charges': round(float(df['monthlycharges'].mean()), 2),
            'avg_tenure': round(float(df['tenure'].mean()), 2),
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(kpis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/high-risk')
def get_high_risk():
    try:
        df = pd.read_csv('data/processed/telco_churn_clean.csv')
        high_risk = df[df['is_high_risk'] == 1].nlargest(
            20, 'total_value'
        )[['customerid', 'tenure', 'monthlycharges',
           'total_value', 'contract', 'churn']]
        return jsonify({
            'count': len(high_risk),
            'customers': high_risk.to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/churn-by-contract')
def churn_by_contract():
    try:
        df = pd.read_csv('data/processed/telco_churn_clean.csv')
        result = df.groupby('contract')['churn'].agg(
            ['mean', 'sum', 'count']
        ).reset_index()
        result['churn_rate'] = (result['mean'] * 100).round(2)
        result = result.rename(columns={
            'sum': 'churned', 'count': 'total'
        })
        return jsonify(
            result[['contract', 'churn_rate',
                    'churned', 'total']].to_dict('records')
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/churn-by-tenure')
def churn_by_tenure():
    try:
        df = pd.read_csv('data/processed/telco_churn_clean.csv')
        df['tenure_group'] = pd.cut(df['tenure'],
            bins=[0, 6, 12, 24, 48, 72],
            labels=['0-6m', '6-12m', '12-24m', '24-48m', '48-72m'])
        result = df.groupby('tenure_group', observed=True)['churn'].agg(
            ['mean', 'count']
        ).reset_index()
        result['churn_rate'] = (result['mean'] * 100).round(2)
        result = result.rename(columns={'count': 'total'})
        return jsonify(
            result[['tenure_group', 'churn_rate',
                    'total']].to_dict('records')
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    if not models:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        feature_cols = models['churn_features']
        missing = [f for f in feature_cols if f not in data]
        if missing:
            return jsonify({
                'error': f'Missing fields: {missing}',
                'required_fields': feature_cols
            }), 400

        X = [[data[f] for f in feature_cols]]
        X_scaled = models['churn_scaler'].transform(X)

        probability = float(models['churn_model'].predict_proba(X_scaled)[0][1])
        prediction = int(models['churn_model'].predict(X_scaled)[0])

        return jsonify({
            'customer_id': data.get('customer_id', 'unknown'),
            'churn_probability': round(probability * 100, 1),
            'prediction': 'WILL CHURN' if prediction == 1 else 'WILL STAY',
            'risk_level': (
                'HIGH' if probability > 0.7 else
                'MEDIUM' if probability > 0.4 else 'LOW'
            ),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard')
def dashboard():
    try:
        return send_file('outputs/nexaiq_dashboard.html')
    except:
        return jsonify({'error': 'Dashboard not built yet. Run build_dashboard.py first'}), 404


@app.route('/api/summary')
def summary():
    try:
        df = pd.read_csv('data/processed/telco_churn_clean.csv')
        scripts_count = len([f for f in os.listdir('scripts')
                             if f.endswith('.py')])
        models_count = len([f for f in os.listdir('models')
                           if f.endswith('.pkl')])

        return jsonify({
            'project': 'NexaIQ',
            'author': 'Gourikrishna',
            'built_in_days': 30,
            'total_customers': int(len(df)),
            'churn_rate': round(float(df['churn'].mean() * 100), 2),
            'monthly_revenue': round(float(df['monthlycharges'].sum()), 2),
            'lost_revenue': round(float(
                df[df['churn']==1]['monthlycharges'].sum()), 2),
            'high_risk_customers': int(df['is_high_risk'].sum()),
            'python_scripts': scripts_count,
            'ml_models_saved': models_count,
            'literature_gaps_addressed': 4,
            'owasp_compliance': '83%',
            'best_model_auc': 0.815,
            'github': 'github.com/Gourikrishna1311/NexaIQ',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*50}")
    print("NEXAIQ API SERVER STARTING")
    print(f"{'='*50}")
    print(f"Local URL: http://localhost:{port}")
    print(f"Endpoints:")
    print(f"  http://localhost:{port}/")
    print(f"  http://localhost:{port}/api/health")
    print(f"  http://localhost:{port}/api/kpis")
    print(f"  http://localhost:{port}/api/high-risk")
    print(f"  http://localhost:{port}/api/churn-by-contract")
    print(f"  http://localhost:{port}/api/churn-by-tenure")
    print(f"  http://localhost:{port}/api/predict")
    print(f"  http://localhost:{port}/api/dashboard")
    print(f"  http://localhost:{port}/api/summary")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, debug=False)