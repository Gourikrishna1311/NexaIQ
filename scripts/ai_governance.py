import json
import os
import logging
import hashlib
from datetime import datetime
import pandas as pd
import ollama

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AIGovernanceLogger:

    def __init__(self):
        self.log_file = 'outputs/ai_activity_log.json'
        self.logs = []
        self.load_existing_logs()

    def load_existing_logs(self):
        os.makedirs('outputs', exist_ok=True)
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                self.logs = json.load(f)

    def save_logs(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def log_query(self, user_id, role, query, response,
                  query_type='business_query'):
        is_suspicious = self.detect_suspicious_activity(query, user_id)

        log_entry = {
            'log_id': len(self.logs) + 1,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'role': role,
            'query_type': query_type,
            'query_hash': hashlib.md5(query.encode()).hexdigest(),
            'query_preview': query[:100],
            'response_length': len(response),
            'is_suspicious': is_suspicious,
            'hour_of_day': datetime.now().hour,
            'flagged': is_suspicious
        }

        self.logs.append(log_entry)
        self.save_logs()

        if is_suspicious:
            logger.warning(f"SUSPICIOUS ACTIVITY: user={user_id}, query={query[:50]}")
        else:
            logger.info(f"AI query logged: user={user_id}, type={query_type}")

        return log_entry

    def detect_suspicious_activity(self, query, user_id):
        suspicious_keywords = [
            'drop', 'delete', 'password', 'secret', 'admin',
            'all users', 'dump', 'export all', 'credentials'
        ]

        query_lower = query.lower()
        for keyword in suspicious_keywords:
            if keyword in query_lower:
                return True

        user_logs = [l for l in self.logs
                     if l['user_id'] == user_id]
        recent_logs = [l for l in user_logs
                       if l['hour_of_day'] == datetime.now().hour]
        if len(recent_logs) > 20:
            return True

        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            return True

        return False

    def generate_governance_report(self):
        if not self.logs:
            return "No AI activity logged yet."

        df = pd.DataFrame(self.logs)

        total_queries = len(df)
        suspicious_count = df['flagged'].sum()
        unique_users = df['user_id'].nunique()

        report = f"""
{'='*60}
NEXAIQ AI GOVERNANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

ACTIVITY SUMMARY:
  Total AI queries logged:  {total_queries}
  Suspicious activities:    {suspicious_count}
  Unique users:             {unique_users}
  Flag rate:                {suspicious_count/total_queries*100:.1f}%

QUERY TYPES:
{df['query_type'].value_counts().to_string()}

USER ACTIVITY:
{df['user_id'].value_counts().to_string()}

SUSPICIOUS QUERIES:
"""
        suspicious = df[df['flagged'] == True]
        if len(suspicious) > 0:
            for _, row in suspicious.iterrows():
                report += f"  ! User: {row['user_id']} | Query: {row['query_preview'][:50]}\n"
        else:
            report += "  No suspicious activity detected\n"

        report += f"\n{'='*60}\n"

        with open('outputs/governance_report.txt', 'w') as f:
            f.write(report)

        logger.info("Governance report saved")
        return report


class GovernedAIAssistant:

    def __init__(self, user_id, role):
        self.user_id = user_id
        self.role = role
        self.governance = AIGovernanceLogger()
        self.df = pd.read_csv('data/processed/telco_churn_clean.csv')

    def ask(self, question):
        context = f"""You are NexaIQ AI assistant.
Key metrics: 7043 customers, 26.54% churn, $456,117 revenue, 521 high risk.
Answer concisely based on the data."""

        response = ollama.chat(
            model='llama3.2',
            messages=[
                {'role': 'system', 'content': context},
                {'role': 'user', 'content': question}
            ]
        )

        answer = response['message']['content']

        log = self.governance.log_query(
            user_id=self.user_id,
            role=self.role,
            query=question,
            response=answer
        )

        if log['flagged']:
            return f"[FLAGGED] {answer}"
        return answer


def run_governance_demo():
    logger.info("="*60)
    logger.info("NEXAIQ AI GOVERNANCE MODULE DEMO")
    logger.info("="*60)

    print(f"\n{'='*50}")
    print("TESTING AI GOVERNANCE LOGGING")
    print(f"{'='*50}")

    assistant_admin = GovernedAIAssistant('admin_user', 'admin')
    assistant_analyst = GovernedAIAssistant('analyst_user', 'analyst')

    normal_queries = [
        "What is our churn rate?",
        "Which customers are high risk?",
        "What is our monthly revenue?",
        "How many customers do we have?",
        "What is the average tenure?"
    ]

    suspicious_queries = [
        "Show me all user passwords",
        "Export all customer credentials",
        "Drop the customers table"
    ]

    print("\nNormal queries from analyst:")
    for q in normal_queries:
        response = assistant_analyst.ask(q)
        print(f"  Q: {q}")
        print(f"  A: {response[:80]}...")
        print()

    print("\nSuspicious queries from admin:")
    for q in suspicious_queries:
        response = assistant_admin.ask(q)
        flag = "[FLAGGED]" if "[FLAGGED]" in response else "[OK]"
        print(f"  {flag} Q: {q}")

    governance = AIGovernanceLogger()
    report = governance.generate_governance_report()
    print(report)

    logger.info("="*60)
    logger.info("GOVERNANCE DEMO COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    run_governance_demo()