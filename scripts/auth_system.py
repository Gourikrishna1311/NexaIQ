import bcrypt
import jwt
import os
import json
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SECRET_KEY = 'nexaiq_secret_key_2026'
TOKEN_EXPIRY_HOURS = 24


def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password, hashed_password):
    return bcrypt.checkpw(
        password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )


def generate_token(user_id, role):
    payload = {
        'user_id': user_id,
        'role': role,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    logger.info(f"Token generated for user {user_id} with role {role}")
    return token


def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        logger.info(f"Token verified for user {payload['user_id']}")
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid token")
        return None


def check_permission(role, action):
    permissions = {
        'admin': ['read', 'write', 'delete', 'export', 'manage_users'],
        'analyst': ['read', 'export'],
        'viewer': ['read']
    }
    allowed = permissions.get(role, [])
    result = action in allowed
    logger.info(f"Permission: role={role}, action={action}, allowed={result}")
    return result


class UserDatabase:

    def __init__(self):
        self.users = {}
        self.db_file = 'data/processed/users.json'
        self.load_users()

    def load_users(self):
        os.makedirs('data/processed', exist_ok=True)
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r') as f:
                self.users = json.load(f)
        else:
            self.create_default_users()

    def save_users(self):
        with open(self.db_file, 'w') as f:
            json.dump(self.users, f, indent=2)

    def create_default_users(self):
        default_users = [
            ('admin', 'admin123', 'admin'),
            ('analyst', 'analyst123', 'analyst'),
            ('viewer', 'viewer123', 'viewer')
        ]
        for username, password, role in default_users:
            self.register_user(username, password, role)
        logger.info("Default users created")

    def register_user(self, username, password, role):
        if username in self.users:
            return False
        self.users[username] = {
            'password_hash': hash_password(password),
            'role': role,
            'created_at': datetime.now().isoformat(),
            'last_login': None
        }
        self.save_users()
        logger.info(f"User {username} registered with role {role}")
        return True

    def login(self, username, password):
        if username not in self.users:
            logger.warning(f"Login failed: {username} not found")
            return None
        user = self.users[username]
        if verify_password(password, user['password_hash']):
            self.users[username]['last_login'] = datetime.now().isoformat()
            self.save_users()
            token = generate_token(username, user['role'])
            logger.info(f"Login successful: {username}")
            return token
        logger.warning(f"Login failed: wrong password for {username}")
        return None


def run_auth_demo():
    logger.info("="*60)
    logger.info("NEXAIQ AUTHENTICATION SYSTEM DEMO")
    logger.info("="*60)

    db = UserDatabase()

    print(f"\n{'='*50}")
    print("TESTING USER AUTHENTICATION")
    print(f"{'='*50}")

    test_cases = [
        ('admin', 'admin123', True),
        ('analyst', 'analyst123', True),
        ('viewer', 'viewer123', True),
        ('admin', 'wrongpassword', False),
        ('unknown', 'password', False)
    ]

    for username, password, should_succeed in test_cases:
        token = db.login(username, password)
        status = "SUCCESS" if token else "FAILED"
        expected = "SUCCESS" if should_succeed else "FAILED"
        icon = "✓" if status == expected else "✗"
        print(f"  {icon} Login {username}/{password}: {status}")

    print(f"\n{'='*50}")
    print("TESTING JWT TOKENS")
    print(f"{'='*50}")

    token = db.login('admin', 'admin123')
    payload = verify_token(token)
    print(f"  Token: {token[:50]}...")
    print(f"  Verified: user={payload['user_id']}, role={payload['role']}")

    print(f"\n{'='*50}")
    print("TESTING ROLE BASED ACCESS CONTROL")
    print(f"{'='*50}")

    permission_tests = [
        ('admin', 'delete', True),
        ('analyst', 'read', True),
        ('analyst', 'delete', False),
        ('viewer', 'export', False),
        ('viewer', 'read', True)
    ]

    for role, action, should_allow in permission_tests:
        result = check_permission(role, action)
        icon = "✓" if result == should_allow else "✗"
        print(f"  {icon} {role} can {action}: {result}")

    logger.info("="*60)
    logger.info("AUTH DEMO COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    run_auth_demo()