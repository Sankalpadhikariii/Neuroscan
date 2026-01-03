import os
import io
import base64
import sqlite3
import secrets
import logging
import random
import string
import json
from pathlib import Path
from functools import wraps
from datetime import datetime, timedelta
import stripe
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from flask import Flask, request, jsonify, session, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
from pdf_report import generate_pdf_report
from email_utilis import send_verification_email, send_welcome_email

# -----------------------------
# Configuration
# -----------------------------
UPLOAD_FOLDER = 'uploads/profile_pictures'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
DB_FILE = "neuroscan_platform.db"
MODEL_PATH = "Brain_Tumor_model.pt"

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load environment variables
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(16)
# -----------------------------
# Stripe Configuration
# -----------------------------
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY

CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    }}
)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["50 per minute"]
)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# ==============================================
# DATABASE & UTILITIES
# ==============================================

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_code(length=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def log_activity(user_type, user_id, action, details=None, hospital_id=None):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            INSERT INTO activity_logs (user_type, user_id, hospital_id, action, details)
            VALUES (?, ?, ?, ?, ?)
        """, (user_type, user_id, hospital_id, action, details))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log activity: {e}")


# ==============================================
# AUTHENTICATION DECORATORS
# ==============================================

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" in session and "user_type" in session:
            return f(*args, **kwargs)
        if "patient_id" in session and "patient_type" in session:
            return f(*args, **kwargs)
        return jsonify({"error": "Not authenticated"}), 401

    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session or session.get("user_type") != "admin":
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)

    return wrapper


def hospital_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session or session.get("user_type") != "hospital":
            return jsonify({"error": "Hospital access required"}), 403
        return f(*args, **kwargs)

    return wrapper


def patient_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "patient_id" not in session:
            return jsonify({"error": "Patient authentication required"}), 403
        return f(*args, **kwargs)

    return wrapper


# ==============================================
# SUBSCRIPTION HELPERS
# ==============================================

def get_hospital_subscription(hospital_id):
    """Get active subscription for a hospital"""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT hs.*, sp.name as plan_name, sp.display_name, sp.price_monthly,
               sp.max_scans_per_month, sp.max_users, sp.max_patients, sp.features
        FROM hospital_subscriptions hs
        JOIN subscription_plans sp ON hs.plan_id = sp.id
        WHERE hs.hospital_id = ? AND hs.status = 'active'
        ORDER BY hs.created_at DESC
        LIMIT 1
    """, (hospital_id,))
    sub = c.fetchone()
    conn.close()
    return dict(sub) if sub else None

# ==============================================
# STRIPE HELPERS
# ==============================================

def get_stripe_price_id(plan_name, billing_cycle):
    price_mapping = {
        'basic_monthly': os.getenv('STRIPE_PRICE_BASIC_MONTHLY'),
        'basic_yearly': os.getenv('STRIPE_PRICE_BASIC_YEARLY'),
        'professional_monthly': os.getenv('STRIPE_PRICE_PRO_MONTHLY'),
        'professional_yearly': os.getenv('STRIPE_PRICE_PRO_YEARLY'),
        'enterprise_monthly': os.getenv('STRIPE_PRICE_ENTERPRISE_MONTHLY'),
        'enterprise_yearly': os.getenv('STRIPE_PRICE_ENTERPRISE_YEARLY'),
    }
    return price_mapping.get(plan_name)


def get_or_create_stripe_customer(hospital_id):
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT stripe_customer_id, email, hospital_name FROM hospitals WHERE id=?", (hospital_id,))
    hospital = c.fetchone()

    if hospital["stripe_customer_id"]:
        conn.close()
        return hospital["stripe_customer_id"]

    customer = stripe.Customer.create(
        email=hospital["email"],
        name=hospital["hospital_name"],
        metadata={"hospital_id": hospital_id}
    )

    c.execute("UPDATE hospitals SET stripe_customer_id=? WHERE id=?",
              (customer.id, hospital_id))
    conn.commit()
    conn.close()

    return customer.id
# -----------------------------
# STRIPE HELPERS
# -----------------------------

def get_stripe_price_id(plan_id, billing_cycle):
    """
    Map subscription plan IDs to Stripe price IDs.
    """
    price_mapping = {
        1: { "monthly": os.getenv('STRIPE_PRICE_BASIC_MONTHLY'), "yearly": os.getenv('STRIPE_PRICE_BASIC_YEARLY') },
        2: { "monthly": os.getenv('STRIPE_PRICE_PRO_MONTHLY'), "yearly": os.getenv('STRIPE_PRICE_PRO_YEARLY') },
        3: { "monthly": os.getenv('STRIPE_PRICE_ENTERPRISE_MONTHLY'), "yearly": os.getenv('STRIPE_PRICE_ENTERPRISE_YEARLY') },
    }

    plan_prices = price_mapping.get(plan_id)
    if not plan_prices:
        return None

    return plan_prices.get(billing_cycle)

def get_current_usage(hospital_id):
    """Get current period usage for a hospital"""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM usage_tracking
        WHERE hospital_id = ? AND is_current = 1
        ORDER BY period_start DESC
        LIMIT 1
    """, (hospital_id,))
    usage = c.fetchone()
    conn.close()
    return dict(usage) if usage else None


def check_usage_limit(hospital_id, resource_type='scans'):
    """
    Check if hospital has reached usage limits
    Returns: (can_use: bool, current_usage: int, limit: int)
    """
    subscription = get_hospital_subscription(hospital_id)
    usage = get_current_usage(hospital_id)

    if not subscription or not usage:
        return False, 0, 0

    if resource_type == 'scans':
        limit = subscription['max_scans_per_month']
        current = usage['scans_used']
    elif resource_type == 'users':
        limit = subscription['max_users']
        current = usage['users_count']
    elif resource_type == 'patients':
        limit = subscription['max_patients']
        current = usage['patients_count']
    else:
        return False, 0, 0

    # -1 means unlimited
    if limit == -1:
        return True, current, limit

    can_use = current < limit
    return can_use, current, limit
@app.route("/api/stripe/config", methods=["GET"])
def stripe_config():
    return jsonify({"publishableKey": STRIPE_PUBLISHABLE_KEY})


def increment_usage(hospital_id, resource_type='scans', amount=1):
    """Increment usage counter"""
    conn = get_db()
    c = conn.cursor()

    field_map = {
        'scans': 'scans_used',
        'users': 'users_count',
        'patients': 'patients_count'
    }

    field = field_map.get(resource_type)
    if not field:
        conn.close()
        return False

    c.execute(f"""
        UPDATE usage_tracking
        SET {field} = {field} + ?, updated_at = CURRENT_TIMESTAMP
        WHERE hospital_id = ? AND is_current = 1
    """, (amount, hospital_id))

    conn.commit()
    conn.close()
    return True


def has_feature(hospital_id, feature_key):
    """Check if hospital's plan includes a feature"""
    subscription = get_hospital_subscription(hospital_id)
    if not subscription:
        return False
    try:
        features = json.loads(subscription['features'])
        return feature_key in features
    except:
        return False


def requires_feature(feature_key):
    """Decorator to check feature access"""

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            hospital_id = session.get("hospital_id")
            if not hospital_id or not has_feature(hospital_id, feature_key):
                return jsonify({
                    "error": "This feature is not available in your current plan",
                    "feature": feature_key,
                    "upgrade_required": True
                }), 403
            return f(*args, **kwargs)

        return wrapper

    return decorator


# ==============================================
# CNN MODEL SETUP
# ==============================================

class CNN_TUMOR(nn.Module):
    def __init__(self, params=None):
        super(CNN_TUMOR, self).__init__()
        if params is None:
            params = {
                "shape_in": (3, 224, 224),
                "initial_filters": 8,
                "num_fc1": 100,
                "num_classes": 4,
                "dropout_rate": 0.25
            }
        Cin, Hin, Win = params["shape_in"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.num_flatten = 7 * 7 * 8 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = self.adaptive_pool(X)
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, p=self.dropout_rate, training=self.training)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)


class_names = ["glioma", "meningioma", "notumor", "pituitary"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    logger.info("Loading model...")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise


# ==============================================
# PUBLIC SUBSCRIPTION ROUTES
# ==============================================

@app.route("/api/subscription/plans", methods=["GET"])
def get_subscription_plans():
    """Get all available subscription plans"""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM subscription_plans
        WHERE is_active = 1
        ORDER BY price_monthly ASC
    """)
    plans = [dict(row) for row in c.fetchall()]

    # Parse features JSON
    for plan in plans:
        try:
            plan['features'] = json.loads(plan['features'])
        except:
            plan['features'] = []

    conn.close()
    return jsonify({"plans": plans})


@app.route("/api/subscription/features", methods=["GET"])
def get_all_features():
    """Get all available features"""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM feature_flags WHERE is_active = 1")
    features = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify({"features": features})


# ==============================================
# ADMIN ROUTES
# ==============================================

@app.route("/admin/login", methods=["POST"])
def admin_login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM admins WHERE username=?", (username,))
    admin = c.fetchone()
    conn.close()

    if not admin or not check_password_hash(admin["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    session["user_id"] = admin["id"]
    session["user_type"] = "admin"
    session["username"] = admin["username"]

    log_activity("admin", admin["id"], "login")

    return jsonify({
        "user": {
            "id": admin["id"],
            "username": admin["username"],
            "email": admin["email"],
            "full_name": admin["full_name"],
            "type": "admin"
        }
    })


@app.route("/admin/dashboard", methods=["GET"])
@admin_required
def admin_dashboard():
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) as count FROM hospitals WHERE status='active'")
    total_hospitals = c.fetchone()["count"]
    c.execute("SELECT COUNT(*) as count FROM hospital_users WHERE status='active'")
    total_doctors = c.fetchone()["count"]
    c.execute("SELECT COUNT(*) as count FROM patients")
    total_patients = c.fetchone()["count"]
    c.execute("SELECT COUNT(*) as count FROM mri_scans")
    total_scans = c.fetchone()["count"]
    c.execute("SELECT COUNT(*) as count FROM bug_reports WHERE status!='resolved'")
    open_bugs = c.fetchone()["count"]

    c.execute("""
        SELECT id, hospital_name, hospital_code, email, city, created_at
        FROM hospitals ORDER BY created_at DESC LIMIT 5
    """)
    recent_hospitals = [dict(row) for row in c.fetchall()]

    c.execute("""
        SELECT h.hospital_name, COUNT(s.id) as scan_count
        FROM hospitals h
        LEFT JOIN mri_scans s ON h.id = s.hospital_id
        GROUP BY h.id ORDER BY scan_count DESC LIMIT 5
    """)
    top_hospitals = [dict(row) for row in c.fetchall()]

    conn.close()

    return jsonify({
        "stats": {
            "total_hospitals": total_hospitals,
            "total_doctors": total_doctors,
            "total_patients": total_patients,
            "total_scans": total_scans,
            "open_bugs": open_bugs
        },
        "recent_hospitals": recent_hospitals,
        "top_hospitals": top_hospitals
    })


@app.route("/admin/hospitals", methods=["GET", "POST"])
@admin_required
def admin_hospitals():
    conn = get_db()
    c = conn.cursor()

    if request.method == "GET":
        c.execute("""
            SELECT h.*, 
                   COUNT(DISTINCT hu.id) as user_count,
                   COUNT(DISTINCT s.id) as scan_count
            FROM hospitals h
            LEFT JOIN hospital_users hu ON h.id = hu.hospital_id
            LEFT JOIN mri_scans s ON h.id = s.hospital_id
            GROUP BY h.id ORDER BY h.created_at DESC
        """)
        hospitals = [dict(row) for row in c.fetchall()]
        conn.close()
        return jsonify({"hospitals": hospitals})

    # POST - Create new hospital
    data = request.json
    hospital_code = generate_code()

    c.execute("""
        INSERT INTO hospitals (
            hospital_name, hospital_code, contact_person, email,
            phone, address, city, country, created_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("hospital_name"), hospital_code,
        data.get("contact_person"), data.get("email"),
        data.get("phone"), data.get("address"),
        data.get("city"), data.get("country"),
        session["user_id"]
    ))

    hospital_id = c.lastrowid
    conn.commit()
    conn.close()

    log_activity("admin", session["user_id"], "create_hospital", f"Created {data.get('hospital_name')}")

    return jsonify({
        "message": "Hospital created",
        "hospital_id": hospital_id,
        "hospital_code": hospital_code
    }), 201


@app.route("/admin/hospitals/<int:hospital_id>", methods=["GET", "PUT", "DELETE"])
@admin_required
def admin_hospital_detail(hospital_id):
    conn = get_db()
    c = conn.cursor()

    if request.method == "GET":
        c.execute("SELECT * FROM hospitals WHERE id=?", (hospital_id,))
        hospital = c.fetchone()
        if not hospital:
            conn.close()
            return jsonify({"error": "Hospital not found"}), 404
        c.execute("SELECT * FROM hospital_users WHERE hospital_id=?", (hospital_id,))
        users = [dict(row) for row in c.fetchall()]
        conn.close()
        return jsonify({"hospital": dict(hospital), "users": users})

    elif request.method == "PUT":
        data = request.json
        c.execute("""
            UPDATE hospitals 
            SET hospital_name=?, contact_person=?, email=?, phone=?, 
                address=?, city=?, country=?, status=?
            WHERE id=?
        """, (
            data.get("hospital_name"), data.get("contact_person"),
            data.get("email"), data.get("phone"),
            data.get("address"), data.get("city"),
            data.get("country"), data.get("status", "active"),
            hospital_id
        ))
        conn.commit()
        conn.close()
        return jsonify({"message": "Hospital updated"})

    else:  # DELETE
        c.execute("DELETE FROM hospitals WHERE id=?", (hospital_id,))
        conn.commit()
        conn.close()
        log_activity("admin", session["user_id"], "delete_hospital", f"Deleted hospital {hospital_id}")
        return jsonify({"message": "Hospital deleted"})


@app.route("/admin/subscriptions", methods=["GET"])
@admin_required
def admin_get_all_subscriptions():
    """Get all subscriptions with analytics"""
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        SELECT hs.*, h.hospital_name, sp.display_name as plan_name,
               sp.price_monthly, ut.scans_used, ut.scans_limit
        FROM hospital_subscriptions hs
        JOIN hospitals h ON hs.hospital_id = h.id
        JOIN subscription_plans sp ON hs.plan_id = sp.id
        LEFT JOIN usage_tracking ut ON hs.id = ut.subscription_id AND ut.is_current = 1
        ORDER BY hs.created_at DESC
    """)
    subscriptions = [dict(row) for row in c.fetchall()]

    c.execute("""
        SELECT 
            SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) as total_revenue,
            COUNT(DISTINCT hospital_id) as paying_hospitals,
            SUM(CASE WHEN status = 'pending' THEN amount ELSE 0 END) as pending_revenue
        FROM payment_transactions
    """)
    stats = dict(c.fetchone())
    conn.close()

    return jsonify({"subscriptions": subscriptions, "stats": stats})
@app.route("/api/stripe/create-checkout-session", methods=["POST"])
@hospital_required
def create_checkout_session():
    data = request.json
    plan_id = data.get("plan_id")
    billing_cycle = data.get("billing_cycle", "monthly")

    # 🔹 Debug log
    print("DEBUG: plan_id =", plan_id, "billing_cycle =", billing_cycle)

    hospital_id = session["hospital_id"]

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM subscription_plans WHERE id=?", (plan_id,))
    plan = c.fetchone()
    conn.close()

    if not plan:
        return jsonify({"error": "Plan not found"}), 404

    price_id = get_stripe_price_id(plan["id"], billing_cycle)
    if not price_id:
        return jsonify({"error": "Stripe price not configured"}), 400

    customer_id = get_or_create_stripe_customer(hospital_id)

    checkout_session = stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=f"{os.getenv('FRONTEND_URL')}/subscription-success",
        cancel_url=f"{os.getenv('FRONTEND_URL')}/subscription-cancelled",
        metadata={
            "hospital_id": hospital_id,
            "plan_id": plan_id,
            "billing_cycle": billing_cycle
        }
    )

    return jsonify({"url": checkout_session.url})


@app.route("/admin/revenue", methods=["GET"])
@admin_required
def admin_revenue_analytics():
    """Get revenue analytics"""
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        SELECT SUM(sp.price_monthly) as mrr
        FROM hospital_subscriptions hs
        JOIN subscription_plans sp ON hs.plan_id = sp.id
        WHERE hs.status = 'active' AND hs.billing_cycle = 'monthly'
    """)
    mrr = c.fetchone()['mrr'] or 0

    c.execute("""
        SELECT SUM(sp.price_yearly) as arr
        FROM hospital_subscriptions hs
        JOIN subscription_plans sp ON hs.plan_id = sp.id
        WHERE hs.status = 'active' AND hs.billing_cycle = 'yearly'
    """)
    arr = c.fetchone()['arr'] or 0

    c.execute("""
        SELECT sp.display_name, COUNT(*) as customers, 
               SUM(sp.price_monthly) as monthly_revenue
        FROM hospital_subscriptions hs
        JOIN subscription_plans sp ON hs.plan_id = sp.id
        WHERE hs.status = 'active'
        GROUP BY sp.id
    """)
    by_plan = [dict(row) for row in c.fetchall()]

    c.execute("""
        SELECT COUNT(*) as churned
        FROM hospital_subscriptions
        WHERE status = 'cancelled' 
        AND cancelled_at >= date('now', '-30 days')
    """)
    churned = c.fetchone()['churned']

    c.execute("SELECT COUNT(*) as total FROM hospital_subscriptions WHERE status = 'active'")
    active = c.fetchone()['total']

    churn_rate = (churned / active * 100) if active > 0 else 0

    conn.close()

    return jsonify({
        "mrr": round(mrr, 2),
        "arr": round(arr + (mrr * 12), 2),
        "by_plan": by_plan,
        "churn_rate": round(churn_rate, 2),
        "active_subscriptions": active
    })


# ==============================================
# HOSPITAL USER ROUTES
# ==============================================

@app.route("/hospital/login", methods=["POST"])
def hospital_login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT hu.*, h.hospital_name, h.hospital_code
        FROM hospital_users hu
        JOIN hospitals h ON hu.hospital_id = h.id
        WHERE hu.username=? AND hu.status='active' AND h.status='active'
    """, (username,))
    user = c.fetchone()
    conn.close()

    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    session["user_id"] = user["id"]
    session["user_type"] = "hospital"
    session["hospital_id"] = user["hospital_id"]
    session["username"] = user["username"]

    log_activity("hospital", user["id"], "login", hospital_id=user["hospital_id"])

    return jsonify({
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "full_name": user["full_name"],
            "hospital_id": user["hospital_id"],
            "hospital_name": user["hospital_name"],
            "hospital_code": user["hospital_code"],
            "role": user["role"],
            "type": "hospital"
        }
    })


@app.route("/hospital/dashboard", methods=["GET"])
@hospital_required
def hospital_dashboard():
    conn = get_db()
    c = conn.cursor()
    hospital_id = session["hospital_id"]

    c.execute("SELECT COUNT(*) as count FROM patients WHERE hospital_id=?", (hospital_id,))
    total_patients = c.fetchone()["count"]
    c.execute("SELECT COUNT(*) as count FROM mri_scans WHERE hospital_id=?", (hospital_id,))
    total_scans = c.fetchone()["count"]
    c.execute("SELECT COUNT(*) as count FROM mri_scans WHERE hospital_id=? AND is_tumor=1", (hospital_id,))
    tumor_detections = c.fetchone()["count"]
    c.execute("SELECT COUNT(*) as count FROM chat_conversations WHERE hospital_id=? AND status='active'",
              (hospital_id,))
    active_chats = c.fetchone()["count"]
    c.execute("""
        SELECT COUNT(*) as count FROM mri_scans 
        WHERE hospital_id=? AND strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')
    """, (hospital_id,))
    scans_this_month = c.fetchone()["count"]

    c.execute("""
        SELECT s.*, p.full_name as patient_name, p.patient_code
        FROM mri_scans s
        JOIN patients p ON s.patient_id = p.id
        WHERE s.hospital_id=?
        ORDER BY s.created_at DESC LIMIT 10
    """, (hospital_id,))
    recent_scans = [dict(row) for row in c.fetchall()]

    conn.close()

    return jsonify({
        "stats": {
            "total_patients": total_patients,
            "total_scans": total_scans,
            "tumor_detected": tumor_detections,
            "scans_this_month": scans_this_month,
            "active_chats": active_chats
        },
        "recent_scans": recent_scans
    })


@app.route("/hospital/subscription", methods=["GET"])
@hospital_required
def get_hospital_subscription_info():
    """Get hospital's current subscription and usage"""
    hospital_id = session["hospital_id"]

    subscription = get_hospital_subscription(hospital_id)
    usage = get_current_usage(hospital_id)

    if not subscription:
        return jsonify({"error": "No active subscription"}), 404

    # Parse features
    try:
        subscription['features'] = json.loads(subscription['features'])
    except:
        subscription['features'] = []

    # Calculate days until renewal
    if subscription['current_period_end']:
        end_date = datetime.strptime(subscription['current_period_end'], '%Y-%m-%d')
        days_remaining = (end_date - datetime.now()).days
    else:
        days_remaining = 0

    return jsonify({
        "subscription": subscription,
        "usage": usage,
        "days_remaining": days_remaining,
        "is_trial": subscription.get('is_trial', 0) == 1
    })


@app.route("/hospital/subscription/upgrade", methods=["POST"])
@hospital_required
def upgrade_subscription():
    """Upgrade to a new plan"""
    data = request.json
    new_plan_id = data.get("plan_id")
    billing_cycle = data.get("billing_cycle", "monthly")

    if not new_plan_id:
        return jsonify({"error": "Plan ID required"}), 400

    hospital_id = session["hospital_id"]

    conn = get_db()
    c = conn.cursor()

    # Get new plan
    c.execute("SELECT * FROM subscription_plans WHERE id = ?", (new_plan_id,))
    new_plan = c.fetchone()
    if not new_plan:
        conn.close()
        return jsonify({"error": "Plan not found"}), 404
    new_plan = dict(new_plan)

    # Get current subscription
    c.execute("""
        SELECT * FROM hospital_subscriptions
        WHERE hospital_id = ? AND status = 'active'
    """, (hospital_id,))
    current_sub = c.fetchone()

    if current_sub:
        current_sub = dict(current_sub)
        old_plan_id = current_sub['plan_id']
        c.execute("""
            UPDATE hospital_subscriptions
            SET status = 'cancelled', cancelled_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (current_sub['id'],))
    else:
        old_plan_id = None

    # Create new subscription
    start_date = datetime.now()
    if billing_cycle == 'yearly':
        end_date = start_date + timedelta(days=365)
        amount = new_plan['price_yearly']
    else:
        end_date = start_date + timedelta(days=30)
        amount = new_plan['price_monthly']

    c.execute("""
        INSERT INTO hospital_subscriptions
        (hospital_id, plan_id, status, billing_cycle,
         current_period_start, current_period_end, next_billing_date)
        VALUES (?, ?, 'active', ?, ?, ?, ?)
    """, (hospital_id, new_plan_id, billing_cycle,
          start_date.date(), end_date.date(), end_date.date()))

    new_sub_id = c.lastrowid

    # Create usage tracking
    c.execute("UPDATE usage_tracking SET is_current = 0 WHERE hospital_id = ?", (hospital_id,))
    c.execute("""
        INSERT INTO usage_tracking
        (hospital_id, subscription_id, period_start, period_end,
         scans_limit, users_limit, patients_limit, is_current)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1)
    """, (hospital_id, new_sub_id, start_date.date(), end_date.date(),
          new_plan['max_scans_per_month'], new_plan['max_users'], new_plan['max_patients']))

    # Log change
    c.execute("""
        INSERT INTO subscription_history
        (hospital_id, subscription_id, action, old_plan_id, new_plan_id, changed_by)
        VALUES (?, ?, 'upgrade', ?, ?, ?)
    """, (hospital_id, new_sub_id, old_plan_id, new_plan_id, session["user_id"]))

    # Create payment transaction
    invoice_number = f"INV-{hospital_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    c.execute("""
        INSERT INTO payment_transactions
        (hospital_id, subscription_id, amount, status, invoice_number, description)
        VALUES (?, ?, ?, 'pending', ?, ?)
    """, (hospital_id, new_sub_id, amount, invoice_number,
          f"Upgrade to {new_plan['display_name']} - {billing_cycle}"))

    conn.commit()
    conn.close()

    log_activity("hospital", session["user_id"], "subscription_upgrade",
                 f"Upgraded to {new_plan['display_name']}", hospital_id)

    return jsonify({
        "message": "Subscription upgraded successfully",
        "subscription_id": new_sub_id,
        "invoice_number": invoice_number,
        "amount": amount,
        "billing_cycle": billing_cycle
    })


@app.route("/hospital/subscription/cancel", methods=["POST"])
@hospital_required
def cancel_subscription():
    """Cancel subscription"""
    hospital_id = session["hospital_id"]
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        UPDATE hospital_subscriptions
        SET auto_renew = 0, updated_at = CURRENT_TIMESTAMP
        WHERE hospital_id = ? AND status = 'active'
    """, (hospital_id,))

    c.execute("""
        SELECT id, plan_id FROM hospital_subscriptions
        WHERE hospital_id = ? AND status = 'active'
    """, (hospital_id,))
    sub = c.fetchone()
    if sub:
        c.execute("""
            INSERT INTO subscription_history
            (hospital_id, subscription_id, action, old_plan_id, changed_by)
            VALUES (?, ?, 'cancel', ?, ?)
        """, (hospital_id, sub['id'], sub['plan_id'], session["user_id"]))

    conn.commit()
    conn.close()

    return jsonify({"message": "Subscription will not auto-renew at end of period"})


@app.route("/hospital/patients", methods=["GET", "POST"])
@hospital_required
def hospital_patients():
    conn = get_db()
    c = conn.cursor()
    hospital_id = session["hospital_id"]

    if request.method == "GET":
        c.execute("""
            SELECT p.*, hu.full_name as doctor_name, COUNT(s.id) as scan_count
            FROM patients p
            LEFT JOIN hospital_users hu ON p.assigned_doctor_id = hu.id
            LEFT JOIN mri_scans s ON p.id = s.patient_id
            WHERE p.hospital_id=?
            GROUP BY p.id ORDER BY p.created_at DESC
        """, (hospital_id,))
        patients = [dict(row) for row in c.fetchall()]
        conn.close()
        return jsonify({"patients": patients})

    # POST - Create patient
    data = request.json
    patient_code = generate_code(6)
    access_code = generate_code(8)

    c.execute("""
        INSERT INTO patients (
            hospital_id, patient_code, full_name, email, phone,
            date_of_birth, gender, address, emergency_contact, 
            emergency_phone, assigned_doctor_id, created_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        hospital_id, patient_code,
        data.get("full_name"), data.get("email"), data.get("phone"),
        data.get("date_of_birth"), data.get("gender"), data.get("address"),
        data.get("emergency_contact"), data.get("emergency_phone"),
        session["user_id"], session["user_id"]
    ))
    patient_id = c.lastrowid

    expires_at = datetime.now() + timedelta(days=30)
    c.execute("""
        INSERT INTO patient_access_codes (patient_id, access_code, expires_at)
        VALUES (?, ?, ?)
    """, (patient_id, access_code, expires_at))

    c.execute("SELECT hospital_name, hospital_code FROM hospitals WHERE id=?", (hospital_id,))
    hospital = c.fetchone()

    c.execute("SELECT p.*, 0 as scan_count FROM patients p WHERE p.id=?", (patient_id,))
    patient = dict(c.fetchone())

    conn.commit()
    conn.close()

    # Send welcome email
    email_sent = False
    try:
        email_sent = send_welcome_email(
            to_email=data.get("email"),
            patient_name=data.get("full_name"),
            patient_code=patient_code,
            access_code=access_code,
            hospital_name=hospital["hospital_name"],
            hospital_code=hospital["hospital_code"]
        )
        if email_sent:
            logger.info(f"✅ Welcome email sent to {data.get('email')}")
        else:
            logger.warning(f"⚠️ Email not configured")
    except Exception as e:
        logger.error(f"❌ Email error: {e}")
        email_sent = False

    log_activity("hospital", session["user_id"], "create_patient", hospital_id=hospital_id)

    return jsonify({
        "message": "Patient created successfully",
        "patient": patient,
        "patient_id": patient_id,
        "patient_code": patient_code,
        "access_code": access_code,
        "email_sent": email_sent
    }), 201


@app.route("/hospital/history", methods=["GET"])
@hospital_required
def hospital_history():
    """Get scan history"""
    conn = get_db()
    c = conn.cursor()
    hospital_id = session["hospital_id"]

    c.execute("""
        SELECT s.*, p.full_name as patient_name, p.patient_code,
               hu.full_name as uploaded_by_name
        FROM mri_scans s
        JOIN patients p ON s.patient_id = p.id
        LEFT JOIN hospital_users hu ON s.uploaded_by = hu.id
        WHERE s.hospital_id=?
        ORDER BY s.created_at DESC
    """, (hospital_id,))

    scans = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify({"scans": scans})


# ==============================================
# MRI PREDICTION (WITH USAGE LIMITS)
# ==============================================

@app.route("/hospital/predict", methods=["POST"])
@hospital_required
def predict():
    """Predict with usage limit enforcement"""
    hospital_id = session["hospital_id"]

    # ✅ CHECK USAGE LIMIT BEFORE PROCESSING
    can_scan, current, limit = check_usage_limit(hospital_id, 'scans')

    if not can_scan:
        return jsonify({
            "error": "Scan limit reached",
            "current_usage": current,
            "limit": limit,
            "upgrade_required": True
        }), 403

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image"}), 400

        patient_id = request.form.get("patient_id")
        if not patient_id:
            return jsonify({"error": "Patient ID required"}), 400

        image_bytes = request.files["image"].read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.exp(output)[0]
            conf, pred = torch.max(probs, 0)

        pred_idx = int(pred.item())
        conf_val = float(conf.item())
        prediction_label = class_names[pred_idx]
        is_tumor = prediction_label != "notumor"

        probabilities = {
            class_names[i]: round(float(probs[i].item()) * 100, 2)
            for i in range(len(class_names))
        }

        conn = get_db()
        c = conn.cursor()
        c.execute("""
            INSERT INTO mri_scans (
                hospital_id, patient_id, uploaded_by, scan_image,
                prediction, confidence, is_tumor, probabilities,
                notes, scan_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            hospital_id, patient_id, session["user_id"],
            base64.b64encode(image_bytes).decode(),
            prediction_label, conf_val, is_tumor, str(probabilities),
            request.form.get("notes", ""),
            request.form.get("scan_date", datetime.now().strftime("%Y-%m-%d"))
        ))
        scan_id = c.lastrowid
        conn.commit()
        conn.close()

        # ✅ INCREMENT USAGE AFTER SUCCESSFUL SCAN
        increment_usage(hospital_id, 'scans', 1)

        log_activity("hospital", session["user_id"], "prediction", hospital_id=hospital_id)

        return jsonify({
            "scan_id": scan_id,
            "prediction": prediction_label,
            "confidence": round(conf_val * 100, 2),
            "is_tumor": is_tumor,
            "probabilities": probabilities,
            "usage": {
                "scans_used": current + 1,
                "scans_limit": limit
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


# ==============================================
# REPORT GENERATION
# ==============================================

@app.route("/generate-report/<int:scan_id>", methods=["GET"])
@login_required
def generate_report(scan_id):
    """Generate PDF report"""
    try:
        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT s.*, p.*, h.hospital_name, h.hospital_code, hu.full_name as doctor_name
            FROM mri_scans s
            JOIN patients p ON s.patient_id = p.id
            JOIN hospitals h ON s.hospital_id = h.id
            LEFT JOIN hospital_users hu ON s.uploaded_by = hu.id
            WHERE s.id=?
        """, (scan_id,))

        row = c.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Scan not found"}), 404

        # Check authorization
        user_type = session.get("user_type")

        if user_type == "hospital":
            if row["hospital_id"] != session.get("hospital_id"):
                return jsonify({"error": "Unauthorized"}), 403
        elif user_type == "patient":
            if row["patient_id"] != session.get("patient_id"):
                return jsonify({"error": "Unauthorized"}), 403
        elif user_type != "admin":
            return jsonify({"error": "Unauthorized"}), 403

        # Prepare data
        scan_data = {
            "id": row["id"],
            "prediction": row["prediction"],
            "confidence": row["confidence"] * 100 if row["confidence"] <= 1 else row["confidence"],
            "is_tumor": row["is_tumor"],
            "probabilities": row["probabilities"],
            "scan_date": row["scan_date"],
            "scan_image": row["scan_image"],
            "notes": row["notes"]
        }

        patient_data = {
            "full_name": row["full_name"],
            "patient_code": row["patient_code"],
            "email": row["email"],
            "phone": row["phone"],
            "date_of_birth": row["date_of_birth"],
            "gender": row["gender"]
        }

        hospital_data = {
            "hospital_name": row["hospital_name"],
            "hospital_code": row["hospital_code"],
            "doctor_name": row["doctor_name"] or "N/A"
        }

        pdf_buffer = generate_pdf_report(scan_data, patient_data, hospital_data)

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'NeuroScan_Report_{scan_id}.pdf'
        )

    except Exception as e:
        logger.error(f"Report error: {e}")
        return jsonify({"error": str(e)}), 500


# ==============================================
# PATIENT ROUTES
# ==============================================

@app.route("/patient/verify", methods=["POST"])
def patient_verify():
    """Step 2: Verify patient"""
    data = request.json
    hospital_code = data.get("hospital_code")
    patient_code = data.get("patient_code")
    access_code = data.get("access_code")

    conn = get_db()
    c = conn.cursor()

    c.execute("""
        SELECT p.*, h.hospital_name, pac.id as access_id
        FROM patients p
        JOIN hospitals h ON p.hospital_id = h.id
        JOIN patient_access_codes pac ON p.id = pac.patient_id
        WHERE h.hospital_code=? AND p.patient_code=? AND pac.access_code=?
            AND pac.expires_at > datetime('now')
    """, (hospital_code, patient_code, access_code))

    patient = c.fetchone()

    if not patient:
        conn.close()
        return jsonify({"error": "Invalid credentials or expired"}), 401

    # Generate verification code
    verification_code = ''.join(random.choices(string.digits, k=6))

    c.execute("UPDATE patient_access_codes SET verification_code=? WHERE id=?",
              (verification_code, patient["access_id"]))
    conn.commit()
    conn.close()

    # Send email
    try:
        send_verification_email(
            to_email=patient['email'],
            verification_code=verification_code,
            patient_name=patient['full_name'],
            hospital_name=patient['hospital_name']
        )
    except Exception as e:
        logger.error(f"Email error: {e}")

    logger.info(f"Verification code: {verification_code}")

    return jsonify({
        "message": "Verification code sent to email",
        "email_hint": patient["email"][:3] + "***" + patient["email"][-10:]
    })


@app.route("/patient/login", methods=["POST"])
def patient_login():
    """Step 3: Login with verification code"""
    data = request.json
    patient_code = data.get("patient_code")
    verification_code = data.get("verification_code")

    conn = get_db()
    c = conn.cursor()

    c.execute("""
        SELECT p.*, h.hospital_name, h.id as hospital_id, pac.id as access_id
        FROM patients p
        JOIN hospitals h ON p.hospital_id = h.id
        JOIN patient_access_codes pac ON p.id = pac.patient_id
        WHERE p.patient_code=? AND pac.verification_code=?
    """, (patient_code, verification_code))

    patient = c.fetchone()

    if not patient:
        conn.close()
        return jsonify({"error": "Invalid verification code"}), 401

    c.execute("""
        UPDATE patient_access_codes 
        SET is_verified=1, verified_at=datetime('now')
        WHERE id=?
    """, (patient["access_id"],))
    conn.commit()
    conn.close()

    session["patient_id"] = patient["id"]
    session["patient_type"] = "patient"
    session["user_id"] = patient["id"]
    session["user_type"] = "patient"
    session["hospital_id"] = patient["hospital_id"]

    return jsonify({
        "patient": {
            "id": patient["id"],
            "full_name": patient["full_name"],
            "patient_code": patient["patient_code"],
            "hospital_name": patient["hospital_name"],
            "type": "patient"
        }
    })


@app.route("/patient/scans", methods=["GET"])
@patient_required
def get_patient_scans():
    """Get patient's scans"""
    patient_id = session.get("patient_id")

    conn = get_db()
    c = conn.cursor()

    c.execute("""
        SELECT s.*, h.hospital_name, hu.full_name as doctor_name
        FROM mri_scans s
        JOIN hospitals h ON s.hospital_id = h.id
        LEFT JOIN hospital_users hu ON s.uploaded_by = hu.id
        WHERE s.patient_id=?
        ORDER BY s.created_at DESC
    """, (patient_id,))

    scans = [dict(row) for row in c.fetchall()]
    conn.close()

    return jsonify({"scans": scans})


# ==============================================
# PROFILE PICTURE ROUTES
# ==============================================

@app.route("/patient/profile-picture", methods=["POST"])
@patient_required
def upload_profile_picture():
    """Upload profile picture"""
    try:
        if 'profile_picture' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['profile_picture']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size > MAX_FILE_SIZE:
            return jsonify({"error": "File too large (max 5MB)"}), 400

        patient_id = session.get("patient_id")

        image_data = file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        file_ext = file.filename.rsplit('.', 1)[1].lower()
        mime_type = f"image/{file_ext}" if file_ext != 'jpg' else "image/jpeg"

        conn = get_db()
        c = conn.cursor()

        c.execute("""
            UPDATE patients 
            SET profile_picture=?, profile_picture_mime=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (image_base64, mime_type, patient_id))

        conn.commit()
        conn.close()

        return jsonify({
            "message": "Profile picture updated",
            "profile_picture": f"data:{mime_type};base64,{image_base64}"
        })

    except Exception as e:
        logger.error(f"Profile picture error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/patient/profile-picture", methods=["DELETE"])
@patient_required
def delete_profile_picture():
    """Delete profile picture"""
    patient_id = session.get("patient_id")

    conn = get_db()
    c = conn.cursor()
    c.execute("""
        UPDATE patients 
        SET profile_picture=NULL, profile_picture_mime=NULL
        WHERE id=?
    """, (patient_id,))
    conn.commit()
    conn.close()

    return jsonify({"message": "Profile picture deleted"})


# ==============================================
# GENERAL ROUTES
# ==============================================

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})


@app.route("/me", methods=["GET"])
def me():
    """Get current user info"""
    if "user_id" not in session and "patient_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_type = session.get("user_type")

    if user_type == "patient":
        patient_id = session.get("patient_id")

        conn = get_db()
        c = conn.cursor()
        c.execute("""
            SELECT p.*, h.hospital_name, h.hospital_code
            FROM patients p
            JOIN hospitals h ON p.hospital_id = h.id
            WHERE p.id=?
        """, (patient_id,))
        patient = c.fetchone()
        conn.close()

        if not patient:
            return jsonify({"error": "Patient not found"}), 404

        profile_picture_url = None
        if patient["profile_picture"] and patient["profile_picture_mime"]:
            profile_picture_url = f"data:{patient['profile_picture_mime']};base64,{patient['profile_picture']}"

        return jsonify({
            "user": {
                "id": patient["id"],
                "type": "patient",
                "full_name": patient["full_name"],
                "patient_code": patient["patient_code"],
                "email": patient["email"],
                "phone": patient["phone"],
                "date_of_birth": patient["date_of_birth"],
                "gender": patient["gender"],
                "hospital_name": patient["hospital_name"],
                "hospital_code": patient["hospital_code"],
                "profile_picture": profile_picture_url
            }
        })

    return jsonify({
        "user": {
            "id": session.get("user_id"),
            "type": session.get("user_type"),
            "username": session.get("username"),
            "hospital_id": session.get("hospital_id")
        }
    })
@app.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig = request.headers.get("Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig, STRIPE_WEBHOOK_SECRET
        )
    except Exception:
        return jsonify({"error": "Invalid webhook"}), 400

    if event["type"] == "checkout.session.completed":
        session_obj = event["data"]["object"]
        if event["type"] == "checkout.session.completed":
            session_obj = event["data"]["object"]

            hospital_id = int(session_obj["metadata"]["hospital_id"])
            plan_id = int(session_obj["metadata"]["plan_id"])
            billing_cycle = session_obj["metadata"]["billing_cycle"]

            stripe_subscription_id = session_obj["subscription"]

            # Fetch subscription from Stripe
            stripe_sub = stripe.Subscription.retrieve(stripe_subscription_id)

            period_start = datetime.fromtimestamp(
                stripe_sub["current_period_start"]
            ).date()

            period_end = datetime.fromtimestamp(
                stripe_sub["current_period_end"]
            ).date()

            conn = get_db()
            c = conn.cursor()

            # Expire old active subscriptions
            c.execute("""
                UPDATE hospital_subscriptions
                SET status='expired'
                WHERE hospital_id=? AND status='active'
            """, (hospital_id,))

            # Create new active subscription
            c.execute("""
                INSERT INTO hospital_subscriptions
                (
                    hospital_id,
                    plan_id,
                    status,
                    billing_cycle,
                    current_period_start,
                    current_period_end,
                    stripe_subscription_id,
                    auto_renew
                )
                VALUES (?, ?, 'active', ?, ?, ?, ?, 1)
            """, (
                hospital_id,
                plan_id,
                billing_cycle,
                period_start,
                period_end,
                stripe_subscription_id
            ))

            subscription_id = c.lastrowid

            # Create usage tracking for this period
            c.execute("""
                INSERT INTO usage_tracking
                (
                    hospital_id,
                    subscription_id,
                    period_start,
                    period_end,
                    scans_used,
                    scans_limit,
                    users_count,
                    users_limit,
                    patients_count,
                    patients_limit,
                    is_current
                )
                SELECT
                    ?, ?, ?, ?, 0,
                    max_scans_per_month,
                    0, max_users,
                    0, max_patients,
                    1
                FROM subscription_plans
                WHERE id=?
            """, (
                hospital_id,
                subscription_id,
                period_start,
                period_end,
                plan_id
            ))

            conn.commit()
            conn.close()

    return jsonify({"status": "ok"})





@app.route("/api/chatbot", methods=["POST"])
@login_required
def chatbot():
    """AI Chatbot endpoint using Ollama"""
    try:
        data = request.json
        message = data.get('message', '')

        # Get chatbot response from Ollama
        response = get_ollama_response(message)

        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({'error': str(e)}), 500


def get_ollama_response(user_message):
    """Get response from local Ollama instance"""
    try:
        # System prompt for NeuroScan context
        system_prompt = """You are NeuroScan AI Assistant, a helpful medical AI specializing in brain tumor detection. 
        You help users understand:
        - Brain tumor types (Glioma, Meningioma, Pituitary)
        - MRI scan interpretation
        - How to use the NeuroScan platform
        - General brain health information

        Keep responses concise, accurate, and professional. Always remind users to consult medical professionals for diagnosis."""

        # Call Ollama API
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2:3b',
                'prompt': f"{system_prompt}\n\nUser: {user_message}\nAssistant:",
                'stream': False
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result['response']
        else:
            logger.error(f"Ollama error: {response.status_code}")
            return "I'm having trouble connecting. Please try again."

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Is it running?")
        return "AI service is currently unavailable. Please make sure Ollama is running."
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return "Sorry, I encountered an error. Please try again."
# ==============================================
# RUN SERVER
# ==============================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)