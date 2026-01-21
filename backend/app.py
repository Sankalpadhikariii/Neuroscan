import os
import io
import base64
import sqlite3
import secrets
import logging
import random
import string
from typing import Optional, List, Dict
from pathlib import Path

<<<<<<< HEAD
import matplotlib
from flask_socketio import SocketIO, emit, join_room, leave_room, send
import socketio
=======
>>>>>>> 60e9c6c55f4a0825e4e279be9d2e0cd535cc68f5
from werkzeug.utils import secure_filename
import mimetypes
from datetime import datetime, timedelta
import stripe
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torchvision.models as models
from functools import wraps
import cv2
from email_utilis import send_verification_email, send_welcome_email, logger
import sqlite3
import io
import os
from datetime import datetime
conn = sqlite3.connect('neuroscan_platform.db')
c = conn.cursor()
matplotlib.use('Agg')
# Check if column exists, if not add it
try:
    c.execute("ALTER TABLE patient_access_codes ADD COLUMN verification_code TEXT")
    c.execute("ALTER TABLE patient_access_codes ADD COLUMN verification_code_expiry DATETIME")
    conn.commit()
    print("✅ Added verification_code columns")
except sqlite3.OperationalError as e:
    print(f"Column might already exist: {e}")
finally:
    conn.close()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, session, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_socketio import SocketIO, emit, join_room, leave_room, send
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import json

from pdf_report import generate_pdf_report
from email_utilis import send_verification_email, send_welcome_email

# -----------------------------
# Configuration
# -----------------------------
UPLOAD_FOLDER = 'uploads/profile_pictures'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
DB_FILE = "neuroscan_platform.db"

MODEL_PATH = Path(__file__).resolve().parent / "weights_60epochs.pt"

CHAT_UPLOAD_FOLDER = 'uploads/chat_attachments'
ALLOWED_CHAT_FILES = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'dcm'}  # dcm for DICOM files
MAX_CHAT_FILE_SIZE = 10 * 1024 * 1024  # 10MB
os.makedirs(CHAT_UPLOAD_FOLDER, exist_ok=True)
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

from flask_session import Session  # Add this import at top

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Configure session
app.config.update(
    SECRET_KEY=SECRET_KEY,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_DOMAIN=None,
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    SESSION_TYPE='filesystem',
    SESSION_FILE_DIR='./flask_session',  # Where to store sessions
    SESSION_PERMANENT=False,
    SESSION_COOKIE_NAME='neuroscan-session',
    SESSION_USE_SIGNER=True,
)

# Initialize Flask-Session
Session(app)

# Create session directory
os.makedirs('./flask_session', exist_ok=True)
# ==============================================
# CRITICAL CORS CONFIGURATION - FIXED
# ==============================================

from flask_cors import CORS, cross_origin

# Allowed origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://192.168.1.70:3000"
]

# âœ… APPLY CORS TO ALL ROUTES WITH EXPLICIT SETTINGS
CORS(app,
     origins=ALLOWED_ORIGINS,
     supports_credentials=True,
     allow_headers=[
         "Content-Type",
         "Authorization",
         "X-Requested-With",
         "Accept",
         "Origin",
         "Access-Control-Request-Method",
         "Access-Control-Request-Headers"
     ],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
     expose_headers=["Content-Type", "Authorization"],
     send_wildcard=False,
     max_age=3600)


# âœ… EXPLICIT AFTER_REQUEST HANDLER
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')

    # Only set CORS headers if origin is in allowed list
    if origin in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, PATCH'
        response.headers[
            'Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Type, Authorization'

    return response


# âœ… EXPLICIT OPTIONS HANDLER FOR PREFLIGHT
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    response = app.make_default_options_response()
    origin = request.headers.get('Origin')

    if origin in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, PATCH'
        response.headers[
            'Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'

    return response
# ==============================================
# SESSION CONFIGURATION - FIXED FOR CROSS-ORIGIN
# ==============================================

app.config.update(
    SECRET_KEY=SECRET_KEY,
    SESSION_COOKIE_NAME='neuroscan_session',
    SESSION_COOKIE_SAMESITE='Lax',  # âœ… CRITICAL for cross-origin
    SESSION_COOKIE_SECURE=False,     # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_DOMAIN=None,      # Let Flask handle it
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    SESSION_TYPE='filesystem'
)
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# Add this function to app.py and call it at startup

def migrate_database_schema():
    """Create missing tables and columns"""
    conn = get_db()
    c = conn.cursor()

    # Create patient_access_codes table
    c.execute("""
        CREATE TABLE IF NOT EXISTS patient_access_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            access_code TEXT NOT NULL,
            verification_code TEXT,
            is_verified INTEGER DEFAULT 0,
            verified_at TIMESTAMP,
            expires_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
        )
    """)

    # Add access_code column to patients if missing
    c.execute("PRAGMA table_info(patients)")
    columns = [col[1] for col in c.fetchall()]

    if 'access_code' not in columns:
        c.execute("ALTER TABLE patients ADD COLUMN access_code TEXT")
        logger.info("✅ Added access_code column to patients")

    # Create indexes
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_patient_access_codes_patient_id 
        ON patient_access_codes(patient_id)
    """)

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_patient_access_codes_access_code 
        ON patient_access_codes(access_code)
    """)

    conn.commit()
    conn.close()
    logger.info("✅ Database migration completed")

# SocketIO with CORS support
socketio = SocketIO(
    app,
    cors_allowed_origins=ALLOWED_ORIGINS,
    async_mode='threading',
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25
)


# ============================================
# CONFIGURATION FLAGS
# ============================================
ENABLE_GRADCAM = False  # Disabled due to GPU memory constraints


# ==============================================
# DATABASE & UTILITIES
# ==============================================

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def get_db_connection():
    # Alias for get_db() - for compatibility
    return get_db()


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


def validate_brain_scan(img, prob_results=None):
    """
    Enhanced validation to check if uploaded image is a brain MRI scan.
    Checks image characteristics before running model.
    """
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(img)

        # CHECK 1: Color check - reject color photos
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            r = img_array[:, :, 0].astype(float)
            g = img_array[:, :, 1].astype(float)
            b = img_array[:, :, 2].astype(float)

            color_diff = (np.abs(r - g).mean() + np.abs(r - b).mean() + np.abs(g - b).mean()) / 3

            if color_diff > 20:
                return False, "Invalid image: This appears to be a color photo, not a grayscale MRI scan.", None

        # CHECK 2: Size check
        height, width = img_array.shape[:2]
        if width < 100 or height < 100:
            return False, f"Image too small: {width}x{height}. Must be at least 100x100 pixels.", None

        # CHECK 3: Aspect ratio
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False, f"Invalid aspect ratio: {aspect_ratio:.2f}. Brain scans are typically square.", None

        # CHECK 4: Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # CHECK 5: Brightness check
        mean_intensity = gray.mean()
        if mean_intensity < 10 or mean_intensity > 245:
            return False, "Invalid brightness: Image is too dark or too bright to be an MRI scan.", None

        # CHECK 6: Contrast check
        std_intensity = gray.std()
        if std_intensity < 15:
            return False, "Insufficient contrast: MRI scans have distinct tissue contrasts.", None

        # CHECK 7: Edge detection - detect screenshots/UI
        edges = cv2.Canny(gray, 50, 150)
        edge_density = (edges > 0).sum() / edges.size

        if edge_density < 0.02:
            return False, "Invalid structure: Image is too simple to be a medical scan.", None

        if edge_density > 0.40:
            return False, "Screenshot detected: This appears to be a UI/screenshot, not a medical scan.", None

        # CHECK 8: Detect UI elements (sharp lines)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        strong_horizontal = (np.abs(sobel_y) > 100).sum() / gray.size
        strong_vertical = (np.abs(sobel_x) > 100).sum() / gray.size

        if strong_horizontal > 0.15 or strong_vertical > 0.15:
            return False, "Screenshot detected: Sharp UI elements found. Please upload an MRI scan.", None

        # CHECK 9: Circular structure detection (brain outline)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30,
            minRadius=30, maxRadius=min(width, height) // 2
        )

        if circles is None or len(circles[0]) == 0:
            return False, "No brain structure detected: No circular/oval brain outline found.", None

        # CHECK 10: Probability check (if provided)
        if prob_results:
            max_prob = max(prob_results.values())
            if max_prob < 50.0:
                return False, "Low confidence: AI model is uncertain. Image may not be a valid MRI scan.", None
            if max_prob < 70.0:
                return True, None, "Warning: Moderate confidence. Please verify image quality."

        # All checks passed
        warnings = []
        if aspect_ratio < 0.75 or aspect_ratio > 1.35:
            warnings.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
        if width < 200 or height < 200:
            warnings.append("Low resolution image")

        warning_text = ". ".join(warnings) if warnings else None
        return True, None, warning_text

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return True, None, f"Validation warning: {str(e)}"


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" in session and "user_type" in session:
            return f(*args, **kwargs)
        if "patient_id" in session and "patient_type" in session:
            return f(*args, **kwargs)
        return jsonify({"error": "Not authenticated"}), 401

    return wrapper


def ensure_usage_tracking_exists(hospital_id):
    """Ensure usage_tracking record exists for hospital"""
    try:
        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT id FROM usage_tracking 
            WHERE hospital_id = ? AND is_current = 1
        """, (hospital_id,))

        if not c.fetchone():
            logger.warning(f"⚠️ Creating missing usage_tracking for hospital {hospital_id}")

            # Get actual counts
            c.execute("SELECT COUNT(*) FROM hospital_users WHERE hospital_id = ?", (hospital_id,))
            users_count = c.fetchone()[0]

            c.execute("SELECT COUNT(*) FROM patients WHERE hospital_id = ?", (hospital_id,))
            patients_count = c.fetchone()[0]

            c.execute("""
                SELECT COUNT(*) FROM mri_scans 
                WHERE hospital_id = ? 
                AND strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')
            """, (hospital_id,))
            scans_count = c.fetchone()[0]

            period_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if period_start.month == 12:
                period_end = period_start.replace(year=period_start.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                period_end = period_start.replace(month=period_start.month + 1, day=1) - timedelta(days=1)

            c.execute("""
                INSERT INTO usage_tracking (
                    hospital_id, scans_used, users_count, patients_count,
                    period_start, period_end, is_current
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (hospital_id, scans_count, users_count, patients_count,
                  period_start.isoformat(), period_end.isoformat()))

            conn.commit()
            logger.info(
                f"✅ Created usage_tracking: {scans_count} scans, {users_count} users, {patients_count} patients")

        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error ensuring usage_tracking: {e}")
        return False
# ==============================================
# HELPER FUNCTIONS FOR SUBSCRIPTION & USAGE
# Added to resolve unresolved references
# ==============================================

def generate_unique_code(table, column, prefix='', length=6):
    """
    Generate a unique code for a table column

    Args:
        table: Database table name (e.g., 'patients', 'hospitals')
        column: Column name to check for uniqueness (e.g., 'patient_code')
        prefix: Optional prefix for the code (e.g., 'P' for patients)
        length: Length of the random portion (default 6)

    Returns:
        str: Unique code with prefix
    """
    conn = get_db()
    c = conn.cursor()

    max_attempts = 100
    for _ in range(max_attempts):
        # Generate random code
        random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
        code = f"{prefix}{random_part}" if prefix else random_part

        # Check if exists
        c.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (code,))
        count = c.fetchone()[0]

        if count == 0:
            conn.close()
            return code

    conn.close()
    raise Exception(f"Could not generate unique code for {table}.{column} after {max_attempts} attempts")


def get_hospital_subscription(hospital_id):
    """
    Get active subscription for a hospital with plan details

    Args:
        hospital_id: Hospital ID

    Returns:
        dict: Subscription details with plan information, or None if no active subscription
    """
    try:
        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT 
                hs.id as subscription_id,
                hs.hospital_id,
                hs.plan_id,
                hs.stripe_subscription_id,
                hs.status,
                hs.current_period_start,
                hs.current_period_end,
                hs.cancel_at_period_end,
                sp.plan_name,
                sp.max_scans_per_month,
                sp.max_users,
                sp.max_patients,
                sp.features,
                sp.price_monthly,
                sp.price_yearly
            FROM hospital_subscriptions hs
            JOIN subscription_plans sp ON hs.plan_id = sp.id
            WHERE hs.hospital_id = ? 
            AND hs.status = 'active'
            AND (hs.current_period_end IS NULL OR hs.current_period_end > datetime('now'))
            ORDER BY hs.created_at DESC
            LIMIT 1
        """, (hospital_id,))

        row = c.fetchone()
        conn.close()

        if row:
            return dict(row)
        else:
            # Return default free plan if no subscription
            return {
                'subscription_id': None,
                'hospital_id': hospital_id,
                'plan_id': 1,
                'plan_name': 'Free',
                'max_scans_per_month': 10,
                'max_users': 2,
                'max_patients': 50,
                'features': '[]',
                'status': 'active'
            }

    except Exception as e:
        logger.error(f"Error getting hospital subscription: {e}")
        return None


def get_current_usage(hospital_id):
    """
    Get current usage statistics for a hospital

    Args:
        hospital_id: Hospital ID

    Returns:
        dict: Usage statistics (scans_used, users_count, patients_count)
    """
    try:
        conn = get_db()
        c = conn.cursor()

        # Get or create usage tracking record for current period
        c.execute("""
            SELECT scans_used, users_count, patients_count, period_start, period_end
            FROM usage_tracking
            WHERE hospital_id = ? AND is_current = 1
        """, (hospital_id,))

        usage = c.fetchone()

        if usage:
            result = {
                'scans_used': usage['scans_used'],
                'users_count': usage['users_count'],
                'patients_count': usage['patients_count'],
                'period_start': usage['period_start'],
                'period_end': usage['period_end']
            }
        else:
            # Create new usage tracking record
            period_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            # Calculate period_end (last day of current month)
            if period_start.month == 12:
                period_end = period_start.replace(year=period_start.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                period_end = period_start.replace(month=period_start.month + 1, day=1) - timedelta(days=1)

            c.execute("""
                INSERT INTO usage_tracking (
                    hospital_id, scans_used, users_count, patients_count,
                    period_start, period_end, is_current
                ) VALUES (?, 0, 0, 0, ?, ?, 1)
            """, (hospital_id, period_start.isoformat(), period_end.isoformat()))

            conn.commit()

            result = {
                'scans_used': 0,
                'users_count': 0,
                'patients_count': 0,
                'period_start': period_start.isoformat(),
                'period_end': period_end.isoformat()
            }

        conn.close()
        return result

    except Exception as e:
        logger.error(f"Error getting current usage: {e}")
        return {
            'scans_used': 0,
            'users_count': 0,
            'patients_count': 0
        }


def get_stripe_price_id(plan_id, billing_cycle='monthly'):
    """
    Map subscription plan IDs to Stripe price IDs
    """
    # CHECK IF REAL STRIPE PRICES ARE CONFIGURED
    basic_monthly = os.getenv('STRIPE_PRICE_BASIC_MONTHLY')

    if not basic_monthly or basic_monthly == 'price_basic_monthly':
        logger.warning("⚠️ Stripe prices not configured - using test mode")
        # RETURN TEST PRICE IDs (create these in Stripe Dashboard first)
        price_mapping = {
            2: {  # Basic
                'monthly': 'price_1QVtExIvP6P9XTkI6aDVGBg3',  # REPLACE WITH YOUR REAL STRIPE PRICE ID
                'yearly': 'price_1QVtExIvP6P9XTkI6aDVGBg3'
            },
            3: {  # Premium
                'monthly': 'price_1QVtExIvP6P9XTkI6aDVGBg3',
                'yearly': 'price_1QVtExIvP6P9XTkI6aDVGBg3'
            },
            4: {  # Enterprise
                'monthly': 'price_1QVtExIvP6P9XTkI6aDVGBg3',
                'yearly': 'price_1QVtExIvP6P9XTkI6aDVGBg3'
            }
        }
    else:
        price_mapping = {
            1: {'monthly': None, 'yearly': None},
            2: {
                'monthly': os.getenv('STRIPE_PRICE_BASIC_MONTHLY'),
                'yearly': os.getenv('STRIPE_PRICE_BASIC_YEARLY')
            },
            3: {
                'monthly': os.getenv('STRIPE_PRICE_PREMIUM_MONTHLY'),
                'yearly': os.getenv('STRIPE_PRICE_PREMIUM_YEARLY')
            },
            4: {
                'monthly': os.getenv('STRIPE_PRICE_ENTERPRISE_MONTHLY'),
                'yearly': os.getenv('STRIPE_PRICE_ENTERPRISE_YEARLY')
            }
        }

    plan_prices = price_mapping.get(plan_id)
    if not plan_prices:
        return None

    return plan_prices.get(billing_cycle)


def get_or_create_stripe_customer(hospital_id, email):
    """
    Get existing Stripe customer ID or create a new one

    Args:
        hospital_id: Hospital ID
        email: Hospital email address

    Returns:
        str: Stripe customer ID
    """
    try:
        conn = get_db()
        c = conn.cursor()

        # Check if hospital already has a Stripe customer ID
        c.execute("SELECT stripe_customer_id FROM hospitals WHERE id = ?", (hospital_id,))
        result = c.fetchone()

        if result and result['stripe_customer_id']:
            conn.close()
            return result['stripe_customer_id']

        # Create new Stripe customer
        customer = stripe.Customer.create(
            email=email,
            metadata={
                'hospital_id': hospital_id
            }
        )

        # Save customer ID to database
        c.execute("""
            UPDATE hospitals 
            SET stripe_customer_id = ?
            WHERE id = ?
        """, (customer.id, hospital_id))

        conn.commit()
        conn.close()

        return customer.id

    except Exception as e:
        logger.error(f"Error creating Stripe customer: {e}")
        raise


def get_detailed_usage(hospital_id):
    """
    Get detailed usage information with limits and blocking status

    Args:
        hospital_id: Hospital ID

    Returns:
        dict: Complete usage information including limits and blocking status
    """
    subscription = get_hospital_subscription(hospital_id)

    if not subscription:
        return {
            'is_blocked': True,
            'block_message': 'No active subscription found',
            'scans_used': 0,
            'max_scans': 0,
            'users_count': 0,
            'max_users': 0,
            'patients_count': 0,
            'max_patients': 0,
            'plan_name': 'None'
        }

    usage = get_current_usage(hospital_id)

    max_scans = subscription['max_scans_per_month']
    max_users = subscription['max_users']
    max_patients = subscription['max_patients']

    scans_used = usage['scans_used']
    users_count = usage['users_count']
    patients_count = usage['patients_count']

    # Check if blocked
    is_blocked = False
    block_message = ""

    if max_scans != -1 and scans_used >= max_scans:
        is_blocked = True
        block_message = f"Monthly scan limit reached ({max_scans} scans). Please upgrade your plan."
    elif max_users != -1 and users_count >= max_users:
        is_blocked = True
        block_message = f"User limit reached ({max_users} users). Please upgrade your plan."
    elif max_patients != -1 and patients_count >= max_patients:
        is_blocked = True
        block_message = f"Patient limit reached ({max_patients} patients). Please upgrade your plan."

    # Calculate usage percentage
    usage_percentage = 0
    if max_scans != -1 and max_scans > 0:
        usage_percentage = (scans_used / max_scans) * 100

    return {
        'is_blocked': is_blocked,
        'block_message': block_message,
        'scans_used': scans_used,
        'max_scans': 'unlimited' if max_scans == -1 else max_scans,
        'users_count': users_count,
        'max_users': 'unlimited' if max_users == -1 else max_users,
        'patients_count': patients_count,
        'max_patients': 'unlimited' if max_patients == -1 else max_patients,
        'plan_name': subscription['plan_name'],
        'usage_percentage': round(usage_percentage, 1)
    }


def increment_usage(hospital_id, resource_type='scans', amount=1):
    """
    Increment usage counter for a hospital

    Args:
        hospital_id: Hospital ID
        resource_type: 'scans', 'users', or 'patients'
        amount: Amount to increment (default 1)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
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

        # Get or create current period usage
        c.execute("""
            SELECT id FROM usage_tracking
            WHERE hospital_id = ? AND is_current = 1
        """, (hospital_id,))

        usage = c.fetchone()

        if not usage:
            # Create new usage record
            get_current_usage(hospital_id)

        # Increment the counter
        c.execute(f"""
            UPDATE usage_tracking
            SET {field} = {field} + ?, updated_at = CURRENT_TIMESTAMP
            WHERE hospital_id = ? AND is_current = 1
        """, (amount, hospital_id))

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Error incrementing usage: {e}")
        return False


def increment_usage(hospital_id, resource_type='scans', amount=1):
    """Increment usage counter for a hospital"""
    try:
        conn = get_db()
        c = conn.cursor()

        field_map = {
            'scans': 'scans_used',
            'users': 'users_count',
            'patients': 'patients_count'
        }

        field = field_map.get(resource_type)
        if not field:
            logger.error(f"❌ Invalid resource_type: {resource_type}")
            conn.close()
            return False

        # ✅ ADD THIS DEBUG LOG
        logger.info(f"📊 Incrementing {resource_type} for hospital {hospital_id}")

        # First, ensure usage tracking record exists
        c.execute("""
            SELECT id, {field} FROM usage_tracking
            WHERE hospital_id = ? AND is_current = 1
        """.format(field=field), (hospital_id,))

        existing = c.fetchone()

        # ✅ ADD THIS DEBUG LOG
        if existing:
            logger.info(f"   Current {field}: {existing[field]}")
        else:
            logger.warning(f"   No current usage_tracking record found! Creating one...")

        if not existing:
            # Create new tracking record
            get_current_usage(hospital_id)

        # Update the counter
        c.execute("""
            UPDATE usage_tracking
            SET {field} = {field} + ?
            WHERE hospital_id = ? AND is_current = 1
        """.format(field=field), (amount, hospital_id))

        rows_affected = c.rowcount

        # ✅ ADD THIS DEBUG LOG
        logger.info(f"   Updated {rows_affected} rows")

        if rows_affected == 0:
            logger.error(f"❌ Failed to update usage! No rows affected")

        conn.commit()

        # ✅ ADD THIS - Verify the update
        c.execute(f"SELECT {field} FROM usage_tracking WHERE hospital_id = ? AND is_current = 1",
                  (hospital_id,))
        new_value = c.fetchone()
        if new_value:
            logger.info(f"   New {field}: {new_value[field]}")

        conn.close()
        return True

    except Exception as e:
        logger.error(f"❌ Error incrementing usage: {e}")
        return False
def has_feature(hospital_id, feature_key):
    """
    Check if hospital's plan includes a specific feature

    Args:
        hospital_id: Hospital ID
        feature_key: Feature identifier (e.g., 'video_call', 'tumor_tracking')

    Returns:
        bool: True if feature is available, False otherwise
    """
    subscription = get_hospital_subscription(hospital_id)
    if not subscription:
        return False

    try:
        features = json.loads(subscription.get('features', '[]'))
        return feature_key in features
    except:
        return False


def create_usage_tracking_table():
    """Create usage_tracking table if it doesn't exist"""
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS usage_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            scans_used INTEGER DEFAULT 0,
            users_count INTEGER DEFAULT 0,
            patients_count INTEGER DEFAULT 0,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            is_current INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(id)
        )
    """)

    # Create index for faster lookups
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_usage_hospital_current 
        ON usage_tracking(hospital_id, is_current)
    """)

    conn.commit()
    conn.close()
    logger.info("✅ usage_tracking table created/verified")


def add_stripe_customer_column():
    """Add stripe_customer_id column to hospitals table if it doesn't exist"""
    try:
        conn = get_db()
        c = conn.cursor()

        # Check if column exists
        c.execute("PRAGMA table_info(hospitals)")
        columns = [column[1] for column in c.fetchall()]

        if 'stripe_customer_id' not in columns:
            c.execute("""
                ALTER TABLE hospitals 
                ADD COLUMN stripe_customer_id TEXT
            """)
            conn.commit()
            logger.info("✅ Added stripe_customer_id column to hospitals table")

        conn.close()
    except Exception as e:
        logger.error(f"Error adding stripe_customer_id column: {e}")


def migrate_usage_tracking_columns():
    """Add missing columns to usage_tracking table"""
    try:
        conn = get_db()
        c = conn.cursor()

        # Check if columns exist
        c.execute("PRAGMA table_info(usage_tracking)")
        columns = [column[1] for column in c.fetchall()]

        if 'scans_limit' not in columns:
            c.execute("ALTER TABLE usage_tracking ADD COLUMN scans_limit INTEGER DEFAULT 0")
            logger.info("✅ Added scans_limit column to usage_tracking")

        if 'users_limit' not in columns:
            c.execute("ALTER TABLE usage_tracking ADD COLUMN users_limit INTEGER DEFAULT 0")
            logger.info("✅ Added users_limit column to usage_tracking")

        if 'patients_limit' not in columns:
            c.execute("ALTER TABLE usage_tracking ADD COLUMN patients_limit INTEGER DEFAULT 0")
            logger.info("✅ Added patients_limit column to usage_tracking")

        if 'subscription_id' not in columns:
            c.execute("ALTER TABLE usage_tracking ADD COLUMN subscription_id INTEGER")
            logger.info("✅ Added subscription_id column to usage_tracking")

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error migrating usage_tracking columns: {e}")


# Initialize tables
create_usage_tracking_table()
add_stripe_customer_column()
migrate_usage_tracking_columns()


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


def create_notifications_table():
    """Create notifications table - Run once during setup"""
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            user_type TEXT NOT NULL,
            hospital_id INTEGER,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            prediction_id INTEGER,
            scan_id INTEGER,
            patient_id INTEGER,
            is_read INTEGER DEFAULT 0,
            priority TEXT DEFAULT 'normal',
            action_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            read_at TIMESTAMP,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE,
            FOREIGN KEY (prediction_id) REFERENCES mri_scans(id) ON DELETE CASCADE,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
        )
    """)

    # Create indexes for performance
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_notifications_user 
        ON notifications(user_id, user_type, is_read)
    """)

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_notifications_hospital 
        ON notifications(hospital_id, created_at)
    """)

    conn.commit()
    conn.close()
    logger.info("✅ Notifications table created")


def create_messages_table():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            hospital_user_id INTEGER NOT NULL,
            sender_type TEXT NOT NULL CHECK (sender_type IN ('patient', 'hospital')),
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_read INTEGER DEFAULT 0,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
            FOREIGN KEY (hospital_user_id) REFERENCES hospital_users(id) ON DELETE CASCADE
        )
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_chat
        ON messages(patient_id, hospital_user_id, created_at)
    """)
    conn.commit()
    conn.close()
    logger.info("✅ Messages table created")


# Call to create tables
create_notifications_table()
create_messages_table()

# SocketIO connected users (for private messaging)
connected_users = {}  # {user_id: sid} for both patient and hospital users


@socketio.on('connect')
def handle_connect(auth=None):  # ← Add auth=None parameter
    """Handle socket connection"""
    try:
        if 'patient_id' in session:
            user_id = session['patient_id']
            user_type = 'patient'
        elif 'user_id' in session and session.get('user_type') == 'hospital':
            user_id = session['user_id']
            user_type = 'hospital'
        else:
            print("⚠️ Unauthenticated connection")
            return False

        connected_users[user_id] = request.sid
        room_name = f"{user_type}_{user_id}"
        join_room(room_name)

        print(f"✅ {user_type.capitalize()} {user_id} connected (SID: {request.sid})")

        # Remove broadcast=True - just use emit()
        emit('connected', {
            'user_id': user_id,
            'user_type': user_type,
            'status': 'online'
        })

        return True
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

    # Store connection
    connected_users[user_id] = request.sid

    # Join user-specific room
    room_name = f"{user_type}_{user_id}"
    join_room(room_name)

    print(f"✅ {user_type.capitalize()} {user_id} connected (SID: {request.sid})")

    # Emit status WITHOUT broadcast parameter
    socketio.emit('user_status', {
        'user_id': user_id,
        'user_type': user_type,
        'status': 'online'
    }, room=room_name)  # <- REMOVE broadcast=True


@socketio.on('disconnect')
def handle_disconnect():
    for key, sid in list(connected_users.items()):
        if sid == request.sid:
            del connected_users[key]
            print(f"User disconnected: {key}")
            break


@socketio.on('join_chat')
def on_join(data):
    patient_id = data['patient_id']
    room = f"chat_{patient_id}"
    join_room(room)
    print(f"Joined room {room}")


@socketio.on('send_message')
def handle_send_message(data):
    patient_id = data['patient_id']
    hospital_user_id = data['hospital_user_id']
    message = data['message']
    sender_type = 'patient' if 'patient_id' in session else 'hospital'
    sender_id = session.get('patient_id') or session.get('user_id')

    # Save message to DB
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO messages (patient_id, hospital_user_id, sender_type, message)
        VALUES (?, ?, ?, ?)
    """, (patient_id, hospital_user_id, sender_type, message))
    message_id = c.lastrowid
    conn.commit()
    conn.close()

    # Emit to room
    room = f"chat_{patient_id}"
    emit('new_message', {
        'id': message_id,
        'sender_type': sender_type,
        'message': message,
        'created_at': datetime.now().isoformat()
    }, room=room)

    # Send notification to recipient
    recipient_type = 'hospital' if sender_type == 'patient' else 'patient'
    recipient_id = hospital_user_id if sender_type == 'patient' else patient_id
    create_notification(
        user_id=recipient_id,
        user_type=recipient_type,
        notification_type='new_message',
        title='New Message',
        message=f"You have a new message from {sender_type}",
        patient_id=patient_id,
        priority='normal',
        action_url=f"/chat/{patient_id}"
    )


def create_enhanced_messages_table():
    """Enhanced messages table with file attachments"""
    conn = get_db()
    c = conn.cursor()

    # Drop old table if exists (for migration)
    # c.execute("DROP TABLE IF EXISTS messages")

    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            hospital_user_id INTEGER NOT NULL,
            sender_type TEXT NOT NULL CHECK (sender_type IN ('patient', 'hospital')),
            message TEXT,
            attachment BLOB,
            attachment_name TEXT,
            attachment_type TEXT,
            attachment_size INTEGER,
            is_read INTEGER DEFAULT 0,
            read_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
            FOREIGN KEY (hospital_user_id) REFERENCES hospital_users(id) ON DELETE CASCADE
        )
    """)

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_chat
        ON messages(patient_id, hospital_user_id, created_at DESC)
    """)

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_unread
        ON messages(patient_id, hospital_user_id, is_read)
    """)

    conn.commit()
    conn.close()
    logger.info("✅ Enhanced messages table created")


# Call this during initialization
create_enhanced_messages_table()

# ==============================================
# ENHANCED SOCKET.IO EVENTS
# ==============================================

# Track typing status
typing_users = {}  # {room_id: {user_id: timestamp}}

# Track online users
online_users = {}  # {user_type_user_id: {sid, patient_id, hospital_user_id}}


@socketio.on('connect')
def handle_connect():
    """Enhanced connection handler with online status"""
    if 'patient_id' in session:
        user_id = session['patient_id']
        user_type = 'patient'
        patient_id = user_id
        hospital_user_id = None
    elif 'user_id' in session and session.get('user_type') == 'hospital':
        user_id = session['user_id']
        user_type = 'hospital'
        patient_id = None
        hospital_user_id = user_id
    else:
        return False

    user_key = f"{user_type}_{user_id}"
    online_users[user_key] = {
        'sid': request.sid,
        'user_id': user_id,
        'user_type': user_type,
        'patient_id': patient_id,
        'hospital_user_id': hospital_user_id
    }

    print(f"✅ {user_type.capitalize()} {user_id} connected (SID: {request.sid})")

    emit('connected', {'status': 'connected', 'user_type': user_type, 'user_id': user_id})


@socketio.on('disconnect')
def handle_disconnect():
    """Enhanced disconnect handler - FIXED"""
    user_key = None
    user_info = None

    for key, info in list(online_users.items()):
        if info['sid'] == request.sid:
            user_key = key
            user_info = info
            del online_users[key]
            print(f"❌ User disconnected: {key}")
            break

    if user_info:
        # FIXED: Use emit instead of socketio.emit
        emit('user_status', {
            'user_type': user_info['user_type'],
            'user_id': user_info['user_id'],
            'status': 'offline'
        }, broadcast=True)


@socketio.on('join_chat')
def on_join_chat(data):
    """Join a specific chat room"""
    patient_id = data.get('patient_id')
    hospital_user_id = data.get('hospital_user_id')

    if not patient_id or not hospital_user_id:
        emit('error', {'message': 'Invalid chat parameters'})
        return

    room = f"chat_{patient_id}_{hospital_user_id}"
    join_room(room)

    print(f"👥 Joined room: {room}")

    # Notify room members
    emit('user_joined', {
        'room': room,
        'timestamp': datetime.now().isoformat()
    }, room=room)

    # Mark messages as read for the joining user
    if 'patient_id' in session:
        mark_messages_read(patient_id, hospital_user_id, 'patient')
    elif 'user_id' in session:
        mark_messages_read(patient_id, hospital_user_id, 'hospital')


@socketio.on('leave_chat')
def on_leave_chat(data):
    """Leave a chat room"""
    patient_id = data.get('patient_id')
    hospital_user_id = data.get('hospital_user_id')
    room = f"chat_{patient_id}_{hospital_user_id}"

    leave_room(room)
    print(f"👋 Left room: {room}")


@socketio.on('typing')
def on_typing(data):
    """Handle typing indicators"""
    patient_id = data.get('patient_id')
    hospital_user_id = data.get('hospital_user_id')
    is_typing = data.get('is_typing', False)

    room = f"chat_{patient_id}_{hospital_user_id}"

    if 'patient_id' in session:
        sender_type = 'patient'
        sender_id = session['patient_id']
    else:
        sender_type = 'hospital'
        sender_id = session['user_id']

    emit('user_typing', {
        'sender_type': sender_type,
        'sender_id': sender_id,
        'is_typing': is_typing
    }, room=room, include_self=False)


@socketio.on('send_message')
def handle_send_message(data):
    """Enhanced message handler with read receipts"""
    patient_id = data.get('patient_id')
    hospital_user_id = data.get('hospital_user_id')
    message = data.get('message', '').strip()

    if not patient_id or not hospital_user_id:
        emit('error', {'message': 'Invalid message parameters'})
        return

    if not message:
        emit('error', {'message': 'Message cannot be empty'})
        return

    sender_type = 'patient' if 'patient_id' in session else 'hospital'
    sender_id = session.get('patient_id') or session.get('user_id')

    # Save message to database
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO messages (
            patient_id, hospital_user_id, sender_type, message
        ) VALUES (?, ?, ?, ?)
    """, (patient_id, hospital_user_id, sender_type, message))

    message_id = c.lastrowid
    timestamp = datetime.now().isoformat()
    conn.commit()
    conn.close()

    room = f"chat_{patient_id}_{hospital_user_id}"

    # Emit to room
    message_data = {
        'id': message_id,
        'patient_id': patient_id,
        'hospital_user_id': hospital_user_id,
        'sender_type': sender_type,
        'sender_id': sender_id,
        'message': message,
        'attachment': None,
        'created_at': timestamp,
        'is_read': False
    }

    emit('new_message', message_data, room=room)

    # Send notification to recipient
    recipient_type = 'hospital' if sender_type == 'patient' else 'patient'
    recipient_id = hospital_user_id if sender_type == 'patient' else patient_id

    # Get sender name
    conn = get_db()
    c = conn.cursor()
    if sender_type == 'patient':
        c.execute("SELECT full_name FROM patients WHERE id=?", (sender_id,))
        sender = c.fetchone()
        sender_name = sender['full_name'] if sender else 'Patient'
    else:
        c.execute("SELECT full_name FROM hospital_users WHERE id=?", (sender_id,))
        sender = c.fetchone()
        sender_name = sender['full_name'] if sender else 'Doctor'
    conn.close()

    # Create notification
    create_notification(
        user_id=recipient_id,
        user_type=recipient_type,
        notification_type='new_message',
        title=f'💬 New message from {sender_name}',
        message=message[:100] + ('...' if len(message) > 100 else ''),
        patient_id=patient_id,
        priority='normal',
        action_url=f'/chat?patient_id={patient_id}&hospital_user_id={hospital_user_id}'
    )

    # Emit notification to recipient if online
    recipient_key = f"{recipient_type}_{recipient_id}"
    if recipient_key in online_users:
        socketio.emit('notification', {
            'type': 'new_message',
            'title': f'New message from {sender_name}',
            'message': message[:100],
            'patient_id': patient_id,
            'hospital_user_id': hospital_user_id
        }, room=online_users[recipient_key]['sid'])


def mark_messages_read(patient_id, hospital_user_id, reader_type):
    """Mark messages as read"""
    conn = get_db()
    c = conn.cursor()

    # Determine which messages to mark as read (ones sent BY the other party)
    if reader_type == 'patient':
        # Patient is reading, so mark hospital's messages as read
        sender_type_to_mark = 'hospital'
    else:
        # Hospital is reading, so mark patient's messages as read
        sender_type_to_mark = 'patient'

    c.execute("""
        UPDATE messages
        SET is_read = 1, read_at = CURRENT_TIMESTAMP
        WHERE patient_id = ?
        AND hospital_user_id = ?
        AND sender_type = ?
        AND is_read = 0
    """, (patient_id, hospital_user_id, sender_type_to_mark))

    conn.commit()
    conn.close()


# ==============================================
# NOTIFICATION ROUTES
# ==============================================

@app.route('/notifications', methods=['GET'])
def get_notifications_api():
    """Get all notifications for the current user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user_id = session['user_id']

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, user_id, type, message, scan_id, is_read, created_at
            FROM notifications
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 50
        ''', (user_id,))

        notifications = []
        for row in cursor.fetchall():
            notifications.append({
                'id': row[0],
                'user_id': row[1],
                'type': row[2],
                'message': row[3],
                'scan_id': row[4],
                'read': bool(row[5]),
                'created_at': row[6]
            })

        conn.close()
        return jsonify({'notifications': notifications})

    except Exception as e:
        logging.error(f"Error fetching notifications: {e}")
        return jsonify({'error': 'Failed to fetch notifications'}), 500


def create_notification_full(user_id, notif_type, message, scan_id=None, user_type='patient'):
    """Helper function to create a notification"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert into the full notifications table with all required fields
        cursor.execute('''
            INSERT INTO notifications (
                user_id, user_type, hospital_id, type, title, message, 
                scan_id, is_read, priority, created_at
            )
            VALUES (?, ?, NULL, ?, ?, ?, ?, 0, 'normal', CURRENT_TIMESTAMP)
        ''', (user_id, user_type, notif_type, 'Scan Analysis', message, scan_id))

        notification_id = cursor.lastrowid
        conn.commit()

        # Get the created notification
        cursor.execute('''
            SELECT id, user_id, user_type, type, message, scan_id, is_read, created_at
            FROM notifications
            WHERE id = ?
        ''', (notification_id,))

        row = cursor.fetchone()
        notification = {
            'id': row[0],
            'user_id': row[1],
            'user_type': row[2],
            'type': row[3],
            'message': row[4],
            'scan_id': row[5],
            'read': bool(row[6]),
            'created_at': row[7]
        }

        conn.close()

        # Emit socket event to the specific user
        socketio.emit('new_notification', notification, room=f'{user_type}_{user_id}')

        return notification

    except Exception as e:
        logging.error(f"Error creating notification: {e}")
        return None


# ==============================================
# CHAT/MESSAGING ROUTES
# ==============================================

@app.route('/messages/<int:patient_id>', methods=['GET'])
def get_messages(patient_id):
    """Get all messages between hospital and patient"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    hospital_user_id = session['user_id']

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, patient_id, hospital_user_id, sender_type, message, created_at, is_read
            FROM messages
            WHERE patient_id = ? AND hospital_user_id = ?
            ORDER BY created_at ASC
        ''', (patient_id, hospital_user_id))

        messages = []
        for row in cursor.fetchall():
            messages.append({
                'id': row[0],
                'patient_id': row[1],
                'hospital_user_id': row[2],
                'sender_type': row[3],
                'sender_id': row[1] if row[3] == 'patient' else row[2],  # For compatibility
                'recipient_id': row[2] if row[3] == 'patient' else row[1],  # For compatibility
                'message': row[4],
                'attachment_url': None,
                'created_at': row[5],
                'read': bool(row[6])
            })

        # Mark messages from patient as read
        cursor.execute('''
            UPDATE messages 
            SET is_read = 1
            WHERE patient_id = ? AND hospital_user_id = ? AND sender_type = 'patient'
        ''', (patient_id, hospital_user_id))

        conn.commit()
        conn.close()

        return jsonify({'messages': messages})

    except Exception as e:
        logging.error(f"Error fetching messages: {e}")
        return jsonify({'error': 'Failed to fetch messages'}), 500


@app.route('/send-message', methods=['POST'])
def send_message():
    """Send a message to a patient"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    hospital_user_id = session['user_id']
    patient_id = request.form.get('recipient_id')  # Still accepting recipient_id for compatibility
    message_text = request.form.get('message')
    attachment = request.files.get('attachment')

    if not patient_id or not message_text:
        return jsonify({'error': 'Missing required fields'}), 400

    attachment_url = None

    # Handle file attachment
    if attachment and attachment.filename:
        filename = secure_filename(attachment.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(CHAT_UPLOAD_FOLDER, filename)

        attachment.save(filepath)
        attachment_url = f'/uploads/chat_attachments/{filename}'

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO messages (patient_id, hospital_user_id, sender_type, message, created_at, is_read)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 0)
        ''', (patient_id, hospital_user_id, 'hospital', message_text))

        message_id = cursor.lastrowid
        conn.commit()

        # Get the created message
        cursor.execute('''
            SELECT id, patient_id, hospital_user_id, sender_type, message, created_at, is_read
            FROM messages
            WHERE id = ?
        ''', (message_id,))

        row = cursor.fetchone()
        new_message = {
            'id': row[0],
            'patient_id': row[1],
            'hospital_user_id': row[2],
            'sender_type': row[3],
            'sender_id': row[2],  # hospital_user_id for compatibility
            'recipient_id': row[1],  # patient_id for compatibility
            'message': row[4],
            'attachment_url': attachment_url,
            'created_at': row[5],
            'read': bool(row[6])
        }

        conn.close()

        # Emit socket event
        socketio.emit('receive_message', new_message,
                      room=f'patient_{patient_id}')
        socketio.emit('receive_message', new_message,
                      room=f'hospital_{hospital_user_id}')

        return jsonify({'success': True, 'message': new_message})

    except Exception as e:
        logging.error(f"Error sending message: {e}")
        return jsonify({'error': 'Failed to send message'}), 500


# ==============================================
# PATIENT SCAN HISTORY ROUTES
# ==============================================
# Add this route to app.py - REPLACE the existing /hospital/patients POST endpoint

@app.route("/hospital/patients", methods=["POST"])
@hospital_required
def create_patient_with_email():
    """Create patient and send credentials email"""
    conn = get_db()
    c = conn.cursor()
    hospital_id = session.get("hospital_id")

    if not hospital_id:
        conn.close()
        return jsonify({"error": "Hospital ID not found in session"}), 401

    data = request.json
    logger.info(f"Creating patient with data: {data}")

    # Validate required fields
    if not data.get("full_name"):
        conn.close()
        return jsonify({"error": "full_name is required"}), 400

    try:
        # Generate codes
        patient_code = generate_unique_code('patients', 'patient_code', prefix='P')
        access_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

        # Insert patient
        c.execute("""
            INSERT INTO patients (
                hospital_id, patient_code, full_name, email, phone,
                date_of_birth, gender, address, emergency_contact, 
                emergency_phone, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            hospital_id, patient_code,
            data.get("full_name"), data.get("email"), data.get("phone"),
            data.get("date_of_birth"), data.get("gender"), data.get("address"),
            data.get("emergency_contact"), data.get("emergency_phone"),
            session["user_id"]
        ))
        patient_id = c.lastrowid

        # Create access code entry with expiration
        expires_at = datetime.now() + timedelta(days=365)  # 1 year validity
        c.execute("""
            INSERT INTO patient_access_codes (patient_id, access_code, expires_at)
            VALUES (?, ?, ?)
        """, (patient_id, access_code, expires_at))

        # Get hospital info for email
        c.execute("SELECT hospital_name, hospital_code FROM hospitals WHERE id=?", (hospital_id,))
        hospital = c.fetchone()

        # Get patient data
        c.execute("SELECT * FROM patients WHERE id=?", (patient_id,))
        patient = dict(c.fetchone())
        patient['scan_count'] = 0

        conn.commit()
        conn.close()

        # Send email
        email_sent = False
        if data.get("email"):
            try:
                from email_utilis import send_patient_credentials_email

                email_sent = send_patient_credentials_email(
                    to_email=data.get("email"),
                    patient_name=data.get("full_name"),
                    hospital_name=hospital["hospital_name"],
                    hospital_code=hospital["hospital_code"],
                    patient_code=patient_code,
                    access_code=access_code
                )

                if email_sent:
                    logger.info(f"✅ Welcome email sent to {data.get('email')}")
            except Exception as e:
                logger.error(f"❌ Email error: {e}")

        log_activity("hospital", session["user_id"], "create_patient", hospital_id=hospital_id)

        return jsonify({
            "message": "Patient created successfully",
            "patient": patient,
            "patient_id": patient_id,
            "patient_code": patient_code,
            "access_code": access_code,
            "email_sent": email_sent
        }), 201

    except Exception as e:
        conn.rollback()
        conn.close()
        logger.error(f"❌ Patient creation error: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/hospital/patient-scans/<int:patient_id>', methods=['GET'])
def get_patient_scans(patient_id):
    """Get all scans for a specific patient"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, patient_id, prediction, confidence, is_tumor, 
                   probabilities, created_at, notes
            FROM scans
            WHERE patient_id = ?
            ORDER BY created_at DESC
        ''', (patient_id,))

        scans = []
        for row in cursor.fetchall():
            probabilities = json.loads(row[5]) if row[5] else {}
            scans.append({
                'id': row[0],
                'patient_id': row[1],
                'prediction': row[2],
                'confidence': row[3],
                'is_tumor': bool(row[4]),
                'probabilities': probabilities,
                'created_at': row[6],
                'notes': row[7]
            })

        conn.close()
        return jsonify({'scans': scans})

    except Exception as e:
        logging.error(f"Error fetching patient scans: {e}")
        return jsonify({'error': 'Failed to fetch scans'}), 500


# ==============================================
# SOCKET.IO EVENT HANDLERS
# ==============================================

@socketio.on('join_room')
def handle_join_room(data):
    """Join a socket.io room for real-time communication"""
    room = data.get('room')
    if room:
        join_room(room)
        emit('joined', {'room': room}, room=room)


@socketio.on('leave_room')
def handle_leave_room(data):
    """Leave a socket.io room"""
    room = data.get('room')
    if room:
        leave_room(room)
        emit('left', {'room': room}, room=room)


@socketio.on('send_message')
def handle_socket_message(data):
    """Handle real-time message sending"""
    room = data.get('room')
    message = data.get('message')

    if room and message:
        emit('receive_message', message, room=room)


@socketio.on('send_notification')
def handle_socket_notification(data):
    """Handle real-time notifications"""
    recipient_id = data.get('recipient_id')
    notification = data.get('notification')

    if recipient_id and notification:
        emit('new_notification', notification, room=f'user_{recipient_id}')


# ==============================================
# DATABASE SCHEMA UPDATES
# ==============================================

def update_database_schema():
    """
    Add new tables for enhanced features
    Call this function once to update your database
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Notifications table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            scan_id INTEGER,
            read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (scan_id) REFERENCES scans(id)
        )
    ''')

    # Messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER NOT NULL,
            recipient_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            attachment_url TEXT,
            read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sender_id) REFERENCES users(id),
            FOREIGN KEY (recipient_id) REFERENCES users(id)
        )
    ''')

    # Add gradcam_path column to scans table if it doesn't exist
    cursor.execute("PRAGMA table_info(scans)")
    columns = [column[1] for column in cursor.fetchall()]

    if 'gradcam_path' not in columns:
        cursor.execute('''
            ALTER TABLE scans
            ADD COLUMN gradcam_path TEXT
        ''')

    if 'notes' not in columns:
        cursor.execute('''
            ALTER TABLE scans
            ADD COLUMN notes TEXT
        ''')

    conn.commit()
    conn.close()

    print("Database schema updated successfully!")


# Modified predict route to include Grad-CAM generation
@app.route('/hospital/predict', methods=['POST'])
def hospital_predict_enhanced(hospital_id=None):
    """Enhanced prediction with Grad-CAM support"""
    if 'user_id' not in session or session.get('user_type') != 'hospital':
        return jsonify({'error': 'Unauthorized'}), 403

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    patient_id = request.form.get('patient_id')
    generate_gradcam = request.form.get('generate_gradcam', 'false').lower() == 'true'

    if not patient_id:
        return jsonify({'error': 'Patient ID required'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Load and preprocess image
        image = Image.open(filepath).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()

        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        prediction = class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item() * 100
        is_tumor = prediction != 'notumor'

        probs_dict = {
            'glioma': probabilities[0].item() * 100,
            'meningioma': probabilities[1].item() * 100,
            'notumor': probabilities[2].item() * 100,
            'pituitary': probabilities[3].item() * 100
        }

        # Generate Grad-CAM if requested
        gradcam_path = None
        if generate_gradcam and ENABLE_GRADCAM:
            try:
                from gradcam_utils import generate_gradcam_from_tensor

                # ENSURE MODEL IS IN EVAL MODE
                model.eval()

                gradcam_img, _ = generate_gradcam_from_tensor(
                    model, input_tensor, image, target_class=predicted_idx
                )

                # SAVE GRADCAM IMAGE
                gradcam_filename = f"gradcam_{timestamp}_{filename}"
                gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
                Image.fromarray(gradcam_img).save(gradcam_path)

                logger.info(f"✅ GradCAM saved: {gradcam_path}")
            except Exception as e:
                logger.error(f"❌ Grad-CAM generation failed: {e}")
                gradcam_path = None  # Continue without GradCAM

        # Get hospital user_id from session
        hospital_user_id = session.get('user_id')

        # Save to database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
    INSERT INTO scans (
        patient_id, prediction, confidence, is_tumor, 
        probabilities, image_path, gradcam_path, uploaded_by, hospital_id, scan_date, created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
''', (
            patient_id, prediction, confidence, is_tumor,
            json.dumps(probs_dict), filepath, gradcam_path, hospital_user_id, 1
        ))

        scan_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Create notification using the simpler function
        create_notification_full(
            patient_id,
            'alert' if is_tumor else 'success',
            f'New scan analysis completed: {prediction} ({confidence:.1f}% confidence)',
            scan_id
        )

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'is_tumor': is_tumor,
            'probabilities': probs_dict,
            'scan_id': scan_id,
            'gradcam_available': gradcam_path is not None
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


# ==============================================
# CHAT FILE UPLOAD ENDPOINTS
# ==============================================

@app.route('/api/chat/upload', methods=['POST'])
@login_required
def upload_chat_file():
    """Upload file/image to chat"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        patient_id = request.form.get('patient_id')
        hospital_user_id = request.form.get('hospital_user_id')
        message_text = request.form.get('message', '').strip()

        if not patient_id or not hospital_user_id:
            return jsonify({'error': 'Missing chat parameters'}), 400

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size > MAX_CHAT_FILE_SIZE:
            return jsonify({'error': 'File too large (max 10MB)'}), 400

        # Check file extension
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

        if file_ext not in ALLOWED_CHAT_FILES:
            return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_CHAT_FILES)}'}), 400

        # Read file data
        file_data = file.read()

        # Determine sender
        sender_type = 'patient' if 'patient_id' in session else 'hospital'
        sender_id = session.get('patient_id') or session.get('user_id')

        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = 'application/octet-stream'

        # Save to database
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            INSERT INTO messages (
                patient_id, hospital_user_id, sender_type, message,
                attachment, attachment_name, attachment_type, attachment_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id, hospital_user_id, sender_type,
            message_text if message_text else None,
            file_data, filename, mime_type, file_size
        ))

        message_id = c.lastrowid
        timestamp = datetime.now().isoformat()
        conn.commit()
        conn.close()

        # Prepare response data
        room = f"chat_{patient_id}_{hospital_user_id}"

        message_data = {
            'id': message_id,
            'patient_id': int(patient_id),
            'hospital_user_id': int(hospital_user_id),
            'sender_type': sender_type,
            'sender_id': sender_id,
            'message': message_text if message_text else None,
            'attachment': {
                'name': filename,
                'type': mime_type,
                'size': file_size,
                'url': f'/api/chat/attachment/{message_id}'
            },
            'created_at': timestamp,
            'is_read': False
        }

        # Emit to room
        socketio.emit('new_message', message_data, room=room)

        # Send notification
        recipient_type = 'hospital' if sender_type == 'patient' else 'patient'
        recipient_id = int(hospital_user_id) if sender_type == 'patient' else int(patient_id)

        conn = get_db()
        c = conn.cursor()
        if sender_type == 'patient':
            c.execute("SELECT full_name FROM patients WHERE id=?", (sender_id,))
            sender = c.fetchone()
            sender_name = sender['full_name'] if sender else 'Patient'
        else:
            c.execute("SELECT full_name FROM hospital_users WHERE id=?", (sender_id,))
            sender = c.fetchone()
            sender_name = sender['full_name'] if sender else 'Doctor'
        conn.close()

        notif_message = f"Sent you a file: {filename}"
        if message_text:
            notif_message = f"{message_text[:50]}... (with file: {filename})"

        create_notification(
            user_id=recipient_id,
            user_type=recipient_type,
            notification_type='new_message',
            title=f'📎 {sender_name} sent a file',
            message=notif_message,
            patient_id=int(patient_id),
            priority='normal',
            action_url=f'/chat?patient_id={patient_id}&hospital_user_id={hospital_user_id}'
        )

        log_activity(sender_type, sender_id, 'send_chat_file', f"Sent {filename}")

        return jsonify({
            'message': 'File uploaded successfully',
            'data': message_data
        })

    except Exception as e:
        logger.error(f"Chat file upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/attachment/<int:message_id>', methods=['GET'])
@login_required
def get_chat_attachment(message_id):
    """Get chat attachment"""
    try:
        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT attachment, attachment_name, attachment_type
            FROM messages
            WHERE id = ?
        """, (message_id,))

        result = c.fetchone()
        conn.close()

        if not result or not result['attachment']:
            return jsonify({'error': 'Attachment not found'}), 404

        # Verify access (user must be part of the conversation)
        # Add authorization check here if needed

        return send_file(
            io.BytesIO(result['attachment']),
            mimetype=result['attachment_type'],
            as_attachment=True,
            download_name=result['attachment_name']
        )

    except Exception as e:
        logger.error(f"Error retrieving attachment: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/messages/<int:patient_id>/<int:hospital_user_id>', methods=['GET'])
@login_required
def get_chat_history(patient_id, hospital_user_id):
    """Get chat history with pagination"""
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        # Verify access
        user_type = session.get('user_type') or 'patient'
        user_id = session.get('user_id') or session.get('patient_id')

        if user_type == 'patient' and user_id != patient_id:
            return jsonify({'error': 'Unauthorized'}), 403

        if user_type == 'hospital' and user_id != hospital_user_id:
            return jsonify({'error': 'Unauthorized'}), 403

        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT id, patient_id, hospital_user_id, sender_type, message,
                   attachment_name, attachment_type, attachment_size,
                   is_read, created_at, read_at
            FROM messages
            WHERE patient_id = ? AND hospital_user_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (patient_id, hospital_user_id, limit, offset))

        messages = []
        for row in c.fetchall():
            msg = dict(row)
            if msg['attachment_name']:
                msg['attachment'] = {
                    'name': msg['attachment_name'],
                    'type': msg['attachment_type'],
                    'size': msg['attachment_size'],
                    'url': f'/api/chat/attachment/{msg["id"]}'
                }
            else:
                msg['attachment'] = None

            del msg['attachment_name']
            del msg['attachment_type']
            del msg['attachment_size']

            messages.append(msg)

        # Get unread count
        c.execute("""
            SELECT COUNT(*) as count
            FROM messages
            WHERE patient_id = ? AND hospital_user_id = ?
            AND is_read = 0
            AND sender_type = ?
        """, (
            patient_id,
            hospital_user_id,
            'hospital' if user_type == 'patient' else 'patient'
        ))

        unread_count = c.fetchone()['count']

        conn.close()

        # Reverse to show oldest first
        messages.reverse()

        return jsonify({
            'messages': messages,
            'unread_count': unread_count,
            'has_more': len(messages) == limit
        })

    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/mark-read', methods=['POST'])
@login_required
def mark_chat_read():
    """Mark chat messages as read"""
    try:
        data = request.json
        patient_id = data.get('patient_id')
        hospital_user_id = data.get('hospital_user_id')

        if not patient_id or not hospital_user_id:
            return jsonify({'error': 'Missing parameters'}), 400

        reader_type = 'patient' if 'patient_id' in session else 'hospital'
        mark_messages_read(patient_id, hospital_user_id, reader_type)

        return jsonify({'message': 'Messages marked as read'})

    except Exception as e:
        logger.error(f"Error marking messages read: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/conversations', methods=['GET'])
@login_required
def get_user_conversations():
    """Get list of conversations for current user"""
    try:
        user_type = session.get('user_type') or 'patient'
        user_id = session.get('user_id') or session.get('patient_id')

        conn = get_db()
        c = conn.cursor()

        if user_type == 'patient':
            # Get all doctors patient has chatted with
            c.execute("""
                SELECT DISTINCT
                    m.hospital_user_id,
                    hu.full_name as doctor_name,
                    hu.email as doctor_email,
                    (SELECT message FROM messages m2 
                     WHERE m2.patient_id = m.patient_id 
                     AND m2.hospital_user_id = m.hospital_user_id 
                     ORDER BY m2.created_at DESC LIMIT 1) as last_message,
                    (SELECT created_at FROM messages m2 
                     WHERE m2.patient_id = m.patient_id 
                     AND m2.hospital_user_id = m.hospital_user_id 
                     ORDER BY m2.created_at DESC LIMIT 1) as last_message_time,
                    (SELECT COUNT(*) FROM messages m2 
                     WHERE m2.patient_id = m.patient_id 
                     AND m2.hospital_user_id = m.hospital_user_id 
                     AND m2.is_read = 0 
                     AND m2.sender_type = 'hospital') as unread_count
                FROM messages m
                JOIN hospital_users hu ON m.hospital_user_id = hu.id
                WHERE m.patient_id = ?
                ORDER BY last_message_time DESC
            """, (user_id,))
        else:
            # Get all patients doctor has chatted with
            c.execute("""
                SELECT DISTINCT
                    m.patient_id,
                    p.full_name as patient_name,
                    p.patient_code,
                    p.email as patient_email,
                    (SELECT message FROM messages m2 
                     WHERE m2.patient_id = m.patient_id 
                     AND m2.hospital_user_id = m.hospital_user_id 
                     ORDER BY m2.created_at DESC LIMIT 1) as last_message,
                    (SELECT created_at FROM messages m2 
                     WHERE m2.patient_id = m.patient_id 
                     AND m2.hospital_user_id = m.hospital_user_id 
                     ORDER BY m2.created_at DESC LIMIT 1) as last_message_time,
                    (SELECT COUNT(*) FROM messages m2 
                     WHERE m2.patient_id = m.patient_id 
                     AND m2.hospital_user_id = m.hospital_user_id 
                     AND m2.is_read = 0 
                     AND m2.sender_type = 'patient') as unread_count
                FROM messages m
                JOIN patients p ON m.patient_id = p.id
                WHERE m.hospital_user_id = ?
                ORDER BY last_message_time DESC
            """, (user_id,))

        conversations = [dict(row) for row in c.fetchall()]
        conn.close()

        return jsonify({'conversations': conversations})

    except Exception as e:
        logger.error(f"Error loading conversations: {e}")
        return jsonify({'error': str(e)}), 500


# ==============================================
# NOTIFICATION HELPER FUNCTIONS
# ==============================================

def create_notification(
        user_id: int,
        user_type: str,
        notification_type: str,
        title: str,
        message: str,
        hospital_id: Optional[int] = None,
        prediction_id: Optional[int] = None,
        scan_id: Optional[int] = None,
        patient_id: Optional[int] = None,
        priority: str = 'normal',
        action_url: Optional[str] = None
) -> int:
    """
    Create a new notification

    Args:
        user_id: ID of the user to notify
        user_type: Type of user ('admin', 'hospital', 'patient')
        notification_type: Notification type ('prediction_complete', 'low_confidence', 'error', 'info', 'warning', 'subscription')
        title: Notification title
        message: Notification message
        hospital_id: Optional hospital ID
        prediction_id: Optional prediction/scan ID
        scan_id: Optional scan ID
        patient_id: Optional patient ID
        priority: 'low', 'normal', 'high', 'urgent'
        action_url: Optional URL for action button

    Returns:
        notification_id: ID of created notification
    """
    try:
        conn = get_db()
        c = conn.cursor()

        c.execute("""
            INSERT INTO notifications (
                user_id, user_type, hospital_id, type, title, message,
                prediction_id, scan_id, patient_id, priority, action_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, user_type, hospital_id, notification_type, title, message,
            prediction_id, scan_id, patient_id, priority, action_url
        ))

        notification_id = c.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"✅ Notification created: ID={notification_id}, User={user_id}, Type={notification_type}")
        return notification_id

    except Exception as e:
        logger.error(f"❌ Error creating notification: {e}")
        return -1


def bulk_create_notifications(notifications: List[Dict]) -> int:
    """
    Create multiple notifications at once

    Args:
        notifications: List of notification dicts with required fields

    Returns:
        count: Number of notifications created
    """
    try:
        conn = get_db()
        c = conn.cursor()

        for notif in notifications:
            c.execute("""
                INSERT INTO notifications (
                    user_id, user_type, hospital_id, type, title, message,
                    prediction_id, scan_id, patient_id, priority, action_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                notif.get('user_id'),
                notif.get('user_type'),
                notif.get('hospital_id'),
                notif.get('type'),
                notif.get('title'),
                notif.get('message'),
                notif.get('prediction_id'),
                notif.get('scan_id'),
                notif.get('patient_id'),
                notif.get('priority', 'normal'),
                notif.get('action_url')
            ))

        count = len(notifications)
        conn.commit()
        conn.close()

        logger.info(f"✅ Bulk created {count} notifications")
        return count

    except Exception as e:
        logger.error(f"❌ Error bulk creating notifications: {e}")
        return 0


def notify_admins(notification_type: str, title: str, message: str, priority: str = 'normal'):
    """Notify all admin users"""
    try:
        conn = get_db()
        c = conn.cursor()

        c.execute("SELECT id FROM users WHERE role IN ('admin', 'superadmin')")
        admins = c.fetchall()
        conn.close()

        for admin in admins:
            create_notification(
                user_id=admin['id'],
                user_type='admin',
                notification_type=notification_type,
                title=title,
                message=message,
                priority=priority
            )

        logger.info(f"✅ Notified {len(admins)} admins")

    except Exception as e:
        logger.error(f"❌ Error notifying admins: {e}")


def notify_hospital_users(hospital_id: int, notification_type: str, title: str, message: str,
                          priority: str = 'normal', exclude_user_id: Optional[int] = None):
    """Notify all users in a hospital"""
    try:
        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT id FROM hospital_users 
            WHERE hospital_id=? AND status='active'
        """, (hospital_id,))

        users = c.fetchall()
        conn.close()

        for user in users:
            if exclude_user_id and user['id'] == exclude_user_id:
                continue

            create_notification(
                user_id=user['id'],
                user_type='hospital',
                notification_type=notification_type,
                title=title,
                message=message,
                hospital_id=hospital_id,
                priority=priority
            )

        logger.info(f"✅ Notified {len(users)} hospital users")

    except Exception as e:
        logger.error(f"❌ Error notifying hospital users: {e}")


# ==============================================
# NOTIFICATION API ENDPOINTS
# ==============================================

@app.route('/api/notifications/<int:notification_id>/read', methods=['PUT'])
@login_required
def mark_notification_read(notification_id):
    """Mark a notification as read"""
    try:
        user_id = session.get('user_id') or session.get('patient_id')
        conn = get_db()
        c = conn.cursor()

        c.execute("SELECT id FROM notifications WHERE id=? AND user_id=?",
                  (notification_id, user_id))

        if not c.fetchone():
            conn.close()
            return jsonify({'error': 'Notification not found'}), 404

        c.execute("UPDATE notifications SET is_read=1, read_at=CURRENT_TIMESTAMP WHERE id=?",
                  (notification_id,))

        conn.commit()
        conn.close()
        return jsonify({'message': 'Notification marked as read'})
    except Exception as e:
        logger.error(f"Error marking notification read: {e}")
        return jsonify({'error': str(e)}), 500


@login_required
def mark_all_notifications_read():
    """Mark all notifications as read for current user"""
    try:
        user_id = session.get('user_id') or session.get('patient_id')
        user_type = session.get('user_type') or 'patient'

        conn = get_db()
        c = conn.cursor()

        c.execute("""
            UPDATE notifications
            SET is_read=1, read_at=CURRENT_TIMESTAMP
            WHERE user_id=? AND user_type=? AND is_read=0
        """, (user_id, user_type))

        count = c.rowcount
        conn.commit()
        conn.close()

        return jsonify({
            'message': f'Marked {count} notifications as read',
            'count': count
        })

    except Exception as e:
        logger.error(f"❌ Error marking all read: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/notifications/<int:notification_id>', methods=['DELETE'])
@login_required
def delete_notification(notification_id):
    """Delete a notification"""
    try:
        user_id = session.get('user_id') or session.get('patient_id')

        conn = get_db()
        c = conn.cursor()

        # Verify ownership
        c.execute("""
            SELECT id FROM notifications
            WHERE id=? AND user_id=?
        """, (notification_id, user_id))

        if not c.fetchone():
            conn.close()
            return jsonify({'error': 'Notification not found'}), 404

        # Delete
        c.execute("DELETE FROM notifications WHERE id=?", (notification_id,))

        conn.commit()
        conn.close()

        return jsonify({'message': 'Notification deleted'})

    except Exception as e:
        logger.error(f"❌ Error deleting notification: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/notifications/clear-all', methods=['DELETE'])
@login_required
def clear_all_notifications():
    """Delete all notifications for current user"""
    try:
        user_id = session.get('user_id') or session.get('patient_id')
        user_type = session.get('user_type') or 'patient'

        conn = get_db()
        c = conn.cursor()

        c.execute("""
            DELETE FROM notifications
            WHERE user_id=? AND user_type=?
        """, (user_id, user_type))

        count = c.rowcount
        conn.commit()
        conn.close()

        return jsonify({
            'message': f'Deleted {count} notifications',
            'count': count
        })

    except Exception as e:
        logger.error(f"❌ Error clearing notifications: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/notifications/unread-count', methods=['GET'])
@login_required
def get_unread_count():
    """Get unread notification count"""
    try:
        user_id = session.get('user_id') or session.get('patient_id')
        user_type = session.get('user_type') or 'patient'

        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT COUNT(*) as count FROM notifications
            WHERE user_id=? AND user_type=? AND is_read=0
        """, (user_id, user_type))

        count = c.fetchone()['count']
        conn.close()

        return jsonify({'unread_count': count})

    except Exception as e:
        logger.error(f"❌ Error getting unread count: {e}")
        return jsonify({'error': str(e)}), 500


# ==============================================
# CHAT API ENDPOINTS
# ==============================================

@app.route('/api/chat/messages', methods=['GET'])
@login_required
def get_chat_messages():
    """Get chat messages for a patient-doctor conversation"""
    try:
        patient_id = request.args.get('patient_id', type=int)
        if not patient_id:
            return jsonify({'error': 'Patient ID required'}), 400

        user_type = session.get('user_type') or 'patient'
        if user_type == 'patient':
            if session.get('patient_id') != patient_id:
                return jsonify({'error': 'Unauthorized'}), 403
            hospital_user_id = None  # Patient view, get all with their doctor
        else:
            hospital_user_id = session.get('user_id')

        conn = get_db()
        c = conn.cursor()

        query = """
            SELECT * FROM messages
            WHERE patient_id = ?
        """
        params = [patient_id]

        if hospital_user_id:
            query += " AND hospital_user_id = ?"
            params.append(hospital_user_id)

        query += " ORDER BY created_at ASC"
        c.execute(query, params)
        messages = [dict(row) for row in c.fetchall()]

        conn.close()

        return jsonify({'messages': messages})

    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        return jsonify({'error': str(e)}), 500


# ==============================================
# INTEGRATE WITH EXISTING ROUTES
# ==============================================

# MODIFY YOUR EXISTING /hospital/predict ROUTE
# Replace the existing predict function with this updated version:

def check_scan_limit(args):
    pass


# ==============================================
# CORRECTED APP UPDATES - NO ERRORS VERSION
# Add these endpoints to your app.py
# ==============================================

# ==============================================
# 1. NOTIFICATION ENDPOINTS
# ==============================================

@app.route('/notifications/mark-all-read', methods=['POST'])
@login_required
def mark_all_notifications_read():
    """Mark all notifications as read for the current user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user_id = session['user_id']
    user_type = session.get('user_type', 'hospital')

    try:
        conn = get_db()
        c = conn.cursor()

        c.execute('''
            UPDATE notifications 
            SET is_read = 1
            WHERE user_id = ? AND user_type = ? AND is_read = 0
        ''', (user_id, user_type))

        affected = c.rowcount
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'marked_count': affected,
            'message': f'{affected} notifications marked as read'
        })

    except Exception as e:
        logger.error(f"Error marking all notifications as read: {e}")
        return jsonify({'error': 'Failed to update notifications'}), 500


@app.route('/notifications/clear-all', methods=['POST'])
@login_required
def clear_all_read_notifications():
    """Delete all read notifications for the current user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user_id = session['user_id']
    user_type = session.get('user_type', 'hospital')

    try:
        conn = get_db()
        c = conn.cursor()

        c.execute('''
            DELETE FROM notifications 
            WHERE user_id = ? AND user_type = ? AND is_read = 1
        ''', (user_id, user_type))

        deleted = c.rowcount
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'deleted_count': deleted,
            'message': f'{deleted} notifications cleared'
        })

    except Exception as e:
        logger.error(f"Error clearing notifications: {e}")
        return jsonify({'error': 'Failed to clear notifications'}), 500


# ==============================================
# 3. GRAD-CAM VISUALIZATION
# ==============================================

@app.route('/generate-gradcam/<int:scan_id>', methods=['GET'])
@login_required
def generate_gradcam_route(scan_id):
    """Generate GradCAM heatmap visualization for a scan"""
    try:
        user_type = session.get('user_type')
        user_id = session.get('user_id') or session.get('patient_id')

        # Get scan data
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            SELECT scan_image, prediction, hospital_id, patient_id, probabilities
            FROM mri_scans WHERE id = ?
        """, (scan_id,))

        scan = c.fetchone()
        conn.close()

        if not scan:
            return jsonify({"error": "Scan not found"}), 404

        # Check authorization
        if user_type == "hospital":
            if scan["hospital_id"] != session.get("hospital_id"):
                return jsonify({"error": "Unauthorized"}), 403
        elif user_type == "patient":
            if scan["patient_id"] != session.get("patient_id"):
                return jsonify({"error": "Unauthorized"}), 403

        # Decode base64 image
        try:
            image_data = base64.b64decode(scan["scan_image"])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            logger.error(f"Image decode error: {e}")
            return jsonify({"error": "Failed to decode scan image"}), 500

        # Prepare input tensor
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Import GradCAM utility
        try:
            from gradcam_utils import generate_gradcam_from_tensor
        except ImportError:
            logger.error("gradcam_utils.py not found")
            return jsonify({"error": "GradCAM module not available"}), 500

        # Generate GradCAM
        try:
            model.eval()  # Ensure model is in eval mode

            overlaid_image, prediction_idx = generate_gradcam_from_tensor(
                model=model,
                input_tensor=input_tensor,
                original_image=image,
                target_class=None  # Use predicted class
            )

            # Convert to bytes
            buffer = io.BytesIO()
            Image.fromarray(overlaid_image).save(buffer, format='PNG')
            buffer.seek(0)

            return send_file(
                buffer,
                mimetype='image/png',
                as_attachment=False,
                download_name=f'gradcam_scan_{scan_id}.png'
            )

        except Exception as e:
            logger.error(f"GradCAM generation error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Failed to generate GradCAM: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"GradCAM route error: {e}")
        return jsonify({"error": str(e)}), 500



# ==============================================
# 4. ENHANCED PREDICTION WITH USAGE TRACKING
# ==============================================

@app.route("/hospital/predict", methods=["POST"])
@login_required
def predict_with_usage():
    """Enhanced prediction with usage tracking and warnings"""

    hospital_id = session.get("hospital_id")
    user_id = session.get("user_id")

    if not hospital_id:
        logger.error("❌ hospital_id not found in session")
        return jsonify({
            "error": "Session error: Hospital ID not found. Please log in again."
        }), 401

    try:
        # Check usage limits BEFORE prediction
        usage_info = get_detailed_usage(hospital_id)

        if usage_info['is_blocked']:
            return jsonify({
                "error": "Usage limit reached",
                "usage": usage_info,
                "upgrade_required": True,
                "message": usage_info['block_message']
            }), 403

        # Validate image upload
        if "image" not in request.files:
            return jsonify({"error": "No image"}), 400

        patient_id = request.form.get("patient_id")
        if not patient_id:
            return jsonify({"error": "Patient ID required"}), 400

        # Read and process image
        image_bytes = request.files["image"].read()
        image_stream = io.BytesIO(image_bytes)
        image_stream.seek(0)
        image = Image.open(image_stream).convert("RGB")

        # Validate image (if you have validation function)
        logger.info("🔍 Processing uploaded image...")

        # Process image and run prediction
        image_stream.seek(0)
        image = Image.open(image_stream).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        logger.info("🧠 Running model prediction...")
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)[0]
            conf, pred = torch.max(probs, 0)

        pred_idx = int(pred.item())
        conf_val = float(conf.item())
        prediction_label = class_names[pred_idx]
        is_tumor = prediction_label != "notumor"

        # Create probability dictionary
        probabilities = {
            class_names[i]: round(float(probs[i].item()) * 100, 2)
            for i in range(len(class_names))
        }

        confidence_percent = round(conf_val * 100, 2)
        logger.info(f"📊 Prediction: {prediction_label} ({confidence_percent}%)")

        # Save to database
        conn = get_db()
        c = conn.cursor()

        c.execute("""
            INSERT INTO mri_scans (
                hospital_id, patient_id, uploaded_by, scan_image,
                prediction, confidence, is_tumor, probabilities,
                notes, scan_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            hospital_id,
            patient_id,
            user_id,
            base64.b64encode(image_bytes).decode(),
            prediction_label,
            conf_val,
            is_tumor,
            json.dumps(probabilities),
            request.form.get("notes", ""),
            request.form.get("scan_date", datetime.now().strftime("%Y-%m-%d"))
        ))
        scan_id = c.lastrowid

        # Get patient info
        c.execute("SELECT full_name, patient_code FROM patients WHERE id=?", (patient_id,))
        patient = c.fetchone()
        patient_code = patient["patient_code"] if patient else "Unknown"

        conn.commit()
        conn.close()

        # Update usage and get new stats
        increment_usage(hospital_id, 'scans', 1)
        updated_usage = get_detailed_usage(hospital_id)

        # Create notification
        if is_tumor:
            if confidence_percent < 70:
                notif_title = '⚠️ Low Confidence Detection'
                notif_message = f'{prediction_label.capitalize()} detected with {confidence_percent}% confidence for patient {patient_code}. Manual review recommended.'
                notif_priority = 'high'
            else:
                notif_title = '🔴 Tumor Detected'
                notif_message = f'{prediction_label.capitalize()} detected with {confidence_percent}% confidence for patient {patient_code}.'
                notif_priority = 'high'
        else:
            notif_title = '✅ Scan Analysis Complete'
            notif_message = f'No tumor detected for patient {patient_code} with {confidence_percent}% confidence.'
            notif_priority = 'normal'

        try:
            create_notification(
                user_id=user_id,
                user_type='hospital',
                notification_type='scan_result',
                title=notif_title,
                message=notif_message,
                hospital_id=hospital_id,
                scan_id=scan_id,
                patient_id=int(patient_id),
                priority=notif_priority,
                action_url=f'/scan/{scan_id}'
            )
        except Exception as notif_error:
            logger.error(f"Failed to create notification: {notif_error}")

        log_activity("hospital", user_id, "prediction", hospital_id=hospital_id)

        # Calculate usage percentage for warning
        usage_percentage = 0
        usage_warning = None

        if updated_usage['max_scans'] != 'unlimited' and updated_usage['max_scans'] > 0:
            usage_percentage = (updated_usage['scans_used'] / updated_usage['max_scans']) * 100

            if usage_percentage >= 90:
                usage_warning = {
                    "level": "critical",
                    "message": f"⚠️ You've used {usage_percentage:.0f}% of your monthly scan limit ({updated_usage['scans_used']}/{updated_usage['max_scans']}). Consider upgrading your plan.",
                    "scans_remaining": updated_usage['max_scans'] - updated_usage['scans_used']
                }
            elif usage_percentage >= 75:
                usage_warning = {
                    "level": "warning",
                    "message": f"You've used {usage_percentage:.0f}% of your monthly scan limit ({updated_usage['scans_used']}/{updated_usage['max_scans']}).",
                    "scans_remaining": updated_usage['max_scans'] - updated_usage['scans_used']
                }

        return jsonify({
            "scan_id": scan_id,
            "prediction": prediction_label,
            "confidence": confidence_percent,
            "is_tumor": is_tumor,
            "probabilities": probabilities,
            "usage": {
                "scans_used": updated_usage['scans_used'],
                "max_scans": updated_usage['max_scans'],
                "percentage": round(usage_percentage, 1) if usage_percentage else None,
                "warning": usage_warning
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


# ==============================================
# PDF GENERATION ENDPOINT
# ==============================================

@app.route('/api/generate-pdf', methods=['POST'])
@login_required
def generate_pdf_endpoint():
    """Generate PDF report for brain tumor analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['file']
        prediction = request.form.get('prediction', 'Unknown')
        confidence = request.form.get('confidence', '0')
        patient_id = request.form.get('patient_id', 'unknown')
        patient_name = request.form.get('patient_name', 'Anonymous Patient')
        probabilities = request.form.get('probabilities', '{}')
        scan_id = request.form.get('scan_id', '')

        # Parse probabilities
        try:
            probs = json.loads(probabilities)
        except:
            probs = {}

        # Ensure confidence doesn't exceed 100
        conf_value = float(confidence)
        if conf_value > 100:
            conf_value = 100

        # Get hospital info from session
        hospital_id = session.get('hospital_id')

        # Prepare scan data
        scan_data = {
            'prediction': prediction,
            'confidence': conf_value,
            'probabilities': probs,
            'is_tumor': prediction.lower() != 'notumor',
            'scan_id': scan_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Prepare patient data
        patient_data = {
            'name': patient_name,
            'id': patient_id,
            'date_analyzed': datetime.now().strftime('%Y-%m-%d')
        }

        # Prepare hospital data
        hospital_data = {}
        if hospital_id:
            conn = get_db()
            c = conn.cursor()
            c.execute("SELECT name, address, phone FROM hospitals WHERE id=?", (hospital_id,))
            hospital_row = c.fetchone()
            conn.close()
            if hospital_row:
                hospital_data = {
                    'name': hospital_row['name'],
                    'address': hospital_row['address'],
                    'phone': hospital_row['phone']
                }

        # Generate PDF
        pdf_buffer = generate_pdf_report(scan_data, patient_data, hospital_data)
        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'NeuroScan_Report_{patient_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )

    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500


# ==============================================
# 5. SUBSCRIPTION LIMITS ENDPOINT (for FeatureGate)
# ==============================================


@app.route('/hospital/subscription-limits', methods=['GET'])
@login_required
def get_subscription_limits():
    """Get subscription limits for the current hospital"""
    hospital_id = session.get('hospital_id')

    if not hospital_id:
        return jsonify({'error': 'Not authorized'}), 403

    try:
        usage_info = get_detailed_usage(hospital_id)

        return jsonify({
            'plan_name': usage_info['plan_name'],
            'scans_used': usage_info['scans_used'],
            'max_scans': usage_info['max_scans'],
            'users_count': usage_info['users_count'],
            'max_users': usage_info['max_users'],
            'patients_count': usage_info['patients_count'],
            'max_patients': usage_info['max_patients']
        })

    except Exception as e:
        logger.error(f"Error getting subscription limits: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin/dashboard', methods=['GET'])
@login_required
@admin_required
def admin_dashboard():
    """Get admin dashboard statistics"""
    try:
        conn = get_db()
        c = conn.cursor()

        # Get total counts
        c.execute("SELECT COUNT(*) FROM users WHERE role IN ('admin', 'superadmin')")
        total_admins = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM hospitals")
        total_hospitals = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM patients")
        total_patients = c.fetchone()[0]

        c.execute("""
            SELECT COUNT(*) FROM hospital_subscriptions 
            WHERE status = 'active'
        """)
        active_subscriptions = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM scans")
        total_scans = c.fetchone()[0]

        c.execute("""
            SELECT COUNT(*) FROM scans 
            WHERE DATE(scan_date) = DATE('now')
        """)
        scans_today = c.fetchone()[0]

        c.execute("""
            SELECT COUNT(*) FROM scans 
            WHERE strftime('%Y-%m', scan_date) = strftime('%Y-%m', 'now')
        """)
        scans_this_month = c.fetchone()[0]

        # Calculate active users (users who logged in within last 30 days)
        c.execute("""
            SELECT COUNT(DISTINCT user_id) FROM activity_logs
            WHERE created_at >= datetime('now', '-30 days')
        """)
        active_users = c.fetchone()[0]

        conn.close()

        return jsonify({
            'stats': {
                'total_admins': total_admins,
                'total_hospitals': total_hospitals,
                'total_patients': total_patients,
                'active_subscriptions': active_subscriptions,
                'total_scans': total_scans,
                'scans_today': scans_today,
                'scans_this_month': scans_this_month,
                'active_users': active_users
            }
        })

    except Exception as e:
        logger.error(f"Error loading admin dashboard: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin/subscription-plans', methods=['GET'])
@login_required
@admin_required
def get_subscription_plans():
    """Get all subscription plans"""
    try:
        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT id, name, display_name, description, 
                   price_monthly, price_yearly, 
                   max_scans_per_month, max_users, max_patients,
                   features, is_active
            FROM subscription_plans
            WHERE is_active = 1
            ORDER BY price_monthly ASC
        """)

        plans = []
        for row in c.fetchall():
            plans.append({
                'id': row[0],
                'name': row[1],
                'display_name': row[2],
                'description': row[3],
                'price_monthly': row[4],
                'price_yearly': row[5],
                'max_scans_per_month': row[6],
                'max_users': row[7],
                'max_patients': row[8],
                'features': json.loads(row[9]) if row[9] else [],
                'is_active': row[10]
            })

        conn.close()
        return jsonify({'plans': plans})

    except Exception as e:
        logger.error(f"Error loading subscription plans: {e}")
        return jsonify({'error': str(e)}), 500


# ==============================================
# ADMIN - GET ALL USERS BY TYPE
# ==============================================

@app.route('/admin/users/admins', methods=['GET'])
@login_required
@admin_required
def get_all_admins():
    """Get all admin users"""
    try:
        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT id, username, email, role, created_at
            FROM users
            WHERE role IN ('admin', 'superadmin')
            ORDER BY created_at DESC
        """)

        admins = []
        for row in c.fetchall():
            admins.append({
                'id': row[0],
                'username': row[1],
                'email': row[2],
                'role': row[3],
                'created_at': row[4]
            })

        conn.close()

        log_activity('admin', session['user_id'], 'viewed_all_admins')

        return jsonify({'admins': admins})

    except Exception as e:
        logger.error(f"Error loading admins: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin/users/hospitals', methods=['GET'])
@login_required
@admin_required
def get_all_hospitals():
    """Get all hospital accounts"""
    try:
        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT h.id, h.hospital_name, h.hospital_code, h.email,
                   h.contact_person, h.phone, h.address, h.created_at,
                   hs.status as subscription_status,
                   sp.display_name as subscription_plan
            FROM hospitals h
            LEFT JOIN hospital_subscriptions hs ON h.id = hs.hospital_id AND hs.status = 'active'
            LEFT JOIN subscription_plans sp ON hs.plan_id = sp.id
            ORDER BY h.created_at DESC
        """)

        hospitals = []
        for row in c.fetchall():
            hospitals.append({
                'id': row[0],
                'hospital_name': row[1],
                'hospital_code': row[2],
                'email': row[3],
                'contact_person': row[4],
                'phone': row[5],
                'address': row[6],
                'created_at': row[7],
                'subscription_status': row[8] or 'none',
                'subscription_plan': row[9] or 'Free'
            })

        conn.close()

        log_activity('admin', session['user_id'], 'viewed_all_hospitals')

        return jsonify({'hospitals': hospitals})

    except Exception as e:
        logger.error(f"Error loading hospitals: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin/users/patients', methods=['GET'])
@login_required
@admin_required
def get_all_patients():
    """Get all patient records"""
    try:
        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT p.id, p.full_name, p.patient_code, p.email, p.phone,
                   p.date_of_birth, p.gender, p.created_at,
                   h.hospital_name, h.hospital_code
            FROM patients p
            LEFT JOIN hospitals h ON p.hospital_id = h.id
            ORDER BY p.created_at DESC
        """)

        patients = []
        for row in c.fetchall():
            patients.append({
                'id': row[0],
                'full_name': row[1],
                'patient_code': row[2],
                'email': row[3],
                'phone': row[4],
                'date_of_birth': row[5],
                'gender': row[6],
                'created_at': row[7],
                'hospital_name': row[8],
                'hospital_code': row[9]
            })

        conn.close()

        log_activity('admin', session['user_id'], 'viewed_all_patients')

        return jsonify({'patients': patients})

    except Exception as e:
        logger.error(f"Error loading patients: {e}")
        return jsonify({'error': str(e)}), 500


# ==============================================
# ADMIN - CREATE NEW USERS
# ==============================================

@app.route('/admin/users/admin', methods=['POST'])
@login_required
@admin_required
def create_admin_user():
    """Create a new admin user"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['username', 'email', 'password', 'role']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Validate role
        if data['role'] not in ['admin', 'superadmin']:
            return jsonify({'error': 'Invalid role'}), 400

        # Only superadmin can create other superadmins
        if data['role'] == 'superadmin' and session.get('role') != 'superadmin':
            return jsonify({'error': 'Only superadmins can create other superadmins'}), 403

        conn = get_db()
        c = conn.cursor()

        # Check if username or email already exists
        c.execute('SELECT id FROM users WHERE username=? OR email=?',
                  (data['username'], data['email']))
        if c.fetchone():
            conn.close()
            return jsonify({'error': 'Username or email already exists'}), 400

        # Create admin user
        hashed_password = generate_password_hash(data['password'])
        c.execute("""
            INSERT INTO users (username, email, password, role)
            VALUES (?, ?, ?, ?)
        """, (data['username'], data['email'], hashed_password, data['role']))

        user_id = c.lastrowid
        conn.commit()
        conn.close()

        log_activity('admin', session['user_id'], 'created_admin_user',
                     f"Created {data['role']} user: {data['username']}")

        return jsonify({
            'message': f"{data['role'].capitalize()} user created successfully",
            'user_id': user_id
        })

    except Exception as e:
        logger.error(f"Error creating admin user: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin/users/hospital', methods=['POST'])
@login_required
@admin_required
def create_hospital_account():
    """Create a new hospital account"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['hospital_name', 'hospital_code', 'email', 'password']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        conn = get_db()
        c = conn.cursor()

        # Check if hospital code or email already exists
        c.execute('SELECT id FROM hospitals WHERE hospital_code=? OR email=?',
                  (data['hospital_code'], data['email']))
        if c.fetchone():
            conn.close()
            return jsonify({'error': 'Hospital code or email already exists'}), 400

        # Create hospital account
        hashed_password = generate_password_hash(data['password'])
        c.execute("""
            INSERT INTO hospitals 
            (hospital_name, hospital_code, email, password, 
             contact_person, phone, address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            data['hospital_name'],
            data['hospital_code'].upper(),
            data['email'],
            hashed_password,
            data.get('contact_person', ''),
            data.get('phone', ''),
            data.get('address', '')
        ))

        hospital_id = c.lastrowid

        # Get subscription plan
        plan_name = data.get('subscription_plan', 'free')
        c.execute('SELECT id FROM subscription_plans WHERE name=?', (plan_name,))
        plan = c.fetchone()

        if plan:
            plan_id = plan[0]

            # Create subscription
            start_date = datetime.now()
            end_date = start_date + timedelta(days=30)

            c.execute("""
                INSERT INTO hospital_subscriptions
                (hospital_id, plan_id, status, billing_cycle,
                 current_period_start, current_period_end,
                 trial_ends_at, is_trial)
                VALUES (?, ?, 'active', 'monthly', ?, ?, ?, ?)
            """, (hospital_id, plan_id, start_date.date(), end_date.date(),
                  end_date.date(), 1 if plan_name == 'free' else 0))

            subscription_id = c.lastrowid

            # Get plan limits
            c.execute("""
                SELECT max_scans_per_month, max_users, max_patients
                FROM subscription_plans WHERE id=?
            """, (plan_id,))
            limits = c.fetchone()

            # Create usage tracking
            c.execute("""
                INSERT INTO usage_tracking
                (hospital_id, subscription_id, period_start, period_end,
                 scans_used, scans_limit, users_count, users_limit,
                 patients_count, patients_limit, is_current)
                VALUES (?, ?, ?, ?, 0, ?, 0, ?, 0, ?, 1)
            """, (hospital_id, subscription_id, start_date.date(), end_date.date(),
                  limits[0], limits[1], limits[2]))

        conn.commit()
        conn.close()

        log_activity('admin', session['user_id'], 'created_hospital',
                     f"Created hospital: {data['hospital_name']}")

        return jsonify({
            'message': 'Hospital account created successfully',
            'hospital_id': hospital_id,
            'hospital_code': data['hospital_code'].upper()
        })

    except Exception as e:
        logger.error(f"Error creating hospital: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin/users/patient', methods=['POST'])
@login_required
@admin_required
def create_patient_record():
    """Create a new patient record"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['full_name', 'patient_code', 'access_code', 'email', 'hospital_id']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        conn = get_db()
        c = conn.cursor()

        # Verify hospital exists
        c.execute('SELECT id, hospital_code FROM hospitals WHERE id=?', (data['hospital_id'],))
        hospital = c.fetchone()
        if not hospital:
            conn.close()
            return jsonify({'error': 'Hospital not found'}), 404

        hospital_code = hospital[1]

        # Check if patient code already exists for this hospital
        c.execute("""
            SELECT id FROM patients 
            WHERE patient_code=? AND hospital_id=?
        """, (data['patient_code'], data['hospital_id']))
        if c.fetchone():
            conn.close()
            return jsonify({'error': 'Patient code already exists for this hospital'}), 400

        # Check if email already exists
        c.execute('SELECT id FROM patients WHERE email=?', (data['email'],))
        if c.fetchone():
            conn.close()
            return jsonify({'error': 'Email already registered'}), 400

        # Create patient record
        c.execute("""
            INSERT INTO patients
            (hospital_id, full_name, patient_code, access_code, email,
             phone, date_of_birth, gender, blood_group, 
             emergency_contact, medical_history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['hospital_id'],
            data['full_name'],
            data['patient_code'].upper(),
            data['access_code'],
            data['email'],
            data.get('phone', ''),
            data.get('date_of_birth', ''),
            data.get('gender', ''),
            data.get('blood_group', ''),
            data.get('emergency_contact', ''),
            data.get('medical_history', '')
        ))

        patient_id = c.lastrowid
        conn.commit()

        # Update usage tracking
        try:
            c.execute("""
                UPDATE usage_tracking
                SET patients_count = patients_count + 1
                WHERE hospital_id = ? AND is_current = 1
            """, (data['hospital_id'],))
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not update usage tracking: {e}")

        conn.close()

        # Send welcome email (optional)
        try:
            send_welcome_email(
                data['email'],
                data['full_name'],
                data['patient_code'],
                data['access_code'],
                'Hospital Name',  # You might want to fetch this
                hospital_code
            )
        except Exception as e:
            logger.warning(f"Could not send welcome email: {e}")

        log_activity('admin', session['user_id'], 'created_patient',
                     f"Created patient: {data['full_name']}")

        return jsonify({
            'message': 'Patient record created successfully',
            'patient_id': patient_id,
            'patient_code': data['patient_code'].upper()
        })

    except Exception as e:
        logger.error(f"Error creating patient: {e}")
        return jsonify({'error': str(e)}), 500


# ==============================================
# ADMIN - UPDATE USERS
# ==============================================

@app.route('/admin/users/admin/<int:user_id>', methods=['PUT'])
@login_required
@admin_required
def update_admin_user(user_id):
    """Update an admin user"""
    try:
        data = request.get_json()

        conn = get_db()
        c = conn.cursor()

        # Check if user exists
        c.execute('SELECT id, role FROM users WHERE id=?', (user_id,))
        user = c.fetchone()
        if not user:
            conn.close()
            return jsonify({'error': 'User not found'}), 404

        # Only superadmin can modify superadmin accounts
        if user[1] == 'superadmin' and session.get('role') != 'superadmin':
            conn.close()
            return jsonify({'error': 'Insufficient permissions'}), 403

        # Build update query
        updates = []
        params = []

        if 'email' in data and data['email']:
            updates.append('email = ?')
            params.append(data['email'])

        if 'username' in data and data['username']:
            updates.append('username = ?')
            params.append(data['username'])

        if 'role' in data and data['role'] in ['admin', 'superadmin']:
            if data['role'] == 'superadmin' and session.get('role') != 'superadmin':
                conn.close()
                return jsonify({'error': 'Only superadmins can grant superadmin role'}), 403
            updates.append('role = ?')
            params.append(data['role'])

        if 'password' in data and data['password']:
            updates.append('password = ?')
            params.append(generate_password_hash(data['password']))

        if not updates:
            conn.close()
            return jsonify({'error': 'No valid fields to update'}), 400

        # Perform update
        query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
        params.append(user_id)

        c.execute(query, params)
        conn.commit()
        conn.close()

        log_activity('admin', session['user_id'], 'updated_admin_user',
                     f"Updated admin user ID: {user_id}")

        return jsonify({'message': 'Admin user updated successfully'})

    except Exception as e:
        logger.error(f"Error updating admin user: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin/users/hospital/<int:hospital_id>', methods=['GET'])
@login_required
@admin_required
def get_hospital_account(hospital_id):
    """Get a single hospital account details"""
    try:
        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT h.id, h.hospital_name, h.hospital_code, h.email,
                   h.contact_person, h.phone, h.address, h.created_at,
                   hs.status as subscription_status,
                   sp.display_name as subscription_plan,
                   sp.name as subscription_plan_name
            FROM hospitals h
            LEFT JOIN hospital_subscriptions hs ON h.id = hs.hospital_id AND hs.status = 'active'
            LEFT JOIN subscription_plans sp ON hs.plan_id = sp.id
            WHERE h.id = ?
        """, (hospital_id,))

        row = c.fetchone()
        conn.close()

        if not row:
            return jsonify({'error': 'Hospital not found'}), 404

        hospital = {
            'id': row[0],
            'hospital_name': row[1],
            'hospital_code': row[2],
            'email': row[3],
            'contact_person': row[4],
            'phone': row[5],
            'address': row[6],
            'created_at': row[7],
            'subscription_status': row[8] or 'none',
            'subscription_plan': row[9] or 'Free',
            'subscription_plan_name': row[10] or 'free',
            'userType': 'hospital'
        }

        return jsonify(hospital)

    except Exception as e:
        logger.error(f"Error fetching hospital: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin/users/hospital/<int:hospital_id>', methods=['PUT'])
@login_required
@admin_required
def update_hospital_account(hospital_id):
    """Update a hospital account"""
    try:
        data = request.get_json()

        conn = get_db()
        c = conn.cursor()

        # Check if hospital exists
        c.execute('SELECT id FROM hospitals WHERE id=?', (hospital_id,))
        if not c.fetchone():
            conn.close()
            return jsonify({'error': 'Hospital not found'}), 404

        # Build update query for hospital
        updates = []
        params = []

        updatable_fields = ['hospital_name', 'email', 'contact_person', 'phone', 'address']
        for field in updatable_fields:
            if field in data and data[field] is not None:
                updates.append(f'{field} = ?')
                params.append(data[field])

        if 'password' in data and data['password']:
            updates.append('password = ?')
            params.append(generate_password_hash(data['password']))

        if updates:
            query = f"UPDATE hospitals SET {', '.join(updates)} WHERE id = ?"
            params.append(hospital_id)
            c.execute(query, params)

        # Update subscription if provided
        if 'subscription_plan' in data:
            c.execute('SELECT id FROM subscription_plans WHERE name=?',
                      (data['subscription_plan'],))
            plan = c.fetchone()

            if plan:
                plan_id = plan[0]

                # Check if hospital has active subscription
                c.execute("""
                    SELECT id FROM hospital_subscriptions
                    WHERE hospital_id=? AND status='active'
                """, (hospital_id,))
                existing_sub = c.fetchone()

                if existing_sub:
                    # Update existing subscription
                    c.execute("""
                        UPDATE hospital_subscriptions
                        SET plan_id=?, updated_at=CURRENT_TIMESTAMP
                        WHERE id=?
                    """, (plan_id, existing_sub[0]))

                    # Update usage limits
                    c.execute("""
                        SELECT max_scans_per_month, max_users, max_patients
                        FROM subscription_plans WHERE id=?
                    """, (plan_id,))
                    limits = c.fetchone()

                    c.execute("""
                        UPDATE usage_tracking
                        SET scans_limit=?, users_limit=?, patients_limit=?
                        WHERE hospital_id=? AND is_current=1
                    """, (limits[0], limits[1], limits[2], hospital_id))
                else:
                    # Create new subscription
                    start_date = datetime.now()
                    end_date = start_date + timedelta(days=30)

                    c.execute("""
                        INSERT INTO hospital_subscriptions
                        (hospital_id, plan_id, status, billing_cycle,
                         current_period_start, current_period_end)
                        VALUES (?, ?, 'active', 'monthly', ?, ?)
                    """, (hospital_id, plan_id, start_date.date(), end_date.date()))

        conn.commit()
        conn.close()

        log_activity('admin', session['user_id'], 'updated_hospital',
                     f"Updated hospital ID: {hospital_id}")

        return jsonify({'message': 'Hospital account updated successfully'})

    except Exception as e:
        logger.error(f"Error updating hospital: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin/users/patient/<int:patient_id>', methods=['PUT'])
@login_required
@admin_required
def update_patient_record(patient_id):
    """Update a patient record"""
    try:
        data = request.get_json()

        conn = get_db()
        c = conn.cursor()

        # Check if patient exists
        c.execute('SELECT id FROM patients WHERE id=?', (patient_id,))
        if not c.fetchone():
            conn.close()
            return jsonify({'error': 'Patient not found'}), 404

        # Build update query
        updates = []
        params = []

        updatable_fields = [
            'full_name', 'email', 'phone', 'date_of_birth', 'gender',
            'blood_group', 'emergency_contact', 'medical_history'
        ]

        for field in updatable_fields:
            if field in data and data[field] is not None:
                updates.append(f'{field} = ?')
                params.append(data[field])

        if not updates:
            conn.close()
            return jsonify({'error': 'No valid fields to update'}), 400

        query = f"UPDATE patients SET {', '.join(updates)} WHERE id = ?"
        params.append(patient_id)

        c.execute(query, params)
        conn.commit()
        conn.close()

        log_activity('admin', session['user_id'], 'updated_patient',
                     f"Updated patient ID: {patient_id}")

        return jsonify({'message': 'Patient record updated successfully'})

    except Exception as e:
        logger.error(f"Error updating patient: {e}")
        return jsonify({'error': str(e)}), 500


# ==============================================
# ADMIN - DELETE USERS
# ==============================================

@app.route("/api/stripe/config", methods=["GET"])
def stripe_config():
    return jsonify({"publishableKey": STRIPE_PUBLISHABLE_KEY})


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


# ==============================================
# USAGE CHECK MIDDLEWARE
# ==============================================

def check_scan_limit(f):
    """
    Decorator to check if hospital can perform scan
    Returns usage info even if blocked (for frontend display)
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        hospital_id = session.get("hospital_id")

        if not hospital_id:
            return jsonify({"error": "Not authenticated"}), 401

        usage_info = get_detailed_usage(hospital_id)

        # Check if blocked
        if usage_info['is_blocked']:
            return jsonify({
                "error": "Usage limit reached",
                "usage": usage_info,
                "upgrade_required": True,
                "message": usage_info['block_message']
            }), 403

        # Proceed with function, but attach usage info
        request.usage_info = usage_info
        return f(*args, **kwargs)

    return wrapper

    return {
        'is_blocked': is_blocked and (not cooldown_active if cooldown_active else True),
        'can_scan': not is_blocked or (cooldown_active == False),
        'scans_used': scans_used,
        'scans_limit': scans_limit,
        'scans_remaining': max(0, scans_limit - scans_used),
        'usage_percent': round(usage_percent, 1),
        'plan_name': subscription['display_name'],
        'plan_id': subscription['plan_id'],
        'is_free_tier': subscription['plan_name'] == 'free',
        'is_trial': subscription.get('is_trial', 0) == 1,
        'days_until_reset': days_until_reset,
        'period_end': usage['period_end'],
        'block_message': block_message,
        'cooldown_active': cooldown_active,
        'cooldown_ends': cooldown_ends.isoformat() if cooldown_ends else None,
        'upgrade_plans': get_upgrade_options(subscription['plan_id'])
    }


def get_upgrade_options(current_plan_id):
    """Get available upgrade plans"""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT id, name, display_name, price_monthly, max_scans_per_month
        FROM subscription_plans
        WHERE id > ? AND is_active = 1
        ORDER BY price_monthly ASC
        LIMIT 3
    """, (current_plan_id,))
    plans = [dict(row) for row in c.fetchall()]
    conn.close()
    return plans


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
                "initial_filters": 16,
                "num_fc1": 64,
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


class VGG19_BrainTumor(nn.Module):
    """VGG19-based model for brain tumor classification"""

    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(VGG19_BrainTumor, self).__init__()

        # Load pretrained VGG19
        self.vgg19 = models.vgg19(pretrained=False)

        # Modify classifier to match the saved checkpoint architecture
        num_features = self.vgg19.classifier[0].in_features  # 25088
        self.vgg19.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),  # Layer 0
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 1024),  # Layer 3 (changed from 4096 to 1024)
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),  # Layer 6 (changed from 1000 to 256)
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)  # Layer 9 (changed from 1000 to 4)
        )

    def forward(self, x):
        return self.vgg19(x)


# Define class names and transforms

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ==============================================
# IMAGE VALIDATION FUNCTION
# ==============================================

def validate_brain_scan(image, probabilities=None):
    """
    Validate if the uploaded image is likely a brain MRI scan.
    Returns (is_valid, error_message, warning_message)
    """
    warnings = []

    # Check 1: Image mode (brain MRIs are typically grayscale or converted from grayscale)
    # Most brain MRIs are grayscale, but we convert to RGB for the model
    # So we check if the image has very low color variation
    if image.mode == 'RGB':
        # Convert to numpy to check color variation
        import numpy as np
        img_array = np.array(image)

        # Check if R, G, B channels are very similar (indicating grayscale-like image)
        r_channel = img_array[:, :, 0]
        g_channel = img_array[:, :, 1]
        b_channel = img_array[:, :, 2]

        # Calculate variance between channels
        rg_diff = np.abs(r_channel.astype(float) - g_channel.astype(float)).mean()
        rb_diff = np.abs(r_channel.astype(float) - b_channel.astype(float)).mean()
        gb_diff = np.abs(g_channel.astype(float) - b_channel.astype(float)).mean()

        avg_diff = (rg_diff + rb_diff + gb_diff) / 3

        # If channels are very different, it's likely a color photo, not MRI
        if avg_diff > 15:  # Threshold for "too colorful"
            return False, "Invalid image type: Brain MRI scans should be grayscale medical images, not color photographs.", None

    # Check 2: Image dimensions (brain scans typically have certain aspect ratios)
    width, height = image.size
    aspect_ratio = width / height if height > 0 else 0

    # Brain MRI scans are usually square or close to square (0.8 to 1.2 ratio)
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        warnings.append(f"Unusual aspect ratio ({aspect_ratio:.2f}). Brain scans are typically more square-shaped.")

    # Check 3: Image size (too small might not be a real scan)
    if width < 100 or height < 100:
        return False, "Image too small: Please upload a high-quality brain MRI scan (minimum 100x100 pixels).", None

    # Check 4: Confidence threshold (if probabilities provided)
    if probabilities:
        max_confidence = max(probabilities.values())

        # If confidence is very low, image likely not a brain scan
        if max_confidence < 50:
            return False, "Low confidence detected: This image does not appear to be a brain MRI scan. Please upload a valid medical brain scan.", None

        # If confidence is moderate, show warning
        if max_confidence < 70:
            warnings.append(
                f"Moderate confidence ({max_confidence:.1f}%). Please verify this is a clear brain MRI scan.")

        # Check if predictions are too evenly distributed (another sign of invalid input)
        prob_values = list(probabilities.values())
        if len(prob_values) == 4:
            # If all probabilities are within 15% of each other, model is "confused"
            prob_range = max(prob_values) - min(prob_values)
            if prob_range < 15:
                warnings.append(
                    "Model uncertain: Predictions are evenly distributed. Image may not be a clear brain scan.")

    # Return result
    warning_msg = " ".join(warnings) if warnings else None
    return True, None, warning_msg


# Load the model
try:
    logger.info("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # Use lightweight CNN architecture that matches stored checkpoints
    model = CNN_TUMOR()

    # Load the state dict from checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
        model.load_state_dict(state_dict)
        logger.info("Loaded CNN_TUMOR weights from checkpoint")
    else:
        # It's already a model object
        model = checkpoint
        logger.info("Checkpoint provided a model instance directly")

    model.to(device)
    model.eval()
    logger.info("✅ Model loaded successfully and moved to device")

except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    import traceback

    traceback.print_exc()


# Model already loaded above (lines 125-165)
# No need to reload here


# ==============================================
# PUBLIC SUBSCRIPTION ROUTES
# ==============================================

@app.route("/api/subscription/plans", methods=["GET"])
def get_public_subscription_plans():
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


# REPLACE the /api/stripe/create-checkout-session endpoint in app.py

@app.route("/api/stripe/create-checkout-session", methods=["POST"])
@hospital_required
def create_checkout_session():
    """Fixed Stripe checkout session creation"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No request data provided"}), 400

        plan_identifier = data.get("plan_id")
        billing_cycle = data.get("billing_cycle", "monthly")

        logger.info(f"🔍 Checkout request: plan={plan_identifier}, cycle={billing_cycle}")

        hospital_id = session.get("hospital_id")
        if not hospital_id:
            return jsonify({"error": "Hospital ID not found in session"}), 400

        conn = get_db()
        c = conn.cursor()

        # Find plan by ID or name
        if isinstance(plan_identifier, str) and not plan_identifier.isdigit():
            c.execute("SELECT * FROM subscription_plans WHERE name=?", (plan_identifier,))
        else:
            c.execute("SELECT * FROM subscription_plans WHERE id=?", (plan_identifier,))

        plan = c.fetchone()

        if not plan:
            conn.close()
            logger.error(f"❌ Plan not found: {plan_identifier}")
            return jsonify({"error": f"Plan '{plan_identifier}' not found"}), 404

        plan = dict(plan)
        logger.info(f"✅ Found plan: {plan['name']} (ID: {plan['id']})")

        # Get Stripe price ID
        price_id = get_stripe_price_id(plan["id"], billing_cycle)
        if not price_id:
            conn.close()
            logger.error(f"❌ No Stripe price configured for plan {plan['id']}")
            return jsonify({"error": "Stripe price not configured for this plan"}), 400

        logger.info(f"💳 Using Stripe price_id: {price_id}")

        # Get hospital email
        c.execute("SELECT email FROM hospitals WHERE id = ?", (hospital_id,))
        hospital_row = c.fetchone()
        hospital_email = hospital_row['email'] if hospital_row else None
        conn.close()

        # Get or create Stripe customer
        customer_id = get_or_create_stripe_customer(hospital_id, hospital_email)
        logger.info(f"👤 Stripe customer_id: {customer_id}")

        if not customer_id:
            return jsonify({"error": "Failed to create Stripe customer"}), 500

        # Create checkout session
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')

        checkout_session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{frontend_url}/subscription-success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{frontend_url}/subscription-cancelled",
            metadata={
                "hospital_id": str(hospital_id),
                "plan_id": str(plan["id"]),
                "billing_cycle": billing_cycle
            }
        )

        logger.info(f"✅ Checkout session created: {checkout_session.id}")
        return jsonify({
            "url": checkout_session.url,
            "sessionId": checkout_session.id
        })

    except stripe.error.StripeError as e:
        logger.error(f"❌ Stripe error: {str(e)}")
        return jsonify({"error": f"Stripe error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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

# Replace your /hospital/login endpoint with this fixed version

@app.route("/hospital/login", methods=["POST"])
def hospital_login():
    """Hospital login with proper session setup"""
    try:
        data = request.json
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400

        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT hu.*, h.hospital_name, h.hospital_code, h.id as hospital_id
            FROM hospital_users hu
            JOIN hospitals h ON hu.hospital_id = h.id
            WHERE hu.username=? AND hu.status='active' AND h.status='active'
        """, (username,))

        user = c.fetchone()
        conn.close()

        if not user or not check_password_hash(user["password"], password):
            logger.warning(f"❌ Failed login attempt: {username}")
            return jsonify({"error": "Invalid credentials"}), 401

        # ✅ Clear any old session data
        session.clear()

        # ✅ Set ALL required session variables
        session["user_id"] = user["id"]
        session["user_type"] = "hospital"
        session["hospital_id"] = user["hospital_id"]
        session["username"] = user["username"]
        session.permanent = True
        session.modified = True

        # ✅ Ensure usage tracking exists
        ensure_usage_tracking_exists(user["hospital_id"])

        # Log the login
        log_activity("hospital", user["id"], "login", hospital_id=user["hospital_id"])

        # Debug log
        logger.info(f"✅ Hospital login successful: user_id={user['id']}, hospital_id={user['hospital_id']}")

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

    except Exception as e:
        logger.error(f"❌ Hospital login error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Login failed. Please try again."}), 500
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

    # Active chats - use a safe default if table doesn't exist
    try:
        c.execute("SELECT COUNT(*) as count FROM chat_conversations WHERE hospital_id=? AND status='active'",
                  (hospital_id,))
        active_chats = c.fetchone()["count"]
    except:
        active_chats = 0

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
        # Handle both datetime strings with time and date-only strings
        period_end = subscription['current_period_end']
        if ' ' in period_end:
            # Contains time information
            end_date = datetime.strptime(period_end, '%Y-%m-%d %H:%M:%S.%f')
        else:
            # Date only
            end_date = datetime.strptime(period_end, '%Y-%m-%d')
        days_remaining = (end_date - datetime.now()).days
    else:
        days_remaining = 0

    return jsonify({
        "subscription": subscription,
        "usage": usage,
        "days_remaining": days_remaining,
        "is_trial": subscription.get('is_trial', 0) == 1
    })


@app.route("/hospital/usage-status", methods=["GET"])
@hospital_required
def get_usage_status():
    """
    Get current usage status
    Called on dashboard load and before actions
    """
    hospital_id = session["hospital_id"]
    usage_info = get_detailed_usage(hospital_id)
    return jsonify(usage_info)


# ==============================================
# COOLDOWN SYSTEM (Optional)
# ==============================================

@app.route("/hospital/claim-free-scan", methods=["POST"])
@hospital_required
def claim_free_scan():
    """
    Allow free tier users to claim 1 scan after 24h cooldown
    """
    hospital_id = session["hospital_id"]
    usage_info = get_detailed_usage(hospital_id)

    if not usage_info['is_free_tier']:
        return jsonify({"error": "Only available for free tier"}), 403

    if usage_info['cooldown_active']:
        return jsonify({
            "error": "Cooldown still active",
            "cooldown_ends": usage_info['cooldown_ends']
        }), 403

    # Reset 1 scan (temporary increase of limit)
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        UPDATE usage_tracking
        SET scans_limit = scans_limit + 1
        WHERE hospital_id = ? AND is_current = 1
    """, (hospital_id,))
    conn.commit()
    conn.close()

    return jsonify({
        "message": "Free scan claimed! 1 additional scan available.",
        "usage": get_detailed_usage(hospital_id)
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
    hospital_id = session.get("hospital_id")

    # ADD THIS DEBUG LINE
    logger.info(f"🔍 Session hospital_id: {hospital_id}, user_id: {session.get('user_id')}")

    if not hospital_id:
        logger.error("❌ No hospital_id in session")
        return jsonify({"error": "Session error: Please log out and log in again"}), 401

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
    logger.info(f"📝 Creating patient with data: {data}")

    # VALIDATE REQUIRED FIELDS
    if not data.get("full_name"):
        logger.error("❌ Missing full_name")
        return jsonify({"error": "full_name is required"}), 400

    # GENERATE CODES
    patient_code = generate_unique_code('patients', 'patient_code', prefix='P')
    access_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    logger.info(f"✓ Generated patient_code: {patient_code}")

    try:
        c.execute("""
            INSERT INTO patients (
                hospital_id, patient_code, full_name, email, phone,
                date_of_birth, gender, address, emergency_contact, 
                emergency_phone, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            hospital_id, patient_code,
            data.get("full_name"), data.get("email"), data.get("phone"),
            data.get("date_of_birth"), data.get("gender"), data.get("address"),
            data.get("emergency_contact"), data.get("emergency_phone"),
            session["user_id"]
        ))
        patient_id = c.lastrowid

        # CREATE ACCESS CODE ENTRY
        expires_at = datetime.now() + timedelta(days=30)
        c.execute("""
            INSERT INTO patient_access_codes (patient_id, access_code, expires_at)
            VALUES (?, ?, ?)
        """, (patient_id, access_code, expires_at))

        # GET HOSPITAL INFO
        c.execute("SELECT hospital_name, hospital_code FROM hospitals WHERE id=?", (hospital_id,))
        hospital = c.fetchone()

        # GET PATIENT DATA
        c.execute("SELECT * FROM patients WHERE id=?", (patient_id,))
        patient = dict(c.fetchone())
        patient['scan_count'] = 0

        conn.commit()
        conn.close()

        # SEND EMAIL
        email_sent = False
        try:
            from email_utilis import send_patient_credentials_email

            email_sent = send_patient_credentials_email(
                to_email=data.get("email"),
                patient_name=data.get("full_name"),
                hospital_name=hospital["hospital_name"],
                hospital_code=hospital["hospital_code"],
                patient_code=patient_code,
                access_code=access_code
            )

            if email_sent:
                logger.info(f"✅ Welcome email sent to {data.get('email')}")
        except Exception as e:
            logger.error(f"❌ Email error: {e}")

        log_activity("hospital", session["user_id"], "create_patient", hospital_id=hospital_id)

        return jsonify({
            "message": "Patient created successfully",
            "patient": patient,
            "patient_id": patient_id,
            "patient_code": patient_code,
            "access_code": access_code,
            "email_sent": email_sent
        }), 201

    except Exception as e:
        conn.rollback()
        conn.close()
        logger.error(f"❌ Patient creation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/hospital/patients/<int:patient_id>/scans", methods=["GET"])
@hospital_required
def hospital_get_patient_scans(patient_id):
    """Get all scans for a patient in hospital portal"""
    conn = get_db()
    c = conn.cursor()
    hospital_id = session.get("hospital_id")

    # Verify patient belongs to hospital
    c.execute("SELECT id FROM patients WHERE id=? AND hospital_id=?", (patient_id, hospital_id))
    if not c.fetchone():
        conn.close()
        return jsonify({"error": "Patient not found"}), 404

    # Get scans
    c.execute("""
        SELECT id, patient_id, prediction, confidence, is_tumor, scan_type, created_at
        FROM mri_scans
        WHERE patient_id=?
        ORDER BY created_at DESC
    """, (patient_id,))

    scans = [dict(row) for row in c.fetchall()]
    conn.close()

    return jsonify({"scans": scans})


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
    """Step 2: Verify patient - FIXED"""
    data = request.json
    hospital_code = data.get("hospital_code")
    patient_code = data.get("patient_code")
    access_code = data.get("access_code")

    if not hospital_code or not patient_code or not access_code:
        return jsonify({"error": "All fields required"}), 400

    conn = get_db()
    c = conn.cursor()

    c.execute("""
        SELECT p.*, h.hospital_name, pac.id as access_id
        FROM patients p
        JOIN hospitals h ON p.hospital_id = h.id
        JOIN patient_access_codes pac ON p.id = pac.patient_id
        WHERE h.hospital_code=? AND p.patient_code=? AND pac.access_code=?
            AND (pac.expires_at IS NULL OR pac.expires_at > datetime('now'))
    """, (hospital_code.upper(), patient_code.upper(), access_code))

    patient_row = c.fetchone()

    if not patient_row:
        conn.close()
        return jsonify({"error": "Invalid credentials or expired access code"}), 401

    # ✅ CONVERT TO DICTIONARY
    patient = dict(patient_row)

    # Generate verification code
    verification_code = ''.join(random.choices(string.digits, k=6))
    expiry = datetime.now() + timedelta(minutes=10)

    c.execute("""
        UPDATE patient_access_codes 
        SET verification_code=?, verification_code_expiry=?
        WHERE id=?
    """, (verification_code, expiry, patient["access_id"]))

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
        logger.info(f"✅ Verification email sent to {patient['email']}")
    except Exception as e:
        logger.error(f"❌ Email error: {e}")

    logger.info(f"🔐 Verification code for {patient['full_name']}: {verification_code}")

    # Mask email
    email = patient["email"]
    email_hint = email[:3] + "***" + email[email.rfind('@'):]

    return jsonify({
        "message": "Verification code sent to your email",
        "email_hint": email_hint
    })

@app.route("/patient/login", methods=["POST"])
def patient_login():
    """Step 3: Login with verification code - FIXED"""
    data = request.json
    patient_code = data.get("patient_code")
    verification_code = data.get("verification_code")

    if not patient_code or not verification_code:
        return jsonify({"error": "Patient code and verification code required"}), 400

    conn = get_db()
    c = conn.cursor()

    c.execute("""
        SELECT p.*, h.hospital_name, h.id as hospital_id, pac.id as access_code_id
        FROM patients p
        JOIN hospitals h ON p.hospital_id = h.id
        JOIN patient_access_codes pac ON p.id = pac.patient_id
        WHERE p.patient_code = ? 
        AND pac.verification_code = ?
        AND (pac.verification_code_expiry IS NULL OR pac.verification_code_expiry > datetime('now'))
    """, (patient_code.upper(), verification_code))

    patient_row = c.fetchone()

    if not patient_row:
        conn.close()
        logger.warning(f"❌ Invalid login attempt: {patient_code}")
        return jsonify({"error": "Invalid or expired verification code"}), 401

    # ✅ CONVERT TO DICTIONARY
    patient = dict(patient_row)

    # Mark as verified
    c.execute("""
        UPDATE patient_access_codes 
        SET is_verified = 1, verified_at = datetime('now')
        WHERE id = ?
    """, (patient["access_code_id"],))

    conn.commit()
    conn.close()

    # ✅ SET SESSION
    session.clear()
    session.permanent = True
    session["patient_id"] = patient["id"]
    session["patient_type"] = "patient"
    session["user_id"] = patient["id"]
    session["user_type"] = "patient"
    session["hospital_id"] = patient["hospital_id"]
    session["full_name"] = patient["full_name"]
    session.modified = True

    logger.info(f"✅ Patient logged in: {patient['full_name']} (ID: {patient['id']})")

    # ✅ USE DICT.GET() NOW
    return jsonify({
        "patient": {
            "id": patient["id"],
            "full_name": patient["full_name"],
            "patient_code": patient["patient_code"],
            "hospital_name": patient["hospital_name"],
            "email": patient.get("email"),  # ✅ Now works
            "phone": patient.get("phone"),  # ✅ Now works
            "date_of_birth": patient.get("date_of_birth"),
            "gender": patient.get("gender"),
            "type": "patient"
        }
    })

# DUPLICATE - @app.route("/patient/scans", methods=["GET"])
# DUPLICATE - @patient_required
# DUPLICATE - def get_patient_scans():
# DUPLICATE - """Get patient's scans"""
# DUPLICATE - patient_id = session.get("patient_id")
# DUPLICATE -     # DUPLICATE - conn = get_db()
# DUPLICATE - c = conn.cursor()
# DUPLICATE -     # DUPLICATE - c.execute("""
# DUPLICATE - SELECT s.*, h.hospital_name, hu.full_name as doctor_name
# DUPLICATE - FROM mri_scans s
# DUPLICATE - JOIN hospitals h ON s.hospital_id = h.id
# DUPLICATE - LEFT JOIN hospital_users hu ON s.uploaded_by = hu.id
# DUPLICATE - WHERE s.patient_id=?
# DUPLICATE - ORDER BY s.created_at DESC
# DUPLICATE - """, (patient_id,))
# DUPLICATE -     # DUPLICATE - scans = [dict(row) for row in c.fetchall()]
# DUPLICATE - conn.close()
# DUPLICATE -     # DUPLICATE - return jsonify({"scans": scans})
# DUPLICATE -  # DUPLICATE - # ==============================================
# PROFILE PICTURE ROUTES
# ==============================================
# DUPLICATE - @app.route("/patient/profile-picture", methods=["POST"])
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
    # Check both user_id and patient_id
    if "user_id" not in session and "patient_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_type = session.get("user_type") or session.get("patient_type")

    if user_type == "patient":
        patient_id = session.get("patient_id") or session.get("user_id")

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
        if patient.get("profile_picture") and patient.get("profile_picture_mime"):
            profile_picture_url = f"data:{patient['profile_picture_mime']};base64,{patient['profile_picture']}"

        return jsonify({
            "user": {
                "id": patient["id"],
                "type": "patient",
                "full_name": patient["full_name"],
                "patient_code": patient["patient_code"],
                "email": patient.get("email"),
                "phone": patient.get("phone"),
                "date_of_birth": patient.get("date_of_birth"),
                "gender": patient.get("gender"),
                "hospital_name": patient["hospital_name"],
                "hospital_code": patient["hospital_code"],
                "profile_picture": profile_picture_url
            }
        })

    # Hospital user
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

    return jsonify({"status": "success"}), 200


# ==============================================
# CHAT ENDPOINTS - ADD THESE TO app.py
# ==============================================

@app.route('/api/chat/send', methods=['POST'])
@login_required
def send_chat_message():
    """Send a new chat message"""
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        hospital_user_id = data.get('hospital_user_id')
        message = data.get('message', '').strip()

        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        if not patient_id or not hospital_user_id:
            return jsonify({'error': 'Patient ID and Hospital User ID required'}), 400

        # Determine sender type
        user_type = session.get('user_type') or 'patient'

        # Verify authorization
        if user_type == 'patient':
            if session.get('patient_id') != int(patient_id):
                return jsonify({'error': 'Unauthorized'}), 403
            sender_type = 'patient'
        elif user_type == 'hospital':
            if session.get('user_id') != int(hospital_user_id):
                return jsonify({'error': 'Unauthorized'}), 403
            sender_type = 'hospital'
        else:
            return jsonify({'error': 'Invalid user type'}), 403

        # Insert message into database
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            INSERT INTO messages 
            (patient_id, hospital_user_id, sender_type, message, is_read)
            VALUES (?, ?, ?, ?, 0)
        """, (patient_id, hospital_user_id, sender_type, message))

        message_id = c.lastrowid

        # Get the created message
        c.execute("""
            SELECT * FROM messages WHERE id = ?
        """, (message_id,))
        new_message = dict(c.fetchone())

        conn.commit()
        conn.close()

        # Emit via SocketIO for real-time delivery
        socketio.emit('new_message', new_message,
                      room=f"chat_{patient_id}_{hospital_user_id}")

        # Create notification for recipient
        if sender_type == 'patient':
            # Notify hospital user
            create_notification(
                user_id=hospital_user_id,
                user_type='hospital',
                notification_type='new_message',
                title='New Message',
                message=f'New message from patient',
                patient_id=patient_id
            )
        else:
            # Notify patient
            create_notification(
                user_id=patient_id,
                user_type='patient',
                notification_type='new_message',
                title='New Message from Doctor',
                message=f'You have a new message',
                hospital_id=session.get('hospital_id')
            )

        return jsonify({
            'success': True,
            'message_id': message_id,
            'message': new_message
        }), 201

    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return jsonify({'error': str(e)}), 500


def create_notification(user_id, user_type, notification_type, title, message,
                        hospital_id=None, patient_id=None, scan_id=None,
                        priority='normal', action_url=None):
    """Helper function to create notifications"""
    try:
        conn = get_db()
        c = conn.cursor()

        c.execute("""
            INSERT INTO notifications 
            (user_id, user_type, hospital_id, type, title, message, 
             patient_id, scan_id, priority, action_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, user_type, hospital_id, notification_type, title,
              message, patient_id, scan_id, priority, action_url))

        notification_id = c.lastrowid
        conn.commit()

        # Get the created notification
        c.execute("SELECT * FROM notifications WHERE id = ?", (notification_id,))
        notification = dict(c.fetchone())
        conn.close()

        # Emit real-time notification via SocketIO
        socketio.emit('notification', notification,
                      room=f"{user_type}_{user_id}")

        return notification_id

    except Exception as e:
        logger.error(f"Error creating notification: {e}")
        return None


# ==============================================
# SOCKETIO EVENTS FOR CHAT
# ==============================================

@socketio.on('join_chat')
def handle_join_chat(data):
    """Join a chat room"""
    try:
        patient_id = data.get('patient_id')
        hospital_user_id = data.get('hospital_user_id')

        if patient_id and hospital_user_id:
            room = f"chat_{patient_id}_{hospital_user_id}"
            join_room(room)
            logger.info(f"User joined room: {room}")

            # Also join notification room
            user_type = data.get('user_type', 'patient')
            if user_type == 'patient':
                user_id = patient_id
            else:
                user_id = hospital_user_id

            notification_room = f"{user_type}_{user_id}"
            join_room(notification_room)

            emit('joined', {'room': room})
    except Exception as e:
        logger.error(f"Error joining chat: {e}")


@socketio.on('leave_chat')
def handle_leave_chat(data):
    """Leave a chat room"""
    try:
        patient_id = data.get('patient_id')
        hospital_user_id = data.get('hospital_user_id')

        if patient_id and hospital_user_id:
            room = f"chat_{patient_id}_{hospital_user_id}"
            leave_room(room)
            logger.info(f"User left room: {room}")
            emit('left', {'room': room})
    except Exception as e:
        logger.error(f"Error leaving chat: {e}")


@socketio.on('send_message')
def handle_send_message(data):
    """Handle real-time message sending via WebSocket"""
    try:
        patient_id = data.get('patient_id')
        hospital_user_id = data.get('hospital_user_id')

        if patient_id and hospital_user_id:
            room = f"chat_{patient_id}_{hospital_user_id}"

            # Broadcast to room (including sender for confirmation)
            emit('new_message', data, room=room, include_self=True)

            # Send confirmation to sender
            emit('message_sent', {
                'temp_id': data.get('temp_id'),
                'message_id': data.get('id'),
                'status': 'delivered'
            })

    except Exception as e:
        logger.error(f"Error in send_message: {e}")
        emit('message_error', {'error': str(e)})


@socketio.on('typing')
def handle_typing(data):
    """Handle typing indicator"""
    try:
        patient_id = data.get('patient_id')
        hospital_user_id = data.get('hospital_user_id')
        user_type = data.get('user_type')

        if patient_id and hospital_user_id:
            room = f"chat_{patient_id}_{hospital_user_id}"

            # Broadcast typing to others in room (not self)
            emit('user_typing', {
                'user_type': user_type
            }, room=room, include_self=False)

    except Exception as e:
        logger.error(f"Error in typing indicator: {e}")


@socketio.on('mark_read')
def handle_mark_read(data):
    """Mark messages as read via WebSocket"""
    try:
        patient_id = data.get('patient_id')
        hospital_user_id = data.get('hospital_user_id')
        user_type = data.get('user_type', 'patient')

        conn = get_db()
        c = conn.cursor()

        # Mark unread messages from the other party as read
        if user_type == 'patient':
            sender_to_mark = 'hospital'
        else:
            sender_to_mark = 'patient'

        c.execute("""
            UPDATE messages 
            SET is_read = 1, read_at = CURRENT_TIMESTAMP
            WHERE patient_id = ? 
            AND hospital_user_id = ?
            AND sender_type = ?
            AND is_read = 0
        """, (patient_id, hospital_user_id, sender_to_mark))

        affected_rows = c.rowcount
        conn.commit()
        conn.close()

        if affected_rows > 0:
            # Notify the sender that their messages were read
            room = f"chat_{patient_id}_{hospital_user_id}"
            emit('messages_read', {
                'patient_id': patient_id,
                'hospital_user_id': hospital_user_id,
                'reader_type': user_type
            }, room=room, include_self=False)

    except Exception as e:
        logger.error(f"Error marking messages as read: {e}")


# ==============================================
# WEBRTC SIGNALING FOR AUDIO/VIDEO CALLS
# ==============================================

@socketio.on('call_user')
def handle_call_user(data):
    """Initiate a call to another user"""
    try:
        caller_id = data.get('caller_id')
        caller_type = data.get('caller_type')  # 'patient' or 'hospital'
        callee_id = data.get('callee_id')
        callee_type = data.get('callee_type')
        call_type = data.get('call_type', 'video')  # 'audio' or 'video'
        offer = data.get('offer')

        # Create unique call room
        call_room = f"call_{caller_id}_{callee_id}_{datetime.now().timestamp()}"

        # Notify the callee about incoming call
        callee_room = f"{callee_type}_{callee_id}"
        emit('incoming_call', {
            'caller_id': caller_id,
            'caller_type': caller_type,
            'call_type': call_type,
            'call_room': call_room,
            'offer': offer
        }, room=callee_room)

        logger.info(f"📞 Call initiated: {caller_type}_{caller_id} -> {callee_type}_{callee_id}")

    except Exception as e:
        logger.error(f"Error initiating call: {e}")
        emit('call_error', {'error': str(e)})


@socketio.on('answer_call')
def handle_answer_call(data):
    """Answer an incoming call"""
    try:
        caller_id = data.get('caller_id')
        caller_type = data.get('caller_type')
        callee_id = data.get('callee_id')
        call_room = data.get('call_room')
        answer = data.get('answer')

        # Join the call room
        join_room(call_room)

        # Send answer back to caller
        caller_room = f"{caller_type}_{caller_id}"
        emit('call_answered', {
            'callee_id': callee_id,
            'answer': answer,
            'call_room': call_room
        }, room=caller_room)

        logger.info(f"✅ Call answered: {callee_id} -> {caller_id}")

    except Exception as e:
        logger.error(f"Error answering call: {e}")
        emit('call_error', {'error': str(e)})


@socketio.on('reject_call')
def handle_reject_call(data):
    """Reject an incoming call"""
    try:
        caller_id = data.get('caller_id')
        caller_type = data.get('caller_type')
        callee_id = data.get('callee_id')

        # Notify caller that call was rejected
        caller_room = f"{caller_type}_{caller_id}"
        emit('call_rejected', {
            'callee_id': callee_id
        }, room=caller_room)

        logger.info(f"❌ Call rejected: {callee_id} rejected {caller_id}")

    except Exception as e:
        logger.error(f"Error rejecting call: {e}")


@socketio.on('ice_candidate')
def handle_ice_candidate(data):
    """Exchange ICE candidates for WebRTC connection"""
    try:
        call_room = data.get('call_room')
        candidate = data.get('candidate')
        sender_id = data.get('sender_id')

        # Broadcast ICE candidate to other peer in the call room
        emit('ice_candidate', {
            'candidate': candidate,
            'sender_id': sender_id
        }, room=call_room, include_self=False)

    except Exception as e:
        logger.error(f"Error handling ICE candidate: {e}")


@socketio.on('join_call')
def handle_join_call(data):
    """Join a call room"""
    try:
        call_room = data.get('call_room')
        user_id = data.get('user_id')
        user_type = data.get('user_type')

        join_room(call_room)

        # Notify others in the call
        emit('user_joined_call', {
            'user_id': user_id,
            'user_type': user_type
        }, room=call_room, include_self=False)

        logger.info(f"👤 User joined call: {user_type}_{user_id} in {call_room}")

    except Exception as e:
        logger.error(f"Error joining call: {e}")


@socketio.on('leave_call')
def handle_leave_call(data):
    """Leave a call room"""
    try:
        call_room = data.get('call_room')
        user_id = data.get('user_id')
        user_type = data.get('user_type')

        # Notify others that user left
        emit('user_left_call', {
            'user_id': user_id,
            'user_type': user_type
        }, room=call_room, include_self=False)

        leave_room(call_room)

        logger.info(f"👋 User left call: {user_type}_{user_id} from {call_room}")

    except Exception as e:
        logger.error(f"Error leaving call: {e}")


@socketio.on('end_call')
def handle_end_call(data):
    """End an active call"""
    try:
        call_room = data.get('call_room')
        user_id = data.get('user_id')

        # Notify all participants that call ended
        emit('call_ended', {
            'ended_by': user_id
        }, room=call_room)

        logger.info(f"📴 Call ended by {user_id} in {call_room}")

    except Exception as e:
        logger.error(f"Error ending call: {e}")


@socketio.on('toggle_audio')
def handle_toggle_audio(data):
    """Toggle audio mute/unmute"""
    try:
        call_room = data.get('call_room')
        user_id = data.get('user_id')
        audio_enabled = data.get('audio_enabled')

        # Notify other participants
        emit('peer_audio_toggled', {
            'user_id': user_id,
            'audio_enabled': audio_enabled
        }, room=call_room, include_self=False)

    except Exception as e:
        logger.error(f"Error toggling audio: {e}")


@socketio.on('toggle_video')
def handle_toggle_video(data):
    """Toggle video on/off"""
    try:
        call_room = data.get('call_room')
        user_id = data.get('user_id')
        video_enabled = data.get('video_enabled')

        # Notify other participants
        emit('peer_video_toggled', {
            'user_id': user_id,
            'video_enabled': video_enabled
        }, room=call_room, include_self=False)

    except Exception as e:
        logger.error(f"Error toggling video: {e}")

        @app.route('/analyze', methods=['POST'])
        @hospital_required  # This ensures only logged-in hospitals can use it
        def analyze_scan():
            try:
                if 'image' not in request.files:
                    return jsonify({"error": "No image uploaded"}), 400
                file = request.files['image']
                patient_id = request.form.get('patient_id')

                # 1. Load and Transform Image
                image_bytes = file.read()
                image_stream = io.BytesIO(image_bytes)
                image_stream.seek(0)  # Rewind to start
                img = Image.open(image_stream).convert('RGB')

                # VALIDATION: Check if this is actually a brain scan
                is_valid, error_msg, warning_msg = validate_brain_scan(img)
                if not is_valid:
                    return jsonify({
                        "error": "Invalid Image",
                        "message": error_msg,
                        "suggestion": "Please upload a grayscale brain MRI scan."
                    }), 400

                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_tensor = transform(img).unsqueeze(0)

                # 2. Prediction with Softmax
                with torch.no_grad():
                    output = model(img_tensor)
                    # Softmax forces the 4 outputs to sum to exactly 1.0 (100%)
                    probabilities = F.softmax(output, dim=1)[0]

                classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

                # 3. Create probability dictionary
                prob_results = {}
                for i, class_name in enumerate(classes):
                    # Convert to percentage and round
                    prob_results[class_name] = round(float(probabilities[i]) * 100, 2)

                # 4. Get highest confidence result
                pred_idx = torch.argmax(probabilities).item()
                prediction = classes[pred_idx]
                confidence = prob_results[prediction]

                # VALIDATION: Check confidence after prediction
                is_valid_result, error_msg_result, warning_msg_result = validate_brain_scan(img, prob_results)
                if not is_valid_result:
                    return jsonify({
                        "error": "Invalid Scan Result",
                        "message": error_msg_result
                    }), 400

                return jsonify({
                    "status": "success",
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": prob_results,
                    "validation_warning": warning_msg_result  # Include warning
                })

            except Exception as e:
                logging.error(f"Analysis error: {str(e)}")
                return jsonify({"error": "Internal server error during analysis"}), 500


@app.route('/hospital/check-feature/<feature_name>', methods=['GET'])
@hospital_required
def check_feature_access(feature_name):
    """Check if hospital has access to a specific feature"""
    hospital_id = session.get('hospital_id')

    try:
        conn = get_db()
        c = conn.cursor()

        # Get hospital's subscription plan
        c.execute("""
            SELECT sp.name, sp.features
            FROM hospital_subscriptions hs
            JOIN subscription_plans sp ON hs.plan_id = sp.id
            WHERE hs.hospital_id = ? AND hs.status = 'active'
            ORDER BY hs.created_at DESC
            LIMIT 1
        """, (hospital_id,))

        sub = c.fetchone()
        conn.close()

        if not sub:
            return jsonify({
                'has_access': False,
                'plan': 'free',
                'required_plans': get_required_plans(feature_name)
            })

        plan_name = sub['name']
        features = json.loads(sub['features']) if sub['features'] else []

        # Check if feature is in plan's features
        has_access = feature_name in features

        return jsonify({
            'has_access': has_access,
            'plan': plan_name,
            'required_plans': get_required_plans(feature_name)
        })

    except Exception as e:
        logger.error(f"Error checking feature access: {e}")
        return jsonify({'error': str(e)}), 500


def get_required_plans(feature_name):
    """Get which plans include a specific feature"""
    feature_plans = {
        'video_call': ['premium', 'enterprise'],
        'ai_chat': ['premium', 'enterprise'],
        'advanced_analytics': ['enterprise'],
        'tumor_tracking': ['premium', 'enterprise']
    }
    return feature_plans.get(feature_name, ['enterprise'])


def mark_all_notifications_read():
    """Mark all notifications as read for the current user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user_id = session['user_id']
    user_type = session.get('user_type', 'hospital')

    try:
        conn = get_db()
        c = conn.cursor()

        c.execute('''
            UPDATE notifications 
            SET is_read = 1
            WHERE user_id = ? AND user_type = ? AND is_read = 0
        ''', (user_id, user_type))

        affected = c.rowcount
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'marked_count': affected,
            'message': f'{affected} notifications marked as read'
        })

    except Exception as e:
        logger.error(f"Error marking all notifications as read: {e}")
        return jsonify({'error': 'Failed to update notifications'}), 500


# 2. ADD THIS ENDPOINT FOR CLEARING ALL NOTIFICATIONS
def clear_all_notifications():
    """Delete all read notifications for the current user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user_id = session['user_id']
    user_type = session.get('user_type', 'hospital')

    try:
        conn = get_db()
        c = conn.cursor()

        c.execute('''
            DELETE FROM notifications 
            WHERE user_id = ? AND user_type = ? AND is_read = 1
        ''', (user_id, user_type))

        deleted = c.rowcount
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'deleted_count': deleted,
            'message': f'{deleted} notifications cleared'
        })

    except Exception as e:
        logger.error(f"Error clearing notifications: {e}")
        return jsonify({'error': 'Failed to clear notifications'}), 500


# Add these routes to your app.py

@app.route('/admin/subscription-plans/<int:plan_id>', methods=['GET', 'PUT', 'DELETE'])
@admin_required
def manage_subscription_plan(plan_id):
    """Get, update, or delete a subscription plan"""
    conn = get_db()
    c = conn.cursor()

    if request.method == 'GET':
        # Get plan details
        c.execute("SELECT * FROM subscription_plans WHERE id=?", (plan_id,))
        plan = c.fetchone()
        conn.close()

        if not plan:
            return jsonify({'error': 'Plan not found'}), 404

        plan_dict = dict(plan)
        try:
            plan_dict['features'] = json.loads(plan_dict['features']) if plan_dict['features'] else []
        except:
            plan_dict['features'] = []

        return jsonify({'plan': plan_dict})

    elif request.method == 'PUT':
        # Update plan
        data = request.json

        try:
            # Validate pricing
            price_monthly = float(data.get('price_monthly', 0))
            price_yearly = float(data.get('price_yearly', 0))

            # Validate limits
            max_scans = int(data.get('max_scans_per_month', -1))
            max_users = int(data.get('max_users', -1))
            max_patients = int(data.get('max_patients', -1))

            # Prepare features
            features = data.get('features', [])
            if isinstance(features, list):
                features_json = json.dumps(features)
            else:
                features_json = features

            # Update query
            c.execute("""
                UPDATE subscription_plans
                SET 
                    name = ?,
                    display_name = ?,
                    description = ?,
                    price_monthly = ?,
                    price_yearly = ?,
                    max_scans_per_month = ?,
                    max_users = ?,
                    max_patients = ?,
                    features = ?,
                    is_active = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                data.get('name'),
                data.get('display_name'),
                data.get('description'),
                price_monthly,
                price_yearly,
                max_scans,
                max_users,
                max_patients,
                features_json,
                data.get('is_active', 1),
                plan_id
            ))

            if c.rowcount == 0:
                conn.close()
                return jsonify({'error': 'Plan not found'}), 404

            conn.commit()

            # Get updated plan
            c.execute("SELECT * FROM subscription_plans WHERE id=?", (plan_id,))
            updated_plan = dict(c.fetchone())
            updated_plan['features'] = json.loads(updated_plan['features']) if updated_plan['features'] else []

            conn.close()

            log_activity('admin', session['user_id'], 'update_plan',
                         f"Updated plan: {data.get('name')}")

            return jsonify({
                'message': 'Plan updated successfully',
                'plan': updated_plan
            })

        except ValueError as e:
            conn.close()
            return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
        except Exception as e:
            conn.close()
            logger.error(f"Error updating plan: {e}")
            return jsonify({'error': str(e)}), 500

    elif request.method == 'DELETE':
        # Soft delete - mark as inactive
        c.execute("""
            UPDATE subscription_plans
            SET is_active = 0, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (plan_id,))

        if c.rowcount == 0:
            conn.close()
            return jsonify({'error': 'Plan not found'}), 404

        conn.commit()
        conn.close()

        log_activity('admin', session['user_id'], 'delete_plan',
                     f"Deleted plan ID: {plan_id}")

        return jsonify({'message': 'Plan deactivated successfully'})


@app.route('/admin/subscription-plans', methods=['POST'])
@admin_required
def create_subscription_plan():
    """Create a new subscription plan"""
    data = request.json

    try:
        conn = get_db()
        c = conn.cursor()

        # Validate required fields
        required = ['name', 'display_name', 'price_monthly', 'price_yearly']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Check if name already exists
        c.execute("SELECT id FROM subscription_plans WHERE name=?", (data['name'],))
        if c.fetchone():
            conn.close()
            return jsonify({'error': 'Plan name already exists'}), 409

        # Prepare features
        features = data.get('features', [])
        if isinstance(features, list):
            features_json = json.dumps(features)
        else:
            features_json = features

        # Insert plan
        c.execute("""
            INSERT INTO subscription_plans (
                name, display_name, description,
                price_monthly, price_yearly,
                max_scans_per_month, max_users, max_patients,
                features, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['name'],
            data['display_name'],
            data.get('description', ''),
            float(data['price_monthly']),
            float(data['price_yearly']),
            int(data.get('max_scans_per_month', -1)),
            int(data.get('max_users', -1)),
            int(data.get('max_patients', -1)),
            features_json,
            data.get('is_active', 1)
        ))

        plan_id = c.lastrowid
        conn.commit()

        # Get created plan
        c.execute("SELECT * FROM subscription_plans WHERE id=?", (plan_id,))
        new_plan = dict(c.fetchone())
        new_plan['features'] = json.loads(new_plan['features']) if new_plan['features'] else []

        conn.close()

        log_activity('admin', session['user_id'], 'create_plan',
                     f"Created plan: {data['name']}")

        return jsonify({
            'message': 'Plan created successfully',
            'plan': new_plan
        }), 201

    except Exception as e:
        logger.error(f"Error creating plan: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin/features', methods=['GET'])
@admin_required
def get_all_features():
    """Get list of all available features"""
    # Predefined feature list
    features = [
        {
            'key': 'video_call',
            'name': 'Video Consultations',
            'description': 'Real-time video calls with patients'
        },
        {
            'key': 'ai_chat',
            'name': 'AI-Powered Chat',
            'description': 'Advanced chat with AI assistance'
        },
        {
            'key': 'tumor_tracking',
            'name': 'Tumor Progression Tracking',
            'description': 'Track tumor changes over time'
        },
        {
            'key': 'advanced_analytics',
            'name': 'Advanced Analytics',
            'description': 'Detailed reports and insights'
        },
        {
            'key': 'priority_support',
            'name': 'Priority Support',
            'description': '24/7 priority customer support'
        },
        {
            'key': 'api_access',
            'name': 'API Access',
            'description': 'RESTful API for integrations'
        },
        {
            'key': 'white_label',
            'name': 'White Label',
            'description': 'Custom branding options'
        }
    ]

    return jsonify({'features': features})


# Also add this database migration to ensure the table exists
def ensure_subscription_tables():
    """Ensure subscription_plans table has all required columns"""
    conn = get_db()
    c = conn.cursor()

    # Check if table exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS subscription_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            description TEXT,
            price_monthly REAL NOT NULL DEFAULT 0,
            price_yearly REAL NOT NULL DEFAULT 0,
            max_scans_per_month INTEGER DEFAULT -1,
            max_users INTEGER DEFAULT -1,
            max_patients INTEGER DEFAULT -1,
            features TEXT DEFAULT '[]',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert default plans if table is empty
    c.execute("SELECT COUNT(*) FROM subscription_plans")
    if c.fetchone()[0] == 0:
        default_plans = [
            {
                'name': 'free',
                'display_name': 'Free',
                'description': 'Perfect for trying out NeuroScan',
                'price_monthly': 0,
                'price_yearly': 0,
                'max_scans_per_month': 10,
                'max_users': 2,
                'max_patients': 50,
                'features': json.dumps([])
            },
            {
                'name': 'basic',
                'display_name': 'Basic',
                'description': 'For small clinics and practices',
                'price_monthly': 99,
                'price_yearly': 950,
                'max_scans_per_month': 100,
                'max_users': 5,
                'max_patients': 500,
                'features': json.dumps(['priority_support'])
            },
            {
                'name': 'premium',
                'display_name': 'Premium',
                'description': 'For growing healthcare facilities',
                'price_monthly': 299,
                'price_yearly': 2990,
                'max_scans_per_month': 500,
                'max_users': 20,
                'max_patients': 2000,
                'features': json.dumps([
                    'video_call', 'ai_chat', 'tumor_tracking',
                    'priority_support', 'advanced_analytics'
                ])
            },
            {
                'name': 'enterprise',
                'display_name': 'Enterprise',
                'description': 'For large hospitals and networks',
                'price_monthly': 999,
                'price_yearly': 9990,
                'max_scans_per_month': -1,
                'max_users': -1,
                'max_patients': -1,
                'features': json.dumps([
                    'video_call', 'ai_chat', 'tumor_tracking',
                    'priority_support', 'advanced_analytics',
                    'api_access', 'white_label'
                ])
            }
        ]

        for plan in default_plans:
            c.execute("""
                INSERT INTO subscription_plans (
                    name, display_name, description,
                    price_monthly, price_yearly,
                    max_scans_per_month, max_users, max_patients,
                    features, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                plan['name'], plan['display_name'], plan['description'],
                plan['price_monthly'], plan['price_yearly'],
                plan['max_scans_per_month'], plan['max_users'], plan['max_patients'],
                plan['features']
            ))

    conn.commit()
    conn.close()
    logger.info("✅ Subscription tables verified")


# Call this when app starts
ensure_subscription_tables()


@app.route("/api/chatbot", methods=["POST"])
def chatbot_api():
    """AI Chatbot endpoint using Ollama"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        # Get Ollama configuration from environment
        ollama_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
        ollama_model = os.getenv('OLLAMA_MODEL', 'llama2')

        # System prompt for medical context
        system_prompt = """You are a helpful medical AI assistant for NeuroScan, a brain tumor detection platform. 
You can answer questions about:
- Brain tumors (glioma, meningioma, pituitary tumors)
- MRI scans and medical imaging
- How to use the NeuroScan platform
- General brain health information

Always be professional, empathetic, and provide accurate medical information. 
If asked about specific diagnoses, remind users to consult with healthcare professionals.
Keep responses concise and helpful."""

        try:
            # Call Ollama API
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": f"{system_prompt}\n\nUser: {user_message}\nAssistant:",
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', 'Sorry, I could not generate a response.')

                return jsonify({
                    "success": True,
                    "response": ai_response,
                    "model": ollama_model
                }), 200
            else:
                # Ollama not available - return fallback response
                return jsonify({
                    "success": True,
                    "response": "I'm currently unavailable. The AI service is not running. Please make sure Ollama is installed and running.",
                    "fallback": True
                }), 200

        except requests.exceptions.ConnectionError:
            # Ollama not running - return helpful message
            return jsonify({
                "success": True,
                "response": """The AI assistant is currently offline. To enable AI chat:

1. Install Ollama from https://ollama.ai
2. Run: ollama pull llama2
3. Start Ollama service
4. Restart NeuroScan

For now, you can still use all other features of the platform.""",
                "fallback": True
            }), 200

        except requests.exceptions.Timeout:
            return jsonify({
                "success": True,
                "response": "The AI is taking too long to respond. Please try again with a shorter question.",
                "fallback": True
            }), 200

    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({
            "success": True,
            "response": f"I encountered an error: {str(e)}. Please try again.",
            "fallback": True
        }), 200


# Alternative: Simple chatbot without Ollama (fallback)
@app.route("/api/chatbot/simple", methods=["POST"])
def simple_chatbot():
    """Simple rule-based chatbot fallback"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip().lower()

        # Simple keyword-based responses
        responses = {
            'hello': "Hello! I'm your NeuroScan AI assistant. How can I help you today?",
            'hi': "Hi there! I can help answer questions about brain tumors and MRI scans.",
            'help': "I can help you with:\n- Brain tumor information\n- MRI scan interpretation\n- Platform features\n- Medical terminology\n\nWhat would you like to know?",
            'glioma': "Glioma is a type of tumor that occurs in the brain and spinal cord. It begins in the glial cells that surround nerve cells. Treatment depends on the type, location, and grade. Please consult with a healthcare provider for specific medical advice.",
            'meningioma': "Meningioma is a tumor that arises from the meninges (membranes surrounding the brain and spinal cord). Most are benign and grow slowly. Treatment may involve observation, surgery, or radiation therapy.",
            'pituitary': "Pituitary tumors are growths in the pituitary gland. Most are benign and can affect hormone production. Symptoms vary based on size and hormone effects. Consult an endocrinologist for proper evaluation.",
            'mri': "MRI (Magnetic Resonance Imaging) uses magnetic fields and radio waves to create detailed images of the brain. It's the gold standard for detecting brain tumors. The scan is painless and typically takes 30-60 minutes.",
            'scan': "To upload a scan:\n1. Click 'Upload MRI' in your dashboard\n2. Select your MRI image file\n3. Wait for AI analysis\n4. View results and probability scores\n\nSupported formats: JPG, PNG",
        }

        # Find matching response
        for keyword, response in responses.items():
            if keyword in user_message:
                return jsonify({
                    "success": True,
                    "response": response,
                    "model": "simple_rules"
                }), 200

        # Default response
        return jsonify({
            "success": True,
            "response": "I'm not sure I understand. Could you rephrase your question? I can help with brain tumor information, MRI scans, and platform features.",
            "model": "simple_rules"
        }), 200

    except Exception as e:
        logger.error(f"Simple chatbot error: {e}")
        return jsonify({"error": str(e)}), 500
# ============================================================
# ADMIN: DELETE PATIENT
# ============================================================
def require_admin(args):
    pass


@app.route("/admin/users/patient/<int:patient_id>", methods=["DELETE"])

def admin_delete_patient(patient_id):
    """Delete a patient (admin only)"""
    try:
        conn = get_db()
        c = conn.cursor()

        # Check if patient exists
        c.execute("SELECT full_name, email FROM patients WHERE id = ?", (patient_id,))
        patient = c.fetchone()

        if not patient:
            return jsonify({"error": "Patient not found"}), 404

        patient_name = patient[0]

        # Delete patient (cascade will handle related records)
        c.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
        conn.commit()

        logger.info(f"🗑️ Admin deleted patient: {patient_name} (ID: {patient_id})")

        return jsonify({
            "success": True,
            "message": f"Patient {patient_name} deleted successfully"
        }), 200

    except Exception as e:
        logger.error(f"Error deleting patient: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# ADMIN: DELETE HOSPITAL
# ============================================================
@app.route("/admin/users/hospital/<int:hospital_id>", methods=["DELETE"])

def admin_delete_hospital(hospital_id):
    """Delete a hospital (admin only)"""
    try:
        conn = get_db()
        c = conn.cursor()

        # Check if hospital exists
        c.execute("SELECT hospital_name, hospital_code FROM hospitals WHERE id = ?", (hospital_id,))
        hospital = c.fetchone()

        if not hospital:
            return jsonify({"error": "Hospital not found"}), 404

        hospital_name = hospital[0]

        # Check if hospital has patients
        c.execute("SELECT COUNT(*) FROM patients WHERE hospital_id = ?", (hospital_id,))
        patient_count = c.fetchone()[0]

        if patient_count > 0:
            return jsonify({
                "error": f"Cannot delete hospital with {patient_count} active patients. Please delete or transfer patients first."
            }), 400

        # Delete hospital (cascade will handle related records)
        c.execute("DELETE FROM hospitals WHERE id = ?", (hospital_id,))
        conn.commit()

        logger.info(f"🗑️ Admin deleted hospital: {hospital_name} (ID: {hospital_id})")

        return jsonify({
            "success": True,
            "message": f"Hospital {hospital_name} deleted successfully"
        }), 200

    except Exception as e:
        logger.error(f"Error deleting hospital: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# ADMIN: DELETE ADMIN USER
# ============================================================
@app.route("/admin/users/admin/<int:admin_id>", methods=["DELETE"])

def admin_delete_admin(admin_id):
    """Delete an admin user (admin only)"""
    try:
        # Prevent self-deletion
        current_admin_id = session.get('user_id')
        if current_admin_id == admin_id:
            return jsonify({"error": "Cannot delete your own admin account"}), 400

        conn = get_db()
        c = conn.cursor()

        # Check if admin exists
        c.execute("SELECT username, email FROM admins WHERE id = ?", (admin_id,))
        admin = c.fetchone()

        if not admin:
            return jsonify({"error": "Admin not found"}), 404

        admin_username = admin[0]

        # Check if this is the last admin
        c.execute("SELECT COUNT(*) FROM admins WHERE is_active = 1")
        admin_count = c.fetchone()[0]

        if admin_count <= 1:
            return jsonify({
                "error": "Cannot delete the last admin account. Create another admin first."
            }), 400

        # Delete admin
        c.execute("DELETE FROM admins WHERE id = ?", (admin_id,))
        conn.commit()

        logger.info(f"🗑️ Admin deleted admin user: {admin_username} (ID: {admin_id})")

        return jsonify({
            "success": True,
            "message": f"Admin {admin_username} deleted successfully"
        }), 200

    except Exception as e:
        logger.error(f"Error deleting admin: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# HELPER: Batch delete patients
# ============================================================
@app.route("/admin/users/patients/batch-delete", methods=["DELETE"])

def admin_batch_delete_patients():
    """Delete multiple patients at once"""
    try:
        data = request.get_json()
        patient_ids = data.get('patient_ids', [])

        if not patient_ids:
            return jsonify({"error": "No patient IDs provided"}), 400

        conn = get_db()
        c = conn.cursor()

        # Delete patients
        placeholders = ','.join('?' * len(patient_ids))
        c.execute(f"DELETE FROM patients WHERE id IN ({placeholders})", patient_ids)
        deleted_count = c.rowcount
        conn.commit()

        logger.info(f"🗑️ Admin batch deleted {deleted_count} patients")

        return jsonify({
            "success": True,
            "message": f"Successfully deleted {deleted_count} patients"
        }), 200

    except Exception as e:
        logger.error(f"Error batch deleting patients: {e}")
        return jsonify({"error": str(e)}), 500

    # ============================================================
    # ADMIN: DELETE PATIENT
    # ============================================================
    @app.route("/admin/users/patient/<int:patient_id>", methods=["DELETE"])
    def admin_delete_patient(patient_id):
        """Delete a patient (admin only)"""
        # Check admin authentication
        if session.get('user_type') != 'admin':
            return jsonify({"error": "Admin access required"}), 403

        try:
            conn = get_db()
            c = conn.cursor()

            # Check if patient exists
            c.execute("SELECT full_name, email FROM patients WHERE id = ?", (patient_id,))
            patient = c.fetchone()

            if not patient:
                return jsonify({"error": "Patient not found"}), 404

            patient_name = patient[0]

            # Delete patient (cascade will handle related records)
            c.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
            conn.commit()

            logger.info(f"🗑️ Admin deleted patient: {patient_name} (ID: {patient_id})")

            return jsonify({
                "success": True,
                "message": f"Patient {patient_name} deleted successfully"
            }), 200

        except Exception as e:
            logger.error(f"Error deleting patient: {e}")
            return jsonify({"error": str(e)}), 500

    # ============================================================
    # ADMIN: DELETE HOSPITAL
    # ============================================================
    @app.route("/admin/users/hospital/<int:hospital_id>", methods=["DELETE"])
    def admin_delete_hospital(hospital_id):
        """Delete a hospital (admin only)"""
        # Check admin authentication
        if session.get('user_type') != 'admin':
            return jsonify({"error": "Admin access required"}), 403

        try:
            conn = get_db()
            c = conn.cursor()

            # Check if hospital exists
            c.execute("SELECT hospital_name, hospital_code FROM hospitals WHERE id = ?", (hospital_id,))
            hospital = c.fetchone()

            if not hospital:
                return jsonify({"error": "Hospital not found"}), 404

            hospital_name = hospital[0]

            # Check if hospital has patients
            c.execute("SELECT COUNT(*) FROM patients WHERE hospital_id = ?", (hospital_id,))
            patient_count = c.fetchone()[0]

            if patient_count > 0:
                return jsonify({
                    "error": f"Cannot delete hospital with {patient_count} active patients. Please delete or transfer patients first."
                }), 400

            # Delete hospital (cascade will handle related records)
            c.execute("DELETE FROM hospitals WHERE id = ?", (hospital_id,))
            conn.commit()

            logger.info(f"🗑️ Admin deleted hospital: {hospital_name} (ID: {hospital_id})")

            return jsonify({
                "success": True,
                "message": f"Hospital {hospital_name} deleted successfully"
            }), 200

        except Exception as e:
            logger.error(f"Error deleting hospital: {e}")
            return jsonify({"error": str(e)}), 500

    # ============================================================
    # ADMIN: DELETE ADMIN USER
    # ============================================================
    @app.route("/admin/users/admin/<int:admin_id>", methods=["DELETE"])
    def admin_delete_admin(admin_id):
        """Delete an admin user (admin only)"""
        # Check admin authentication
        if session.get('user_type') != 'admin':
            return jsonify({"error": "Admin access required"}), 403

        try:
            # Prevent self-deletion
            current_admin_id = session.get('user_id')
            if current_admin_id == admin_id:
                return jsonify({"error": "Cannot delete your own admin account"}), 400

            conn = get_db()
            c = conn.cursor()

            # Check if admin exists
            c.execute("SELECT username, email FROM admins WHERE id = ?", (admin_id,))
            admin = c.fetchone()

            if not admin:
                return jsonify({"error": "Admin not found"}), 404

            admin_username = admin[0]

            # Check if this is the last admin
            c.execute("SELECT COUNT(*) FROM admins WHERE is_active = 1")
            admin_count = c.fetchone()[0]

            if admin_count <= 1:
                return jsonify({
                    "error": "Cannot delete the last admin account. Create another admin first."
                }), 400

            # Delete admin
            c.execute("DELETE FROM admins WHERE id = ?", (admin_id,))
            conn.commit()

            logger.info(f"🗑️ Admin deleted admin user: {admin_username} (ID: {admin_id})")

            return jsonify({
                "success": True,
                "message": f"Admin {admin_username} deleted successfully"
            }), 200

        except Exception as e:
            logger.error(f"Error deleting admin: {e}")
            return jsonify({"error": str(e)}), 500

        # ============================================================
        # ADMIN: DELETE PATIENT
        # ============================================================
        @app.route("/admin/users/patient/<int:patient_id>", methods=["DELETE"])
        def admin_delete_patient(patient_id):
            """Delete a patient (admin only)"""
            # Check admin authentication
            if session.get('user_type') != 'admin':
                return jsonify({"error": "Admin access required"}), 403

            try:
                conn = get_db()
                c = conn.cursor()

                # Check if patient exists
                c.execute("SELECT full_name, email FROM patients WHERE id = ?", (patient_id,))
                patient = c.fetchone()

                if not patient:
                    return jsonify({"error": "Patient not found"}), 404

                patient_name = patient[0]

                # Delete patient (cascade will handle related records)
                c.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
                conn.commit()

                logger.info(f"🗑️ Admin deleted patient: {patient_name} (ID: {patient_id})")

                return jsonify({
                    "success": True,
                    "message": f"Patient {patient_name} deleted successfully"
                }), 200

            except Exception as e:
                logger.error(f"Error deleting patient: {e}")
                return jsonify({"error": str(e)}), 500

        # ============================================================
        # ADMIN: DELETE HOSPITAL
        # ============================================================
        @app.route("/admin/users/hospital/<int:hospital_id>", methods=["DELETE"])
        def admin_delete_hospital(hospital_id):
            """Delete a hospital (admin only)"""
            # Check admin authentication
            if session.get('user_type') != 'admin':
                return jsonify({"error": "Admin access required"}), 403

            try:
                conn = get_db()
                c = conn.cursor()

                # Check if hospital exists
                c.execute("SELECT hospital_name, hospital_code FROM hospitals WHERE id = ?", (hospital_id,))
                hospital = c.fetchone()

                if not hospital:
                    return jsonify({"error": "Hospital not found"}), 404

                hospital_name = hospital[0]

                # Check if hospital has patients
                c.execute("SELECT COUNT(*) FROM patients WHERE hospital_id = ?", (hospital_id,))
                patient_count = c.fetchone()[0]

                if patient_count > 0:
                    return jsonify({
                        "error": f"Cannot delete hospital with {patient_count} active patients. Please delete or transfer patients first."
                    }), 400

                # Delete hospital (cascade will handle related records)
                c.execute("DELETE FROM hospitals WHERE id = ?", (hospital_id,))
                conn.commit()

                logger.info(f"🗑️ Admin deleted hospital: {hospital_name} (ID: {hospital_id})")

                return jsonify({
                    "success": True,
                    "message": f"Hospital {hospital_name} deleted successfully"
                }), 200

            except Exception as e:
                logger.error(f"Error deleting hospital: {e}")
                return jsonify({"error": str(e)}), 500

        # ============================================================
        # ADMIN: DELETE ADMIN USER
        # ============================================================
        @app.route("/admin/users/admin/<int:admin_id>", methods=["DELETE"])
        def admin_delete_admin(admin_id):
            """Delete an admin user (admin only)"""
            # Check admin authentication
            if session.get('user_type') != 'admin':
                return jsonify({"error": "Admin access required"}), 403

            try:
                # Prevent self-deletion
                current_admin_id = session.get('user_id')
                if current_admin_id == admin_id:
                    return jsonify({"error": "Cannot delete your own admin account"}), 400

                conn = get_db()
                c = conn.cursor()

                # Check if admin exists
                c.execute("SELECT username, email FROM admins WHERE id = ?", (admin_id,))
                admin = c.fetchone()

                if not admin:
                    return jsonify({"error": "Admin not found"}), 404

                admin_username = admin[0]

                # Check if this is the last admin
                c.execute("SELECT COUNT(*) FROM admins WHERE is_active = 1")
                admin_count = c.fetchone()[0]

                if admin_count <= 1:
                    return jsonify({
                        "error": "Cannot delete the last admin account. Create another admin first."
                    }), 400

                # Delete admin
                c.execute("DELETE FROM admins WHERE id = ?", (admin_id,))
                conn.commit()

                logger.info(f"🗑️ Admin deleted admin user: {admin_username} (ID: {admin_id})")

                return jsonify({
                    "success": True,
                    "message": f"Admin {admin_username} deleted successfully"
                }), 200

            except Exception as e:
                logger.error(f"Error deleting admin: {e}")
                return jsonify({"error": str(e)}), 500

            @app.before_request
            def log_session_debug():
                """Debug middleware to check session state"""
                if request.endpoint and 'hospital' in request.endpoint:
                    app.logger.info(f"🔍 Request: {request.method} {request.endpoint}")
                    app.logger.info(f"   Session: {dict(session)}")
                    app.logger.info(f"   user_type: {session.get('user_type')}")
                    app.logger.info(f"   user_id: {session.get('user_id')}")
                    app.logger.info(f"   hospital_id: {session.get('hospital_id')}")
                    app.logger.info(f"   Cookies: {request.cookies.keys()}")


# OpenCV and image processing
try:
    import cv2
    import numpy as np
    from PIL import Image

    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    logger.warning("⚠️ OpenCV not available. Grad-CAM will be disabled.")

# Matplotlib for visualization
try:
    import matplotlib

    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("⚠️ Matplotlib not available. Grad-CAM will be disabled.")

# ReportLab for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, Image as RLImage, PageBreak
    )
    from reportlab.lib import colors

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("⚠️ ReportLab not available. PDF generation will be disabled.")

# PyTorch for Grad-CAM
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch imports failed. Grad-CAM will be disabled.")

# ===============================================================================
# 1. GRAD-CAM CLASS (Only if dependencies available)
# ===============================================================================

if CV_AVAILABLE and MATPLOTLIB_AVAILABLE and TORCH_AVAILABLE:

    class GradCAM:
        """Generate Grad-CAM visualizations for CNN models"""

        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None

            # Register hooks
            self.target_layer.register_forward_hook(self.save_activation)
            self.target_layer.register_backward_hook(self.save_gradient)

        def save_activation(self, module, input, output):
            self.activations = output.detach()

        def save_gradient(self, module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def generate_cam(self, input_tensor, target_class=None):
            """Generate Class Activation Map"""
            self.model.eval()
            output = self.model(input_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            class_loss = output[0, target_class]
            class_loss.backward()

            gradients = self.gradients
            activations = self.activations

            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            return cam.squeeze().cpu().numpy()


    def generate_gradcam_visualization(model, image_path, save_path=None, device='cpu'):
        """Generate and save Grad-CAM visualization"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            input_tensor = transform(img).unsqueeze(0).to(device)

            if hasattr(model, 'features'):
                target_layer = model.features[-1]
            else:
                target_layer = list(model.children())[-2]

            gradcam = GradCAM(model, target_layer)
            cam = gradcam.generate_cam(input_tensor)

            cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))

            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            overlay = heatmap * 0.4 + img_array * 0.6
            overlay = np.uint8(overlay)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_array)
            axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(heatmap)
            axes[1].set_title('Activation Map', fontsize=14, fontweight='bold')
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
            axes[2].axis('off')

            plt.tight_layout()

            if save_path is None:
                save_path = image_path.replace('.jpg', '_gradcam.jpg').replace('.png', '_gradcam.png')

            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Grad-CAM saved to: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"❌ Grad-CAM generation failed: {e}")
            raise


# ===============================================================================
# 2. GRAD-CAM ENDPOINT (with availability check)
# ===============================================================================

# Add this corrected endpoint to app.py (replace existing)
@app.route("/gradcam/<int:scan_id>", methods=["GET"])
@login_required
def get_gradcam_image(scan_id):
    """Generate and return GradCAM visualization"""
    try:
        user_type = session.get('user_type')
        user_id = session.get('user_id') or session.get('patient_id')

        conn = get_db()
        c = conn.cursor()

        # Get scan with authorization check
        if user_type == 'hospital':
            c.execute("""
                SELECT s.*, p.hospital_id 
                FROM mri_scans s
                JOIN patients p ON s.patient_id = p.id
                WHERE s.id = ? AND p.hospital_id = ?
            """, (scan_id, session.get('hospital_id')))
        elif user_type == 'patient':
            c.execute("""
                SELECT s.*, p.hospital_id 
                FROM mri_scans s
                JOIN patients p ON s.patient_id = p.id
                WHERE s.id = ? AND p.id = ?
            """, (scan_id, session.get('patient_id')))
        else:
            c.execute("SELECT * FROM mri_scans WHERE id = ?", (scan_id,))

        scan = c.fetchone()
        conn.close()

        if not scan:
            return jsonify({"error": "Scan not found"}), 404

        # Check if scan_image exists (base64)
        if not scan.get('scan_image'):
            return jsonify({"error": "Scan image not found"}), 404

        # Decode base64 image
        try:
            image_data = base64.b64decode(scan['scan_image'])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            logger.error(f"Image decode error: {e}")
            return jsonify({"error": "Failed to decode scan image"}), 500

        # Generate GradCAM
        try:
            # Import here to avoid circular imports
            from gradcam_utils import generate_gradcam_from_tensor

            # Prepare image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            input_tensor = transform(image).unsqueeze(0).to(device)

            # Generate GradCAM
            model.eval()
            overlaid_image, prediction_idx = generate_gradcam_from_tensor(
                model=model,
                input_tensor=input_tensor,
                original_image=image,
                target_class=None
            )

            # Convert to bytes
            buffer = io.BytesIO()
            Image.fromarray(overlaid_image).save(buffer, format='PNG')
            buffer.seek(0)

            return send_file(
                buffer,
                mimetype='image/png',
                as_attachment=False,
                download_name=f'gradcam_scan_{scan_id}.png'
            )

        except ImportError:
            return jsonify({
                "error": "GradCAM module not available",
                "message": "gradcam_utils.py is missing"
            }), 503
        except Exception as e:
            logger.error(f"GradCAM generation error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Failed to generate GradCAM: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"GradCAM route error: {e}")
        return jsonify({"error": str(e)}), 500
# ===============================================================================
# 3. PDF GENERATION (with availability check)
# ===============================================================================

if REPORTLAB_AVAILABLE:

    def generate_pdf_report(scan_data, patient_data, hospital_data, output_path):
        """Generate comprehensive PDF report"""
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )

            elements = []
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1e40af'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )

            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#1e40af'),
                spaceAfter=12,
                spaceBefore=12,
                fontName='Helvetica-Bold'
            )

            # Header
            elements.append(Paragraph("NEUROSCAN", title_style))
            elements.append(Paragraph("Brain Tumor Detection Report", styles['Heading2']))
            elements.append(Spacer(1, 0.2 * inch))

            # Report metadata
            report_info = [
                ['Report Generated:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
                ['Report ID:', f"RPT-{scan_data['id']:06d}"],
                ['Hospital:', hospital_data.get('hospital_name', 'N/A')]
            ]

            report_table = Table(report_info, colWidths=[2 * inch, 4 * inch])
            report_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))

            elements.append(report_table)
            elements.append(Spacer(1, 0.3 * inch))

            # Patient Information
            elements.append(Paragraph("Patient Information", heading_style))

            patient_info = [
                ['Patient Name:', patient_data.get('full_name', 'N/A')],
                ['Patient ID:', patient_data.get('patient_code', 'N/A')],
                ['Date of Birth:', patient_data.get('date_of_birth', 'N/A')],
                ['Gender:', patient_data.get('gender', 'N/A')],
                ['Phone:', patient_data.get('phone', 'N/A')],
            ]

            patient_table = Table(patient_info, colWidths=[2 * inch, 4 * inch])
            patient_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))

            elements.append(patient_table)
            elements.append(Spacer(1, 0.3 * inch))

            # Diagnosis
            elements.append(Paragraph("Diagnosis Results", heading_style))

            prediction = scan_data['prediction']
            confidence = float(scan_data['confidence'])

            if prediction.lower() == 'no tumor':
                result_color = colors.HexColor('#10b981')
                result_text = '✓ No Tumor Detected'
            else:
                result_color = colors.HexColor('#ef4444')
                result_text = f'⚠ {prediction.title()} Detected'

            diagnosis_style = ParagraphStyle(
                'Diagnosis',
                parent=styles['Normal'],
                fontSize=18,
                textColor=result_color,
                spaceAfter=12,
                fontName='Helvetica-Bold',
                alignment=TA_CENTER
            )

            elements.append(Paragraph(result_text, diagnosis_style))
            elements.append(Paragraph(f"Confidence: {confidence:.1f}%", styles['Normal']))
            elements.append(Spacer(1, 0.2 * inch))

            # Probabilities
            if scan_data.get('probabilities'):
                probs = scan_data['probabilities']

                prob_data = [
                    ['Classification', 'Probability'],
                    ['Glioma', f"{float(probs.get('glioma', 0)):.2f}%"],
                    ['Meningioma', f"{float(probs.get('meningioma', 0)):.2f}%"],
                    ['Pituitary Tumor', f"{float(probs.get('pituitary', 0)):.2f}%"],
                    ['No Tumor', f"{float(probs.get('no_tumor', 0)):.2f}%"],
                ]

                prob_table = Table(prob_data, colWidths=[3 * inch, 2 * inch])
                prob_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))

                elements.append(prob_table)

            # Build PDF
            doc.build(elements)
            logger.info(f"✅ PDF generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"❌ PDF generation failed: {e}")
            raise


# ===============================================================================
# 4. PDF DOWNLOAD ENDPOINT
# ===============================================================================

@app.route("/scan/<int:scan_id>/download-report", methods=["GET"])
@hospital_required
def download_scan_report(scan_id):
    """Generate and download PDF report"""

    if not REPORTLAB_AVAILABLE:
        return jsonify({
            "error": "PDF generation not available. Install reportlab package."
        }), 503

    try:
        hospital_id = session.get("hospital_id")

        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT s.*, p.*, h.hospital_name
            FROM mri_scans s
            JOIN patients p ON s.patient_id = p.id
            JOIN hospitals h ON p.hospital_id = h.id
            WHERE s.id = ? AND p.hospital_id = ?
        """, (scan_id, hospital_id))

        row = c.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Scan not found"}), 404

        data = dict(row)

        scan_data = {
            'id': data['id'],
            'prediction': data['prediction'],
            'confidence': data['confidence'],
            'probabilities': json.loads(data.get('probabilities', '{}')),
            'created_at': data['created_at'],
            'image_path': data.get('image_path', '')
        }

        patient_data = {
            'full_name': data['full_name'],
            'patient_code': data['patient_code'],
            'date_of_birth': data.get('date_of_birth', 'N/A'),
            'gender': data.get('gender', 'N/A'),
            'phone': data.get('phone', 'N/A')
        }

        hospital_data = {
            'hospital_name': data['hospital_name']
        }

        pdf_dir = "reports"
        os.makedirs(pdf_dir, exist_ok=True)

        pdf_filename = f"NeuroScan_Report_Scan{scan_id}.pdf"
        pdf_path = os.path.join(pdf_dir, pdf_filename)

        generate_pdf_report(scan_data, patient_data, hospital_data, pdf_path)

        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=pdf_filename
        )

    except Exception as e:
        logger.error(f"❌ PDF download error: {e}")
        return jsonify({"error": str(e)}), 500


# ===============================================================================
# 5. TUMOR PROGRESSION ENDPOINT (No special dependencies)
# ===============================================================================

@app.route("/patient/<int:patient_id>/progression", methods=["GET"])
@hospital_required
def get_tumor_progression(patient_id):
    """Get tumor progression analysis"""
    try:
        hospital_id = session.get("hospital_id")

        conn = get_db()
        c = conn.cursor()

        c.execute("""
            SELECT id FROM patients 
            WHERE id = ? AND hospital_id = ?
        """, (patient_id, hospital_id))

        if not c.fetchone():
            conn.close()
            return jsonify({"error": "Patient not found"}), 404

        c.execute("""
            SELECT id, prediction, confidence, probabilities, created_at
            FROM mri_scans
            WHERE patient_id = ?
            ORDER BY created_at ASC
        """, (patient_id,))

        scans = [dict(row) for row in c.fetchall()]
        conn.close()

        if len(scans) < 2:
            return jsonify({
                "message": "At least 2 scans required for progression analysis",
                "scans_count": len(scans)
            })

        progression_data = []

        for i in range(1, len(scans)):
            prev_scan = scans[i - 1]
            curr_scan = scans[i]

            prev_probs = json.loads(prev_scan.get('probabilities', '{}'))
            curr_probs = json.loads(curr_scan.get('probabilities', '{}'))

            prev_tumor_prob = (
                    float(prev_probs.get('glioma', 0)) +
                    float(prev_probs.get('meningioma', 0)) +
                    float(prev_probs.get('pituitary', 0))
            )

            curr_tumor_prob = (
                    float(curr_probs.get('glioma', 0)) +
                    float(curr_probs.get('meningioma', 0)) +
                    float(curr_probs.get('pituitary', 0))
            )

            tumor_prob_change = curr_tumor_prob - prev_tumor_prob
            confidence_change = float(curr_scan['confidence']) - float(prev_scan['confidence'])

            prev_date = datetime.fromisoformat(prev_scan['created_at'].replace('Z', '+00:00'))
            curr_date = datetime.fromisoformat(curr_scan['created_at'].replace('Z', '+00:00'))
            days_between = (curr_date - prev_date).days

            if tumor_prob_change > 5:
                trend = 'increasing'
                severity = 'high' if tumor_prob_change > 15 else 'medium'
            elif tumor_prob_change < -5:
                trend = 'decreasing'
                severity = 'low'
            else:
                trend = 'stable'
                severity = 'low'

            progression_data.append({
                'scan_index': i,
                'previous_scan_id': prev_scan['id'],
                'current_scan_id': curr_scan['id'],
                'days_between': days_between,
                'tumor_probability_change': round(tumor_prob_change, 2),
                'confidence_change': round(confidence_change, 2),
                'trend': trend,
                'severity': severity
            })

        latest_metric = progression_data[-1]

        summary = {
            'total_scans': len(scans),
            'latest_trend': latest_metric['trend'],
            'requires_attention': latest_metric['trend'] == 'increasing'
        }

        return jsonify({
            'summary': summary,
            'progression': progression_data,
            'scans': scans
        })

    except Exception as e:
        logger.error(f"❌ Progression tracking error: {e}")
        return jsonify({"error": str(e)}), 500
# ==============================================
# RUN SERVER WITH SOCKETIO
# ==============================================
migrate_database_schema()
if __name__ == "__main__":
    print("🚀 Starting NeuroScan Platform with Real-time Chat...")
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=True,
        allow_unsafe_werkzeug=True  # Needed for debug mode with SocketIO
    )