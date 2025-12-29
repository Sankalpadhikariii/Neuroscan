import os
import io
import base64
import sqlite3
import secrets
import logging
import random
import string
from pathlib import Path
from functools import wraps
from datetime import datetime, timedelta

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

from pdf_report import generate_pdf_report
from email_utilis import send_verification_email, send_welcome_email

# -----------------------------
# Environment
# -----------------------------
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(16)

# -----------------------------
# App Setup
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# -----------------------------
# Database
# -----------------------------
DB_FILE = "neuroscan_platform.db"


def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


# -----------------------------
# Auth Decorators
# -----------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session or "user_type" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        return f(*args, **kwargs)

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


# -----------------------------
# Utility Functions
# -----------------------------
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
    except:
        pass


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

    # Total statistics
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

    # Recent hospitals
    c.execute("""
        SELECT id, hospital_name, hospital_code, email, city, created_at
        FROM hospitals
        ORDER BY created_at DESC
        LIMIT 5
    """)
    recent_hospitals = [dict(row) for row in c.fetchall()]

    # Usage by hospital (top 5)
    c.execute("""
        SELECT h.hospital_name, COUNT(s.id) as scan_count
        FROM hospitals h
        LEFT JOIN mri_scans s ON h.id = s.hospital_id
        GROUP BY h.id
        ORDER BY scan_count DESC
        LIMIT 5
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
            GROUP BY h.id
            ORDER BY h.created_at DESC
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
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("hospital_name"),
        hospital_code,
        data.get("contact_person"),
        data.get("email"),
        data.get("phone"),
        data.get("address"),
        data.get("city"),
        data.get("country"),
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
        return jsonify({
            "hospital": dict(hospital),
            "users": users
        })

    elif request.method == "PUT":
        data = request.json
        c.execute("""
            UPDATE hospitals 
            SET hospital_name=?, contact_person=?, email=?, phone=?, 
                address=?, city=?, country=?, status=?
            WHERE id=?
        """, (
            data.get("hospital_name"),
            data.get("contact_person"),
            data.get("email"),
            data.get("phone"),
            data.get("address"),
            data.get("city"),
            data.get("country"),
            data.get("status", "active"),
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

    # Statistics
    c.execute("SELECT COUNT(*) as count FROM patients WHERE hospital_id=?", (hospital_id,))
    total_patients = c.fetchone()["count"]

    c.execute("SELECT COUNT(*) as count FROM mri_scans WHERE hospital_id=?", (hospital_id,))
    total_scans = c.fetchone()["count"]

    c.execute("""
        SELECT COUNT(*) as count FROM mri_scans 
        WHERE hospital_id=? AND is_tumor=1
    """, (hospital_id,))
    tumor_detections = c.fetchone()["count"]

    c.execute("""
        SELECT COUNT(*) as count FROM chat_conversations 
        WHERE hospital_id=? AND status='active'
    """, (hospital_id,))
    active_chats = c.fetchone()["count"]

    # Scans this month
    c.execute("""
        SELECT COUNT(*) as count FROM mri_scans 
        WHERE hospital_id=? 
        AND strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')
    """, (hospital_id,))
    scans_this_month = c.fetchone()["count"]

    # Recent scans
    c.execute("""
        SELECT s.*, p.full_name as patient_name, p.patient_code
        FROM mri_scans s
        JOIN patients p ON s.patient_id = p.id
        WHERE s.hospital_id=?
        ORDER BY s.created_at DESC
        LIMIT 10
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


@app.route("/hospital/patients", methods=["GET", "POST"])
@hospital_required
def hospital_patients():
    conn = get_db()
    c = conn.cursor()
    hospital_id = session["hospital_id"]

    if request.method == "GET":
        c.execute("""
            SELECT p.*, 
                   hu.full_name as doctor_name,
                   COUNT(s.id) as scan_count
            FROM patients p
            LEFT JOIN hospital_users hu ON p.assigned_doctor_id = hu.id
            LEFT JOIN mri_scans s ON p.id = s.patient_id
            WHERE p.hospital_id=?
            GROUP BY p.id
            ORDER BY p.created_at DESC
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
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        hospital_id,
        patient_code,
        data.get("full_name"),
        data.get("email"),
        data.get("phone"),
        data.get("date_of_birth"),
        data.get("gender"),
        data.get("address"),
        data.get("emergency_contact"),
        data.get("emergency_phone"),
        session["user_id"],
        session["user_id"]
    ))
    patient_id = c.lastrowid

    # Create access code for chat
    expires_at = datetime.now() + timedelta(days=30)
    c.execute("""
        INSERT INTO patient_access_codes (patient_id, access_code, expires_at)
        VALUES (?, ?, ?)
    """, (patient_id, access_code, expires_at))

    # Get hospital details for email
    c.execute("""
        SELECT hospital_name, hospital_code 
        FROM hospitals 
        WHERE id=?
    """, (hospital_id,))
    hospital = c.fetchone()

    # Get the created patient with scan_count
    c.execute("""
        SELECT p.*, 0 as scan_count
        FROM patients p
        WHERE p.id=?
    """, (patient_id,))
    patient = dict(c.fetchone())

    conn.commit()
    conn.close()

    # Send welcome email with credentials
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
            logger.warning(f"⚠️ Email not sent (not configured). Credentials shown in UI.")

    except Exception as e:
        logger.error(f"❌ Failed to send welcome email: {e}")
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
    """Get scan history for the hospital"""
    conn = get_db()
    c = conn.cursor()
    hospital_id = session["hospital_id"]

    c.execute("""
        SELECT 
            s.*,
            p.full_name as patient_name,
            p.patient_code,
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
# MRI SCAN / PREDICTION ROUTES (CNN Model)
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


MODEL_PATH = "Brain_Tumor_model.pt"
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


@app.route("/hospital/predict", methods=["POST"])
@hospital_required
def predict():
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
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session["hospital_id"],
            patient_id,
            session["user_id"],
            base64.b64encode(image_bytes).decode(),
            prediction_label,
            conf_val,
            is_tumor,
            str(probabilities),
            request.form.get("notes", ""),
            request.form.get("scan_date", datetime.now().strftime("%Y-%m-%d"))
        ))
        scan_id = c.lastrowid
        conn.commit()
        conn.close()

        log_activity("hospital", session["user_id"], "prediction", hospital_id=session["hospital_id"])

        return jsonify({
            "scan_id": scan_id,
            "prediction": prediction_label,
            "confidence": round(conf_val * 100, 2),
            "is_tumor": is_tumor,
            "probabilities": probabilities
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/generate-report/<int:scan_id>", methods=["GET"])
@login_required
def generate_report(scan_id):
    """Generate and download PDF report for a scan"""
    try:
        conn = get_db()
        c = conn.cursor()

        # Get scan data
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
        if session.get("user_type") == "hospital":
            if row["hospital_id"] != session.get("hospital_id"):
                return jsonify({"error": "Unauthorized"}), 403
        elif session.get("user_type") != "admin":
            return jsonify({"error": "Unauthorized"}), 403

        # Prepare data for PDF
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

        # Generate PDF
        try:
            pdf_buffer = generate_pdf_report(scan_data, patient_data, hospital_data)
        except Exception as e:
            logger.error(f"Error with reportlab, using simple PDF: {e}")
            from pdf_report import generate_simple_pdf_report
            pdf_buffer = generate_simple_pdf_report(scan_data, patient_data, hospital_data)

        # Return PDF
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'NeuroScan_Report_{scan_id}.pdf'
        )

    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return jsonify({"error": "Failed to generate report"}), 500


# ==============================================
# PATIENT CHAT ROUTES
# ==============================================

@app.route("/patient/verify", methods=["POST"])
def patient_verify():
    """Step 2: Verify patient with access code and send verification code"""
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

    c.execute("""
        UPDATE patient_access_codes 
        SET verification_code=?
        WHERE id=?
    """, (verification_code, patient["access_id"]))
    conn.commit()
    conn.close()

    # Send email with verification code
    try:
        send_verification_email(
            to_email=patient['email'],
            verification_code=verification_code,
            patient_name=patient['full_name'],
            hospital_name=patient['hospital_name']
        )
    except Exception as e:
        logger.error(f"Error sending verification email: {e}")

    # Always log for debugging (in case email fails)
    logger.info(f"Verification code for {patient['email']}: {verification_code}")

    return jsonify({
        "message": "Verification code sent to email",
        "email_hint": patient["email"][:3] + "***" + patient["email"][-10:]
    })


@app.route("/patient/login", methods=["POST"])
def patient_login():
    """Step 3: Login patient with verification code"""
    data = request.json
    patient_code = data.get("patient_code")
    verification_code = data.get("verification_code")

    conn = get_db()
    c = conn.cursor()

    c.execute("""
        SELECT p.*, h.hospital_name, pac.id as access_id
        FROM patients p
        JOIN hospitals h ON p.hospital_id = h.id
        JOIN patient_access_codes pac ON p.id = pac.patient_id
        WHERE p.patient_code=? AND pac.verification_code=?
    """, (patient_code, verification_code))

    patient = c.fetchone()

    if not patient:
        conn.close()
        return jsonify({"error": "Invalid verification code"}), 401

    # Mark as verified
    c.execute("""
        UPDATE patient_access_codes 
        SET is_verified=1, verified_at=datetime('now')
        WHERE id=?
    """, (patient["access_id"],))
    conn.commit()
    conn.close()

    session["patient_id"] = patient["id"]
    session["patient_type"] = "patient"

    return jsonify({
        "patient": {
            "id": patient["id"],
            "full_name": patient["full_name"],
            "patient_code": patient["patient_code"],
            "hospital_name": patient["hospital_name"],
            "type": "patient"
        }
    })


@app.route("/chat/send", methods=["POST"])
@login_required
def send_message():
    """Send chat message (from patient or doctor)"""
    data = request.json
    message = data.get("message")

    # Determine sender
    if session.get("user_type") == "hospital":
        conversation_id = data.get("conversation_id")
        sender_type = "doctor"
        sender_id = session["user_id"]
    elif session.get("patient_type") == "patient":
        conversation_id = data.get("conversation_id")
        sender_type = "patient"
        sender_id = session["patient_id"]
    else:
        return jsonify({"error": "Invalid sender"}), 400

    conn = get_db()
    c = conn.cursor()

    c.execute("""
        INSERT INTO chat_messages (conversation_id, sender_type, sender_id, message)
        VALUES (?, ?, ?, ?)
    """, (conversation_id, sender_type, sender_id, message))

    c.execute("""
        UPDATE chat_conversations 
        SET last_message_at=datetime('now')
        WHERE id=?
    """, (conversation_id,))

    conn.commit()
    message_id = c.lastrowid
    conn.close()

    return jsonify({"message_id": message_id})


@app.route("/chat/messages/<int:conversation_id>", methods=["GET"])
@login_required
def get_messages(conversation_id):
    """Get chat messages"""
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        SELECT * FROM chat_messages
        WHERE conversation_id=?
        ORDER BY created_at ASC
    """, (conversation_id,))

    messages = [dict(row) for row in c.fetchall()]
    conn.close()

    return jsonify({"messages": messages})


# ==============================================
# GENERAL ROUTES
# ==============================================

@app.route("/me", methods=["GET"])
def me():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    return jsonify({
        "user": {
            "id": session.get("user_id"),
            "type": session.get("user_type"),
            "username": session.get("username")
        }
    })


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})


# ==============================================
# RUN
# ==============================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)