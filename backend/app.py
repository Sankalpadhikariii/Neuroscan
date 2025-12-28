import os
import io
import base64
import sqlite3
import secrets
import logging
from pathlib import Path
from functools import wraps
import requests
from gradcam_utilis import generate_gradcam_visualization
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
    default_limits=["20 per minute"]
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
DB_FILE = "brain_tumor.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prediction TEXT,
            confidence REAL,
            is_tumor BOOLEAN,
            image_data TEXT,
            patient_name TEXT,
            patient_age INTEGER,
            patient_gender TEXT,
            patient_id TEXT,
            scan_date TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


init_db()


# -----------------------------
# Auth Decorators
# -----------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        return f(*args, **kwargs)

    return wrapper


# -----------------------------
# Auth Routes
# -----------------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not all([username, email, password]):
        return jsonify({"error": "Missing fields"}), 400

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT id FROM users WHERE username=? OR email=?", (username, email))
    if c.fetchone():
        conn.close()
        return jsonify({"error": "User exists"}), 400

    hashed = generate_password_hash(password)
    c.execute(
        "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
        (username, email, hashed)
    )
    conn.commit()

    user_id = c.lastrowid
    conn.close()

    session["user_id"] = user_id
    session["username"] = username
    session["role"] = "user"

    return jsonify({
        "user": {
            "id": user_id,
            "username": username,
            "role": "user"
        }
    })


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "SELECT id, password, role FROM users WHERE username=?",
        (username,)
    )
    user = c.fetchone()
    conn.close()

    if not user or not check_password_hash(user[1], password):
        return jsonify({"error": "Invalid credentials"}), 401

    session["user_id"] = user[0]
    session["username"] = username
    session["role"] = user[2]

    return jsonify({
        "user": {
            "id": user[0],
            "username": username,
            "role": user[2]
        }
    })


@app.route("/me", methods=["GET"])
def me():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    return jsonify({
        "user": {
            "id": session["user_id"],
            "username": session["username"],
            "role": session["role"]
        }
    })


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})


@app.route("/history", methods=["GET"])
@login_required
def history():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        SELECT prediction, confidence, created_at
        FROM predictions
        WHERE user_id=?
        ORDER BY created_at DESC
    """, (session["user_id"],))

    rows = c.fetchall()
    conn.close()

    return jsonify({
        "predictions": [
            {
                "prediction": r[0],
                "confidence": round(r[1] * 100, 2),
                "created_at": r[2]
            } for r in rows
        ]
    })


# -----------------------------
# CNN Model Definition
# -----------------------------
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


# -----------------------------
# Model Loading
# -----------------------------
MODEL_PATH = "Brain_Tumor_model.pt"
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    logger.info("Loading model from file...")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    logger.info(f"Model loaded successfully: {type(model)}")

    logger.info("Model structure:")
    for name, child in model.named_children():
        logger.info(f"  - {name}: {child.__class__.__name__}")

except Exception as e:
    logger.error(f"Error loading model: {e}")
    import traceback

    logger.error(traceback.format_exc())
    raise

model.to(device)
model.eval()


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image"}), 400

        image_bytes = request.files["image"].read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        patient_name = request.form.get('patient_name', '')
        patient_age = request.form.get('patient_age', '')
        patient_gender = request.form.get('patient_gender', '')
        patient_id = request.form.get('patient_id', '')
        scan_date = request.form.get('scan_date', '')
        notes = request.form.get('notes', '')

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.exp(output)[0]
            conf, pred = torch.max(probs, 0)

        pred_idx = int(pred.item())
        conf_val = float(conf.item())
        prediction_label = class_names[pred_idx]
        is_tumor = prediction_label != "notumor"

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            INSERT INTO predictions (
                user_id, prediction, confidence, is_tumor, image_data,
                patient_name, patient_age, patient_gender, patient_id, scan_date, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"],
            prediction_label,
            conf_val,
            is_tumor,
            base64.b64encode(image_bytes).decode(),
            patient_name,
            int(patient_age) if patient_age else None,
            patient_gender,
            patient_id,
            scan_date,
            notes
        ))
        conn.commit()
        prediction_id = c.lastrowid
        conn.close()

        return jsonify({
            "prediction_id": prediction_id,
            "prediction": prediction_label,
            "confidence": round(conf_val * 100, 2),
            "is_tumor": is_tumor,
            "probabilities": {
                class_names[i]: round(float(probs[i].item()) * 100, 2)
                for i in range(len(class_names))
            }
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# -----------------------------
# PDF Generation Route
# -----------------------------
@app.route("/generate-report/<int:prediction_id>", methods=["GET"])
@login_required
def generate_report(prediction_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        c.execute("""
            SELECT 
                p.prediction, p.confidence, p.image_data,
                p.patient_name, p.patient_age, p.patient_gender,
                p.patient_id, p.scan_date, p.notes,
                p.created_at
            FROM predictions p
            WHERE p.id = ? AND p.user_id = ?
        """, (prediction_id, session["user_id"]))

        row = c.fetchone()

        if not row:
            conn.close()
            return jsonify({"error": "Prediction not found"}), 404

        image_data = row[2]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.exp(output)[0]

        probabilities = {
            class_names[i]: round(float(probs[i].item()) * 100, 2)
            for i in range(len(class_names))
        }

        conn.close()

        prediction_data = {
            'prediction': row[0],
            'confidence': round(row[1] * 100, 2),
            'probabilities': probabilities
        }

        patient_info = {
            'name': row[3] or 'N/A',
            'age': row[4] or 'N/A',
            'gender': row[5] or 'N/A',
            'patient_id': row[6] or f'AUTO-{prediction_id}',
            'scan_date': row[7] or row[9].split()[0],
            'notes': row[8] or ''
        }

        pdf_buffer = generate_pdf_report(prediction_data, patient_info, image_data)

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'NeuroScan_Report_{patient_info["patient_id"]}_{prediction_id}.pdf'
        )

    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Chatbot Route
# -----------------------------
@app.route("/api/chatbot", methods=["POST", "OPTIONS"])
def chatbot():
    if request.method == "OPTIONS":
        return '', 200

    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    try:
        data = request.json
        user_message = data.get('message', '').strip()
        conversation_history = data.get('history', [])

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            SELECT prediction, confidence, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (session["user_id"],))
        latest_prediction = c.fetchone()
        conn.close()

        system_context = """You are a medical AI assistant for NeuroScan brain tumor detection system.

Your role:
- Explain brain tumor types (glioma, meningioma, pituitary tumors)
- Help interpret MRI scans
- Describe NeuroScan's CNN capabilities
- Provide general medical information

Guidelines:
- Give accurate medical information
- Remind users you're an AI, not a replacement for doctors
- Encourage consulting healthcare professionals
- Be empathetic and supportive
- If discussing diagnosis results, emphasize need for professional review

System info:
- Detects 4 classes: glioma, meningioma, pituitary tumor, normal tissue
- Uses CNN (VGG19 architecture) with ~94% accuracy
- Provides confidence scores and probability distributions
"""

        if latest_prediction:
            pred_type, confidence, scan_date = latest_prediction
            system_context += f"\n\nUser's latest scan: {pred_type} ({round(confidence * 100, 2)}%) on {scan_date}"

        full_prompt = f"{system_context}\n\n"

        for msg in conversation_history[-10:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if content:
                prefix = "User" if role == 'user' else "Assistant"
                full_prompt += f"{prefix}: {content}\n"

        full_prompt += f"User: {user_message}\nAssistant:"

        logger.info("Calling Ollama...")

        api_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7
                }
            },
            timeout=60
        )

        if api_response.status_code != 200:
            logger.error(f"Ollama error: {api_response.status_code}")
            return jsonify({
                "error": "AI service unavailable",
                "response": "Ollama is not responding. Make sure it's running: 'ollama serve'"
            }), 500

        api_data = api_response.json()
        assistant_message = api_data.get("response", "").strip()

        if not assistant_message:
            assistant_message = "I apologize, but I couldn't generate a response. Please try rephrasing your question."

        logger.info(f"Response generated: {len(assistant_message)} chars")

        return jsonify({
            "response": assistant_message,
            "success": True
        }), 200

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama")
        return jsonify({
            "error": "Connection error",
            "response": "Cannot connect to Ollama. Please run: ollama serve"
        }), 503

    except requests.exceptions.Timeout:
        logger.error("Ollama timeout")
        return jsonify({
            "error": "Request timeout",
            "response": "AI is taking too long. Please try again."
        }), 504

    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "response": "An error occurred. Please try again."
        }), 500


@app.route("/api/chatbot/health", methods=["GET"])
def chatbot_health():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.ok:
            models = response.json().get("models", [])
            return jsonify({
                "status": "online",
                "service": "Ollama (Local)",
                "models": [m.get("name") for m in models]
            }), 200
    except:
        pass

    return jsonify({
        "status": "offline",
        "service": "Ollama",
        "message": "Ollama not running. Execute: ollama serve"
    }), 503


# -----------------------------
# Admin Routes
# -----------------------------
def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        if session.get("role") != "superadmin":
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)

    return wrapper


@app.route("/admin/stats", methods=["GET"])
@admin_required
def admin_stats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM predictions WHERE is_tumor=1")
    tumor_detections = c.fetchone()[0]

    rate = round((tumor_detections / total_predictions) * 100, 2) if total_predictions else 0

    c.execute("""
        SELECT prediction, COUNT(*)
        FROM predictions
        GROUP BY prediction
    """)
    predictions_by_type = dict(c.fetchall())

    c.execute("""
        SELECT DATE(created_at), COUNT(*)
        FROM predictions
        WHERE created_at >= DATE('now', '-7 days')
        GROUP BY DATE(created_at)
    """)
    recent_activity = [
        {"date": row[0], "count": row[1]}
        for row in c.fetchall()
    ]

    conn.close()

    return jsonify({
        "total_users": total_users,
        "total_predictions": total_predictions,
        "tumor_detections": tumor_detections,
        "tumor_detection_rate": rate,
        "predictions_by_type": predictions_by_type,
        "recent_activity": recent_activity
    })


@app.route("/admin/users", methods=["GET"])
@admin_required
def admin_users():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        SELECT u.id, u.username, u.email, u.role, u.created_at,
               COUNT(p.id) as prediction_count
        FROM users u
        LEFT JOIN predictions p ON u.id = p.user_id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """)

    users = [
        {
            "id": r[0],
            "username": r[1],
            "email": r[2],
            "role": r[3],
            "created_at": r[4],
            "prediction_count": r[5]
        }
        for r in c.fetchall()
    ]

    conn.close()
    return jsonify({"users": users})


@app.route("/admin/predictions", methods=["GET"])
@admin_required
def admin_predictions():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        SELECT p.id, u.username, p.prediction, p.confidence,
               p.is_tumor, p.created_at
        FROM predictions p
        JOIN users u ON p.user_id = u.id
        ORDER BY p.created_at DESC
        LIMIT 100
    """)

    predictions = [
        {
            "id": r[0],
            "username": r[1],
            "prediction": r[2],
            "confidence": round(r[3] * 100, 2),
            "is_tumor": bool(r[4]),
            "created_at": r[5]
        }
        for r in c.fetchall()
    ]

    conn.close()
    return jsonify({"predictions": predictions})


# -----------------------------
# Grad-CAM Explainability Route
# -----------------------------
@app.route("/gradcam/<int:prediction_id>", methods=["GET"])
@login_required
def get_gradcam(prediction_id):
    """Generate Grad-CAM heatmap for a prediction"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        c.execute("""
            SELECT image_data, prediction
            FROM predictions
            WHERE id = ? AND user_id = ?
        """, (prediction_id, session["user_id"]))

        row = c.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Prediction not found"}), 404

        image_data = row[0]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        gradcam_result = generate_gradcam_visualization(
            model=model,
            image_tensor=tensor,
            original_image=image,
            target_layer=model.conv4,
            class_names=class_names
        )

        return jsonify(gradcam_result)

    except Exception as e:
        logger.error(f"Grad-CAM error: {str(e)}")
        return jsonify({"error": f"Grad-CAM failed: {str(e)}"}), 500

@app.route("/admin/users/<int:user_id>", methods=["DELETE"])
@admin_required
def delete_user(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("DELETE FROM predictions WHERE user_id=?", (user_id,))
    c.execute("DELETE FROM users WHERE id=?", (user_id,))

    conn.commit()
    conn.close()
    return jsonify({"message": "User deleted"})


@app.route("/admin/users/<int:user_id>/role", methods=["PUT"])
@admin_required
def update_role(user_id):
    role = request.json.get("role")
    if role not in ["user", "superadmin"]:
        return jsonify({"error": "Invalid role"}), 400

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
    conn.commit()
    conn.close()

    return jsonify({"message": "Role updated"})


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)