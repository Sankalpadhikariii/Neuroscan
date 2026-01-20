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

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

CORS(app,
     resources={
         r"/*": {
             "origins": ["http://localhost:3000", "http://localhost:3001", "http://192.168.1.70:3000"],
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization"],
             "supports_credentials": True,
             "expose_headers": ["Content-Type"]
         }
     }
     )

app.secret_key = SECRET_KEY

# SocketIO with permissive CORS for development
socketio = SocketIO(
    app,

    cors_allowed_origins="http://localhost:3000",

    logger=False,
    engineio_logger=False
)

# Limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["50 per minute"]
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model loading moved to lines 3009-3079 where VGG19_BrainTumor class is properly defined
# This early loading is not needed and causes conflicts

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
    """Enhanced disconnect handler"""
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
        # Notify others that user is offline
        socketio.emit('user_status', {
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


@app.route('/notifications/<int:notification_id>/mark-read', methods=['POST'])
def mark_notification_read(notification_id):
    """Mark a notification as read"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user_id = session['user_id']

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE notifications 
            SET is_read = 1
            WHERE id = ? AND user_id = ?
        ''', (notification_id, user_id))

        conn.commit()
        conn.close()

        return jsonify({'success': True})

    except Exception as e:
        logging.error(f"Error marking notification as read: {e}")
        return jsonify({'error': 'Failed to update notification'}), 500


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
# GRAD-CAM ROUTES
# ==============================================

@app.route('/gradcam/<int:scan_id>', methods=['GET'])
def get_gradcam(scan_id):
    """Get Grad-CAM visualization for a scan"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get scan info
        cursor.execute('''
            SELECT gradcam_path, image_path 
            FROM scans 
            WHERE id = ?
        ''', (scan_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({'error': 'Scan not found'}), 404

        gradcam_path = row[0]

        if gradcam_path and os.path.exists(gradcam_path):
            return send_file(gradcam_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Grad-CAM not available'}), 404

    except Exception as e:
        logging.error(f"Error fetching Grad-CAM: {e}")
        return jsonify({'error': 'Failed to fetch Grad-CAM'}), 500


# ==============================================
# PATIENT SCAN HISTORY ROUTES
# ==============================================

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
        if generate_gradcam:
            try:
                from gradcam_utils import generate_gradcam_from_tensor
                gradcam_img, _ = generate_gradcam_from_tensor(
                    model, input_tensor, image, target_class=predicted_idx
                )

                gradcam_filename = f"gradcam_{timestamp}_{filename}"
                gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
                Image.fromarray(gradcam_img).save(gradcam_path)
            except Exception as e:
                logging.error(f"Grad-CAM generation failed: {e}")

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
# @app.route('/api/notifications', methods=['GET'])
# @login_required
# def get_notifications_api():
# """Get notifications for current user"""
# try:
# user_id = session.get('user_id') or session.get('patient_id')
# user_type = session.get('user_type') or 'patient'

# Get query parameters
# limit = request.args.get('limit', 50, type=int)
# offset = request.args.get('offset', 0, type=int)
# unread_only = request.args.get('unread_only', 'false').lower() == 'true'

# conn = get_db()
# c = conn.cursor()

# Build query
# query = """
#   SELECT * FROM notifications
#  WHERE user_id=? AND user_type=?
# """
# params = [user_id, user_type]

# if unread_only:
#   query += " AND is_read=0"

# query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
# params.extend([limit, offset])

# c.execute(query, params)
# notifications = [dict(row) for row in c.fetchall()]

# Get unread count
# c.execute("""
##  SELECT COUNT(*) as count FROM notifications
##WHERE user_id=? AND user_type=? AND is_read=0
##""", (user_id, user_type))

##unread_count = c.fetchone()['count']

## conn.close()

## return jsonify({
##'notifications': notifications,
##'unread_count': unread_count,
##'total': len(notifications)
## })

##except Exception as e:
## logger.error(f"❌ Error getting notifications: {e}")
##return jsonify({'error': str(e)}), 500


# DUPLICATE - @app.route('/api/notifications/<int:notification_id>/read', methods=['PUT'])
# DUPLICATE - @login_required
# DUPLICATE - def mark_notification_read(notification_id):
# DUPLICATE - """Mark a notification as read"""
# DUPLICATE - try:
# DUPLICATE - user_id = session.get('user_id') or session.get('patient_id')
# DUPLICATE -         # DUPLICATE - conn = get_db()
# DUPLICATE - c = conn.cursor()
# DUPLICATE -         # Verify ownership
# DUPLICATE - c.execute("""
# DUPLICATE - SELECT id FROM notifications
# DUPLICATE - WHERE id=? AND user_id=?
# DUPLICATE - """, (notification_id, user_id))
# DUPLICATE -         # DUPLICATE - if not c.fetchone():
# DUPLICATE - conn.close()
# DUPLICATE - return jsonify({'error': 'Notification not found'}), 404
# DUPLICATE -         # Mark as read
# DUPLICATE - c.execute("""
# DUPLICATE - UPDATE notifications
# DUPLICATE - SET is_read=1, read_at=CURRENT_TIMESTAMP
# DUPLICATE - WHERE id=?
# DUPLICATE - """, (notification_id,))
# DUPLICATE -         # DUPLICATE - conn.commit()
# DUPLICATE - conn.close()
# DUPLICATE -         # DUPLICATE - return jsonify({'message': 'Notification marked as read'})
# DUPLICATE -     # DUPLICATE - except Exception as e:
# DUPLICATE - logger.error(f"❌ Error marking notification read: {e}")
# DUPLICATE - return jsonify({'error': str(e)}), 500
# DUPLICATE -  # DUPLICATE - @app.route('/api/notifications/read-all', methods=['PUT'])
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


# ============================================
# FIND YOUR /hospital/predict ROUTE (around line 1390)
# REPLACE THE ENTIRE FUNCTION with this:
# ============================================

# Replace your /hospital/predict endpoint in app.py with this fixed version
# ==============================================
# REPLACE YOUR EXISTING /hospital/predict ROUTE WITH THIS
# Search for "@app.route("/hospital/predict", methods=["POST"])"
# and replace the entire function with this fixed version
# ==============================================

@app.route("/hospital/predict", methods=["POST"])
@hospital_required
@check_scan_limit
def predict():
    """Fixed prediction endpoint with proper notification handling"""

    # Get hospital_id from session
    hospital_id = session.get("hospital_id")
    user_id = session.get("user_id")

    # Verify hospital_id exists
    if not hospital_id:
        logger.error("❌ hospital_id not found in session")
        return jsonify({
            "error": "Session error: Hospital ID not found. Please log in again."
        }), 401

    try:
        # Validate image upload
        if "image" not in request.files:
            return jsonify({"error": "No image"}), 400

        patient_id = request.form.get("patient_id")
        if not patient_id:
            return jsonify({"error": "Patient ID required"}), 400

        # ============================================
        # STEP 1: Read and open the image
        # ============================================
        image_bytes = request.files["image"].read()
        image_stream = io.BytesIO(image_bytes)
        image_stream.seek(0)
        image = Image.open(image_stream).convert("RGB")

        # ============================================
        # STEP 2: VALIDATE IMAGE *BEFORE* PREDICTION
        # ============================================
        logger.info("🔍 Validating uploaded image...")
        is_valid, error_msg, warning_msg = validate_brain_scan(image)

        if not is_valid:
            logger.warning(f"❌ Validation failed: {error_msg}")
            return jsonify({
                "error": "Invalid Image",
                "message": error_msg,
                "suggestion": "Please upload a grayscale brain MRI scan from medical imaging equipment."
            }), 400

        if warning_msg:
            logger.info(f"⚠️ Warning: {warning_msg}")

        logger.info("✅ Image validation passed")

        # ============================================
        # STEP 3: Process image and run prediction
        # ============================================
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

        # ============================================
        # STEP 4: Secondary validation with probabilities
        # ============================================
        is_valid_result, error_msg_result, warning_msg_result = validate_brain_scan(image, probabilities)

        if not is_valid_result:
            logger.warning(f"❌ Post-prediction validation failed: {error_msg_result}")
            return jsonify({
                "error": "Invalid Scan Result",
                "message": error_msg_result,
                "suggestion": "The AI model could not confidently classify this as a brain scan."
            }), 400

        # Combine warnings
        if warning_msg_result:
            warning_msg = f"{warning_msg}. {warning_msg_result}" if warning_msg else warning_msg_result

        confidence_percent = round(conf_val * 100, 2)
        logger.info(f"📊 Prediction: {prediction_label} ({confidence_percent}%)")

        # ============================================
        # STEP 5: Save to database WITH HOSPITAL_ID
        # ============================================
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

        # ============================================
        # STEP 6: Update usage and notifications
        # ============================================
        increment_usage(hospital_id, 'scans', 1)

        # 🔥 FIX: Properly formatted notification creation
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

        # Create notification with ALL required parameters
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

        return jsonify({
            "scan_id": scan_id,
            "prediction": prediction_label,
            "confidence": confidence_percent,
            "is_tumor": is_tumor,
            "probabilities": probabilities,
            "validation_warning": warning_msg
        })

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()

        # Send error notification with ALL required parameters
        try:
            create_notification(
                user_id=user_id,
                user_type='hospital',
                notification_type='error',
                title='❌ Prediction Failed',
                message=f'Failed to analyze MRI scan: {str(e)}',
                hospital_id=hospital_id,
                priority='high'
            )
        except Exception as notif_error:
            logger.error(f"Failed to create error notification: {notif_error}")

        return jsonify({
            "error": str(e)
        }), 500


# ==============================================
# SUBSCRIPTION NOTIFICATIONS
# ==============================================

def notify_usage_limit_warning(hospital_id: int, usage_percent: float):
    """Notify when approaching usage limits"""
    notify_hospital_users(
        hospital_id=hospital_id,
        notification_type='usage_warning',
        title='⚠️ Usage Limit Warning',
        message=f'You have used {usage_percent}% of your monthly scan limit. Consider upgrading your plan.',
        priority='normal'
    )


def notify_usage_limit_reached(hospital_id: int):
    """Notify when usage limit is reached"""
    notify_hospital_users(
        hospital_id=hospital_id,
        notification_type='usage_limit',
        title='🚫 Monthly Limit Reached',
        message='You have reached your monthly scan limit. Upgrade your plan to continue scanning.',
        priority='high'
    )


def notify_subscription_expiring(hospital_id: int, days_remaining: int):
    """Notify when subscription is expiring"""
    notify_hospital_users(
        hospital_id=hospital_id,
        notif_type='subscription_expiring',
        title='⏰ Subscription Expiring Soon',
        message=f'Your subscription will expire in {days_remaining} days. Renew to avoid service interruption.',
        priority='high'
    )


# ==============================================
# SETUP COMMAND
# ==============================================

@app.cli.command()
def setup_notifications():
    """CLI command to setup notifications table"""
    create_notifications_table()
    print("✅ Notifications system initialized")


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
        1: {"monthly": os.getenv('STRIPE_PRICE_BASIC_MONTHLY'), "yearly": os.getenv('STRIPE_PRICE_BASIC_YEARLY')},
        2: {"monthly": os.getenv('STRIPE_PRICE_PRO_MONTHLY'), "yearly": os.getenv('STRIPE_PRICE_PRO_YEARLY')},
        3: {"monthly": os.getenv('STRIPE_PRICE_ENTERPRISE_MONTHLY'),
            "yearly": os.getenv('STRIPE_PRICE_ENTERPRISE_YEARLY')},
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

@app.route('/admin/users/admin/<int:user_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_admin_user(user_id):
    """Delete an admin user"""
    try:
        # Prevent deleting yourself
        if user_id == session['user_id']:
            return jsonify({'error': 'Cannot delete your own account'}), 400

        conn = get_db()
        c = conn.cursor()

        # Check if user exists and get role
        c.execute('SELECT id, role FROM users WHERE id=?', (user_id,))
        user = c.fetchone()
        if not user:
            conn.close()
            return jsonify({'error': 'User not found'}), 404

        # Only superadmin can delete superadmin accounts
        if user[1] == 'superadmin' and session.get('role') != 'superadmin':
            conn.close()
            return jsonify({'error': 'Insufficient permissions'}), 403

        # Delete user
        c.execute('DELETE FROM users WHERE id=?', (user_id,))
        conn.commit()
        conn.close()

        log_activity('admin', session['user_id'], 'deleted_admin_user',
                     f"Deleted admin user ID: {user_id}")

        return jsonify({'message': 'Admin user deleted successfully'})

    except Exception as e:
        logger.error(f"Error deleting admin user: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin/users/hospital/<int:hospital_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_hospital_account(hospital_id):
    """Delete a hospital account and all related data"""
    try:
        conn = get_db()
        c = conn.cursor()

        # Check if hospital exists
        c.execute('SELECT id FROM hospitals WHERE id=?', (hospital_id,))
        if not c.fetchone():
            conn.close()
            return jsonify({'error': 'Hospital not found'}), 404

        # Delete hospital (cascades to related tables if FK set)
        c.execute('DELETE FROM hospitals WHERE id=?', (hospital_id,))
        conn.commit()
        conn.close()

        log_activity('admin', session['user_id'], 'deleted_hospital',
                     f"Deleted hospital ID: {hospital_id}")

        return jsonify({'message': 'Hospital account deleted successfully'})

    except Exception as e:
        logger.error(f"Error deleting hospital: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin/users/patient/<int:patient_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_patient_record(patient_id):
    """Delete a patient record"""
    try:
        conn = get_db()
        c = conn.cursor()

        # Check if patient exists
        c.execute('SELECT id, hospital_id FROM patients WHERE id=?', (patient_id,))
        patient = c.fetchone()
        if not patient:
            conn.close()
            return jsonify({'error': 'Patient not found'}), 404

        hospital_id = patient[1]

        # Delete patient (cascades to scans)
        c.execute('DELETE FROM patients WHERE id=?', (patient_id,))

        # Update usage tracking
        try:
            c.execute("""
                UPDATE usage_tracking
                SET patients_count = GREATEST(0, patients_count - 1)
                WHERE hospital_id = ? AND is_current = 1
            """, (hospital_id,))
        except Exception as e:
            logger.warning(f"Could not update usage tracking: {e}")

        conn.commit()
        conn.close()

        log_activity('admin', session['user_id'], 'deleted_patient',
                     f"Deleted patient ID: {patient_id}")

        return jsonify({'message': 'Patient record deleted successfully'})

    except Exception as e:
        logger.error(f"Error deleting patient: {e}")
        return jsonify({'error': str(e)}), 500


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


def get_detailed_usage(hospital_id):
    """
    Get comprehensive usage information for a hospital
    Returns detailed status including limits, cooldowns, etc.
    """
    conn = get_db()
    c = conn.cursor()

    # Get subscription
    c.execute("""
        SELECT hs.*, sp.name as plan_name, sp.display_name, 
               sp.max_scans_per_month, sp.price_monthly
        FROM hospital_subscriptions hs
        JOIN subscription_plans sp ON hs.plan_id = sp.id
        WHERE hs.hospital_id = ? AND hs.status = 'active'
        ORDER BY hs.created_at DESC
        LIMIT 1
    """, (hospital_id,))
    subscription = c.fetchone()

    if not subscription:
        conn.close()
        return {
            'is_blocked': True,
            'block_message': 'No active subscription',
            'can_scan': False
        }

    subscription = dict(subscription)

    # Get usage
    c.execute("""
        SELECT * FROM usage_tracking
        WHERE hospital_id = ? AND is_current = 1
        ORDER BY period_start DESC
        LIMIT 1
    """, (hospital_id,))
    usage = c.fetchone()

    if not usage:
        conn.close()
        return {
            'is_blocked': True,
            'block_message': 'Usage tracking not found',
            'can_scan': False
        }

    usage = dict(usage)

    scans_used = usage['scans_used']
    scans_limit = usage['scans_limit']

    # Check for unlimited
    if scans_limit == -1:
        conn.close()
        return {
            'is_blocked': False,
            'can_scan': True,
            'scans_used': scans_used,
            'scans_limit': -1,
            'is_unlimited': True,
            'plan_name': subscription['display_name'],
            'usage_percent': 0
        }

    # Calculate usage percentage
    usage_percent = (scans_used / scans_limit * 100) if scans_limit > 0 else 0

    # Check if blocked
    is_blocked = scans_used >= scans_limit

    # Check cooldown (for free tier)
    cooldown_active = False
    cooldown_ends = None

    if subscription['plan_name'] == 'free' and is_blocked:
        # Check last scan time
        c.execute("""
            SELECT created_at FROM mri_scans
            WHERE hospital_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (hospital_id,))
        last_scan = c.fetchone()

        if last_scan:
            last_scan_time = datetime.strptime(last_scan['created_at'], '%Y-%m-%d %H:%M:%S')
            cooldown_ends = last_scan_time + timedelta(hours=24)
            cooldown_active = datetime.now() < cooldown_ends

    # Period end date
    period_end = datetime.strptime(usage['period_end'], '%Y-%m-%d')
    days_until_reset = (period_end - datetime.now()).days

    # Generate appropriate message
    if is_blocked:
        if subscription['plan_name'] == 'free':
            if cooldown_active:
                hours_left = int((cooldown_ends - datetime.now()).total_seconds() / 3600)
                block_message = f"Free scan available in {hours_left} hours. Upgrade for unlimited scans!"
            else:
                block_message = f"Monthly limit reached. Resets in {days_until_reset} days or upgrade now!"
        else:
            block_message = f"Monthly limit of {scans_limit} scans reached. Upgrade to get more!"
    elif usage_percent >= 80:
        remaining = scans_limit - scans_used
        block_message = f"Only {remaining} scans remaining this month. Consider upgrading!"
    else:
        block_message = None

    conn.close()

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
    plan_identifier = data.get("plan_id")  # Can be ID or name
    billing_cycle = data.get("billing_cycle", "monthly")

    print(f"DEBUG: plan_identifier = {plan_identifier}, billing_cycle = {billing_cycle}")

    hospital_id = session["hospital_id"]

    conn = get_db()
    c = conn.cursor()

    # Try to find plan by ID (numeric) OR by name (string)
    if isinstance(plan_identifier, str) and not plan_identifier.isdigit():
        # Plan name provided (e.g., "basic", "professional")
        c.execute("SELECT * FROM subscription_plans WHERE name=?", (plan_identifier,))
    else:
        # Numeric ID provided
        c.execute("SELECT * FROM subscription_plans WHERE id=?", (plan_identifier,))

    plan = c.fetchone()

    if not plan:
        conn.close()
        print(f"ERROR: Plan not found for identifier: {plan_identifier}")
        return jsonify({"error": f"Plan '{plan_identifier}' not found"}), 404

    plan = dict(plan)
    print(f"SUCCESS: Found plan - {plan['name']} (ID: {plan['id']})")

    # Get the Stripe price ID
    price_id = get_stripe_price_id(plan["id"], billing_cycle)
    if not price_id:
        conn.close()
        print(f"ERROR: No Stripe price configured for plan {plan['id']}, cycle {billing_cycle}")
        return jsonify({"error": "Stripe price not configured for this plan"}), 400

    print(f"DEBUG: Using Stripe price_id: {price_id}")

    # Get or create Stripe customer
    customer_id = get_or_create_stripe_customer(hospital_id)
    print(f"DEBUG: Stripe customer_id: {customer_id}")

    try:
        # Create Stripe checkout session
        checkout_session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/subscription-cancelled",
            metadata={
                "hospital_id": hospital_id,
                "plan_id": plan["id"],
                "billing_cycle": billing_cycle
            }
        )

        conn.close()
        print(f"SUCCESS: Checkout session created: {checkout_session.id}")
        return jsonify({"url": checkout_session.url, "sessionId": checkout_session.id})

    except stripe.error.StripeError as e:
        conn.close()
        print(f"STRIPE ERROR: {str(e)}")
        return jsonify({"error": f"Stripe error: {str(e)}"}), 500
    except Exception as e:
        conn.close()
        print(f"UNEXPECTED ERROR: {str(e)}")
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
    """Fixed hospital login with proper session setup"""
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
        return jsonify({"error": "Invalid credentials"}), 401

    # ✅ CRITICAL FIX: Set ALL required session variables
    session.clear()  # Clear any old session data
    session["user_id"] = user["id"]
    session["user_type"] = "hospital"
    session["hospital_id"] = user["hospital_id"]  # ⚠️ MUST BE SET
    session["username"] = user["username"]
    session.permanent = True  # Make session persist

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
            "hospital_id": user["hospital_id"],  # ✅ Include in response
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
@check_scan_limit
def predict():
    hospital_id = session["hospital_id"]
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image"}), 400

        patient_id = request.form.get("patient_id")
        if not patient_id:
            return jsonify({"error": "Patient ID required"}), 400

        # 1. Read and process the image
        image_bytes = request.files["image"].read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        # 2. Run the AI Prediction
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)[0]
            conf, pred = torch.max(probs, 0)

        # 3. Format the results
        probabilities = {
            class_names[i]: round(float(probs[i].item()) * 100, 2)
            for i in range(len(class_names))
        }
        prediction_label = class_names[int(pred.item())]
        conf_val = float(conf.item())

        # 4. VALIDATION: Check if AI confidence is too low (Non-Brain Image)
        # We pass 'image' and the 'probabilities' we just calculated
        is_valid, error_msg, warning_msg = validate_brain_scan(image, probabilities)

        if not is_valid:
            return jsonify({
                "error": "Invalid Scan Result",
                "message": error_msg,
                "suggestion": "The AI does not recognize this as a brain MRI scan."
            }), 400

        # 5. Save to Database (Only if validation passed)
        is_tumor = prediction_label != "notumor"
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

        # 6. Send response back to Frontend
        increment_usage(hospital_id, 'scans', 1)
        log_activity("hospital", session["user_id"], "prediction", hospital_id=hospital_id)

        return jsonify({
            "scan_id": scan_id,
            "prediction": prediction_label,
            "confidence": round(conf_val * 100, 2),
            "probabilities": probabilities,
            "validation_warning": warning_msg  # <--- This triggers the yellow warning in UI
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


@app.route('/api/chat/conversations', methods=['GET'])
@login_required
def get_conversations():
    """Get all conversations for current user"""
    try:
        user_type = session.get('user_type') or 'patient'

        conn = get_db()
        c = conn.cursor()

        if user_type == 'hospital':
            # Get all patients this hospital user has messaged
            hospital_user_id = session.get('user_id')

            c.execute("""
                SELECT DISTINCT
                    p.id as patient_id,
                    p.full_name as patient_name,
                    p.patient_code,
                    p.email as patient_email,
                    (
                        SELECT m.message 
                        FROM messages m 
                        WHERE m.patient_id = p.id 
                        AND m.hospital_user_id = ?
                        ORDER BY m.created_at DESC 
                        LIMIT 1
                    ) as last_message,
                    (
                        SELECT m.created_at 
                        FROM messages m 
                        WHERE m.patient_id = p.id 
                        AND m.hospital_user_id = ?
                        ORDER BY m.created_at DESC 
                        LIMIT 1
                    ) as last_message_time,
                    (
                        SELECT COUNT(*) 
                        FROM messages m 
                        WHERE m.patient_id = p.id 
                        AND m.hospital_user_id = ?
                        AND m.sender_type = 'patient'
                        AND m.is_read = 0
                    ) as unread_count
                FROM patients p
                INNER JOIN messages m ON m.patient_id = p.id
                WHERE m.hospital_user_id = ?
                GROUP BY p.id
                ORDER BY last_message_time DESC
            """, (hospital_user_id, hospital_user_id, hospital_user_id, hospital_user_id))

            conversations = [dict(row) for row in c.fetchall()]

        else:  # patient
            # Get the doctor assigned to this patient
            patient_id = session.get('patient_id')

            c.execute("""
                SELECT DISTINCT
                    hu.id as hospital_user_id,
                    hu.username as doctor_name,
                    hu.email as doctor_email,
                    h.hospital_name,
                    (
                        SELECT m.message 
                        FROM messages m 
                        WHERE m.patient_id = ?
                        AND m.hospital_user_id = hu.id
                        ORDER BY m.created_at DESC 
                        LIMIT 1
                    ) as last_message,
                    (
                        SELECT m.created_at 
                        FROM messages m 
                        WHERE m.patient_id = ?
                        AND m.hospital_user_id = hu.id
                        ORDER BY m.created_at DESC 
                        LIMIT 1
                    ) as last_message_time,
                    (
                        SELECT COUNT(*) 
                        FROM messages m 
                        WHERE m.patient_id = ?
                        AND m.hospital_user_id = hu.id
                        AND m.sender_type = 'hospital'
                        AND m.is_read = 0
                    ) as unread_count
                FROM hospital_users hu
                INNER JOIN hospitals h ON hu.hospital_id = h.id
                INNER JOIN messages m ON m.hospital_user_id = hu.id
                WHERE m.patient_id = ?
                GROUP BY hu.id
                ORDER BY last_message_time DESC
            """, (patient_id, patient_id, patient_id, patient_id))

            conversations = [dict(row) for row in c.fetchall()]

        conn.close()

        return jsonify({
            'conversations': conversations
        })

    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
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


# ==============================================
# RUN SERVER WITH SOCKETIO
# ==============================================

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