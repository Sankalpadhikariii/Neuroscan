"""
New Database Schema for Multi-Tenant Hospital Management System
Run this script to migrate your database to the new structure
"""

import sqlite3
from datetime import datetime

DB_FILE = "neuroscan_platform.db"


def create_new_database():
    """Create complete multi-tenant database schema"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # ==============================================
    # ADMIN TABLES
    # ==============================================

    # Admin accounts (your team)
    c.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT,
            role TEXT DEFAULT 'admin',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)

    # ==============================================
    # HOSPITAL TABLES
    # ==============================================

    # Hospital accounts (your clients)
    c.execute("""
        CREATE TABLE IF NOT EXISTS hospitals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_name TEXT NOT NULL,
            hospital_code TEXT UNIQUE NOT NULL,
            contact_person TEXT,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            address TEXT,
            city TEXT,
            country TEXT,
            status TEXT DEFAULT 'active',
            subscription_tier TEXT DEFAULT 'basic',
            max_scans_per_month INTEGER DEFAULT 100,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER,
            FOREIGN KEY (created_by) REFERENCES admins(id)
        )
    """)

    # Hospital users (doctors/staff at each hospital)
    c.execute("""
        CREATE TABLE IF NOT EXISTS hospital_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT DEFAULT 'doctor',
            specialization TEXT,
            license_number TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE
        )
    """)

    # ==============================================
    # PATIENT TABLES
    # ==============================================

    # Patient records
    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            patient_code TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT,
            date_of_birth DATE,
            gender TEXT,
            address TEXT,
            emergency_contact TEXT,
            emergency_phone TEXT,
            assigned_doctor_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE,
            FOREIGN KEY (assigned_doctor_id) REFERENCES hospital_users(id),
            FOREIGN KEY (created_by) REFERENCES hospital_users(id)
        )
    """)

    # Patient verification codes (for chat access)
    c.execute("""
        CREATE TABLE IF NOT EXISTS patient_access_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            access_code TEXT UNIQUE NOT NULL,
            verification_code TEXT,
            is_verified BOOLEAN DEFAULT 0,
            expires_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            verified_at TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
        )
    """)

    # ==============================================
    # MRI SCAN TABLES
    # ==============================================

    # MRI Scans (predictions)
    c.execute("""
        CREATE TABLE IF NOT EXISTS mri_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            patient_id INTEGER NOT NULL,
            uploaded_by INTEGER NOT NULL,
            scan_image TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            is_tumor BOOLEAN NOT NULL,
            probabilities TEXT,
            notes TEXT,
            scan_date DATE,
            status TEXT DEFAULT 'completed',
            reviewed_by INTEGER,
            reviewed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
            FOREIGN KEY (uploaded_by) REFERENCES hospital_users(id),
            FOREIGN KEY (reviewed_by) REFERENCES hospital_users(id)
        )
    """)

    # ==============================================
    # CHAT SYSTEM TABLES
    # ==============================================

    # Chat conversations
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            hospital_id INTEGER NOT NULL,
            status TEXT DEFAULT 'active',
            last_message_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
            FOREIGN KEY (doctor_id) REFERENCES hospital_users(id),
            FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE
        )
    """)

    # Chat messages
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            sender_type TEXT NOT NULL,
            sender_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            is_read BOOLEAN DEFAULT 0,
            read_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES chat_conversations(id) ON DELETE CASCADE
        )
    """)

    # ==============================================
    # ANALYTICS & LOGS TABLES
    # ==============================================

    # Usage statistics
    c.execute("""
        CREATE TABLE IF NOT EXISTS usage_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            date DATE NOT NULL,
            total_scans INTEGER DEFAULT 0,
            total_tumor_detected INTEGER DEFAULT 0,
            total_chats INTEGER DEFAULT 0,
            total_patients INTEGER DEFAULT 0,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE,
            UNIQUE(hospital_id, date)
        )
    """)

    # Activity logs
    c.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_type TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            hospital_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,
            ip_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Bug reports / feedback
    c.execute("""
        CREATE TABLE IF NOT EXISTS bug_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER,
            reported_by INTEGER,
            user_type TEXT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            severity TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'open',
            assigned_to INTEGER,
            resolved_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(id),
            FOREIGN KEY (assigned_to) REFERENCES admins(id)
        )
    """)

    # Create indexes for better performance
    c.execute("CREATE INDEX IF NOT EXISTS idx_hospital_users_hospital ON hospital_users(hospital_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_patients_hospital ON patients(hospital_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_scans_hospital ON mri_scans(hospital_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_scans_patient ON mri_scans(patient_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON chat_messages(conversation_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_usage_hospital_date ON usage_stats(hospital_id, date)")

    conn.commit()
    conn.close()

    print("✅ Database schema created successfully!")
    print(f"📁 Database file: {DB_FILE}")
    print("\nTables created:")
    print("  ✓ admins")
    print("  ✓ hospitals")
    print("  ✓ hospital_users")
    print("  ✓ patients")
    print("  ✓ patient_access_codes")
    print("  ✓ mri_scans")
    print("  ✓ chat_conversations")
    print("  ✓ chat_messages")
    print("  ✓ usage_stats")
    print("  ✓ activity_logs")
    print("  ✓ bug_reports")


def create_default_admin():
    """Create default admin account for first login"""
    from werkzeug.security import generate_password_hash

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Check if admin exists
    c.execute("SELECT id FROM admins WHERE username='admin'")
    if c.fetchone():
        print("⚠️  Admin account already exists")
        conn.close()
        return

    # Create default admin
    password_hash = generate_password_hash("admin123")
    c.execute("""
        INSERT INTO admins (username, email, password, full_name, role)
        VALUES (?, ?, ?, ?, ?)
    """, ("admin", "admin@neuroscan.com", password_hash, "System Administrator", "superadmin"))

    conn.commit()
    conn.close()

    print("\n✅ Default admin created:")
    print("   Username: admin")
    print("   Password: admin123")
    print("   ⚠️  CHANGE THIS PASSWORD IMMEDIATELY!")


def create_sample_hospital():
    """Create a sample hospital for testing"""
    from werkzeug.security import generate_password_hash
    import random
    import string

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Create sample hospital
    hospital_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

    c.execute("""
        INSERT INTO hospitals (
            hospital_name, hospital_code, contact_person, email, 
            phone, city, country, created_by
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "City General Hospital",
        hospital_code,
        "Dr. John Smith",
        "admin@cityhospital.com",
        "+1234567890",
        "New York",
        "USA",
        1  # Created by default admin
    ))

    hospital_id = c.lastrowid

    # Create sample doctor
    password_hash = generate_password_hash("doctor123")
    c.execute("""
        INSERT INTO hospital_users (
            hospital_id, username, email, password, full_name,
            role, specialization
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        hospital_id,
        "dr.smith",
        "dr.smith@cityhospital.com",
        password_hash,
        "Dr. John Smith",
        "doctor",
        "Neurology"
    ))

    conn.commit()
    conn.close()

    print("\n✅ Sample hospital created:")
    print(f"   Hospital: City General Hospital")
    print(f"   Code: {hospital_code}")
    print(f"   Doctor Username: dr.smith")
    print(f"   Doctor Password: doctor123")


if __name__ == "__main__":
    print("=" * 60)
    print("NeuroScan Multi-Tenant Database Setup")
    print("=" * 60)
    print()

    create_new_database()
    create_default_admin()
    create_sample_hospital()

    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Backup your old database (brain_tumor.db)")
    print("2. Run the new backend with this database")
    print("3. Login as admin to create hospitals")
    print("=" * 60)