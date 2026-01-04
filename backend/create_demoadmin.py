#!/usr/bin/env python3
"""
Create Demo Data for NeuroScan Admin Portal
This script creates sample accounts for demonstration purposes
"""

import sqlite3
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta

DB_FILE = "neuroscan_platform.db"


def create_demo_accounts():
    """Create demo accounts for all three user types"""
    print("\n" + "=" * 70)
    print("🎭 Creating Demo Accounts for NeuroScan")
    print("=" * 70)
    print()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # 1. Create demo admin
    print("1️⃣ Creating demo admin account...")
    try:
        c.execute("""
            INSERT INTO users (username, email, password, role)
            VALUES (?, ?, ?, ?)
        """, ('demo_admin', 'admin@neuroscan.demo', generate_password_hash('admin123'), 'admin'))
        print("   ✅ Demo admin created")
        print("      Username: demo_admin")
        print("      Password: admin123")
    except sqlite3.IntegrityError:
        print("   ℹ️  Demo admin already exists")

    # 2. Create demo hospital
    print("\n2️⃣ Creating demo hospital account...")
    try:
        c.execute("""
            INSERT INTO hospitals 
            (hospital_name, hospital_code, email, password, 
             contact_person, phone, address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'Demo General Hospital',
            'DEMO001',
            'hospital@neuroscan.demo',
            generate_password_hash('hospital123'),
            'Dr. John Smith',
            '+1-555-0123',
            '123 Medical Center Dr, Healthcare City'
        ))
        hospital_id = c.lastrowid
        print("   ✅ Demo hospital created")
        print("      Hospital Code: DEMO001")
        print("      Password: hospital123")

        # Assign free plan to demo hospital
        c.execute("SELECT id FROM subscription_plans WHERE name='free'")
        plan = c.fetchone()

        if plan:
            plan_id = plan[0]
            start_date = datetime.now()
            end_date = start_date + timedelta(days=30)

            c.execute("""
                INSERT INTO hospital_subscriptions
                (hospital_id, plan_id, status, billing_cycle,
                 current_period_start, current_period_end,
                 trial_ends_at, is_trial)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (hospital_id, plan_id, 'active', 'monthly',
                  start_date.date(), end_date.date(),
                  end_date.date(), 1))

            subscription_id = c.lastrowid

            # Create usage tracking
            c.execute("""
                INSERT INTO usage_tracking
                (hospital_id, subscription_id, period_start, period_end,
                 scans_used, scans_limit, users_count, users_limit,
                 patients_count, patients_limit, is_current)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (hospital_id, subscription_id, start_date.date(), end_date.date(),
                  0, 10, 0, 2, 0, 50, 1))

            print("   ✅ Assigned free trial subscription")
        else:
            print("   ⚠️  Warning: Could not find free plan")

    except sqlite3.IntegrityError:
        print("   ℹ️  Demo hospital already exists")
        c.execute("SELECT id FROM hospitals WHERE hospital_code='DEMO001'")
        result = c.fetchone()
        hospital_id = result[0] if result else None

    # 3. Create demo patient
    if hospital_id:
        print("\n3️⃣ Creating demo patient account...")
        try:
            c.execute("""
                INSERT INTO patients
                (hospital_id, full_name, patient_code, access_code, email,
                 phone, date_of_birth, gender)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hospital_id,
                'Jane Doe',
                'PAT001',
                '123456',
                'patient@neuroscan.demo',
                '+1-555-0124',
                '1985-03-15',
                'female'
            ))

            patient_id = c.lastrowid

            # Create patient access code entry
            expires_at = datetime.now() + timedelta(days=30)
            c.execute("""
                INSERT INTO patient_access_codes 
                (patient_id, access_code, expires_at)
                VALUES (?, ?, ?)
            """, (patient_id, '123456', expires_at))

            print("   ✅ Demo patient created")
            print("      Hospital Code: DEMO001")
            print("      Patient Code: PAT001")
            print("      Access Code: 123456")
        except sqlite3.IntegrityError as e:
            print(f"   ℹ️  Demo patient already exists ({e})")
    else:
        print("\n3️⃣ Skipping patient creation (hospital not found)")

    conn.commit()
    conn.close()

    print("\n" + "=" * 70)
    print("✅ Demo Data Created Successfully!")
    print("=" * 70)
    print("\n📝 LOGIN CREDENTIALS:")
    print("-" * 70)
    print("ADMIN:")
    print("  • Username: demo_admin")
    print("  • Password: admin123")
    print()
    print("HOSPITAL:")
    print("  • Hospital Code: DEMO001")
    print("  • Password: hospital123")
    print()
    print("PATIENT:")
    print("  • Hospital Code: DEMO001")
    print("  • Patient Code: PAT001")
    print("  • Access Code: 123456")
    print("-" * 70)
    print("\n🚀 You can now log in with these credentials!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    try:
        # Check if subscription_plans table exists
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='subscription_plans'")
        if not c.fetchone():
            conn.close()
            print("\n❌ ERROR: Subscription tables not found!")
            print("Please run 'python setup_database.py' first.\n")
            exit(1)

        # Check if plans exist
        c.execute("SELECT COUNT(*) FROM subscription_plans")
        if c.fetchone()[0] == 0:
            conn.close()
            print("\n❌ ERROR: No subscription plans found!")
            print("Please run 'python setup_database.py' first.\n")
            exit(1)

        conn.close()

        # Create demo accounts
        create_demo_accounts()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)