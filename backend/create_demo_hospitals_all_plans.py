#!/usr/bin/env python3
"""
Create Demo Hospitals with Different Subscription Plans
Perfect for demonstrating subscription tier differences during defense
"""

import sqlite3
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta

DB_FILE = "neuroscan_platform.db"


def create_demo_hospitals_all_tiers():
    """Create demo hospitals for each subscription tier"""
    print("\n" + "=" * 70)
    print("🏥 Creating Demo Hospitals - All Subscription Tiers")
    print("=" * 70)
    print()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Get all subscription plans
    c.execute(
        "SELECT id, name, display_name, max_scans_per_month, max_users, max_patients FROM subscription_plans ORDER BY price_monthly ASC")
    plans = c.fetchall()

    demo_hospitals = [
        {
            'hospital_name': 'Free Trial Clinic',
            'hospital_code': 'FREE001',
            'email': 'free@neuroscan.demo',
            'password': 'free123',
            'contact_person': 'Dr. Sarah Johnson',
            'phone': '+1-555-1001',
            'address': '100 Trial Street, Test City',
            'plan_name': 'free'
        },
        {
            'hospital_name': 'Basic Health Center',
            'hospital_code': 'BASIC001',
            'email': 'basic@neuroscan.demo',
            'password': 'basic123',
            'contact_person': 'Dr. Michael Chen',
            'phone': '+1-555-2001',
            'address': '200 Basic Avenue, Healthcare Town',
            'plan_name': 'basic'
        },
        {
            'hospital_name': 'Professional Medical Center',
            'hospital_code': 'PRO001',
            'email': 'pro@neuroscan.demo',
            'password': 'pro123',
            'contact_person': 'Dr. Emily Rodriguez',
            'phone': '+1-555-3001',
            'address': '300 Professional Boulevard, Med City',
            'plan_name': 'professional'
        },
        {
            'hospital_name': 'Enterprise Medical Network',
            'hospital_code': 'ENTER001',
            'email': 'enterprise@neuroscan.demo',
            'password': 'enterprise123',
            'contact_person': 'Dr. James Anderson',
            'phone': '+1-555-4001',
            'address': '400 Enterprise Plaza, Healthcare District',
            'plan_name': 'enterprise'
        }
    ]

    created_hospitals = []

    for hospital_data in demo_hospitals:
        plan_name = hospital_data['plan_name']

        # Get plan details
        c.execute("""
            SELECT id, display_name, price_monthly, max_scans_per_month, 
                   max_users, max_patients 
            FROM subscription_plans WHERE name=?
        """, (plan_name,))
        plan = c.fetchone()

        if not plan:
            print(f"   ⚠️  Plan '{plan_name}' not found, skipping {hospital_data['hospital_name']}")
            continue

        plan_id, plan_display_name, price, max_scans, max_users, max_patients = plan

        print(f"\n🏥 Creating: {hospital_data['hospital_name']}")
        print(f"   Plan: {plan_display_name} (${price}/month)")

        try:
            # Create hospital
            c.execute("""
                INSERT INTO hospitals 
                (hospital_name, hospital_code, email, password, 
                 contact_person, phone, address)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                hospital_data['hospital_name'],
                hospital_data['hospital_code'],
                hospital_data['email'],
                generate_password_hash(hospital_data['password']),
                hospital_data['contact_person'],
                hospital_data['phone'],
                hospital_data['address']
            ))

            hospital_id = c.lastrowid

            # Create subscription
            start_date = datetime.now()
            end_date = start_date + timedelta(days=30)

            c.execute("""
                INSERT INTO hospital_subscriptions
                (hospital_id, plan_id, status, billing_cycle,
                 current_period_start, current_period_end,
                 trial_ends_at, is_trial)
                VALUES (?, ?, 'active', 'monthly', ?, ?, ?, ?)
            """, (hospital_id, plan_id, start_date.strftime('%Y-%m-%d'),
                  end_date.strftime('%Y-%m-%d'),
                  end_date.strftime('%Y-%m-%d'), 1 if plan_name == 'free' else 0))

            subscription_id = c.lastrowid

            # Create usage tracking
            c.execute("""
                INSERT INTO usage_tracking
                (hospital_id, subscription_id, period_start, period_end,
                 scans_used, scans_limit, users_count, users_limit,
                 patients_count, patients_limit, is_current)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (hospital_id, subscription_id,
                  start_date.strftime('%Y-%m-%d'),
                  end_date.strftime('%Y-%m-%d'),
                  0, max_scans, 0, max_users, 0, max_patients, 1))

            print(f"   ✅ Created successfully")
            print(f"      Code: {hospital_data['hospital_code']}")
            print(f"      Password: {hospital_data['password']}")
            print(f"      Limits: {max_scans if max_scans != -1 else 'Unlimited'} scans, "
                  f"{max_users if max_users != -1 else 'Unlimited'} users, "
                  f"{max_patients if max_patients != -1 else 'Unlimited'} patients")

            created_hospitals.append({
                'name': hospital_data['hospital_name'],
                'code': hospital_data['hospital_code'],
                'password': hospital_data['password'],
                'plan': plan_display_name,
                'price': price
            })

        except sqlite3.IntegrityError as e:
            print(f"   ℹ️  Already exists: {hospital_data['hospital_name']}")
            print(f"      (Use code: {hospital_data['hospital_code']}, password: {hospital_data['password']})")

    conn.commit()
    conn.close()

    # Print summary
    print("\n" + "=" * 70)
    print("✅ Demo Hospitals Created!")
    print("=" * 70)
    print("\n📋 HOSPITAL LOGIN CREDENTIALS:")
    print("-" * 70)

    for hosp in created_hospitals:
        print(f"\n{hosp['name']}")
        print(f"  Plan: {hosp['plan']} (${hosp['price']}/month)")
        print(f"  Code: {hosp['code']}")
        print(f"  Password: {hosp['password']}")

    if not created_hospitals:
        # Show credentials for existing hospitals
        print("\nAll hospitals already exist. Use these credentials:")
        print("\n🆓 FREE TIER:")
        print("  Code: FREE001")
        print("  Password: free123")
        print("\n💼 BASIC TIER:")
        print("  Code: BASIC001")
        print("  Password: basic123")
        print("\n⭐ PROFESSIONAL TIER:")
        print("  Code: PRO001")
        print("  Password: pro123")
        print("\n🏆 ENTERPRISE TIER:")
        print("  Code: ENTER001")
        print("  Password: enterprise123")

    print("\n" + "-" * 70)
    print("\n🎓 FOR YOUR DEFENSE DEMO:")
    print("-" * 70)
    print("\n1. Login as Admin (demo_admin / admin123)")
    print("2. Go to 'Hospitals' tab")
    print("3. Show the different hospitals with different plans")
    print("4. Click 'Edit' on any hospital to upgrade/downgrade plans")
    print("5. Show how limits change based on subscription tier")
    print("\n" + "=" * 70)
    print()


def create_patients_for_demo_hospitals():
    """Create sample patients for each demo hospital"""
    print("\n" + "=" * 70)
    print("👥 Creating Sample Patients for Demo Hospitals")
    print("=" * 70)
    print()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Get all demo hospitals
    demo_codes = ['FREE001', 'BASIC001', 'PRO001', 'ENTER001']

    for code in demo_codes:
        c.execute("SELECT id, hospital_name FROM hospitals WHERE hospital_code=?", (code,))
        hospital = c.fetchone()

        if not hospital:
            continue

        hospital_id, hospital_name = hospital

        print(f"\n🏥 {hospital_name} ({code})")

        # Create 2 sample patients per hospital
        patients = [
            {
                'full_name': f'John Smith ({code})',
                'patient_code': f'{code}_PAT001',
                'access_code': '111111',
                'email': f'john.smith.{code.lower()}@neuroscan.demo',
                'phone': '+1-555-0001',
                'date_of_birth': '1980-01-15',
                'gender': 'male'
            },
            {
                'full_name': f'Mary Johnson ({code})',
                'patient_code': f'{code}_PAT002',
                'access_code': '222222',
                'email': f'mary.johnson.{code.lower()}@neuroscan.demo',
                'phone': '+1-555-0002',
                'date_of_birth': '1975-06-20',
                'gender': 'female'
            }
        ]

        for patient_data in patients:
            try:
                c.execute("""
                    INSERT INTO patients
                    (hospital_id, full_name, patient_code, access_code, email,
                     phone, date_of_birth, gender)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hospital_id,
                    patient_data['full_name'],
                    patient_data['patient_code'],
                    patient_data['access_code'],
                    patient_data['email'],
                    patient_data['phone'],
                    patient_data['date_of_birth'],
                    patient_data['gender']
                ))

                patient_id = c.lastrowid

                # Create patient access code
                expires_at = datetime.now() + timedelta(days=30)
                c.execute("""
                    INSERT INTO patient_access_codes 
                    (patient_id, access_code, expires_at)
                    VALUES (?, ?, ?)
                """, (patient_id, patient_data['access_code'], expires_at.strftime('%Y-%m-%d %H:%M:%S')))

                print(f"   ✅ {patient_data['full_name']}")
                print(f"      Patient Code: {patient_data['patient_code']}")
                print(f"      Access Code: {patient_data['access_code']}")

            except sqlite3.IntegrityError:
                print(f"   ℹ️  {patient_data['full_name']} already exists")

    conn.commit()
    conn.close()

    print("\n" + "=" * 70)
    print("✅ Sample Patients Created!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    try:
        # Check if subscription_plans exist
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM subscription_plans")
        if c.fetchone()[0] == 0:
            conn.close()
            print("\n❌ ERROR: No subscription plans found!")
            print("Please run 'python setup_database.py' first.\n")
            exit(1)
        conn.close()

        # Create demo hospitals with all subscription tiers
        create_demo_hospitals_all_tiers()

        # Ask if user wants to create sample patients
        print("\n📝 Would you like to create sample patients for these hospitals?")
        response = input("This will help with full demo (y/n): ").lower()

        if response in ['y', 'yes']:
            create_patients_for_demo_hospitals()

        print("\n🎉 Setup complete! You can now:")
        print("   1. Login as admin (demo_admin / admin123)")
        print("   2. View all hospitals in the admin portal")
        print("   3. See different subscription tiers in action")
        print("   4. Upgrade/downgrade plans to show flexibility")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)