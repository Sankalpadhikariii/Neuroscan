"""
Complete Database Fix for Patient Login
Adds ALL missing columns needed for patient authentication
"""

import sqlite3
import os

DB_PATH = 'neuroscan_platform.db'


def fix_patient_access_codes():
    """Add all missing columns to patient_access_codes table"""

    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return False

    print("🔧 Fixing patient_access_codes table...")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        # Check existing columns
        c.execute("PRAGMA table_info(patient_access_codes)")
        columns = [row[1] for row in c.fetchall()]

        print(f"\n📋 Current columns ({len(columns)}):")
        for col in columns:
            print(f"   • {col}")

        # List of all columns needed
        required_columns = {
            'verification_code': 'TEXT',
            'verification_code_expiry': 'DATETIME',
            'is_verified': 'INTEGER DEFAULT 0',
            'hospital_id': 'INTEGER',
            'patient_code': 'TEXT',
        }

        print(f"\n🔍 Checking required columns...\n")

        changes_made = False

        for col_name, col_type in required_columns.items():
            if col_name not in columns:
                print(f"➕ Adding {col_name}...")
                c.execute(f"ALTER TABLE patient_access_codes ADD COLUMN {col_name} {col_type}")
                conn.commit()
                print(f"✅ Added {col_name}")
                changes_made = True
            else:
                print(f"✅ {col_name} already exists")

        if not changes_made:
            print("\n✨ All columns already exist!")

        # Verify final state
        c.execute("PRAGMA table_info(patient_access_codes)")
        final_columns = [row[1] for row in c.fetchall()]

        print(f"\n📋 Final columns ({len(final_columns)}):")
        for col in final_columns:
            print(f"   • {col}")

        print("\n" + "=" * 60)
        print("✅ DATABASE FIX COMPLETE!")
        print("=" * 60)
        print("\n🎯 Next steps:")
        print("   1. Restart Flask server: python app.py")
        print("   2. Try patient login")
        print("   3. Verification code should work now!")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("NeuroScan Complete Database Fix")
    print("=" * 60 + "\n")

    success = fix_patient_access_codes()

    if not success:
        print("\n⚠️ Fix failed. Check the error above.")
    else:
        print("\n✅ All done! Restart your server.")