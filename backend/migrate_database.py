#!/usr/bin/env python3
"""
Database Migration Script: Add Profile Picture Support to Patients Table
Run this script to add profile picture columns to your database
"""

import sqlite3
import sys

DB_FILE = "neuroscan_platform.db"


def run_migration():
    """Add profile picture columns to patients table"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        print("🔄 Starting database migration...")

        # Check if columns already exist
        cursor.execute("PRAGMA table_info(patients)")
        columns = [col[1] for col in cursor.fetchall()]

        migrations_needed = []

        if 'profile_picture' not in columns:
            migrations_needed.append(('profile_picture', 'TEXT'))

        if 'profile_picture_mime' not in columns:
            migrations_needed.append(('profile_picture_mime', 'VARCHAR(50)'))

        if 'updated_at' not in columns:
            # SQLite doesn't support DEFAULT CURRENT_TIMESTAMP in ALTER TABLE
            # We'll add it without a default, then use triggers for new inserts
            migrations_needed.append(('updated_at', 'TIMESTAMP'))

        if not migrations_needed:
            print("✅ All columns already exist. No migration needed.")
            conn.close()
            return True

        # Apply migrations
        for column_name, column_type in migrations_needed:
            try:
                sql = f"ALTER TABLE patients ADD COLUMN {column_name} {column_type}"
                cursor.execute(sql)
                print(f"✅ Added column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"⚠️  Column {column_name} already exists, skipping...")
                else:
                    raise

        # Update existing rows to set updated_at to current time
        if 'updated_at' in [col[0] for col in migrations_needed]:
            try:
                cursor.execute("""
                    UPDATE patients 
                    SET updated_at = CURRENT_TIMESTAMP 
                    WHERE updated_at IS NULL
                """)
                print("✅ Initialized updated_at for existing records")
            except sqlite3.Error as e:
                print(f"⚠️  Warning updating existing records: {e}")

        # Create trigger to auto-update updated_at on profile changes
        try:
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS update_patient_timestamp 
                AFTER UPDATE OF profile_picture, profile_picture_mime ON patients
                BEGIN
                    UPDATE patients 
                    SET updated_at = CURRENT_TIMESTAMP 
                    WHERE id = NEW.id;
                END;
            """)
            print("✅ Created trigger for automatic timestamp updates")
        except sqlite3.Error as e:
            print(f"⚠️  Trigger creation warning: {e}")

        # Create index for better performance
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patients_profile_picture 
                ON patients(id)
            """)
            print("✅ Created index for profile pictures")
        except sqlite3.Error as e:
            print(f"⚠️  Index creation warning: {e}")

        conn.commit()
        conn.close()

        print("✅ Migration completed successfully!")
        return True

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def verify_migration():
    """Verify the migration was successful"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(patients)")
        columns = {col[1]: col[2] for col in cursor.fetchall()}

        required_columns = ['profile_picture', 'profile_picture_mime', 'updated_at']

        print("\n🔍 Verifying migration...")
        all_present = True

        for col in required_columns:
            if col in columns:
                print(f"✅ Column '{col}' exists (type: {columns[col]})")
            else:
                print(f"❌ Column '{col}' is missing!")
                all_present = False

        # Check if trigger exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='trigger' AND name='update_patient_timestamp'
        """)
        trigger = cursor.fetchone()
        if trigger:
            print("✅ Auto-update trigger exists")
        else:
            print("⚠️  Auto-update trigger not found (not critical)")

        conn.close()

        if all_present:
            print("\n🎉 All required columns are present!")
        else:
            print("\n⚠️  Some columns are missing. Please check the migration.")

        return all_present

    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("NeuroScan Database Migration: Profile Picture Support")
    print("=" * 60)
    print()

    success = run_migration()

    if success:
        verify_migration()
    else:
        print("\n❌ Migration failed. Please check the errors above.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Migration complete! You can now restart your Flask app.")
    print("=" * 60)