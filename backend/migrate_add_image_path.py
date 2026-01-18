#!/usr/bin/env python3
"""
Database Migration Script: Add image_path column to scans table
Run this script to add the image_path column to your scans table
"""

import sqlite3
import sys

DB_FILE = "neuroscan_platform.db"


def run_migration():
    """Add image_path column to scans table"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        print("🔄 Starting database migration...")
        print(f"📁 Database: {DB_FILE}")

        # Check if columns already exist
        cursor.execute("PRAGMA table_info(scans)")
        columns = [col[1] for col in cursor.fetchall()]

        print("\n📊 Current columns in 'scans' table:")
        for col in columns:
            print(f"   - {col}")
        print()

        migrations_needed = []

        if 'image_path' not in columns:
            migrations_needed.append(('image_path', 'TEXT'))
            print("ℹ️  image_path column is missing")

        if 'gradcam_path' not in columns:
            migrations_needed.append(('gradcam_path', 'TEXT'))
            print("ℹ️  gradcam_path column is missing")

        if not migrations_needed:
            print("✅ All required columns already exist. No migration needed.")
            conn.close()
            return True

        print(f"\n🔧 Adding {len(migrations_needed)} column(s)...\n")

        # Apply migrations
        for column_name, column_type in migrations_needed:
            try:
                sql = f"ALTER TABLE scans ADD COLUMN {column_name} {column_type}"
                cursor.execute(sql)
                print(f"✅ Added column: {column_name} ({column_type})")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"⚠️  Column {column_name} already exists, skipping...")
                else:
                    print(f"❌ Error adding column {column_name}: {e}")
                    raise

        conn.commit()
        conn.close()

        print("\n✅ Migration completed successfully!")
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

        cursor.execute("PRAGMA table_info(scans)")
        columns = {col[1]: col[2] for col in cursor.fetchall()}

        required_columns = ['image_path', 'gradcam_path']

        print("\n🔍 Verifying migration...")
        all_present = True

        for col in required_columns:
            if col in columns:
                print(f"✅ Column '{col}' exists (type: {columns[col]})")
            else:
                print(f"❌ Column '{col}' is missing!")
                all_present = False

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
    print("NeuroScan Database Migration: Add image_path to scans")
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
