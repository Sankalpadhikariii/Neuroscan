#!/usr/bin/env python3
"""
Database Migration Script for NeuroScan
Adds patient info columns to existing predictions table

Run this ONCE to update your database schema
"""

import sqlite3
import os

DB_FILE = "brain_tumor.db"


def migrate_database():
    print("=" * 60)
    print("NeuroScan Database Migration")
    print("=" * 60)
    print()

    if not os.path.exists(DB_FILE):
        print(f"❌ Error: Database file '{DB_FILE}' not found!")
        print("   Make sure you're in the backend directory.")
        return

    print(f"📁 Found database: {DB_FILE}")
    print()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Check existing columns
    c.execute("PRAGMA table_info(predictions)")
    existing_columns = [row[1] for row in c.fetchall()]

    print("📊 Current columns in 'predictions' table:")
    for col in existing_columns:
        print(f"   - {col}")
    print()

    # List of new columns to add
    new_columns = [
        ("patient_name", "TEXT"),
        ("patient_age", "INTEGER"),
        ("patient_gender", "TEXT"),
        ("patient_id", "TEXT"),
        ("scan_date", "TEXT"),
        ("notes", "TEXT")
    ]

    print("🔧 Adding missing columns...")
    print()

    added_count = 0
    skipped_count = 0

    for col_name, col_type in new_columns:
        if col_name not in existing_columns:
            try:
                sql = f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}"
                c.execute(sql)
                print(f"   ✅ Added: {col_name} ({col_type})")
                added_count += 1
            except sqlite3.OperationalError as e:
                print(f"   ⚠️  Error adding {col_name}: {e}")
        else:
            print(f"   ⏭️  Skipped: {col_name} (already exists)")
            skipped_count += 1

    conn.commit()

    # Verify the changes
    print()
    print("🔍 Verifying changes...")
    c.execute("PRAGMA table_info(predictions)")
    updated_columns = [row[1] for row in c.fetchall()]

    print()
    print("📊 Updated columns in 'predictions' table:")
    for col in updated_columns:
        print(f"   - {col}")

    conn.close()

    print()
    print("=" * 60)
    print(f"✅ Migration Complete!")
    print(f"   - Added: {added_count} new columns")
    print(f"   - Skipped: {skipped_count} existing columns")
    print("=" * 60)
    print()
    print("🚀 You can now restart your Flask server!")
    print()


if __name__ == "__main__":
    try:
        migrate_database()
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Migration Failed!")
        print(f"Error: {e}")
        print("=" * 60)
        print()
        import traceback

        traceback.print_exc()