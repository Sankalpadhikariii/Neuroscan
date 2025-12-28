#!/usr/bin/env python3
"""
Script to create a superadmin user for NeuroScan
Run this once to create your first superadmin account
"""

import sqlite3
from werkzeug.security import generate_password_hash
import getpass

DB_FILE = "brain_tumor.db"


def create_superadmin():
    print("=" * 50)
    print("NeuroScan - Create Superadmin Account")
    print("=" * 50)
    print()

    # Get user input
    username = input("Enter superadmin username: ").strip()
    email = input("Enter superadmin email: ").strip()

    # Get password securely
    while True:
        password = getpass.getpass("Enter password: ")
        password_confirm = getpass.getpass("Confirm password: ")

        if password != password_confirm:
            print("❌ Passwords don't match. Try again.\n")
            continue

        if len(password) < 6:
            print("❌ Password must be at least 6 characters.\n")
            continue

        break

    # Check if user already exists
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute('SELECT id, role FROM users WHERE username=? OR email=?', (username, email))
    existing_user = c.fetchone()

    if existing_user:
        print(f"\n⚠️  User already exists!")
        user_id, current_role = existing_user

        if current_role == 'superadmin':
            print(f"   User '{username}' is already a superadmin.")
            conn.close()
            return

        # Offer to upgrade existing user
        upgrade = input(f"\n   Would you like to upgrade '{username}' to superadmin? (yes/no): ").lower()

        if upgrade in ['yes', 'y']:
            c.execute('UPDATE users SET role=? WHERE id=?', ('superadmin', user_id))
            conn.commit()
            conn.close()
            print(f"\n✅ Successfully upgraded '{username}' to superadmin!")
            return
        else:
            conn.close()
            print("\n❌ Operation cancelled.")
            return

    # Create new superadmin
    try:
        hashed_password = generate_password_hash(password)
        c.execute('''INSERT INTO users (username, email, password, role) 
                     VALUES (?, ?, ?, ?)''',
                  (username, email, hashed_password, 'superadmin'))
        conn.commit()
        user_id = c.lastrowid
        conn.close()

        print("\n" + "=" * 50)
        print("✅ Superadmin created successfully!")
        print("=" * 50)
        print(f"Username: {username}")
        print(f"Email: {email}")
        print(f"Role: superadmin")
        print(f"User ID: {user_id}")
        print("\nYou can now login with these credentials.")
        print("=" * 50)

    except Exception as e:
        conn.close()
        print(f"\n❌ Error creating superadmin: {e}")


if __name__ == "__main__":
    try:
        create_superadmin()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")