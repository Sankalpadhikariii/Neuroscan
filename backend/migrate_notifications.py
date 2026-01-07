import sqlite3
import sys
from pathlib import Path

DB_FILE = "neuroscan_platform.db"


def migrate():
    """Add notifications table to existing database"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        # Check if table already exists
        c.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='notifications'
        """)

        if c.fetchone():
            print("⚠️  Notifications table already exists. Skipping creation.")
            conn.close()
            return True

        # Create notifications table
        print("📝 Creating notifications table...")
        c.execute("""
            CREATE TABLE notifications (
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

        # Create indexes
        print("📊 Creating indexes...")
        c.execute("""
            CREATE INDEX idx_notifications_user 
            ON notifications(user_id, user_type, is_read)
        """)

        c.execute("""
            CREATE INDEX idx_notifications_hospital 
            ON notifications(hospital_id, created_at)
        """)

        c.execute("""
            CREATE INDEX idx_notifications_created 
            ON notifications(created_at DESC)
        """)

        conn.commit()
        conn.close()

        print("✅ Migration completed successfully!")
        print("✅ Notifications table created with indexes")
        return True

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("🔧 Starting database migration...")
    print(f"📍 Database: {DB_FILE}")

    if not Path(DB_FILE).exists():
        print(f"❌ Error: Database file '{DB_FILE}' not found!")
        sys.exit(1)

    success = migrate()
    sys.exit(0 if success else 1)