import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_FILE = "neuroscan_platform.db"


def migrate_patient_access_codes():
    """Add missing columns to patient_access_codes table"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Get existing columns
    c.execute("PRAGMA table_info(patient_access_codes)")
    columns = [col[1] for col in c.fetchall()]
    logger.info(f"Existing columns: {columns}")

    # Add missing columns
    migrations = [
        ("verification_code", "ALTER TABLE patient_access_codes ADD COLUMN verification_code TEXT"),
        ("verification_code_expiry", "ALTER TABLE patient_access_codes ADD COLUMN verification_code_expiry DATETIME"),
        ("verified_at", "ALTER TABLE patient_access_codes ADD COLUMN verified_at DATETIME"),
        ("is_verified", "ALTER TABLE patient_access_codes ADD COLUMN is_verified INTEGER DEFAULT 0")
    ]

    for col_name, sql in migrations:
        if col_name not in columns:
            try:
                c.execute(sql)
                logger.info(f"✅ Added column: {col_name}")
            except sqlite3.OperationalError as e:
                logger.warning(f"Column {col_name} might already exist: {e}")
        else:
            logger.info(f"⏭️  Column {col_name} already exists")

    conn.commit()
    conn.close()
    logger.info("✅ Migration completed!")


if __name__ == "__main__":
    migrate_patient_access_codes()