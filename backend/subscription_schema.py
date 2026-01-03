
import sqlite3
from datetime import datetime, timedelta

DB_FILE = "neuroscan_platform.db"


def create_subscription_tables():
    """Create subscription and billing related tables"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # ==============================================
    # SUBSCRIPTION PLANS
    # ==============================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS subscription_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            description TEXT,
            price_monthly REAL NOT NULL,
            price_yearly REAL NOT NULL,
            max_scans_per_month INTEGER NOT NULL,
            max_users INTEGER NOT NULL,
            max_patients INTEGER NOT NULL,
            features TEXT, -- JSON string of features
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ==============================================
    # HOSPITAL SUBSCRIPTIONS
    # ==============================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS hospital_subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            plan_id INTEGER NOT NULL,
            status TEXT DEFAULT 'active', -- active, cancelled, expired, suspended
            billing_cycle TEXT DEFAULT 'monthly', -- monthly, yearly
            current_period_start DATE NOT NULL,
            current_period_end DATE NOT NULL,
            next_billing_date DATE,
            auto_renew BOOLEAN DEFAULT 1,
            trial_ends_at DATE,
            is_trial BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            cancelled_at TIMESTAMP,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE,
            FOREIGN KEY (plan_id) REFERENCES subscription_plans(id)
        )
    """)

    # ==============================================
    # USAGE TRACKING
    # ==============================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS usage_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            subscription_id INTEGER NOT NULL,
            period_start DATE NOT NULL,
            period_end DATE NOT NULL,
            scans_used INTEGER DEFAULT 0,
            scans_limit INTEGER NOT NULL,
            users_count INTEGER DEFAULT 0,
            users_limit INTEGER NOT NULL,
            patients_count INTEGER DEFAULT 0,
            patients_limit INTEGER NOT NULL,
            is_current BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE,
            FOREIGN KEY (subscription_id) REFERENCES hospital_subscriptions(id),
            UNIQUE(hospital_id, period_start)
        )
    """)

    # ==============================================
    # PAYMENT TRANSACTIONS
    # ==============================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS payment_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            subscription_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            currency TEXT DEFAULT 'USD',
            status TEXT NOT NULL, -- pending, completed, failed, refunded
            payment_method TEXT, -- card, bank_transfer, etc.
            transaction_id TEXT UNIQUE, -- External payment gateway transaction ID
            invoice_number TEXT UNIQUE,
            description TEXT,
            payment_date TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT, -- JSON string for additional payment info
            FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE,
            FOREIGN KEY (subscription_id) REFERENCES hospital_subscriptions(id)
        )
    """)

    # ==============================================
    # INVOICES
    # ==============================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            subscription_id INTEGER NOT NULL,
            invoice_number TEXT UNIQUE NOT NULL,
            amount REAL NOT NULL,
            tax_amount REAL DEFAULT 0,
            total_amount REAL NOT NULL,
            currency TEXT DEFAULT 'USD',
            status TEXT DEFAULT 'draft', -- draft, sent, paid, overdue, cancelled
            issue_date DATE NOT NULL,
            due_date DATE NOT NULL,
            paid_date DATE,
            items TEXT NOT NULL, -- JSON string of line items
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE,
            FOREIGN KEY (subscription_id) REFERENCES hospital_subscriptions(id)
        )
    """)

    # ==============================================
    # FEATURE FLAGS
    # ==============================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS feature_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_key TEXT UNIQUE NOT NULL,
            feature_name TEXT NOT NULL,
            description TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ==============================================
    # PLAN FEATURES (many-to-many)
    # ==============================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS plan_features (
            plan_id INTEGER NOT NULL,
            feature_key TEXT NOT NULL,
            is_enabled BOOLEAN DEFAULT 1,
            limit_value INTEGER, -- For features with limits
            FOREIGN KEY (plan_id) REFERENCES subscription_plans(id) ON DELETE CASCADE,
            FOREIGN KEY (feature_key) REFERENCES feature_flags(feature_key),
            PRIMARY KEY (plan_id, feature_key)
        )
    """)

    # ==============================================
    # SUBSCRIPTION CHANGE HISTORY
    # ==============================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS subscription_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            subscription_id INTEGER NOT NULL,
            action TEXT NOT NULL, -- upgrade, downgrade, cancel, renew, suspend
            old_plan_id INTEGER,
            new_plan_id INTEGER,
            reason TEXT,
            changed_by INTEGER, -- admin user who made the change
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT, -- JSON for additional info
            FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE,
            FOREIGN KEY (subscription_id) REFERENCES hospital_subscriptions(id),
            FOREIGN KEY (old_plan_id) REFERENCES subscription_plans(id),
            FOREIGN KEY (new_plan_id) REFERENCES subscription_plans(id)
        )
    """)

    # ==============================================
    # INDEXES
    # ==============================================
    c.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_hospital ON hospital_subscriptions(hospital_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON hospital_subscriptions(status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_usage_hospital ON usage_tracking(hospital_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_usage_current ON usage_tracking(is_current)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_payments_hospital ON payment_transactions(hospital_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_payments_status ON payment_transactions(status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_invoices_hospital ON invoices(hospital_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(status)")

    conn.commit()
    conn.close()

    print("✅ Subscription tables created successfully!")


def seed_subscription_plans():
    """Create default subscription plans"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    plans = [
        {
            'name': 'free',
            'display_name': 'Free Trial',
            'description': 'Perfect for testing the platform',
            'price_monthly': 0.0,
            'price_yearly': 0.0,
            'max_scans_per_month': 10,
            'max_users': 2,
            'max_patients': 50,
            'features': '["basic_scan", "pdf_reports", "email_support"]'
        },
        {
            'name': 'basic',
            'display_name': 'Basic',
            'description': 'For small clinics and practices',
            'price_monthly': 99.0,
            'price_yearly': 990.0,  # 2 months free
            'max_scans_per_month': 100,
            'max_users': 5,
            'max_patients': 500,
            'features': '["basic_scan", "pdf_reports", "email_support", "patient_portal", "chat_support"]'
        },
        {
            'name': 'professional',
            'display_name': 'Professional',
            'description': 'For medium-sized hospitals',
            'price_monthly': 299.0,
            'price_yearly': 2990.0,  # 2 months free
            'max_scans_per_month': 500,
            'max_users': 20,
            'max_patients': 2000,
            'features': '["basic_scan", "advanced_analytics", "pdf_reports", "email_support", "patient_portal", "chat_support", "api_access", "priority_support", "gradcam_visualization"]'
        },
        {
            'name': 'enterprise',
            'display_name': 'Enterprise',
            'description': 'For large hospitals and networks',
            'price_monthly': 799.0,
            'price_yearly': 7990.0,  # 2 months free
            'max_scans_per_month': -1,  # Unlimited
            'max_users': -1,  # Unlimited
            'max_patients': -1,  # Unlimited
            'features': '["basic_scan", "advanced_analytics", "pdf_reports", "email_support", "patient_portal", "chat_support", "api_access", "priority_support", "gradcam_visualization", "custom_branding", "dedicated_support", "sla_guarantee", "custom_integration"]'
        }
    ]

    for plan in plans:
        try:
            c.execute("""
                INSERT INTO subscription_plans 
                (name, display_name, description, price_monthly, price_yearly, 
                 max_scans_per_month, max_users, max_patients, features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plan['name'], plan['display_name'], plan['description'],
                plan['price_monthly'], plan['price_yearly'],
                plan['max_scans_per_month'], plan['max_users'],
                plan['max_patients'], plan['features']
            ))
            print(f"✅ Created plan: {plan['display_name']}")
        except sqlite3.IntegrityError:
            print(f"⚠️  Plan already exists: {plan['display_name']}")

    conn.commit()
    conn.close()


def seed_feature_flags():
    """Create default feature flags"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    features = [
        ('basic_scan', 'Basic MRI Scan Analysis', 'Standard brain tumor detection'),
        ('advanced_analytics', 'Advanced Analytics Dashboard', 'Detailed statistics and trends'),
        ('pdf_reports', 'PDF Report Generation', 'Generate downloadable PDF reports'),
        ('patient_portal', 'Patient Portal Access', 'Allow patients to view their results'),
        ('chat_support', 'Chat Support', 'In-app chat support'),
        ('email_support', 'Email Support', 'Email support from team'),
        ('api_access', 'API Access', 'RESTful API for integration'),
        ('priority_support', 'Priority Support', '24/7 priority support'),
        ('gradcam_visualization', 'GradCAM Visualization', 'AI attention heatmaps'),
        ('custom_branding', 'Custom Branding', 'White-label branding options'),
        ('dedicated_support', 'Dedicated Account Manager', 'Personal account manager'),
        ('sla_guarantee', 'SLA Guarantee', '99.9% uptime guarantee'),
        ('custom_integration', 'Custom Integration', 'Custom PACS/EHR integration')
    ]

    for feature_key, feature_name, description in features:
        try:
            c.execute("""
                INSERT INTO feature_flags (feature_key, feature_name, description)
                VALUES (?, ?, ?)
            """, (feature_key, feature_name, description))
            print(f"✅ Created feature: {feature_name}")
        except sqlite3.IntegrityError:
            print(f"⚠️  Feature already exists: {feature_name}")

    conn.commit()
    conn.close()


def update_existing_hospitals():
    """Add free trial subscription to existing hospitals"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Get free plan ID
    c.execute("SELECT id FROM subscription_plans WHERE name='free'")
    free_plan = c.fetchone()

    if not free_plan:
        print("❌ Free plan not found. Run seed_subscription_plans first.")
        conn.close()
        return

    free_plan_id = free_plan[0]

    # Get hospitals without subscriptions
    c.execute("""
        SELECT h.id FROM hospitals h
        LEFT JOIN hospital_subscriptions hs ON h.id = hs.hospital_id
        WHERE hs.id IS NULL
    """)
    hospitals = c.fetchall()

    for (hospital_id,) in hospitals:
        # Create free trial subscription
        start_date = datetime.now()
        end_date = start_date + timedelta(days=30)

        c.execute("""
            INSERT INTO hospital_subscriptions
            (hospital_id, plan_id, status, billing_cycle, 
             current_period_start, current_period_end, 
             trial_ends_at, is_trial)
            VALUES (?, ?, 'active', 'monthly', ?, ?, ?, 1)
        """, (hospital_id, free_plan_id,
              start_date.date(), end_date.date(),
              end_date.date()))

        subscription_id = c.lastrowid

        # Create initial usage tracking
        c.execute("""
            INSERT INTO usage_tracking
            (hospital_id, subscription_id, period_start, period_end,
             scans_used, scans_limit, users_count, users_limit,
             patients_count, patients_limit, is_current)
            VALUES (?, ?, ?, ?, 0, 10, 0, 2, 0, 50, 1)
        """, (hospital_id, subscription_id,
              start_date.date(), end_date.date()))

        print(f"✅ Added free trial to hospital ID: {hospital_id}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("NeuroScan Subscription System Setup")
    print("=" * 60)
    print()

    create_subscription_tables()
    seed_subscription_plans()
    seed_feature_flags()
    update_existing_hospitals()

    print("\n" + "=" * 60)
    print("✅ Subscription system setup complete!")
    print("=" * 60)
    print("\nSubscription Plans Created:")
    print("  • Free Trial: $0/month (10 scans, 2 users)")
    print("  • Basic: $99/month (100 scans, 5 users)")
    print("  • Professional: $299/month (500 scans, 20 users)")
    print("  • Enterprise: $799/month (Unlimited)")
    print("\nNext steps:")
    print("1. Run this migration: python subscription_schema.py")
    print("2. Update Flask backend with subscription logic")
    print("3. Add payment integration (Stripe/PayPal)")
    print("4. Update frontend with pricing page")
    print("=" * 60)