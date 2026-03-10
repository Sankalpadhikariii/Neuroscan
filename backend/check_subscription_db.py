import sys
import pprint

# Append the directory so it finds models
sys.path.insert(0, r"E:\8th sem\Updated\Neuroscan\backend")

from app import app, get_db

with app.app_context():
    conn = get_db()
    c = conn.cursor()

    print("\n--- Subscription Plans ---")
    c.execute("SELECT id, name, display_name FROM subscription_plans")
    plans = c.fetchall()
    for row in plans:
        print(dict(row))

    print("\n--- Recent Hospital Subscriptions ---")
    c.execute("SELECT * FROM hospital_subscriptions ORDER BY id DESC LIMIT 5")
    subs = c.fetchall()
    for row in subs:
        print(dict(row))

    print("\n--- Recent Transactions ---")
    c.execute("SELECT * FROM payment_transactions ORDER BY id DESC LIMIT 5")
    txns = c.fetchall()
    for row in txns:
        print(dict(row))

    conn.close()
