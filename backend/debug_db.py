import sqlite3
import pprint
import os

db_path = os.path.join(os.getcwd(), 'instance', 'neuroscan.db')
print("Checking DB at:", db_path)
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("--- hospital_subscriptions ---")
try:
    c.execute("SELECT * FROM hospital_subscriptions ORDER BY id DESC LIMIT 5")
    for row in c.fetchall():
        print(dict(row))
except Exception as e:
    print("Error querying hospital_subscriptions:", e)

print("\n--- payment_transactions ---")
try:
    c.execute("SELECT * FROM payment_transactions ORDER BY id DESC LIMIT 5")
    for row in c.fetchall():
        print(dict(row))
except Exception as e:
    print("Error querying payment_transactions:", e)

print("\n--- subscription_plans ---")
try:
    c.execute("SELECT id, name, display_name FROM subscription_plans")
    for row in c.fetchall():
        print(dict(row))
except Exception as e:
    print("Error querying subscription_plans:", e)
