import re

file_path = "app.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Remove import stripe
content = content.replace("import stripe\n", "")

# 2. Replace Config
stripe_config = """# -----------------------------
# Stripe Configuration
# -----------------------------
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")"""

khalti_config = """# -----------------------------
# Khalti Configuration
# -----------------------------
KHALTI_PUBLIC_KEY = os.getenv("KHALTI_PUBLIC_KEY")
KHALTI_SECRET_KEY = os.getenv("KHALTI_SECRET_KEY")"""

content = content.replace(stripe_config, khalti_config)

# 3. Remove Stripe Config Route
config_route = """@app.route("/api/stripe/config", methods=["GET"])
def stripe_config():
    return jsonify({"publishableKey": STRIPE_PUBLISHABLE_KEY})"""
content = content.replace(config_route, "")

# 4. Remove Stripe helpers
# Find start of `def get_or_create_stripe_customer` and end of `def get_stripe_price_id`
pattern_helpers = re.compile(r'def get_or_create_stripe_customer\(hospital_id\):.*?return plan_prices\.get\(billing_cycle\)\n\n', re.DOTALL)
content = re.sub(pattern_helpers, '', content)

# 5. Replace Endpoints
# From `@app.route("/api/stripe/create-checkout-session"` to end of webhook
khalti_endpoints = '''
# ==============================================
# KHALTI PAYMENT INTEGRATION
# ==============================================

@app.route("/api/khalti/initiate-payment", methods=["POST"])
@hospital_required
def initiate_khalti_payment():
    data = request.json
    plan_identifier = data.get("plan_id")
    billing_cycle = data.get("billing_cycle", "monthly")
    hospital_id = session.get("hospital_id")

    if not plan_identifier:
        return jsonify({"error": "plan_id is required"}), 400

    conn = get_db()
    c = conn.cursor()

    if isinstance(plan_identifier, str) and not plan_identifier.isdigit():
        c.execute("SELECT * FROM subscription_plans WHERE name=?", (plan_identifier,))
    else:
        c.execute("SELECT * FROM subscription_plans WHERE id=?", (plan_identifier,))

    plan = c.fetchone()
    if not plan:
        conn.close()
        return jsonify({"error": f"Plan '{plan_identifier}' not found"}), 404

    plan = dict(plan)
    
    c.execute("SELECT hospital_name, email FROM hospitals WHERE id=?", (hospital_id,))
    hospital = c.fetchone()
    conn.close()

    amount_rs = plan['price_yearly'] if billing_cycle == 'yearly' else plan['price_monthly']
    amount_paisa = int(amount_rs * 100)

    if amount_paisa == 0:
        return jsonify({"error": "Cannot initiate payment for free plan"}), 400

    khalti_url = "https://a.khalti.com/api/v2/epayment/initiate/"
    purchase_order_id = f"SUB-{hospital_id}-{plan['id']}-{int(time.time())}"
    return_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/subscription-success?plan_id={plan['id']}&billing_cycle={billing_cycle}&hospital_id={hospital_id}"

    payload = {
        "return_url": return_url,
        "website_url": os.getenv('FRONTEND_URL', 'http://localhost:3000'),
        "amount": amount_paisa,
        "purchase_order_id": purchase_order_id,
        "purchase_order_name": f"{plan['display_name']} ({billing_cycle})",
        "customer_info": {
            "name": hospital["hospital_name"] or "Hospital",
            "email": hospital["email"] or "hospital@example.com",
            "phone": "9800000000"
        }
    }
    
    headers = {
        "Authorization": f"Key {os.getenv('KHALTI_SECRET_KEY')}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(khalti_url, json=payload, headers=headers)
        res_data = response.json()
        
        if response.status_code == 200:
            return jsonify({
                "url": res_data.get("payment_url"),
                "pidx": res_data.get("pidx")
            })
        else:
            logger.error(f"Khalti Initiate Error: {res_data}")
            return jsonify({"error": "Payment gateway initialization failed"}), 400
    except Exception as e:
        logger.error(f"Khalti Exception: {e}")
        return jsonify({"error": "Error connecting to payment gateway"}), 500


@app.route("/api/khalti/verify-payment", methods=["POST"])
@hospital_required
def verify_khalti_payment():
    data = request.json
    pidx = data.get("pidx")
    plan_id = data.get("plan_id")
    billing_cycle = data.get("billing_cycle", "monthly")
    hospital_id = session.get("hospital_id")

    if not pidx or not plan_id:
        return jsonify({"error": "pidx and plan_id are required"}), 400

    khalti_lookup_url = "https://a.khalti.com/api/v2/epayment/lookup/"
    headers = {
        "Authorization": f"Key {os.getenv('KHALTI_SECRET_KEY')}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(khalti_lookup_url, json={"pidx": pidx}, headers=headers)
        res_data = response.json()

        if response.status_code != 200 or res_data.get("status") != "Completed":
            return jsonify({"error": "Payment verification failed or not completed", "details": res_data}), 400
            
        TransactionID = res_data.get("transaction_id", pidx)
        
        conn = get_db()
        c = conn.cursor()
        
        c.execute("SELECT id FROM payment_transactions WHERE transaction_id = ?", (TransactionID,))
        if c.fetchone():
            conn.close()
            usage_info = get_detailed_usage(hospital_id)
            return jsonify({"success": True, "message": "Subscription already activated", "usage": usage_info})

        c.execute("SELECT * FROM subscription_plans WHERE id = ?", (plan_id,))
        new_plan = c.fetchone()
        if not new_plan:
            conn.close()
            return jsonify({"error": "Plan not found"}), 404
        new_plan = dict(new_plan)

        c.execute("SELECT id, plan_id FROM hospital_subscriptions WHERE hospital_id = ? AND status = 'active'", (hospital_id,))
        current_sub = c.fetchone()
        old_plan_id = None

        if current_sub:
            old_plan_id = current_sub[1]
            c.execute("UPDATE hospital_subscriptions SET status = 'cancelled', cancelled_at = CURRENT_TIMESTAMP WHERE id = ?", (current_sub[0],))

        start_date = datetime.now()
        if billing_cycle == 'yearly':
            end_date = start_date + timedelta(days=365)
            amount = new_plan['price_yearly']
        else:
            end_date = start_date + timedelta(days=30)
            amount = new_plan['price_monthly']

        c.execute("""
            INSERT INTO hospital_subscriptions
            (hospital_id, plan_id, status, billing_cycle,
             current_period_start, current_period_end, next_billing_date)
            VALUES (?, ?, 'active', ?, ?, ?, ?)
        """, (hospital_id, plan_id, billing_cycle, start_date.date(), end_date.date(), end_date.date()))
        new_sub_id = c.lastrowid

        c.execute("SELECT scans_used FROM usage_tracking WHERE hospital_id = ? AND is_current = 1", (hospital_id,))
        old_usage = c.fetchone()
        carry_over_scans = old_usage[0] if old_usage else 0

        c.execute("UPDATE usage_tracking SET is_current = 0 WHERE hospital_id = ?", (hospital_id,))
        c.execute("DELETE FROM usage_tracking WHERE hospital_id = ? AND period_start = ?", (hospital_id, start_date.date()))
        
        c.execute("""
            INSERT INTO usage_tracking
            (hospital_id, subscription_id, period_start, period_end,
             scans_used, scans_limit, users_limit, patients_limit, is_current)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (hospital_id, new_sub_id, start_date.date(), end_date.date(), carry_over_scans, new_plan['max_scans_per_month'], new_plan['max_users'], new_plan['max_patients']))
        
        invoice_number = f"INV-{hospital_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        c.execute("""
            INSERT INTO payment_transactions
            (hospital_id, subscription_id, amount, status,
             transaction_id, invoice_number, description, payment_date)
            VALUES (?, ?, ?, 'completed', ?, ?, ?, CURRENT_TIMESTAMP)
        """, (hospital_id, new_sub_id, amount, TransactionID, invoice_number, f"Upgrade to {new_plan['display_name']} - {billing_cycle}"))

        conn.commit()
        conn.close()

        logger.info(f"✅ Verified Khalti Session: Hospital {hospital_id} upgraded to {new_plan['display_name']}")
        usage_info = get_detailed_usage(hospital_id)
        
        return jsonify({
            "success": True,
            "message": f"Successfully upgraded to {new_plan['display_name']}!",
            "plan_name": new_plan['display_name'],
            "usage": usage_info
        })

    except Exception as e:
        logger.error(f"❌ Khalti verification error: {e}")
        return jsonify({"error": str(e)}), 500
'''

pattern_endpoints = re.compile(r'@app\.route\("/api/stripe/create-checkout-session".*?return jsonify\(\{"status": "success"\}\), 200\n', re.DOTALL)
content = re.sub(pattern_endpoints, khalti_endpoints, content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Replacement successful!")
