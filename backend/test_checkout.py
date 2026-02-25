from app import app
import traceback

with app.test_client() as client:
    try:
        with client.session_transaction() as sess:
            # Fake session for hospital 1
            sess['hospital_id'] = 1
            sess['user_id'] = 1
            sess['user_type'] = 'hospital'
            
        print("Sending POST request to /api/stripe/create-checkout-session...")
        res = client.post(
            '/api/stripe/create-checkout-session', 
            json={"plan_id": 2, "billing_cycle": "monthly"}
        )
        print("Status Code:", res.status_code)
        print("Response Data:", res.get_data(as_text=True))
    except Exception as e:
        print("Exception occurred:")
        traceback.print_exc()
