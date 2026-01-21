"""
NeuroScan OTP System Test Suite
Complete testing script for email OTP authentication
"""

import requests
import time
import json
from datetime import datetime

# Configuration
API_BASE = "http://localhost:5000"
TEST_EMAIL = "your-test-email@gmail.com"  # Change this to your test email

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


class NeuroScanTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = []

    def print_header(self, text):
        print(f"\n{BLUE}{BOLD}{'=' * 70}")
        print(f"{text}")
        print(f"{'=' * 70}{RESET}\n")

    def print_test(self, name, passed, message=""):
        status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        print(f"{status} - {name}")
        if message:
            print(f"     {YELLOW}{message}{RESET}")
        self.test_results.append((name, passed))

    def print_summary(self):
        self.print_header("TEST SUMMARY")
        passed = sum(1 for _, p in self.test_results if p)
        total = len(self.test_results)
        percentage = (passed / total * 100) if total > 0 else 0

        print(f"Total Tests: {total}")
        print(f"Passed: {GREEN}{passed}{RESET}")
        print(f"Failed: {RED}{total - passed}{RESET}")
        print(f"Success Rate: {GREEN if percentage == 100 else YELLOW}{percentage:.1f}%{RESET}\n")

    def test_server_connection(self):
        """Test 1: Verify server is running"""
        self.print_header("TEST 1: Server Connection")
        try:
            response = self.session.get(f"{API_BASE}/")
            self.print_test(
                "Server Connection",
                response.status_code in [200, 404],  # 404 is ok if no root route
                f"Server responded with status {response.status_code}"
            )
            return True
        except Exception as e:
            self.print_test("Server Connection", False, f"Error: {str(e)}")
            return False

    def test_patient_verify_invalid(self):
        """Test 2: Verify endpoint rejects invalid credentials"""
        self.print_header("TEST 2: Invalid Credentials Rejection")

        data = {
            "hospital_code": "INVALID",
            "patient_code": "INVALID",
            "access_code": "INVALID"
        }

        try:
            response = self.session.post(
                f"{API_BASE}/patient/verify",
                json=data,
                headers={"Content-Type": "application/json"}
            )

            self.print_test(
                "Invalid Credentials Rejected",
                response.status_code == 401,
                f"Status: {response.status_code}"
            )
            return True
        except Exception as e:
            self.print_test("Invalid Credentials Test", False, f"Error: {str(e)}")
            return False

    def test_patient_verify_valid(self, hospital_code, patient_code, access_code):
        """Test 3: Verify valid credentials and OTP generation"""
        self.print_header("TEST 3: Valid Credentials & OTP Generation")

        data = {
            "hospital_code": hospital_code,
            "patient_code": patient_code,
            "access_code": access_code
        }

        try:
            response = self.session.post(
                f"{API_BASE}/patient/verify",
                json=data,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                self.print_test("Credentials Accepted", True, f"Email hint: {result.get('email_hint')}")
                self.print_test("OTP Sent", True, f"Expires in: {result.get('expires_in_minutes')} minutes")
                return True, result
            else:
                self.print_test("Credentials Accepted", False, response.text)
                return False, None
        except Exception as e:
            self.print_test("Valid Credentials Test", False, f"Error: {str(e)}")
            return False, None

    def test_otp_login_invalid(self, patient_code):
        """Test 4: Test login with invalid OTP"""
        self.print_header("TEST 4: Invalid OTP Rejection")

        data = {
            "patient_code": patient_code,
            "verification_code": "999999"
        }

        try:
            response = self.session.post(
                f"{API_BASE}/patient/login",
                json=data,
                headers={"Content-Type": "application/json"}
            )

            self.print_test(
                "Invalid OTP Rejected",
                response.status_code == 401,
                f"Status: {response.status_code}"
            )
            return True
        except Exception as e:
            self.print_test("Invalid OTP Test", False, f"Error: {str(e)}")
            return False

    def test_otp_login_valid(self, patient_code, otp):
        """Test 5: Test login with valid OTP"""
        self.print_header("TEST 5: Valid OTP Login")

        data = {
            "patient_code": patient_code,
            "verification_code": otp
        }

        try:
            response = self.session.post(
                f"{API_BASE}/patient/login",
                json=data,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                self.print_test("OTP Login Success", True, f"Patient: {result.get('patient', {}).get('full_name')}")
                return True, result
            else:
                self.print_test("OTP Login", False, response.text)
                return False, None
        except Exception as e:
            self.print_test("Valid OTP Login", False, f"Error: {str(e)}")
            return False, None

    def test_resend_otp(self, hospital_code, patient_code, access_code):
        """Test 6: Test OTP resend functionality"""
        self.print_header("TEST 6: OTP Resend")

        data = {
            "hospital_code": hospital_code,
            "patient_code": patient_code,
            "access_code": access_code
        }

        try:
            response = self.session.post(
                f"{API_BASE}/patient/resend-otp",
                json=data,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                self.print_test("OTP Resend", True, result.get('message'))
                return True
            else:
                self.print_test("OTP Resend", False, response.text)
                return False
        except Exception as e:
            self.print_test("OTP Resend", False, f"Error: {str(e)}")
            return False

    def test_otp_expiration(self, patient_code):
        """Test 7: Test OTP expiration (simulated)"""
        self.print_header("TEST 7: OTP Expiration Check")

        # Note: This is a status check, not actual expiration test
        data = {"patient_code": patient_code}

        try:
            response = self.session.post(
                f"{API_BASE}/patient/verify-otp-status",
                json=data,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                self.print_test(
                    "OTP Status Check",
                    True,
                    f"Valid: {result.get('valid')}, Remaining: {result.get('minutes_remaining', 'N/A')} min"
                )
                return True
            else:
                self.print_test("OTP Status Check", False, response.text)
                return False
        except Exception as e:
            self.print_test("OTP Expiration Test", False, f"Error: {str(e)}")
            return False

    def test_rate_limiting(self, hospital_code, patient_code, access_code):
        """Test 8: Test rate limiting on verify endpoint"""
        self.print_header("TEST 8: Rate Limiting")

        data = {
            "hospital_code": hospital_code,
            "patient_code": patient_code,
            "access_code": access_code
        }

        print(f"{YELLOW}Making 6 rapid requests to test rate limiting...{RESET}")

        rate_limited = False
        for i in range(6):
            try:
                response = self.session.post(
                    f"{API_BASE}/patient/verify",
                    json=data,
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code == 429:  # Too Many Requests
                    rate_limited = True
                    break
                time.sleep(0.1)
            except Exception as e:
                print(f"Request {i + 1} error: {e}")

        self.print_test(
            "Rate Limiting Active",
            rate_limited,
            "Rate limit triggered after multiple requests" if rate_limited else "Rate limit not detected"
        )
        return rate_limited


def run_interactive_test():
    """Interactive test mode - requires user input"""
    tester = NeuroScanTester()

    print(f"\n{BOLD}{BLUE}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                                                                   ║")
    print("║           🧠 NeuroScan OTP System Test Suite 🧠                  ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{RESET}\n")

    # Test 1: Server Connection
    if not tester.test_server_connection():
        print(f"\n{RED}Server is not running. Please start the Flask server first.{RESET}\n")
        return

    # Test 2: Invalid Credentials
    tester.test_patient_verify_invalid()

    # Get test credentials from user
    print(f"\n{YELLOW}{'=' * 70}")
    print("MANUAL TEST SECTION - Enter Patient Credentials")
    print(f"{'=' * 70}{RESET}\n")
    print("Please provide valid patient credentials for testing:")
    print("(These should be from a patient you created via hospital portal)\n")

    hospital_code = input("Hospital Code: ").strip()
    patient_code = input("Patient Code: ").strip()
    access_code = input("Access Code: ").strip()

    if not all([hospital_code, patient_code, access_code]):
        print(f"\n{RED}❌ All credentials required. Test aborted.{RESET}\n")
        return

    # Test 3: Valid Credentials
    success, verify_result = tester.test_patient_verify_valid(hospital_code, patient_code, access_code)

    if not success:
        print(f"\n{RED}Unable to proceed without valid credentials{RESET}\n")
        tester.print_summary()
        return

    print(f"\n{YELLOW}📧 Check your email ({verify_result.get('email_hint')}) for the OTP code{RESET}")
    print(f"{YELLOW}⏰ You have 10 minutes to enter the code{RESET}\n")

    # Test 4: Invalid OTP
    tester.test_otp_login_invalid(patient_code)

    # Test 5: Valid OTP
    otp = input("\nEnter the 6-digit OTP from your email: ").strip()

    if len(otp) == 6 and otp.isdigit():
        tester.test_otp_login_valid(patient_code, otp)
    else:
        print(f"{RED}Invalid OTP format{RESET}")

    # Test 6: Resend OTP
    print(f"\n{YELLOW}Testing resend functionality...{RESET}")
    time.sleep(2)
    tester.test_resend_otp(hospital_code, patient_code, access_code)

    # Test 7: OTP Status
    tester.test_otp_expiration(patient_code)

    # Test 8: Rate Limiting
    print(f"\n{YELLOW}Would you like to test rate limiting? This will make multiple rapid requests.{RESET}")
    test_rate = input("Test rate limiting? (y/n): ").strip().lower()

    if test_rate == 'y':
        tester.test_rate_limiting(hospital_code, patient_code, access_code)

    # Print Summary
    tester.print_summary()

    print(f"\n{GREEN}Testing complete! Check the results above.{RESET}")
    print(f"{YELLOW}Note: Some tests require manual verification (email delivery, expiration){RESET}\n")


def run_automated_test():
    """Automated test mode - requires pre-configured test account"""
    print(f"\n{YELLOW}Automated testing requires a pre-configured test account.")
    print("Please set up test credentials in the script first.{RESET}\n")
    # TODO: Implement automated testing with test fixtures


if __name__ == "__main__":
    print("\nSelect test mode:")
    print("1. Interactive Test (recommended)")
    print("2. Automated Test (requires setup)")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        run_interactive_test()
    elif choice == "2":
        run_automated_test()
    else:
        print("Exiting...\n")