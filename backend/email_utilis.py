"""
Email Utility Module for NeuroScan
Sends verification codes and notifications to patients
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime


def send_verification_email(to_email, verification_code, patient_name, hospital_name):
    """
    Send verification code email to patient

    Args:
        to_email: Patient's email address
        verification_code: 6-digit verification code
        patient_name: Patient's full name
        hospital_name: Hospital name

    Returns:
        bool: True if email sent successfully, False otherwise
    """

    # Email configuration from environment variables
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'your-email@example.com')
    SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', 'your-app-password')
    SENDER_NAME = os.getenv('SENDER_NAME', 'NeuroScan Medical System')

    # Create message
    message = MIMEMultipart("alternative")
    message["Subject"] = f"Your NeuroScan Verification Code: {verification_code}"
    message["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
    message["To"] = to_email

    # Create HTML email content
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9fafb;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
                border-radius: 10px 10px 0 0;
            }}
            .content {{
                background: white;
                padding: 30px;
                border-radius: 0 0 10px 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .code-box {{
                background-color: #f3f4f6;
                border: 2px solid #667eea;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                margin: 30px 0;
            }}
            .code {{
                font-size: 32px;
                font-weight: bold;
                color: #667eea;
                letter-spacing: 8px;
                font-family: 'Courier New', monospace;
            }}
            .warning {{
                background-color: #fef3c7;
                border-left: 4px solid #fbbf24;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            .footer {{
                text-align: center;
                color: #6b7280;
                font-size: 12px;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #e5e7eb;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 style="margin: 0; font-size: 28px;">🧠 NeuroScan</h1>
                <p style="margin: 10px 0 0 0; font-size: 14px; opacity: 0.9;">
                    Brain Tumor Detection System
                </p>
            </div>

            <div class="content">
                <h2 style="color: #111827; margin-top: 0;">Hello {patient_name},</h2>

                <p style="color: #374151; font-size: 16px;">
                    You are attempting to log in to your NeuroScan patient portal at 
                    <strong>{hospital_name}</strong>.
                </p>

                <p style="color: #374151; font-size: 16px;">
                    Please use the following verification code to complete your login:
                </p>

                <div class="code-box">
                    <p style="margin: 0 0 10px 0; color: #6b7280; font-size: 14px;">
                        Your Verification Code
                    </p>
                    <div class="code">{verification_code}</div>
                    <p style="margin: 10px 0 0 0; color: #6b7280; font-size: 12px;">
                        Valid for 10 minutes
                    </p>
                </div>

                <div class="warning">
                    <p style="margin: 0; color: #78350f; font-size: 14px;">
                        <strong>⚠️ Security Notice:</strong> If you did not attempt to log in, 
                        please contact {hospital_name} immediately. Do not share this code with anyone.
                    </p>
                </div>

                <p style="color: #6b7280; font-size: 14px;">
                    This is an automated message. Please do not reply to this email.
                </p>

                <div class="footer">
                    <p style="margin: 0;">
                        © {datetime.now().year} NeuroScan Medical System<br>
                        Powered by AI & Deep Learning
                    </p>
                    <p style="margin: 10px 0 0 0;">
                        {hospital_name}
                    </p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    # Create plain text version as fallback
    text = f"""
NeuroScan - Brain Tumor Detection System

Hello {patient_name},

You are attempting to log in to your NeuroScan patient portal at {hospital_name}.

Your Verification Code: {verification_code}

This code is valid for 10 minutes.

Security Notice: If you did not attempt to log in, please contact {hospital_name} immediately. 
Do not share this code with anyone.

This is an automated message. Please do not reply to this email.

---
© {datetime.now().year} NeuroScan Medical System
{hospital_name}
    """

    # Attach both HTML and plain text versions
    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")
    message.attach(part1)
    message.attach(part2)

    # Send email
    try:
        # Check if email is configured
        if SENDER_EMAIL == 'your-email@example.com' or not SENDER_PASSWORD or SENDER_PASSWORD == 'your-app-password':
            print(f"\n{'=' * 60}")
            print("EMAIL NOT CONFIGURED - Verification code will be logged")
            print(f"{'=' * 60}")
            print(f"To: {to_email}")
            print(f"Patient: {patient_name}")
            print(f"Verification Code: {verification_code}")
            print(f"{'=' * 60}\n")
            return False

        # Create SMTP session
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Enable TLS encryption
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, message.as_string())

        print(f"✓ Verification email sent to {to_email}")
        return True

    except Exception as e:
        print(f"✗ Error sending email: {e}")
        print(f"Verification code for {to_email}: {verification_code}")
        return False


def send_welcome_email(to_email, patient_name, patient_code, access_code, hospital_name, hospital_code):
    """
    Send welcome email with login credentials to new patient

    Args:
        to_email: Patient's email address
        patient_name: Patient's full name
        patient_code: Patient's unique code
        access_code: Patient's access code for login
        hospital_name: Hospital name
        hospital_code: Hospital code

    Returns:
        bool: True if email sent successfully, False otherwise
    """

    # Email configuration
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'your-email@example.com')
    SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', 'your-app-password')
    SENDER_NAME = os.getenv('SENDER_NAME', 'NeuroScan Medical System')

    # Create message
    message = MIMEMultipart("alternative")
    message["Subject"] = f"Welcome to NeuroScan Patient Portal - {hospital_name}"
    message["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
    message["To"] = to_email

    # Create HTML email
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f9fafb; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
            .content {{ background: white; padding: 30px; border-radius: 0 0 10px 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .credentials {{ background-color: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 20px 0; }}
            .code {{ font-family: 'Courier New', monospace; font-size: 18px; font-weight: bold; color: #667eea; }}
            .footer {{ text-align: center; color: #6b7280; font-size: 12px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 style="margin: 0;">🧠 Welcome to NeuroScan</h1>
                <p style="margin: 10px 0 0 0;">Your Medical Portal is Ready</p>
            </div>

            <div class="content">
                <h2 style="color: #111827;">Hello {patient_name},</h2>

                <p>Welcome to the NeuroScan patient portal at <strong>{hospital_name}</strong>.</p>

                <p>You can now access your medical records, view scan results, and communicate with your healthcare provider through our secure portal.</p>

                <div class="credentials">
                    <h3 style="margin-top: 0; color: #111827;">Your Login Credentials</h3>

                    <p><strong>Hospital Code:</strong> <span class="code">{hospital_code}</span></p>
                    <p><strong>Patient Code:</strong> <span class="code">{patient_code}</span></p>
                    <p><strong>Access Code:</strong> <span class="code">{access_code}</span></p>
                </div>

                <p><strong>How to Log In:</strong></p>
                <ol>
                    <li>Visit the patient portal login page</li>
                    <li>Enter the Hospital Code</li>
                    <li>Enter your Patient Code and Access Code</li>
                    <li>Check your email for a verification code</li>
                    <li>Enter the verification code to complete login</li>
                </ol>

                <p style="background-color: #fef3c7; padding: 15px; border-left: 4px solid #fbbf24; border-radius: 4px;">
                    <strong>⚠️ Keep These Credentials Safe:</strong> Store this information securely. 
                    You will need all three codes to access your medical portal.
                </p>

                <div class="footer">
                    <p>© {datetime.now().year} NeuroScan Medical System<br>{hospital_name}</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    # Plain text version
    text = f"""
Welcome to NeuroScan Patient Portal

Hello {patient_name},

Welcome to the NeuroScan patient portal at {hospital_name}.

Your Login Credentials:
- Hospital Code: {hospital_code}
- Patient Code: {patient_code}
- Access Code: {access_code}

How to Log In:
1. Visit the patient portal login page
2. Enter the Hospital Code
3. Enter your Patient Code and Access Code
4. Check your email for a verification code
5. Enter the verification code to complete login

Keep these credentials safe!

---
© {datetime.now().year} NeuroScan Medical System
{hospital_name}
    """

    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")
    message.attach(part1)
    message.attach(part2)

    # Send email
    try:
        if SENDER_EMAIL == 'your-email@example.com' or not SENDER_PASSWORD:
            print(f"\n{'=' * 60}")
            print("EMAIL NOT CONFIGURED - Credentials will be shown in UI")
            print(f"{'=' * 60}\n")
            return False

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, message.as_string())

        print(f"✓ Welcome email sent to {to_email}")
        return True

    except Exception as e:
        print(f"✗ Error sending email: {e}")
        return False