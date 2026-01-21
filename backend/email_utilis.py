import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


def send_patient_credentials_email(to_email, patient_name, hospital_name, hospital_code, patient_code, access_code):
    """Send patient credentials email"""

    # Check if email is configured
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = os.getenv('SMTP_PORT')
    smtp_user = os.getenv('SMTP_USER')
    smtp_pass = os.getenv('SMTP_PASSWORD')

    if not all([smtp_server, smtp_port, smtp_user, smtp_pass]):
        logger.warning("⚠️ Email not configured - skipping email send")
        logger.info(f"📧 Would have sent to {to_email}:")
        logger.info(f"   Hospital Code: {hospital_code}")
        logger.info(f"   Patient Code: {patient_code}")
        logger.info(f"   Access Code: {access_code}")
        return False

    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = smtp_user
        msg['To'] = to_email
        msg['Subject'] = f'Welcome to {hospital_name} - NeuroScan Patient Portal'

        html = f"""
        <html>
          <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
              <h2 style="color: #6366f1;">Welcome to NeuroScan Patient Portal</h2>

              <p>Dear {patient_name},</p>

              <p>Your account has been created at <strong>{hospital_name}</strong>. Use the credentials below to access your patient portal:</p>

              <div style="background: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <p><strong>Hospital Code:</strong> <code style="background: #e2e8f0; padding: 4px 8px; border-radius: 4px;">{hospital_code}</code></p>
                <p><strong>Patient Code:</strong> <code style="background: #e2e8f0; padding: 4px 8px; border-radius: 4px;">{patient_code}</code></p>
                <p><strong>Access Code:</strong> <code style="background: #e2e8f0; padding: 4px 8px; border-radius: 4px;">{access_code}</code></p>
              </div>

              <p><strong>Important:</strong> Keep these credentials secure. You'll need all three to log in.</p>

              <p>Visit the patient portal at: <a href="http://localhost:3000">http://localhost:3000</a></p>

              <p style="color: #64748b; font-size: 12px; margin-top: 30px;">
                This is an automated message. Please do not reply.
              </p>
            </div>
          </body>
        </html>
        """

        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        logger.info(f"✅ Email sent successfully to {to_email}")
        return True

    except Exception as e:
        logger.error(f"❌ Email error: {e}")
        return False


def send_verification_email(to_email, verification_code, patient_name, hospital_name):
    """Send verification code email"""

    smtp_server = os.getenv('SMTP_SERVER')
    if not smtp_server:
        logger.warning("⚠️ Email not configured")
        logger.info(f"📧 Verification code for {to_email}: {verification_code}")
        return False

    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = os.getenv('SMTP_USER')
        msg['To'] = to_email
        msg['Subject'] = 'Your NeuroScan Verification Code'

        html = f"""
        <html>
          <body style="font-family: Arial, sans-serif;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
              <h2>NeuroScan Login Verification</h2>
              <p>Hello {patient_name},</p>
              <p>Your verification code is:</p>
              <div style="font-size: 32px; font-weight: bold; background: #f0f4ff; padding: 20px; text-align: center; border-radius: 8px; letter-spacing: 8px;">
                {verification_code}
              </div>
              <p style="color: #dc2626; margin-top: 20px;"><strong>This code expires in 10 minutes.</strong></p>
            </div>
          </body>
        </html>
        """

        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP(smtp_server, int(os.getenv('SMTP_PORT'))) as server:
            server.starttls()
            server.login(os.getenv('SMTP_USER'), os.getenv('SMTP_PASSWORD'))
            server.send_message(msg)

        return True
    except Exception as e:
        logger.error(f"Email error: {e}")
        return False


def send_welcome_email(to_email, patient_name, patient_code, access_code, hospital_name, hospital_code):
    """Alias for send_patient_credentials_email"""
    return send_patient_credentials_email(to_email, patient_name, hospital_name, hospital_code, patient_code,
                                          access_code)