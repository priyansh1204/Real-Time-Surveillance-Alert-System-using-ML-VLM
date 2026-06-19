import smtplib
import ssl
from email.message import EmailMessage
import os


class AlertSystem:
    def __init__(self, config):
        self.email_config = config.get("email", {})
        self.enabled = self.email_config.get("enabled", False)

        if self.enabled:
            self.smtp_server = self.email_config["smtp_server"]
            self.smtp_port = self.email_config["smtp_port"]
            self.username = self.email_config["username"]
            self.password = self.email_config["password"]
            self.recipients = self.email_config["recipients"]

            print("✅ Email alert system initialized")
        else:
            print("⚠️ Email alerts are disabled")

    def send_alert(self, object_type, confidence, image_path, timestamp, message=None):
        if not self.enabled:
            print("⚠️ Email system disabled")
            return

        try:
            msg = EmailMessage()
            msg["From"] = self.username
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = f"🚨 SECURITY ALERT: {object_type.upper()} DETECTED"

            body = f"""
SECURITY ALERT 🚨

Object Detected : {object_type}
Confidence      : {confidence:.2%}
Timestamp       : {timestamp}

==============================
AI BEHAVIOR ANALYSIS
==============================
{message if message else "No behavior analysis available."}

==============================
Evidence File
==============================
{image_path}

⚠️ Please take immediate action.
"""

            msg.set_content(body)

            # Attach image / video if exists
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    file_data = f.read()
                    file_name = os.path.basename(image_path)

                msg.add_attachment(
                    file_data,
                    maintype="application",
                    subtype="octet-stream",
                    filename=file_name,
                )

            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.username, self.password)
                server.send_message(msg)

            print("✅ EMAIL SENT SUCCESSFULLY")

        except Exception as e:
            print("❌ EMAIL FAILED:", str(e))
