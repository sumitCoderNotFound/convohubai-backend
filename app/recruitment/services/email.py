"""
Lightweight email sender (Phase 5) using the stdlib smtplib and the existing SMTP
settings. No new dependency. If SMTP is not configured, send_email returns False so
callers can fall back to copyable links instead of failing.
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app.core.config import settings


def email_configured() -> bool:
    return bool(settings.smtp_host and settings.emails_from_email)


def send_email(to: str, subject: str, html: str, text: str = "") -> bool:
    """Synchronous send. Call via asyncio.to_thread from async code. Best-effort."""
    if not email_configured() or not to:
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{settings.emails_from_name} <{settings.emails_from_email}>"
        msg["To"] = to
        if text:
            msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))
        context = ssl.create_default_context()
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as server:
            server.starttls(context=context)
            if settings.smtp_user:
                server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.emails_from_email, [to], msg.as_string())
        return True
    except Exception:
        return False


def invite_email_html(interview_name: str, invite_url: str, brand_name: str = "") -> str:
    brand = brand_name or "the hiring team"
    return f"""
    <div style="font-family:system-ui,Arial,sans-serif;max-width:520px;margin:0 auto;color:#1f2937">
      <h2 style="color:#111827">You're invited to an interview</h2>
      <p>{brand} has invited you to complete an AI interview for <strong>{interview_name}</strong>.</p>
      <p>It takes a few minutes and you can do it whenever suits you.</p>
      <p style="margin:28px 0">
        <a href="{invite_url}" style="background:#4f46e5;color:#fff;padding:12px 22px;border-radius:10px;text-decoration:none;font-weight:600">Start interview</a>
      </p>
      <p style="color:#6b7280;font-size:13px">Or paste this link into your browser:<br>{invite_url}</p>
    </div>
    """
