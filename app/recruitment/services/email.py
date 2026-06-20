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


_OUTCOME_COPY = {
    "selected": ("Good news about your application",
                 "We were impressed with your interview and would like to take your application forward. Someone from the team will be in touch with next steps."),
    "advance": ("Your application is moving forward",
                "Thanks for completing your interview. Your application is progressing to the next stage and we'll follow up soon."),
    "rejected": ("Update on your application",
                 "Thank you for taking the time to interview with us. After careful consideration we won't be moving forward on this occasion. We wish you the very best in your search."),
    "reminder": ("A reminder to complete your interview",
                 "This is a friendly reminder to complete your AI interview. It only takes a few minutes."),
    "completed": ("We've received your interview",
                  "Thank you for completing your interview. Our team will review it and be in touch."),
    "score_ready": ("Your interview has been reviewed",
                    "Your interview has been reviewed by our team. We'll be in touch with next steps."),
}


def outcome_email_html(kind: str, candidate_name: str = "", job_title: str = "", brand_name: str = "", link: str = "") -> str:
    title, body = _OUTCOME_COPY.get(kind, _OUTCOME_COPY["completed"])
    brand = brand_name or "The hiring team"
    hello = f"Hi {candidate_name}," if candidate_name else "Hi,"
    role = f" for <strong>{job_title}</strong>" if job_title else ""
    cta = f'<p style="margin:24px 0"><a href="{link}" style="background:#4f46e5;color:#fff;padding:12px 22px;border-radius:10px;text-decoration:none;font-weight:600">Open</a></p>' if link else ""
    return f"""
    <div style="font-family:system-ui,Arial,sans-serif;max-width:520px;margin:0 auto;color:#1f2937">
      <h2 style="color:#111827">{title}</h2>
      <p>{hello}</p>
      <p>{body.replace('your interview', 'your interview' + role, 1) if role else body}</p>
      {cta}
      <p style="color:#6b7280;font-size:13px">{brand}</p>
    </div>
    """


# ---------------- Custom template support (Phase 13) ----------------
TEMPLATE_KINDS = ["invite", "selected", "advance", "rejected", "reminder", "completed", "score_ready"]
TEMPLATE_VARIABLES = ["candidate_name", "job_title", "interview_name", "brand_name", "link"]

DEFAULT_TEMPLATES = {
    "invite": {
        "subject": "You're invited to interview for {interview_name}",
        "body_html": "<p>Hi {candidate_name},</p><p>{brand_name} has invited you to complete a short AI interview for <strong>{interview_name}</strong>.</p><p><a href=\"{link}\">Start your interview</a></p>",
    },
    "selected": {"subject": "Good news about your application", "body_html": "<p>Hi {candidate_name},</p><p>We were impressed with your interview for {job_title} and would like to take your application forward.</p><p>{brand_name}</p>"},
    "advance": {"subject": "Your application is moving forward", "body_html": "<p>Hi {candidate_name},</p><p>Thanks for completing your interview for {job_title}. Your application is progressing to the next stage.</p><p>{brand_name}</p>"},
    "rejected": {"subject": "Update on your application", "body_html": "<p>Hi {candidate_name},</p><p>Thank you for interviewing for {job_title}. We won't be moving forward on this occasion, and we wish you well.</p><p>{brand_name}</p>"},
    "reminder": {"subject": "A reminder to complete your interview", "body_html": "<p>Hi {candidate_name},</p><p>This is a reminder to complete your interview for {interview_name}.</p><p><a href=\"{link}\">Continue</a></p>"},
    "completed": {"subject": "We received your interview", "body_html": "<p>Hi {candidate_name},</p><p>Thank you for completing your interview. Our team will review it and be in touch.</p><p>{brand_name}</p>"},
    "score_ready": {"subject": "Your interview has been reviewed", "body_html": "<p>Hi {candidate_name},</p><p>Your interview has been reviewed. We'll be in touch with next steps.</p><p>{brand_name}</p>"},
}


def render_custom(subject: str, body_html: str, ctx: dict):
    """Replace {placeholder} tokens in a custom template."""
    def sub(t: str) -> str:
        for k, v in (ctx or {}).items():
            t = t.replace("{" + k + "}", str(v if v is not None else ""))
        return t
    return sub(subject or ""), sub(body_html or "")
