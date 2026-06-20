"""
PDF score report generator (Phase 10) using reportlab Platypus.
Returns PDF bytes for an interview result: candidate, job, overall score,
per-criterion breakdown with evidence, integrity, and transcript.
"""
from io import BytesIO
from typing import Any, Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

PRIMARY = colors.HexColor("#4f46e5")
MUTED = colors.HexColor("#6b7280")


def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("HBig", parent=s["Title"], fontSize=20, spaceAfter=4))
    s.add(ParagraphStyle("Sub", parent=s["Normal"], textColor=MUTED, fontSize=10, spaceAfter=2))
    s.add(ParagraphStyle("H2b", parent=s["Heading2"], textColor=PRIMARY, fontSize=13, spaceBefore=12, spaceAfter=6))
    s.add(ParagraphStyle("Body", parent=s["Normal"], fontSize=10, leading=14))
    s.add(ParagraphStyle("Ev", parent=s["Normal"], fontSize=9, textColor=MUTED, leading=12))
    return s


def build_score_report(data: Dict[str, Any]) -> bytes:
    """
    data = {
      candidate_name, candidate_email, job_title, interview_name, status,
      overall_score, recommendation, summary, needs_human_review, risk_level,
      risk_signals: {..}, scored_at,
      criteria: [{name, weight, raw_score, evidence, reasoning}],
      transcript: [{question, answer}],
    }
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18 * mm, bottomMargin=18 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm, title="Interview Report")
    st = _styles()
    story: List[Any] = []

    story.append(Paragraph(data.get("candidate_name") or "Candidate", st["HBig"]))
    sub = " · ".join(x for x in [data.get("candidate_email"), data.get("job_title"), data.get("interview_name")] if x)
    if sub:
        story.append(Paragraph(sub, st["Sub"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 8))

    # Score summary
    score = data.get("overall_score")
    score_txt = f"{round(score)}/100" if score is not None else "Not scored"
    rec = data.get("recommendation") or "—"
    rows = [
        ["Overall score", score_txt],
        ["Recommendation", str(rec).title()],
        ["Status", str(data.get("status") or "").title()],
    ]
    if data.get("needs_human_review"):
        rows.append(["Flag", "Needs human review"])
    if data.get("risk_level") and data.get("risk_level") != "low":
        sig = data.get("risk_signals") or {}
        sig_txt = ", ".join(f"{k.replace('_',' ')}: {v}" for k, v in sig.items()) or "—"
        rows.append(["Integrity risk", f"{data['risk_level']} ({sig_txt})"])
    t = Table(rows, colWidths=[40 * mm, 120 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#f3f4f6")),
    ]))
    story.append(t)
    if data.get("summary"):
        story.append(Spacer(1, 6))
        story.append(Paragraph(data["summary"], st["Body"]))

    # Criteria
    criteria = data.get("criteria") or []
    if criteria:
        from app.recruitment.services.insights import key_points
        kp = key_points(criteria)
        if kp["strengths"] or kp["concerns"]:
            story.append(Paragraph("Summary", st["H2b"]))
            if kp["strengths"]:
                story.append(Paragraph("Key strengths", ParagraphStyle("ks", parent=st["Body"], fontSize=11, spaceBefore=4, spaceAfter=2)))
                for s in kp["strengths"]:
                    story.append(Paragraph(f"+ {s['name']} ({s['score']}/100). {s['evidence']}", st["Ev"]))
            if kp["concerns"]:
                story.append(Paragraph("Areas to probe", ParagraphStyle("kc", parent=st["Body"], fontSize=11, spaceBefore=6, spaceAfter=2)))
                for c in kp["concerns"]:
                    story.append(Paragraph(f"- {c['name']} ({c['score']}/100). {c['evidence']}", st["Ev"]))

        story.append(Paragraph("Per-criterion assessment", st["H2b"]))
        for c in criteria:
            raw = c.get("raw_score")
            head = f"{c.get('name','Criterion')} — {round(raw) if raw is not None else '—'} (weight {c.get('weight',0)}%)"
            story.append(Paragraph(head, ParagraphStyle("c", parent=st["Body"], fontSize=11, spaceBefore=6, spaceAfter=1)))
            if c.get("evidence"):
                story.append(Paragraph(f"<b>Evidence:</b> {c['evidence']}", st["Ev"]))
            if c.get("reasoning"):
                story.append(Paragraph(c["reasoning"], st["Ev"]))

    # Delivery & sentiment (advisory)
    delivery = data.get("delivery") or {}
    if delivery.get("total_words"):
        story.append(Paragraph("Delivery (advisory)", st["H2b"]))
        wpm = delivery.get("words_per_minute")
        line = f"Pace: {delivery.get('pace_label','-')}" + (f" (~{wpm} wpm)" if wpm else "")
        line += f" · Filler words: {delivery.get('filler_count',0)}"
        line += f" · Overall tone: {delivery.get('sentiment_label','neutral')}"
        story.append(Paragraph(line, st["Ev"]))
        story.append(Paragraph("These delivery metrics are advisory and should not by themselves affect a hiring decision.", st["Ev"]))

    # Transcript
    transcript = data.get("transcript") or []
    if transcript:
        story.append(Paragraph("Transcript", st["H2b"]))
        for turn in transcript:
            q = turn.get("question")
            a = turn.get("answer")
            if q:
                story.append(Paragraph(f"<b>Q:</b> {q}", st["Ev"]))
            story.append(Paragraph(f"{a or '(no answer)'}", ParagraphStyle("a", parent=st["Body"], spaceAfter=6)))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    story.append(Paragraph("Generated by ConvoHubAI Recruitment. Decisions require human review.", st["Sub"]))

    doc.build(story)
    return buf.getvalue()
