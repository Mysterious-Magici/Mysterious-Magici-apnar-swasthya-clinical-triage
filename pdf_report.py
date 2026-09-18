"""
pdf_report.py
------------------------------------------------------------------
Builds the downloadable / printable PDF containing:
  * patient parameters
  * triage result and probability table
  * the full chat transcript
  * the safety disclaimer

Uses fpdf2 (pure Python, installs anywhere, no system libraries).
------------------------------------------------------------------
"""

import os
import re
from datetime import datetime

from fpdf import FPDF

LOGO_CANDIDATES = ["logo.png", "logo.jpeg", "logo.jpg"]


def _clean(text) -> str:
    """Strip markdown marks and characters the built-in PDF fonts cannot print."""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)         # **bold**
    text = re.sub(r"\*(.*?)\*", r"\1", text)             # *italic*
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)   # ### headings
    text = text.replace("`", "").replace("> ", "")
    swaps = {
        "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2022": "-", "\u2192": "->",
        "\u00b0": " deg", "\u00b2": "2", "\u2079": "9", "\u00d7": "x",
        "\u2264": "<=", "\u2265": ">=", "\u2026": "...",
    }
    for bad, good in swaps.items():
        text = text.replace(bad, good)
    # the standard fonts are Latin-1 only
    return text.encode("latin-1", "ignore").decode("latin-1")


class TriagePDF(FPDF):
    # ---------- page furniture ----------
    def header(self):
        for logo in LOGO_CANDIDATES:
            if os.path.exists(logo):
                try:
                    self.image(logo, x=10, y=8, w=16)
                    break
                except Exception:
                    pass
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(15, 118, 110)
        self.cell(0, 8, "Apnar Swasthya - Clinical Triage Report", align="C")
        self.ln(8)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 5, "Automated Disease Triage & Differential Diagnostic Support",
                  align="C")
        self.ln(7)
        self.set_draw_color(200, 200, 200)
        self.line(10, 28, 200, 28)
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 4, "Decision-support prototype - not a medical diagnosis. "
                        "Confirm with a registered medical practitioner.", align="C")
        self.ln(4)
        self.cell(0, 4, f"Page {self.page_no()}/{{nb}}", align="C")

    # ---------- helpers ----------
    def section(self, title):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(15, 118, 110)
        self.cell(0, 7, _clean(title))
        self.ln(8)
        self.set_text_color(0, 0, 0)

    def line_text(self, text, size=9, style=""):
        self.set_font("Helvetica", style, size)
        self.cell(0, 6, _clean(text))
        self.ln(6)

    def body(self, text, size=9):
        self.set_font("Helvetica", "", size)
        self.multi_cell(0, 5, _clean(text))


def build_pdf(profile: dict, triage: dict, messages: list,
              top_disease: str, confidence: float, status: str,
              model_mode: str = "XGBoost") -> bytes:
    pdf = TriagePDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.line_text(f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}"
                  f"   |   Triage engine: {model_mode}")

    # ---------------- 1. triage summary ----------------
    pdf.section("1. Triage Summary")
    pdf.line_text(f"Primary differential: {top_disease}", size=13, style="B")
    pdf.line_text(f"Model confidence: {confidence:.1f}%    |    Status: {status}", size=10)
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(240, 249, 246)
    pdf.cell(110, 6, "Condition", border=1, fill=True)
    pdf.cell(40, 6, "Probability (%)", border=1, fill=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 9)
    for disease, prob in sorted(triage.items(), key=lambda kv: kv[1], reverse=True):
        pdf.cell(110, 6, _clean(disease), border=1)
        pdf.cell(40, 6, f"{prob * 100:.2f}", border=1)
        pdf.ln(6)

    # ---------------- 2. patient parameters ----------------
    pdf.section("2. Patient Parameters")
    items = list(profile.items())
    for i in range(0, len(items), 2):
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(45, 6, _clean(items[i][0]))
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(45, 6, _clean(items[i][1]))
        if i + 1 < len(items):
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(45, 6, _clean(items[i + 1][0]))
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(45, 6, _clean(items[i + 1][1]))
        pdf.ln(6)

    # ---------------- 3. transcript ----------------
    pdf.section("3. Clinical Copilot Transcript")
    if not messages:
        pdf.body("No conversation was recorded in this session.")
    for msg in messages:
        clinician = msg.get("role") == "user"
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(30, 64, 120) if clinician else pdf.set_text_color(15, 118, 110)
        pdf.cell(0, 6, "Clinician:" if clinician else "Copilot:")
        pdf.ln(6)
        pdf.set_text_color(0, 0, 0)
        pdf.body(msg.get("content", ""))
        pdf.ln(2)

    # ---------------- 4. disclaimer ----------------
    pdf.section("4. Disclaimer")
    pdf.body(
        "Apnar Swasthya is an educational clinical decision-support prototype. The prediction is "
        "produced by a machine-learning model trained on the INDIA_HEALTHCARE_DATA_2026 dataset and "
        "reflects statistical association only. It does not constitute a diagnosis, a prescription "
        "or medical advice, and it must never delay or replace assessment by a registered medical "
        "practitioner. In an emergency, contact emergency services (108 in India) immediately."
    )

    out = pdf.output()
    return bytes(out)
