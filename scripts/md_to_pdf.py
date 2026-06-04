"""Convert system_description.md to a clean PDF using reportlab.

Designed for the limited markdown subset our description uses:
  - YAML front-matter (skipped)
  - # / ## / ### headings
  - **bold**, *italic*
  - bullet lists (-)
  - tables (pipe-delimited)
  - plain paragraphs

Usage:  python scripts/md_to_pdf.py system_description.md output.pdf
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem,
)


def inline_md(text: str) -> str:
    """Convert basic inline markdown to reportlab HTML markup."""
    # backticks -> mono
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    # **bold**
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    # *italic*
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def main() -> None:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])

    raw = src.read_text(encoding="utf-8").splitlines()
    # Skip YAML front-matter.
    if raw and raw[0].strip() == "---":
        end = next(i for i in range(1, len(raw)) if raw[i].strip() == "---")
        raw = raw[end + 1:]

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16,
                        spaceBefore=12, spaceAfter=8, textColor=colors.black)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                        spaceBefore=10, spaceAfter=6)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11,
                        spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10,
                          leading=13, spaceAfter=4)

    story = []

    i = 0
    while i < len(raw):
        line = raw[i].rstrip()

        if not line.strip():
            i += 1
            continue

        if line.startswith("# "):
            story.append(Paragraph(inline_md(line[2:]), h1))
        elif line.startswith("## "):
            story.append(Paragraph(inline_md(line[3:]), h2))
        elif line.startswith("### "):
            story.append(Paragraph(inline_md(line[4:]), h3))
        elif line.startswith("|"):
            # Collect table.
            tbl_rows = []
            while i < len(raw) and raw[i].startswith("|"):
                row = [c.strip() for c in raw[i].strip("|").split("|")]
                tbl_rows.append(row)
                i += 1
            # Drop separator row (---).
            tbl_rows = [r for r in tbl_rows if not all(re.match(r"^:?-+:?$", c) for c in r)]
            tbl_data = [[Paragraph(inline_md(c), body) for c in r] for r in tbl_rows]
            tbl = Table(tbl_data, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(Spacer(1, 4))
            story.append(tbl)
            story.append(Spacer(1, 6))
            continue
        elif line.lstrip().startswith("- "):
            items = []
            while i < len(raw) and raw[i].lstrip().startswith("- "):
                items.append(ListItem(Paragraph(inline_md(raw[i].lstrip()[2:]), body),
                                      leftIndent=12))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=18))
            continue
        else:
            # Paragraph: gather consecutive non-empty, non-special lines.
            para = [line]
            i += 1
            while i < len(raw):
                nxt = raw[i].rstrip()
                if not nxt.strip(): break
                if nxt.startswith(("#", "|", "- ")): break
                para.append(nxt)
                i += 1
            story.append(Paragraph(inline_md(" ".join(para)), body))
            continue
        i += 1

    doc = SimpleDocTemplate(str(out), pagesize=LETTER,
                            leftMargin=0.9*inch, rightMargin=0.9*inch,
                            topMargin=0.9*inch, bottomMargin=0.9*inch,
                            title="OpenBracket WMT26 System Description")
    doc.build(story)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
