"""Generate the three Milestone 1 sample documents.

The documents describe one fictional project from three sources and deliberately
contain the cases the contradiction engine must handle later:

* proposed LKR 7M (proposal.docx) vs approved LKR 5M (annual report PDF)
  -> CONTEXTUALLY_DIFFERENT, not a contradiction
* approved LKR 5M (PDF) vs approved LKR 5.5M (budget.xlsx)
  -> CONTRADICTED / numerical
* completion June 2025 (PDF) vs September 2025 (xlsx)
  -> CONTRADICTED / temporal

Usage:  python scripts/make_sample_documents.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from openpyxl import Workbook
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PDF_NAME = "project_alpha_annual_report.pdf"
DOCX_NAME = "project_alpha_proposal.docx"
XLSX_NAME = "project_alpha_budget.xlsx"


def make_pdf(path: Path) -> None:
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Annual Report 2025", styles["Title"]),
        Paragraph("1. Overview", styles["Heading1"]),
        Paragraph(
            "This report summarises the activities of the Research Office of the "
            "University of Sri Jayewardenepura (USJ) during 2025.",
            styles["BodyText"],
        ),
        PageBreak(),
        Paragraph("2. Project Alpha", styles["Heading1"]),
        Paragraph(
            "Project Alpha received an approved allocation of LKR 5 million in January 2025.",
            styles["BodyText"],
        ),
        Paragraph(
            "Project Alpha was completed in June 2025 under the supervision of Dr. A. Perera.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]
    table = Table(
        [
            ["Project", "Approved budget (LKR)", "Status"],
            ["Project Alpha", "5,000,000", "Completed"],
            ["Project Beta", "2,500,000", "Active"],
        ]
    )
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "black")]))
    story.append(table)
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def make_docx(path: Path) -> None:
    doc = Document()
    doc.add_heading("Project Alpha — Funding Proposal", level=0)
    doc.add_heading("Budget request", level=1)
    doc.add_paragraph(
        "The project team proposes a budget of LKR 7 million for Project Alpha, "
        "submitted in November 2024."
    )
    doc.add_paragraph("Sri Jayewardenepura University will host the project.")
    table = doc.add_table(rows=3, cols=2)
    table.style = "Table Grid"
    for r, (k, v) in enumerate(
        [("Item", "Proposed cost (LKR)"), ("Equipment", "4,000,000"), ("Staff", "3,000,000")]
    ):
        table.cell(r, 0).text = k
        table.cell(r, 1).text = v
    doc.save(str(path))


def make_xlsx(path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Budget"
    ws.append(["Project", "Modality", "Amount (LKR)", "Date"])
    ws.append(["Project Alpha", "approved", 5_500_000, "2025-01"])
    ws.append(["Project Alpha", "actual", 4_800_000, "2025-09"])
    ws.append(["Project Beta", "approved", 2_500_000, "2025-03"])

    ms = wb.create_sheet("Milestones")
    ms.append(["Project", "Milestone", "Date"])
    ms.append(["Project Alpha", "Completed", "2025-09"])
    wb.save(str(path))


def main(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / PDF_NAME, out_dir / DOCX_NAME, out_dir / XLSX_NAME]
    make_pdf(paths[0])
    make_docx(paths[1])
    make_xlsx(paths[2])
    return paths


if __name__ == "__main__":
    default = Path(__file__).parents[1] / "sample_documents"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    for p in main(target):
        print(p)
