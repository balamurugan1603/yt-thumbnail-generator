"""
Batch thumbnail generation with Excel report.
Usage: python run_batch.py
"""

import asyncio
import logging
import os
from datetime import datetime
import time

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from settings.config import RetryConfig
from pipeline.batch_pipeline import generate_thumbnails_batch, BatchSummary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/batch.log"),
    ],
)
logger = logging.getLogger(__name__)

# Prompts
PROMPTS = [
    "I Tested AI for 30 Days",
    "Why Most Startups Fail",
    "Build a SaaS in a Weekend",
    "The Dark Side of Productivity",
    "Can You Learn Coding Fast?",
    "This Morning Routine Changed Everything",
    "Inside a $1M App Idea",
    "How I Beat Procrastination",
    "AI vs Human Creativity",
    "Master Python in 10 Minutes",
    "The Truth About Passive Income",
    "I Tried No Social Media",
    "Make Money with Open Source?",
    "Design Thumbnails That Click",
    "The Science of Deep Work",
    "Build Your Own GPT App",
    "Why Your UI Looks Bad",
    "Secrets of High CTR Videos",
    "Startup Ideas That Work",
    "Fix Your Sleep in 7 Days",
    "From Zero to ML Engineer",
    "How Billionaires Think",
    "The Minimalist Workspace Setup",
    "Avoid These Coding Mistakes",
    "What Makes Content Go Viral?",
    "AI Automation for Developers",
    "Is Remote Work Dying?",
    "Learn System Design Fast",
    "Quiz: Guess the Country by Emoji",
    "Only 1% Can Solve This Riddle",
]

ARTIFACTS_DIR = r"artifacts\runs\repair_prompt"  # artifacts
REPORTS_DIR = r"artifacts\runs\repair_prompt"  # reports
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Excel styles
HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF", size=10)
HEADER_FILL = PatternFill("solid", start_color="1F4E79")
PASS_FILL = PatternFill("solid", start_color="E2EFDA")
FAIL_FILL = PatternFill("solid", start_color="FCE4D6")
SUMMARY_FILL = PatternFill("solid", start_color="D6E4F0")
WHITE_FILL = PatternFill("solid", start_color="FFFFFF")
CELL_FONT = Font(name="Arial", size=9)
BOLD_FONT = Font(name="Arial", size=9, bold=True)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
THIN_BORDER = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)

# (header label, column width)
COLUMNS = [
    ("Prompt", 28),
    ("Status", 9),
    ("Image Path", 38),
    ("Duration (s)", 12),
    ("Attempt", 8),
    ("Clutter", 9),
    ("Edge Density", 13),
    ("Contrast", 9),
    ("Primary Contrast Ratio", 18),
    ("Secondary Contrast Ratio", 18),
    ("Artifact", 9),
    ("Artifact Probability", 14),
    ("Prompt-Image Similarity", 18),
    ("CLIP Score", 14),
    ("Error", 35),
]


# Helpers
def _check_passed(check_results, name):
    for c in check_results or []:
        if c.name == name:
            return "PASS" if c.passed else "FAIL"
    return "-"


def _check_metric(check_results, name, field):
    for c in check_results or []:
        if c.name == name:
            val = getattr(c, field, None)
            return round(val, 4) if isinstance(val, float) else (val or "-")
    return "-"


def _style(cell, fill=None, font=None, align=None):
    if fill:
        cell.fill = fill
    if font:
        cell.font = font
    if align:
        cell.alignment = align
    cell.border = THIN_BORDER


# Excel writer
def write_excel_report(summary: BatchSummary, path: str):
    wb = Workbook()

    # Results sheet
    ws = wb.active
    ws.title = "Results"
    ws.freeze_panes = "A3"

    # Title
    last_col = get_column_letter(len(COLUMNS))
    ws.merge_cells(f"A1:{last_col}1")
    t = ws["A1"]
    t.value = f"Thumbnail Batch Report  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    t.font = Font(name="Arial", bold=True, size=12, color="FFFFFF")
    t.fill = PatternFill("solid", start_color="0D3B66")
    t.alignment = CENTER
    t.border = THIN_BORDER
    ws.row_dimensions[1].height = 24

    # Headers
    for ci, (label, width) in enumerate(COLUMNS, start=1):
        col = get_column_letter(ci)
        cell = ws[f"{col}2"]
        cell.value = label
        _style(cell, fill=HEADER_FILL, font=HEADER_FONT, align=CENTER)
        ws.column_dimensions[col].width = width
    ws.row_dimensions[2].height = 28

    # Data rows
    for ri, r in enumerate(summary.results, start=3):
        ws.row_dimensions[ri].height = 18

        best = getattr(r, "best_attempt", None)
        check_results = getattr(best, "check_results", []) if best else []
        attempt_num = getattr(best, "attempt_num", "-") if best else "-"
        row_fill = PASS_FILL if r.success else FAIL_FILL

        values = [
            r.prompt,
            "SUCCESS" if r.success else "FAILED",
            getattr(r, "image_path", None) or "-",
            round(r.duration, 2),
            attempt_num,
            _check_passed(check_results, "clutter"),
            _check_metric(check_results, "clutter", "edge_density"),
            _check_passed(check_results, "contrast"),
            _check_metric(check_results, "contrast", "primary_contrast_ratio"),
            _check_metric(check_results, "contrast", "secondary_contrast_ratio"),
            _check_passed(check_results, "artifact"),
            _check_metric(check_results, "artifact", "artifact_probability"),
            _check_passed(check_results, "clip"),
            _check_metric(check_results, "clip", "clip_score"),
            r.error or "",
        ]

        # Columns that show PASS/FAIL and should be coloured individually
        PASS_FAIL_COLS = {6, 8, 10, 12}

        for ci, val in enumerate(values, start=1):
            col = get_column_letter(ci)
            cell = ws[f"{col}{ri}"]
            cell.value = val
            align = LEFT if ci in (1, 3, len(COLUMNS)) else CENTER
            if ci in PASS_FAIL_COLS:
                fill = (
                    PASS_FILL
                    if val == "PASS"
                    else (FAIL_FILL if val == "FAIL" else row_fill)
                )
            else:
                fill = row_fill
            _style(cell, fill=fill, font=CELL_FONT, align=align)

    # Summary sheet
    ws2 = wb.create_sheet("Summary")
    ws2.column_dimensions["A"].width = 22
    ws2.column_dimensions["B"].width = 18

    rows = [
        ("Batch Summary", None),
        ("Run at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Total prompts", summary.total),
        ("Succeeded", summary.succeeded),
        ("Failed", summary.failed),
        ("Success rate", "=B4/B3"),
        ("Total duration", f"{summary.duration:.1f}s"),
        ("Avg per image", "=B7/B3"),
    ]

    for ri, (label, value) in enumerate(rows, start=1):
        a, b = ws2[f"A{ri}"], ws2[f"B{ri}"]
        a.value = label
        b.value = value
        ws2.row_dimensions[ri].height = 18

        if ri == 1:
            ws2.merge_cells("A1:B1")
            a.font = Font(name="Arial", bold=True, size=11, color="FFFFFF")
            a.fill = PatternFill("solid", start_color="1F4E79")
            a.alignment = CENTER
        else:
            fill = SUMMARY_FILL if ri % 2 == 0 else WHITE_FILL
            a.font = BOLD_FONT
            b.font = CELL_FONT
            for cell in (a, b):
                cell.fill = fill
            a.alignment = LEFT
            b.alignment = CENTER

        if ri == 6:
            b.number_format = "0.0%"
        if ri == 8:
            b.number_format = "0.0s"

        for cell in (a, b):
            cell.border = THIN_BORDER

    wb.save(path)
    logger.info("Report saved: %s", path)


def save_images(summary: BatchSummary):
    for r in summary.results:
        if r.success and r.image:
            path = os.path.join(ARTIFACTS_DIR, f"{int(time.time()*1000)}.png")
            r.image.save(path)
            r.image_path = path
        else:
            r.image_path = None


async def main():
    summary = await generate_thumbnails_batch(PROMPTS, retry_cfg=RetryConfig())

    save_images(summary)

    report_path = os.path.join(
        REPORTS_DIR, f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )
    write_excel_report(summary, report_path)

    print(f"\n{'='*55}")
    print(f"  Total   : {summary.total}")
    print(f"  Success : {summary.succeeded}  ({summary.success_rate:.0%})")
    print(f"  Failed  : {summary.failed}")
    print(f"  Time    : {summary.duration:.1f}s")
    print(f"  Report  : {report_path}")
    print(f"{'='*55}\n")

    for r in summary.results:
        tag = "OK  " if r.success else "FAIL"
        loc = r.image_path if r.success else r.error
        print(f"  [{tag}] {r.prompt!r:<35} {r.duration:.1f}s  ->  {loc}")


if __name__ == "__main__":
    asyncio.run(main())
