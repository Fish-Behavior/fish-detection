"""Generate tests/fixtures/synth_db.xlsx: a deterministic 10-row mini workbook.

Column headers are copied verbatim from the real `00_NTT_DataBase.xlsx`
(including its trailing-whitespace/embedded-newline quirks, e.g.
`"NTT \\nTime (min):"`) so catalog.py's header-handling is exercised the same
way it will be against real data, without touching the restricted real file
(PRD Clarification C14).

Covers, across 10 rows: a normal row, a combo-treatment row (`+ methylone`),
two blank-template rows (blank `Compund:`, excluded per FR-001), an exact
duplicate-subject pair (mirrors the real `Subject # 320` case, PRD C13), a row
with all 8 movement/zone columns blank, a 10-minute-exposure row (mirrors the
real 3-of-353 short-duration edge case), and rows spanning more than one
strain.

Run directly to (re)generate the committed fixture:
    python tests/fixtures/make_synth_db.py
"""

from __future__ import annotations

from pathlib import Path

import openpyxl

OUTPUT_PATH = Path(__file__).with_name("synth_db.xlsx")

# Verbatim from the real workbook (see docs/progress.md §0 inventory).
HEADERS = [
    "Date of EXP:",
    "Subject #:",
    "Strain:",
    "Sex (M/F):",
    "Age (~):",
    "Compund:",
    "Conc. (mM):",
    "Agent Exposure Time (min):",
    "NTT \nTime (min):",
    "UV \nExposure\nTime (min)",
    "TDM (Full Arena):",
    "TDM (Top Half):",
    "TDM (Bot Half):",
    "Velocity (Full Arena)",
    "Velocity (Top Half)",
    "Velocity (Bot Half)",
    "Time Spent (Top):",
    "Time Spent (Bot):",
    "H2O (Before):",
    "H2O (After):",
    "Brain Tissue:",
    "Body Tissue:",
]

CASPER = "Casper (roya9; mitfaw2)"
WILD_TYPE = "Wild-type (AB)"

# One tuple per row, in HEADERS order. `None` = blank cell.
ROWS: list[tuple] = [
    # 1. normal row
    (260101, 22, CASPER, "F", 4, "GT-1-42", "0.03", 20, 10, None, 500.0, 200.0, 300.0, 2.1, 2.4, 1.9, 400.0, 200.0, 5.0, 4.8, "Y", None),
    # 2. combo-treatment row (compound name carries "+ methylone", matches real folder naming edge case)
    (260101, 45, CASPER, "M", 4, "FD-2-66 + methylone", "0.03 + 0.01", 20, 10, None, 480.0, 190.0, 290.0, 2.0, 2.3, 1.8, 390.0, 190.0, 5.0, 4.7, "Y", None),
    # 3. blank-template row (blank Compund: -> excluded per FR-001)
    (None, 100, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    # 4/5. exact duplicate-subject pair (mirrors real Subject #320, PRD C13)
    (251216, 320, CASPER, "M", 5, "FD-2-66", "0.2", 20, 10, None, 460.0, 180.0, 280.0, 1.9, 2.2, 1.7, 380.0, 180.0, 5.1, 4.9, "Y", None),
    (251216, 320, CASPER, "M", 5, "FD-2-66", "0.2", 20, 10, None, 460.0, 180.0, 280.0, 1.9, 2.2, 1.7, 380.0, 180.0, 5.1, 4.9, "Y", None),
    # 6. all 8 movement/zone columns blank (TDM x3, Velocity x3, Time Spent x2) - still a real trial (Compund set)
    (260102, 55, CASPER, "F", 4, "Veh", "0", 20, 10, None, None, None, None, None, None, None, None, None, 5.0, 4.8, "Y", None),
    # 7. 10-minute-exposure row (mirrors the real 3-of-353 short-duration edge case)
    (260102, 66, CASPER, "M", 4, "MTA-5-62", "0.05", 10, 10, None, 300.0, 120.0, 180.0, 1.5, 1.7, 1.3, 250.0, 120.0, 5.0, 4.9, "Y", None),
    # 8. different strain (Wild-type), spot-check coverage per PRD §10 risk note
    (260103, 77, WILD_TYPE, "F", 5, "DOB", "0.03", 20, 10, None, 470.0, 185.0, 285.0, 2.0, 2.2, 1.8, 385.0, 185.0, 5.0, 4.8, "Y", None),
    # 9. second blank-template row
    (None, 88, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    # 10. normal row, another compound family
    (260103, 99, CASPER, "M", 4, "Fentanyl", "0.03", 20, 10, None, 490.0, 195.0, 295.0, 2.1, 2.3, 1.9, 395.0, 195.0, 5.0, 4.9, "Y", None),
]


def generate(output_path: Path = OUTPUT_PATH) -> Path:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(HEADERS)
    for row in ROWS:
        sheet.append(list(row))
    workbook.save(output_path)
    return output_path


if __name__ == "__main__":
    path = generate()
    print(f"wrote {path}")
