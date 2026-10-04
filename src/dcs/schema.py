"""The files `prepds` writes, restated as constants (D-013: `dcs` never imports `prepds`).

Source of each fact: the prepds export and review code as of the plan's §1
check (frames.parquet, segments.csv, manifest.json, provenance.json,
accepted_index.parquet) and the trial workbook it reads. If prepds changes
its output, the gold reader's schema checks (EC-25) fail with a message
naming the file and column, and this module is updated.
"""

from __future__ import annotations

# --- behavior states (prepds BehaviorState, in its order) -------------------------------------
CONTROLLED_SWIM = "Controlled Swim"
ERRATIC_MOVEMENT = "Erratic Movement"
FREEZING_DRIFT = "Freezing/Drift"
LISTING_LORR = "Listing/LORR"
SURFACE_BREACH = "Surface Breach"
DEAD = "Dead"
UNDETERMINED = "Undetermined"  # pipeline bookkeeping; never allowed in an Accepted video (EC-22)
STATES = (CONTROLLED_SWIM, ERRATIC_MOVEMENT, FREEZING_DRIFT, LISTING_LORR, SURFACE_BREACH, DEAD, UNDETERMINED)

# Frame and segment label provenance (prepds FrameSource).
SOURCE_AUTO = "auto"
SOURCE_MANUAL = "manual"
SOURCES = (SOURCE_AUTO, SOURCE_MANUAL)

# Review status in manifest.json (prepds ReviewStatus).
REVIEW_NOT_PROCESSED = "NOT_PROCESSED"
REVIEW_PROCESSED_AUTO = "PROCESSED_AUTO"
REVIEW_EDITED = "EDITED"
REVIEW_ACCEPTED = "ACCEPTED"
REVIEW_REJECTED = "REJECTED"
REVIEW_STATUSES = (REVIEW_NOT_PROCESSED, REVIEW_PROCESSED_AUTO, REVIEW_EDITED, REVIEW_ACCEPTED, REVIEW_REJECTED)

# --- per-video gold folder: <accepted>/<video_id>/ --------------------------------------------
FRAMES_FILE = "frames.parquet"
SEGMENTS_FILE = "segments.csv"
MANIFEST_FILE = "manifest.json"
PROVENANCE_FILE = "provenance.json"
STRIP_FILE = "strip.png"  # written by prepds, never read by dcs
INDEX_FILE = "accepted_index.parquet"

# frames.parquet columns in file order, with the dtype pandas reports after reading.
FRAME_DTYPES = {
    "frame_idx": "int32",
    "t_sec": "float32",
    "x": "float32",
    "y": "float32",
    "orientation_deg": "float32",  # null on undetected frames
    "depth_from_surface": "float32",  # pixel row from the frame top; null on undetected frames
    "detected": "bool",
    "velocity": "float32",
    "acceleration": "float32",
    "angular_velocity": "float32",
    "meander": "float32",
    "is_immobile": "bool",
    "state": "category",
    "source": "category",
    "confidence": "float32",
}
# Columns that hold a 0.0 sentinel (not null) on undetected frames.
ZERO_ON_UNDETECTED = ("x", "y", "velocity", "acceleration", "angular_velocity", "meander")
NULL_ON_UNDETECTED = ("orientation_deg", "depth_from_surface")
# Kinematic features. prepds (features.py) also writes 0.0 where a detected run has too little
# history: speed and turn need one earlier frame of the run, acceleration and meander two.
KINEMATIC_COLUMNS = ("velocity", "acceleration", "angular_velocity", "meander")
ZERO_ON_FIRST_FRAME = ("velocity", "angular_velocity")
ZERO_ON_FIRST_TWO_FRAMES = ("acceleration", "meander")
# `confidence` is in the schema but prepds labels set it to null on every frame (labeling.py:
# "nothing probabilistic remains"); a reader must not rely on it.

SEGMENT_COLUMNS = ("start_s", "end_s", "duration_s", "state", "source")

MANIFEST_KEYS = (
    "subject_id",
    "sex",
    "compound",
    "concentration_mM",
    "video_path",
    "video_duration_s",
    "video_fps",
    "pipeline_version",
    "calibration_profile_version",
    "processed_at",
    "review_status",
    "reviewer",
    "reviewed_at",
    "edited",
    "edit_count",
    "review_flags",
)
PROVENANCE_KEYS = ("reviewer", "reviewed_at", "source", "calibration_profile_version", "pipeline_version")

INDEX_COLUMNS = (
    "video_id",
    "subject_id",
    "sex",
    "compound",
    "concentration_mM",
    "strain",
    "age",
    "date",
    "agent_exposure_min",
    "video_path",
    "video_duration_s",
    "video_fps",
    "pipeline_version",
    "calibration_profile_version",
    "reviewer",
    "reviewed_at",
    "provenance",
    "edit_count",
    "frames_path",
    "segments_path",
    "strip_path",
    "manifest_path",
)
# Index fields that stay null until `prepds export-index` runs with a catalog (PRD change C4, EC-27).
WORKBOOK_INDEX_FIELDS = ("strain", "age", "date", "agent_exposure_min")

# --- unreviewed prepds output (D-033): <PDS_OUTPUT_DIR>/{trials_catalog.parquet, processed/<video_id>/} ---
PROCESSED_DIR_NAME = "processed"
DETECTIONS_FILE = "detections.parquet"  # written beside the frames only by the model tracker (D-003, EC-29)
CATALOG_FILE = "trials_catalog.parquet"  # written by `prepds catalog`, one row per workbook trial
CATALOG_COLUMNS = (
    "subject_id",
    "sex",
    "strain",
    "age",
    "compound",
    "concentration_mM",
    "date",
    "agent_exposure_min",
    "video_path",
    "match_status",
)
MATCH_MATCHED = "matched"  # catalog rows with a video; others (e.g. "no_video") have no folder

# --- trial workbook (the input prepds catalogs; dcs reads only the NTT columns, D-004) -----------
WORKBOOK_SHEET = "Sheet1"
WB_DATE = "Date of EXP:"  # YYMMDD number
WB_SUBJECT = "Subject #:"
WB_STRAIN = "Strain:"
WB_SEX = "Sex (M/F):"
WB_AGE = "Age (~):"
WB_COMPOUND = "Compund:"  # sic: the real workbook's spelling
WB_DOSE = "Conc. (mM):"
WB_AGENT_EXPOSURE = "Agent Exposure Time (min):"
WB_NTT_TIME = "NTT \nTime (min):"  # real headers carry embedded newlines; readers strip them
WB_UV_TIME = "UV \nExposure\nTime (min)"
NTT_HEADERS = (
    "TDM (Full Arena):",
    "TDM (Top Half):",
    "TDM (Bot Half):",
    "Velocity (Full Arena)",
    "Velocity (Top Half)",
    "Velocity (Bot Half)",
    "Time Spent (Top):",
    "Time Spent (Bot):",
)
WB_OTHER_HEADERS = ("H2O (Before):", "H2O (After):", "Brain Tissue:", "Body Tissue:")
WORKBOOK_HEADERS = (
    WB_DATE,
    WB_SUBJECT,
    WB_STRAIN,
    WB_SEX,
    WB_AGE,
    WB_COMPOUND,
    WB_DOSE,
    WB_AGENT_EXPOSURE,
    WB_NTT_TIME,
    WB_UV_TIME,
    *NTT_HEADERS,
    *WB_OTHER_HEADERS,
)


def video_id(sex: str, subject_id: str) -> str:
    """Gold folder name and index key, e.g. `F_0042`."""
    return f"{sex}_{subject_id}"
