"""Settings of the synthetic gold dataset (plan U2): design sizes, effect strengths and edge-case knobs.

dcs.synthetic builds the files from a SynthConfig. Every check runs when the
config is created, so an invalid config never gets as far as writing files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dcs import schema

DOSES = ("0.1", "0.3", "1", "3")
COMPOUND_LETTERS = "ABCDEFGHIJ"  # S and T are reserved for the knob groups in dcs.synthetic_design
DATES_PER_DOSE = 2  # every main compound+dose runs on this many dates
SINGLE_DATE_FISH = 6  # the default min_class_size: the class is kept and scheme A must pin it
TINY_CLASS_FISH = 5  # one below the default min_class_size: the class filter drops it
MAX_SUBJECTS = 9999  # video ids carry a 4-digit subject number
MIXED_FPS_FACTOR = 1.5  # frame rate of the mixed-fps fish relative to config.fps
MAX_FRAMES_PER_FISH = 1_000_000  # about 28 hours at 10 fps; guards against runaway configs

# Knob names (each is a SynthConfig field).
KNOB_MISSING_FILE = "missing_file"
KNOB_UNDETERMINED = "undetermined"
KNOB_NULL_DATE = "null_date"
KNOB_MISSING_FRAME_COLUMN = "missing_frame_column"
KNOB_WRONG_DTYPE_COLUMN = "wrong_dtype_column"
KNOB_MIXED_FPS = "mixed_fps"
KNOB_LOW_DETECTED_FISH = "low_detected_fish"
KNOB_RARE_STATE_FISH = "rare_state_fish"
KNOB_MESSY_LABELS = "messy_labels"
KNOB_MIXED_PROFILES = "mixed_profiles"
KNOB_SINGLE_DATE_COMPOUND = "single_date_compound"
KNOB_TINY_CLASS = "tiny_class"

# Knobs that name the fish they change (SynthResult.targets). Single-fish knobs never share a fish:
# the active head knobs take main-design fish 1, 2, ... in this order, the tail knobs the last main
# fish, and messy labels every other vehicle fish plus the combination group.
HEAD_KNOBS = (
    KNOB_MISSING_FILE,
    KNOB_UNDETERMINED,
    KNOB_NULL_DATE,
    KNOB_MISSING_FRAME_COLUMN,
    KNOB_WRONG_DTYPE_COLUMN,
    KNOB_MIXED_FPS,
)
TAIL_KNOBS = (KNOB_LOW_DETECTED_FISH, KNOB_RARE_STATE_FISH)
SINGLE_FISH_KNOBS = (*HEAD_KNOBS, *TAIL_KNOBS, KNOB_MESSY_LABELS)
# Group knobs change a whole date or add a group, so they may include single-knob fish.
GROUP_KNOBS = (KNOB_MIXED_PROFILES, KNOB_SINGLE_DATE_COMPOUND, KNOB_TINY_CLASS)
TARGETED_KNOBS = (*SINGLE_FISH_KNOBS, *GROUP_KNOBS)

MISSING_FRAMES = "frames"
MISSING_SEGMENTS = "segments"
MISSING_FILE_CHOICES = (MISSING_FRAMES, MISSING_SEGMENTS)


@dataclass(frozen=True)
class SynthConfig:
    seed: int = 0
    n_compounds: int = 3
    doses_per_compound: int = 2
    fish_per_dose_date: int = 3
    n_dates: int = 6
    vehicle_per_date: int = 2
    duration_s: float = 120.0
    fps: float = 10.0
    compound_effect: float = 1.0  # 0 = no compound signal
    date_effect: float = 0.5  # 0 = no date (batch) effect on speed
    manual_fraction: float = 0.3  # share of fish with a reviewer-edited window
    ntt_missing_fraction: float = 0.2
    framing_shift_px: float = 0.0  # depth offset added per date index (camera framing, EC-31)
    rare_state_fish: int = 0  # fish that show Listing/LORR (EC-23)
    low_detected_fish: int = 0  # fish with about half their frames undetected (EC-11)
    single_date_compound: bool = False  # EC-5
    tiny_class: bool = False  # EC-4
    constant_meander: bool = False  # EC-20
    messy_labels: bool = False  # EC-7, EC-8: compound case variants and a combination dose spelled two ways
    mixed_profiles: bool = False  # EC-21: the last date uses a model-tracker profile
    mixed_fps: bool = False  # EC-26
    null_date: bool = False  # EC-27
    undetermined: bool = False  # EC-22
    missing_file: str | None = None  # EC-1: "frames" or "segments"
    missing_frame_column: str | None = None  # EC-25
    wrong_dtype_column: str | None = None  # EC-25

    def __post_init__(self) -> None:
        problems = [message for failed, message in self._checks() if failed]
        if problems:
            raise ValueError("Invalid SynthConfig: " + "; ".join(problems))

    @property
    def main_fish(self) -> int:
        """Fish in the main design: every compound x dose x DATES_PER_DOSE dates x fish_per_dose_date."""
        return self.n_compounds * self.doses_per_compound * DATES_PER_DOSE * self.fish_per_dose_date

    @property
    def total_fish(self) -> int:
        extra = SINGLE_DATE_FISH * self.single_date_compound + TINY_CLASS_FISH * self.tiny_class
        extra += DATES_PER_DOSE * self.fish_per_dose_date * self.messy_labels  # the combination group
        return self.main_fish + self.vehicle_per_date * self.n_dates + extra

    @property
    def active_head_knobs(self) -> tuple[str, ...]:
        return tuple(knob for knob in HEAD_KNOBS if getattr(self, knob) not in (None, False))

    def _checks(self) -> list[tuple[bool, str]]:
        knob_fish = len(self.active_head_knobs) + self.rare_state_fish + self.low_detected_fish
        floats = (self.duration_s, self.fps, self.compound_effect, self.date_effect, self.framing_shift_px)
        finite = all(math.isfinite(value) for value in floats)
        return [
            (self.seed < 0, "seed must be >= 0"),
            (not 1 <= self.n_compounds <= len(COMPOUND_LETTERS), f"n_compounds must be 1-{len(COMPOUND_LETTERS)}"),
            (not 1 <= self.doses_per_compound <= len(DOSES), f"doses_per_compound must be 1-{len(DOSES)}"),
            (self.n_dates < max(2, 2 * self.doses_per_compound), "n_dates must be >= 2 x doses_per_compound"),
            (self.fish_per_dose_date < 1 or self.vehicle_per_date < 1, "group sizes must be >= 1"),
            (self.total_fish > MAX_SUBJECTS, f"at most {MAX_SUBJECTS} fish (4-digit subject numbers)"),
            (not finite, "duration_s, fps, effects and framing_shift_px must be finite numbers"),
            (finite and not (self.duration_s > 0 and self.fps > 0), "duration_s and fps must be > 0"),
            (finite and self.duration_s * self.fps * MIXED_FPS_FACTOR > MAX_FRAMES_PER_FISH, "too many frames per fish"),
            (not 0 <= self.manual_fraction <= 1 or not 0 <= self.ntt_missing_fraction <= 1, "fractions must be 0-1"),
            (min(self.rare_state_fish, self.low_detected_fish) < 0, "fish counts must be >= 0"),
            (self.main_fish < knob_fish, "not enough main-design fish for the knobs that are on"),
            (self.missing_file not in (None, *MISSING_FILE_CHOICES), f"missing_file must be one of {MISSING_FILE_CHOICES}"),
            (self.missing_frame_column not in (None, *schema.FRAME_DTYPES), "missing_frame_column is not a frames column"),
            (self.wrong_dtype_column not in (None, *schema.FRAME_DTYPES), "wrong_dtype_column is not a frames column"),
        ]
