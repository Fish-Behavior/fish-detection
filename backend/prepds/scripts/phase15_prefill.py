"""Phase 15: pre-label the sampled frames with an open-vocabulary detector (OWLv2, Apache-2.0) so annotators
correct a box instead of drawing one. Writes outputs/phase15/prefill.json (image -> best box + score, or null).
Needs the [ml] extra and a CUDA torch build. The boxes are suggestions only: they sometimes include the
fish's reflection, and are never used as ground truth without a human check.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

OUT = Path("outputs/phase15")
MODEL = "google/owlv2-base-patch16-ensemble"
PROMPTS = [["a fish", "a zebrafish"]]
MIN_SCORE = 0.05


def main() -> None:
    samples = json.loads((OUT / "sample.json").read_text())["samples"]
    processor = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL).cuda().eval().half()
    result: dict[str, dict | None] = {}
    for sample in samples:
        image = Image.open(OUT / "frames" / sample["image"]).convert("RGB")
        side = max(image.size)  # OWLv2 pads to a square at the bottom/right
        inputs = processor(text=PROMPTS, images=image, return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            outputs = model(**inputs)
        found = processor.post_process_grounded_object_detection(
            outputs, threshold=MIN_SCORE, target_sizes=torch.tensor([[side, side]]).cuda(), text_labels=PROMPTS
        )[0]
        if len(found["scores"]) == 0:
            result[sample["image"]] = None
            continue
        best = int(found["scores"].argmax())
        x0, y0, x1, y1 = (float(v) for v in found["boxes"][best])
        width, height = image.size
        result[sample["image"]] = {
            "score": float(found["scores"][best]),
            "box": [max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)],
        }
    with open(OUT / "prefill.json", "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
    found_n = sum(v is not None for v in result.values())
    print(f"prefilled {found_n}/{len(result)} frames with a box")


if __name__ == "__main__":
    main()
