"""Generate the safe, deterministic illustration used by the frontend demo.

Run with the repository's Python environment (OpenCV + NumPy). No research data.
The trajectory matches position()/sampleFrame() in src/model.ts.
"""
import math
from pathlib import Path

import cv2
import numpy as np


def generate():
    output = Path(__file__).resolve().parents[1] / "public" / "demo-tank.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps = 640, 360, 30
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("A browser-compatible H.264 encoder is required to regenerate this asset.")
    yy, xx = np.mgrid[0:height, 0:width]
    glow = np.exp(-((xx - 320) ** 2 / 120000 + (yy - 140) ** 2 / 35000))
    base = np.zeros((height, width, 3), dtype=np.uint8)
    for c, value in enumerate([49, 53, 50]):
        base[:, :, c] = value + glow * 48
    cv2.rectangle(base, (30, 45), (610, 330), (130, 143, 139), 2)
    cv2.line(base, (32, 64), (608, 64), (165, 180, 168), 2)
    cv2.line(base, (32, 324), (608, 324), (118, 128, 127), 3)
    for x in range(42, 599, 40):
        cv2.line(base, (x, 324), (x + 20, 328), (97, 106, 105), 1)
    try:
        for i in range(fps * 12):
            t = i / fps
            frame = base.copy()
            x = 320 + 170 * math.sin(t * .5)
            y = 180 + 65 * math.sin(t * .9)
            angle = math.atan2(58.5 * math.cos(t * .9), 85 * math.cos(t * .5))

            def point(a, b):
                return (round(x + a * math.cos(angle) - b * math.sin(angle)),
                        round(y + a * math.sin(angle) + b * math.cos(angle)))

            tail = np.array([point(-17, 0), point(-32, -10 + 3 * math.sin(t * 12)), point(-30, 10 + 3 * math.sin(t * 12))], np.int32)
            cv2.fillPoly(frame, [tail], (123, 164, 177), cv2.LINE_AA)
            cv2.ellipse(frame, (round(x), round(y)), (21, 8), math.degrees(angle), 0, 360, (172, 205, 213), -1, cv2.LINE_AA)
            cv2.line(frame, point(-15, 0), point(14, 0), (80, 126, 143), 3, cv2.LINE_AA)
            cv2.circle(frame, point(14, -3), 2, (17, 30, 37), -1, cv2.LINE_AA)
            cv2.putText(frame, "SYNTHETIC / 001", (45, 31), cv2.FONT_HERSHEY_SIMPLEX, .42, (190, 206, 201), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    print(f"Generated {output.name}: 12 seconds, 640 x 360, H.264")


if __name__ == "__main__":
    generate()
