# Lightweight model plan: chat, vision and tabular, near-real-time

Planning checklist. It builds on PRD Part II and III.2. Nothing here changes results; it says what to prepare first
and in what order.

**Compute:** primary is the Dell GB10 (aarch64, 128 GB unified memory, over SSH). Fallback is the RTX workstation
(RTX 4070 Ti, 12 GB, or the home PC with an RTX 3060 Ti).

## The three parts and how they fit

| Part | What it is | Trained here? | Data it needs |
|---|---|---|---|
| A. Chat reasoning | Pre-trained local language model that picks query tools and words the answer (`dcs ask`) | No | None for training; a test question set |
| B. Vision | Fish detector (Keypoint R-CNN, run `m3`), later a clip-level behavior model | Yes (fine-tune) | Labeled frames, reviewed state labels |
| C. Tabular | Compound/dose classifier on features from tracks (`dcs`) | Yes (small) | Reviewed, balanced trials |

Data flow: **video -> B (tracks and states) -> features -> C (compound/dose) -> A (explains results in English)**.
The compound classifier reads features, not pixels, so "vision plus tabular" means B feeds C. It is not one merged network.

## Near-real-time: what is realistic

- The detector runs about 485 frames/s at stride 5 on the 4070 Ti (a 20-minute video in about 75 s), so live tracking is
  feasible. GB10 speed is untested.
- The compound classifier needs behavior summarized over a window (distance, velocity, freezing share), not one frame.
  Live use therefore means a **rolling window** (for example the last 60 s, refreshed every few seconds) with the
  prediction labeled "provisional until the trial ends". It does not mean a per-frame answer.
- The live camera is "Not started" in the README status (Phase 2). Do not promise real-time before items B1-B4 and
  C1-C3 below are done.

## Order of work

### Step 0: decisions from you (before anything else)
- [ ] Confirm the live-streaming target: a rolling-window estimate, as above, or something else.
- [ ] Confirm the camera: resolution, frame rate, same rig as the recorded videos or a new one.
- [ ] Confirm what the live output must show (state per second, tracks, compound guess with confidence).
- [ ] Confirm the data agreement allows a hosted model, or keep everything local (the current setting).

### Step 1: GB10 environment (blocks everything)
- [ ] Run the GB10 runbook in `docs/classifier_progress.md` (machine, GPU, CUDA, Python, torch table).
- [ ] Install an aarch64 CUDA build of torch and torchvision. This is the likeliest snag; check `torch.cuda.is_available()`.
- [ ] Install the `ml` extra, then run the test suite on the GB10.
- [ ] Time one video through the detector on the GB10 and compare with 75 s on the 4070 Ti.
- [ ] Repeat the check on the RTX workstation so the fallback is known to work.

### Step 2: Part A, chat model (no training)
- [ ] Install a model server on the GB10 (Ollama, llama.cpp or vLLM).
- [ ] Shortlist 2-3 open-weights instruct models that support tool calling (about 7-30B). Check licences.
- [ ] Set `chat.model` and `chat.base_url`; keep `allow_remote` false.
- [ ] Write a test set of 20-30 questions: ask about compounds, comparisons to vehicle, model results, one fish's timeline,
      and some out-of-scope questions it should refuse.
- [ ] For each, record the expected tool and the right number (from the tools directly, not from the model).
- [ ] Score each candidate: right tool, right arguments, no invented numbers, latency.
- [ ] Pick the model. Keep it untrained (D-071, W9); revisit only if tool use is poor.

### Step 3: Part B, vision (needs your labeling time)
- [ ] B1. Human-review a sample of the `outputs_r3` run (T085 is not done). Nothing downstream is trustworthy until this happens.
- [ ] B2. Label more frames: aim for several hundred more, including no-fish and reflection frames and the weak groups.
- [ ] B3. Freeze a held-out frame set that no tuning touches.
- [ ] B4. Retrain the detector on the GB10 (write-once run dir); report recall@IoU 0.5, keypoint error, false positives.
- [ ] B5. Later: a clip-level behavior model, only once enough seconds are Accepted. Listing/LORR needs reviewer labels first.
- [ ] B6. Check the live path: a single inference process fed by a frame reader, with a dropped-frame policy.

### Step 4: Part C, tabular classifier
- [ ] C1. Accept reviewed videos so the training table uses reviewed labels (none are Accepted yet).
- [ ] C2. Fix the confound: recording date predicts compound better than behavior. Record more trials per compound
      spread across dates and cameras (W2, W5).
- [ ] C3. Re-run `dcs train` with date-held-out validation on the GB10. Gate: beat the date-only baseline, otherwise
      the live view shows vehicle-vs-drug only.
- [ ] C4. Add rolling-window features: compute the same features on partial windows and check that the classifier is
      stable on them, not only on full trials.

### Step 5: join and stream (Phase 2)
- [ ] Live frame source -> detector -> states -> rolling features -> classifier -> chat/status.
- [ ] Latency target (propose after Step 1 timings), a fallback when the detector lags, and logging of every
      prediction with its window.
- [ ] Label every live prediction "estimate, unvalidated" until the owner and pharmacy sign off (PRD G-gates).

## Risks to keep in view
1. Today no model names the compound better than the recording date. Real-time cannot fix a data problem.
2. Listing/LORR and Dead are not detected automatically, so a live view will miss them.
3. The detector flickers or swaps to the reflection on some videos; speed features depend on stable tracks.
4. GB10 is aarch64, so wheels and speed are untested.
5. The chat API has no authentication (loopback only); a live page must not expose it.

## What to prepare first (short list)
1. Step 0 answers. 2. GB10 environment and timings. 3. Review of the `outputs_r3` sample. 4. Chat test question set.
5. More labeled frames and trials across dates.
