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
- [x] Live-streaming target: detect fish with all the current features (tracks, states, metrics). Rolling-window
      estimates for anything needing a window. Exact spec not fixed yet; assumed to match the recorded-video output.
- [x] Camera: exact resolution and frame rate unknown. Likely a **two-camera rig**: top-down plus a 90-degree side view.
      Open items: resolution/fps, how the two views are synced, and whether the detector and states need retraining
      on a new viewpoint (the current ones are trained on one view; side view also affects the LORR limit).
- [x] Live output: the same outputs as the offline run, produced as actions are detected: live diagrams plus the
      reasoning, with every result written automatically to `temp/output` as CSV so the chatbot can read it and answer
      questions. At the end the user chooses to **save or discard** the session (kept sessions can be used for
      reference or later training).
- [x] Hosting: the model is hosted in Docker, everything local (`allow_remote` stays false).

Implied tasks: a `temp/output` session writer (CSV, append as it runs); a chat tool that reads the current session
CSV; a save/discard step that moves a kept session out of `temp/`; a Docker service for the model server.

### Step 1: GB10 environment (blocks everything)
- [ ] Run the GB10 runbook in `docs/classifier_progress.md` (machine, GPU, CUDA, Python, torch table).
- [ ] Install an aarch64 CUDA build of torch and torchvision. This is the likeliest snag; check `torch.cuda.is_available()`.
- [ ] Install the `ml` extra, then run the test suite on the GB10.
- [ ] Time one video through the detector on the GB10 and compare with 75 s on the 4070 Ti.
- [ ] Repeat the check on the RTX workstation so the fallback is known to work.

#### How to do Step 1 (your instructions)

The base install is already written: "Training on the GB10" in `docs/instructions.md`, part 1. Follow it, then add the
extras below. Do all of it over SSH inside `tmux`. Paste the outputs into `docs/classifier_progress.md` section 6
(hardware record) when done. Real names, dates and results stay out of git.

1. **Machine facts** (record them): `uname -m` (expect aarch64), `nvidia-smi`, `nvcc --version` (or the CUDA version
   `nvidia-smi` prints), `python3 --version` (3.11 or newer), `free -h`, `df -h .`, and `docker --version` plus
   `docker info | grep -i runtime` (Step 2 hosts the model in Docker, so the nvidia runtime must be listed).
2. **Clone and venv:** `git clone`, `git checkout claude/pensive-planck-wnkmn3` (or master once merged), `python3 -m venv .venv`,
   `source .venv/bin/activate`.
3. **torch first:** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`. If the wheel is
   not found, try `cu128`, or use NVIDIA's aarch64 PyTorch container. Check:
   `python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"`
   must print `True`. If it prints `False`, stop and write down the exact output; do not continue.
4. **Project:** `pip install -e ".[dev,ml]"` (the detector needs `ml`, which also pulls `transformers`; the `dcs`
   runbook uses `train`). Then `pytest tests -q`. Skips on missing GPU are fine; failures are not.
5. **Time the detector** on one video, using the same command and stride as the 4070 Ti figure (about 75 s):
   `python -m prepds run --tracker model:<run dir> --stride 5 --device cuda --limit 1` with the m3 profile
   (`--config cal-2026-09-24-r4-model-m3.yaml`) and its own `PDS_OUTPUT_DIR`. Write down wall time and frames/s. Run it
   twice; the first run includes warm-up.
6. **Fallback:** repeat steps 3 to 5 on the RTX workstation (or the 3060 Ti PC) so we know it also works.
7. **Report back** with: the step 1 facts, the `torch.cuda.is_available()` line, the pytest summary line, both timings,
   and any error text. That is enough to start Step 2 (chat model in Docker) next session.

If something fails, do not work around it silently. Paste the exact error. The usual snags are: no aarch64 CUDA wheel
(use the NVIDIA container), a CUDA version mismatch between torch and the driver, and Docker without GPU access.

### Step 2: Part A, chat model (no training)

Prepared without the GB10 (needs only the table and a chat server). The scorer is `python -m dcs.chat_eval`; the 27
questions are in `config/chat_eval_questions.yaml` (placeholders, so no real names are committed).

Prepared now:
- [x] Test set: 27 questions covering every tool, unknown compound/measure/fish, and 4 out-of-scope or rule-breaking
      prompts that must get no tool call and no numbers.
- [x] Expected tool and arguments per question; the right number is computed by calling the tool directly, not by the model.
- [x] Scorer: right tool, right arguments, right number in the answer, invented numbers, latency (test: `tests/dcs/test_dcs_chat_eval.py`).

When the GB10 is available:
- [ ] Install a model server on the GB10 (Ollama, llama.cpp or vLLM), loopback only; Docker per Step 0.
- [ ] Pull 2-3 candidates (about 7-30B, instruct, tool calling). Starting list, **verify each licence and its tool-calling
      support on the model card before use**: Qwen3 30B-A3B or 32B, gpt-oss-20b, Mistral Small 24B. A 70B (Llama 3.3)
      fits in 128 GB memory but is slower; try it only if the small ones fail.
- [ ] Set `DCS_CHAT_BASE_URL` in `.env` and `chat.model` in the override file; `allow_remote` stays false.
- [ ] Run `python -m dcs.chat_eval <model1> <model2> --show-answers`, once with a training run present (so the 4 run
      questions are included) and read the failed answers yourself.
- [ ] Pick the model: most right tools and numbers, zero invented numbers, no refusal failures, acceptable latency.
      Keep it untrained (D-071, W9); revisit only if tool use is poor.

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
