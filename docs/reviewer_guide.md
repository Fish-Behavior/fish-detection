# Reviewer guide: checking zebrafish videos in FishLab

For: pharmacy / lab staff who check the computer's behavior labels. No programming needed.

## 1. Why we need you

The software watches each recording, follows the fish and writes down what the fish is doing second by second
(swimming, frozen, darting, and so on). It makes mistakes. Some behaviors it cannot recognise at all.

Your job is to watch the video next to the colored timeline, fix what is wrong, and **accept** the video when you
agree with it. Accepted videos become the trusted data the drug-prediction model learns from. Videos nobody has
accepted are not trusted yet. So **your judgment is the ground truth**: when the computer and you disagree, you win.

You are not testing the software. A video you reject, or a mistake you find, helps us.

## 2. What the colors mean

Each second of the video gets one behavior. The timeline shows them as colors.

| Behavior | What to look for | Who finds it |
|---|---|---|
| **Controlled Swim** (blue) | Steady, purposeful swimming at a moderate pace | Computer, from speed |
| **Erratic Movement** (orange) | Fast darting, sudden turns, jerky swimming | Computer, from speed |
| **Freezing/Drift** (green) | Almost still, or drifting slowly | Computer, from speed |
| **Listing/LORR** (purple) | The fish is tilted, on its side or upside down (loss of the righting reflex) | **You only** |
| **Surface Breach** (pink) | The fish is at or through the water surface | **You only** |
| **Dead** (dark gray) | Motionless from some point until the end, no gill or body movement | **You only** |
| **Undetermined** (light gray) | The computer lost the fish and could not decide | **You must fix these** |

The computer decides the first three only from how fast the fish moves. It never decides Listing, Surface Breach or
Dead, so those appear only if you add them. These descriptions are short and written by the software team. **If your
lab defines a behavior differently, follow your lab's definition and tell us.**

Things that confuse the computer, so watch for them:
- **Reflection.** The fish's reflection at the water surface or glass can look like a second fish. The tracker may follow the reflection.
- **A motionless fish** can be "lost" by the tracker and become Undetermined or wrongly Freezing.
- **Fish at the surface or against the wall** are hard to see. If you cannot tell what the fish is doing, say so (section 6).

## 3. Opening the app

1. The team will give you a web address (it looks like `http://…:port`) and a list of video names to review (for example `F_0079`).
2. Open the address in a normal browser. The app has no password and only works on the lab computer or a connection the team set up. Do not share the address.
3. On the left you see **Dashboard** and **Review**, and a list of videos (**Sessions**). Click a video name to open it.
4. In the **Reviewer** box type your own name. You need it to save anything. It is recorded with every change.

The **Filter sessions** box above the list can show **Needs review** or **With corrections**.

## 4. Checking one video

1. Open the video from the list and go to the **Review** tab.
2. Press play and watch the fish with the colored timeline below. Click on the timeline to jump to a moment.
3. Look at **Prepds review flags** (a box on the page). These are places the computer is unsure of, for example "no movement until the end of the video". Click one to jump to it.
4. **Possible Listing/LORR** shows advisory hints. These are only guesses and often wrong. Check each one yourself.
5. **Why this label?** explains how the computer chose the label for the second you are on. Use it if a label surprises you.
6. Go through the whole video. For each stretch of the timeline ask: *does the color match what I see the fish doing?*

### Fixing a wrong stretch

1. Under **Relabel a time range**, set the start and end time of the stretch (or click a flag or segment to fill it in).
2. Choose the correct behavior from the list.
3. Click **Stage range**. Repeat for every wrong stretch.
4. Click **Save staged edits**. (**Discard staged edits** throws away what you have not saved. **Restore labels** returns to the computer's labels.)

Nothing is final until you save, and you cannot accept while staged edits are unsaved.

You do **not** need the tracking tools (waterline, ROI, tracking point, detector box, keypoints). They are for the
software team. If you touch them by accident, use **Restore** or tell us.

## 5. Accepting or rejecting

At the bottom you will find **Review decision** with two buttons:

- **Accept review**: you checked the whole video and agree with it as it now stands.
- **Reject review**: the video is not usable (for example, the fish is not visible, or the recording is broken). The team will look at it.

Rules the app enforces:
- **You cannot accept while any Undetermined remains.** Change each Undetermined stretch to a real behavior, or reject the video. There is no way around this on purpose.
- **Dead is a serious claim.** The app asks you to confirm. A fish that is still at the end of the video is not always dead (it can be a long Listing/LORR). Only choose Dead if you are sure.
- **Accepting is final.** An accepted video cannot be edited again. If you made a mistake after accepting, tell the team and do not try to fix it yourself.

Accept means "the labels are right". It does **not** mean the software, tracking or the model is validated.

## 6. When you are not sure

Do not guess. A wrong accepted video is worse than a missing one, because the model will learn from it.

- **Cannot tell what the fish is doing in a stretch:** do not accept the video. Reject it, and write the video name, the time and what you saw in the shared list the team gave you.
- **Disagree with a definition, or the colors look systematically wrong for a drug:** write it down. This is useful, not a mistake.
- **The page shows an error, freezes or says "Could not connect to the API":** try **Reload from backend**. If it continues, tell the team and stop.

## 7. Checklist before pressing Accept

- [ ] My name is in the Reviewer box.
- [ ] I watched the whole video, not just the flagged parts.
- [ ] Every Undetermined stretch is fixed.
- [ ] The timeline colors match what the fish does.
- [ ] Any Listing/LORR, Surface Breach or Dead I saw is on the timeline.
- [ ] I saved my edits (nothing staged).

## 8. What happens after you accept

Accepted videos are copied into the `accepted` folder with an index. The classifier (`dcs`) reads only accepted videos.
Once enough videos are accepted across compounds, the team re-runs the training, so your work directly decides how
reliable the predictions become. You do not run any of that.

## Notes for the team (not for reviewers)

- Written from the current review app source and [instructions.md](instructions.md). It has **not** been tested with a real reviewer, and the screenshots and exact button positions were not checked in a browser. Walk through it once yourself and correct anything that differs.
- Accepting happens in the prepds review app only. `dcs` has no accept step.
- Decide before handing over: the list of videos per reviewer (the sample should cover compounds and dates, see the open review task T085 in [progress.md](../backend/prepds/docs/progress.md)), whether two reviewers check the same videos to measure agreement, and how the reviewer reaches the app (the app only binds to a loopback address, so it needs a browser on the machine or an SSH tunnel).
- Confirm with the lab the written definitions of each behavior in section 2.
