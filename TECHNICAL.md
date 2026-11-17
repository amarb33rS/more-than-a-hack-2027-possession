# Offline football possession from automatic tracking

More than A Hack 2027 - technical document - 17 November 2026

## 1. Approach and submitted assets

We predict home possession (H), away possession (A), or dead ball (D) for each in-play frame of the supplied BAR-ATH tactical recording. Frames outside the organizer's playing-period bounds receive X. The system is offline and uses future context within a period.

The submitted pipeline is:

`tracking -> geometry, masks and motion -> tree + three temporal networks`

`equal probability blend -> Viterbi per half -> frame,label CSV`

We use the supplied tactical automatic tracking TXT, its player/team metadata XML, and the organizer's public game configuration. Coordinates are converted from normalized XY to centered metres using the supplied pitch size. Missing positions retain explicit observation masks; they are never replaced by fabricated players, interpolated balls, or dead-ball labels.

The ATD record ID is converted to zero-based video index by subtracting one. Provided per-half configuration gives output intervals [11185, 90652) and [110808, 190198). Features and decoding do not cross halftime. The output has exactly 199,319 ordered frames, numbered 0 through 199318.

Smart Tagging, tactical/broadcast video and homography are not inputs to the submitted possession predictions. Video and homography were investigated separately, but recovery reliability and a possession benefit were not established. No video detector, PFF-adapted model, experimental history head, or noise-adapted model contributes to the CSV.

## 2. Labelled training data and provenance

The only labelled dataset used to train the shipped models is IDSSE / Sportec Open Tracking & Event Data: Bassek, Rein, Weber et al., *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12, 195 (2025), DOI 10.1038/s41597-025-04505-y. The public PySport mirror declares CC BY 4.0.

The underlying DFL data were collected by Sportec Solutions with TRACAB optical tracking. Supervision is the provider-supplied BallStatus and BallPossession, mapped programmatically: inactive status -> D; active status with home/away owner -> H/A; other combinations -> unknown and excluded from supervised loss. We did not create or hand-correct those labels. These provider definitions are useful control proxies, not a guarantee of exact agreement with the competition's annotation policy.

| Split | Fixtures | Amount and role |
|---|---|---|
| Training | J03WOH, J03WOY, J03WPY | 425,961 labelled period frames; 3 matches / 6 halves |
| Development | J03WN1, J03WQQ | Checkpoint, model and decoder selection; synthetic stress tests |
| Final external check | J03WMX, J03WR9 | One frozen configuration evaluated once; no subsequent tuning |

No labels from the provided game were used for training or validation. No private target ground truth, public-window tuning, human event feed, manual target tracking correction, or hand-authored possession labels enter the submitted system. Provider status/ownership is supervision only and never a model feature.

<!-- PAGEBREAK -->

## 3. Features, models and training

The tree is a histogram gradient-boosted classifier with 66 geometry/context features: ball visibility and gap age, position/speed, nearest-player distances by team, team position/count/speed, nearby-player counts, and one-second past/future context. It uses 100 boosting iterations, at most 15 leaves, L2 regularization 1, and 85,198 sampled clean/corrupted training rows. Its motion derivative lag is one frame.

Each neural member uses a shared pointwise player MLP, mean/max pooling by team, ball and explicit geometry features, and a residual temporal convolution stack. Dilations 1, 2, 4, 8, 16 and 32 give a 127-frame direct receptive field. Pooling is invariant to player-slot permutations. Neural motion features use five-frame differences with observation-continuity checks; missing intervals are not bridged by velocity estimates. Global gap ages/existence are computed for the complete period before chunking.

Neural training uses 512-frame windows, 100 windows per half per epoch, mixed clean and synthetically corrupted observations, and team-swap augmentation. Corruption includes missing-ball blocks and player dropout. AdamW uses learning rate 0.001 and weight decay 0.001. Loss combines state cross-entropy and 0.2 times auxiliary boundary BCE. The generic boundary head is not used by the final decoder: development evidence was stronger for stoppages than for home/away turnovers.

| Member | Selected checkpoint | Probability weight |
|---|---|---:|
| Tree | Original mixed-condition fit | 0.25 |
| Neural seed 42 | Epoch 70 | 0.25 |
| Neural seed 43 | Epoch 100 | 0.25 |
| Neural seed 44 | Epoch 90 | 0.25 |

Runs had 100-epoch ceilings with development checks and early-stopping support. Checkpoint selection used spell-level evidence. Different selected epochs mean different actual optimizer budgets. Included frozen weights reproduce the entry without retraining.

## 4. Possession policy and decoding

Class probabilities are blended before one Viterbi pass per half. Each frame contributes its log H/A/D probability; each state change incurs a fixed cost of 3. There is no hard minimum spell duration and no manual frame-specific ownership rule.

Loose balls and challenges are resolved from learned external control labels and temporal evidence. The nearest player does not automatically become the possessor, and a missing ball does not force D. Stoppages and restarts are predicted by the same state model, without event-tag overrides. These decisions inherit the provider-supervision limitations rather than claiming a manually encoded referee or control policy.

The ranking metric is spell F1, so stable frame classifications alone are insufficient: boundary errors can split or merge possessions. Lower switching costs, alternative boundary coupling and duration-weighted training did not pass bounded development comparisons. We retained the original static decoder.

<!-- PAGEBREAK -->

## 5. Validation and rejected alternatives

The selected ensemble's development spell F1 was 0.647848 on the mean of clean and original dropout conditions, 0.562012 with synthetic coordinate noise, and 0.594680 when unlabelled target ball-availability patterns were replayed on external labelled geometry. Replay uses missingness only, not target possession supervision, and does not recreate every event-dependent tracking failure.

After freezing the candidate, final external predictions were committed before reading their supervision. Only this frozen configuration was evaluated:

| Final external fixture | Spell F1 | Macro F1 | Predicted / reference spells |
|---|---:|---:|---:|
| J03WMX | 0.641648 | 0.938195 | 363 / 511 |
| J03WR9 | 0.730000 | 0.949849 | 310 / 370 |
| Equal-match mean | 0.685824 | 0.944022 | - |

These are external provider-labelled results, not competition scores or target accuracy. The final fixtures were not reused to select later alternatives. Repeated development comparisons and recurring teams limit independence.

Additional work was rejected rather than silently substituted into the entry:

- PFF 2022 World Cup source-adaptation pilots did not meet improvement/regression gates. Those provider-derived control proxies and adapted weights are excluded from the submitted models.
- Coherent observation augmentation, ensemble-aware loss, and explicit observation-history readouts failed their declared acceptance checks. The history readout reduced its stress primary from 0.568711 to 0.554444.
- Coordinate-noise consistency training improved its standalone recovery-noise primary from 0.438611 to 0.474681, but gained only 0.027278 over its matched control (required 0.03) and regressed dropout/live-dead safety checks. It was not promoted.
- External ball-detector fine-tuning used martinjolif/football-ball-detection, revision 1341bc97314029853764141e6f9fffca1fed8bd8: 72 training source frames, 24 development frames and a separately frozen 213-frame clip confirmation. These are other-game ball boxes, not possession labels. Match/pretraining independence is unknown. The detector and its COCO-pretrained YOLOv8n weights are excluded from the entry.
- SoccerTrack v2 source geometry/calibration was audited, with provider statuses quarantined rather than converted into training control labels. Its event-interpolated ball track was excluded as independent detector truth. SkillCorner partial home/away signals also remained isolated.

## 6. Limitations and next improvements

Approximately 41% of target in-play frames have no ball observation. Three training fixtures, all with Fortuna Duesseldorf at home, offer narrow team/division diversity. Short spells and transitions are under-recovered. Label semantics, missingness, identity errors and domain shift can reduce transfer; target accuracy remains unknown.

Ideal external ball restoration helped in a controlled synthetic study, but recovery with about 2.8 m radial RMS error was worse than preserving missingness. This is not a safe-error threshold or an actual video gain. Reliable aligned ball measurements, calibrated abstention, and uncertainty-aware modelling need new external recovery/possession evidence before fusion. Larger networks or more epochs alone do not resolve that evidence gap.

<!-- PAGEBREAK -->

## 7. Reproduction

Verified runtime: Windows x64, Python 3.12.14, CPU PyTorch 2.8.0. Other operating-system/runtime combinations are unverified. Dependencies are pinned in requirements.txt. All four trained artifacts, inference code, input hashes, a synthetic verification fixture and original third-party notices are included in the code repository. Raw competition assets are supplied separately by the organizer and must not be committed.

From the public-code repository root, create the environment and verify the release:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe verify_release.py
.venv\Scripts\python.exe -m unittest `
    modeling.test_pipeline modeling.test_predict_frozen_target -v
```

Obtain the organizer repository at revision 3de6b60aeabd2475d95879a651d38b72bc4fba00 and place the original competition assets in its data folder. This pins the exact game-configuration/input contract rather than relying on a moving branch.

```powershell
.venv\Scripts\python.exe reproduce.py `
    --assets-repo C:\PATH\more-than-a-hack-2027 `
    --out outputs\submission
```

The wrapper verifies source/model hashes and the three allowed inputs, regenerates the full CSV, validates its structure, and requires this exact SHA256:

`bc4320370b362aa195c0b99bd39fe1137abc42290e09a3c4293f1a7f1d61a1b1`

The expected header is frame,label with exactly 199,319 data rows. Renaming the file to the registered team name does not change its bytes. The original clean extraction and independently installed environment both reproduced the exact CSV. Final preparation repeats the full reproduction and validates the exact upload copy with the current organizer validator; verification evidence accompanies the local submission package.

Pinned dependencies: NumPy 2.3.5; pandas 3.0.1; SciPy 1.16.3; scikit-learn 1.7.2; torch 2.8.0; threadpoolctl 3.8.0. No GPU, paid service, detector weights or target label file is required to regenerate the entry.

## 8. Sources and attribution

- Organizer specification, code and MIT licence: https://github.com/metrica-sports/more-than-a-hack-2027
- IDSSE paper and provider-field definitions: https://doi.org/10.1038/s41597-025-04505-y
- IDSSE public mirror (CC BY 4.0): https://huggingface.co/datasets/pysport/idsse-data
- Dataset licence: https://creativecommons.org/licenses/by/4.0/
- Excluded ball-box research source: https://huggingface.co/datasets/martinjolif/football-ball-detection
- Excluded PFF research source: https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset
- Excluded SoccerTrack research source: https://atomscott.github.io/SoccerTrack-v2/

The selected candidate's canonical SHA256 is ef898c9ec6f380b9e80e7ddcf629bdae1d6a0695b910232695d8d8badd4fb5fb. Manifest path relocation is packaging only; numerical models, weights, features, class order, split and decoder are unchanged.
