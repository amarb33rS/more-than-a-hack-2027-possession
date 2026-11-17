# Football possession: frozen tracking ensemble

This repository regenerates the entry for More than A Hack 2027: one H/A/D/X
label for each of the 199,319 frames of the provided BAR-ATH tactical recording.
The selected system blends a tree and three temporal neural networks equally,
then applies Viterbi decoding with switch cost 3. It uses automatic tracking;
video-detector and later adaptation experiments are excluded.

## Runtime and installation

Verified platform: Windows x64, Python 3.12.14, CPU PyTorch 2.8.0. No GPU is
required. Other operating-system/runtime combinations are unverified. Use the
included trained checkpoints for exact reproduction; retraining is optional
and is not expected to reproduce checkpoint bytes.

From this repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe verify_release.py
.venv\Scripts\python.exe -m unittest `
    modeling.test_pipeline modeling.test_predict_frozen_target -v
```

`.gitattributes` preserves exact bytes during Git checkout, including on Windows.

`FILES.json` pins the published source, documentation, models and synthetic
fixtures. `verify_release.py` checks these hashes before loading the included
trusted model artifacts, then checks bit-exact inference on synthetic geometry.

## Original inputs

The competition assets are not redistributable and are not included. Obtain
them from the organizer. Clone the organizer's public code separately and use
revision `3de6b60aeabd2475d95879a651d38b72bc4fba00`:

```powershell
git -c core.autocrlf=false clone https://github.com/metrica-sports/more-than-a-hack-2027 C:\PATH\more-than-a-hack-2027
git -C C:\PATH\more-than-a-hack-2027 -c core.autocrlf=false checkout 3de6b60aeabd2475d95879a651d38b72bc4fba00
```

Unpack the supplied assets into that organizer repository's `data` folder.
The entry reads only `games.py`, `BAR-ATH_tactical_atd.txt` and
`BAR-ATH_tactical_atd.xml`. `input-contract.json` pins their content hashes.
Videos, Smart Tagging and target possession annotations are not read.

## Regenerate the exact CSV

```powershell
.venv\Scripts\python.exe reproduce.py `
    --assets-repo C:\PATH\more-than-a-hack-2027 `
    --out outputs\submission
```

The command verifies the release and input hashes, predicts both complete
periods independently, checks CSV structure, and requires this exact SHA256 for
`outputs/submission/possession.csv`:

```text
bc4320370b362aa195c0b99bd39fe1137abc42290e09a3c4293f1a7f1d61a1b1
```

The header is `frame,label`; there are 199,319 rows, frames 0 through 199318,
and labels H/A/D/X. Renaming the resulting file to the registered team name
does not change the hash. An organizer format check is:

```powershell
.venv\Scripts\python.exe C:\PATH\more-than-a-hack-2027\evaluation\validate_submission.py `
    outputs\submission\possession.csv
```

This is validation, not scoring. No target ground truth is required. The
provided public target window was not used to train or tune this entry.

## Method, data and evidence

- `TECHNICAL.md`: submitted approach, training data and label provenance,
  possession policy, selection, rejected alternatives, limitations and execution.
- `MODEL-CARD.md`: canonical model/configuration metadata.
- `models/`: the four original hash-pinned trained artifacts and external
  development companion reports. Reports describe other, openly licensed
  fixtures; they do not contain target annotations or tracking/video assets.
- `candidate/`: relocated manifest and freeze record. Weights, class order,
  features, split and switch cost remain unchanged.
- `modeling/`: inference plus training/adapter reference source.
- `verification/`: synthetic geometry and expected outputs only.
- `NOTICE.md`, `LICENSE-METRICA`, `modeling/OFFICIAL-METRICS-LICENSE`:
  source/data attribution and preserved third-party terms.

The only labelled training data for the shipped models is IDSSE / Sportec:
425,961 provider-labelled period frames from three other matches, declared
CC BY 4.0. The split and exact mapping are documented in `TECHNICAL.md`.
The one frozen external final evaluation scored mean spell F1 0.685824 and
macro F1 0.944022. These are not competition scores or target accuracy.

## Publication boundary

Only this code tree belongs in the public repository. Keep the organizer data,
local submission staging/checks, generated outputs and environments outside it.
The included `.gitignore` excludes data and generated outputs. Submit the
separate team-named CSV and PDF/Markdown through the organizer's emailed form.
