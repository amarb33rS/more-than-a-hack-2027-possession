# Source, data and artifact provenance

The organizer's metric implementation is included under its original MIT
notice in LICENSE-METRICA and modeling/OFFICIAL-METRICS-LICENSE. Organizer
code licensing does not grant redistribution rights for competition data.
No competition tracking/video assets or target annotations are included.

Training source for the shipped models: IDSSE / Sportec, Bassek et al. (2025),
An integrated dataset of spatiotemporal and event data in elite soccer,
Scientific Data 12, 195. https://doi.org/10.1038/s41597-025-04505-y
Public mirror: https://huggingface.co/datasets/pysport/idsse-data
Declared licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
Credit the authors and DFL/Sportec Solutions. We mapped provider status/owner
fields programmatically to H/A/D, introduced synthetic observation corruption,
and trained the included models. Raw external data is not included; companion
reports contain development metrics on those other fixtures.

PyTorch, NumPy, pandas, SciPy, scikit-learn and threadpoolctl are independently
installed dependencies governed by their upstream terms. Their libraries are
not vendored. No YOLO/video-detector model or its dependencies are shipped.
Project-specific source and weights are provided for inspection/reproduction
of this competition entry. Consult the repository owner for additional reuse
permissions beyond the competition; third-party terms remain applicable.
