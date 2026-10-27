# Classifying Illustrated Maps

DATASCI 281, Fall 2026 - Cheyenne Tsai, Cierra Jagan, Leona Yung, Siddharthen Sridhar

We classify the geographic scale of maps scanned from fiction published 1800-1928:

| Class | Label | Examples | Count |
|---|---|---|---|
| 0 | Small | estates, buildings, floor plans | 210 (8%) |
| 1 | Medium | cities, towns | 553 (21%) |
| 2 | Large | regions, countries, continents | 1,859 (71%) |

Dataset: "Castles, Battlefields, and Continents" (Bax, Mimno, and Wilkens, 2025) - [data on eCommons](https://ecommons.cornell.edu/items/6d470f13-bac5-4b0e-94f2-4eac93d677cb), [paper](https://doi.org/10.63744/oYbvYsUA743D)

## Setup

```bash
pip install -r requirements.txt
python scripts/download_dataset.py
```

This downloads about 1.2 GB into `data/ecommons/` (about 2.5 GB on disk, since the zips are kept). It works from any directory. `data/` is in `.gitignore` - never commit it; GitHub rejects files over 100 MB.

| Flag | Effect |
|---|---|
| `--metadata-only` | CSVs only, no images (about 14 MB) |
| `--force` | Re-download files that already exist |
| `--with-models` | Also get `EfficientNet.zip` (1.7 GB) - the paper's map-vs-not-map models, not ImageNet weights. Usually skip. |

If an unzip gets interrupted, delete the half-extracted folder and re-run; otherwise the script skips it. Don't delete the zips to save space - the script checks for them and will download again.

## Data layout

```
data/ecommons/
  final_maps/Bax_LiteratureMaps_final_maps/final_maps/*.jpg    2,622 map images
  Gemini/Bax_LiteratureMaps_Gemini/Gemini_fict-scale_full.csv  labels ("Scale" column: 0/1/2)
  MARC/Bax_LiteratureMaps_MARC/MARC_info_parallel.csv          book metadata (title, author, year)
  CLIP/, non-map_sample/                                       other outputs from the paper
```

Loading the labels:

```python
from pathlib import Path
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".git").exists())  # repo root
DATA = ROOT / "data" / "ecommons"
IMG_ROOT = DATA / "final_maps" / "Bax_LiteratureMaps_final_maps"

labels = pd.read_csv(DATA / "Gemini" / "Bax_LiteratureMaps_Gemini" / "Gemini_fict-scale_full.csv")

# The dataset authors say to exclude these two (cartoons that contain a map)
EXCLUDE = {"32000002642652_00000033.jpg", "39015030849908_00000219.jpg"}
labels = labels[~labels["Image"].str.split("/").str[-1].isin(EXCLUDE)]

# "Image" already starts with "final_maps/", so join it to IMG_ROOT, not the inner folder
labels["path"] = labels["Image"].map(lambda p: IMG_ROOT / p)
```

## Notes

- **Split by work, not by image.** Many novels were scanned more than once (Treasure Island has 23 copies), so the same map can otherwise land in both train and test. The book ID is the filename prefix before the first `_`.
- Counts in the table above include the two excluded images.

## Repo contents

- `scripts/download_dataset.py` - downloads and unzips the dataset
- `scripts/Cierra_EDA.ipynb` - image sizes, channels, class balance
- `scripts/Cierra_Duplicate_Check.ipynb` - finds works scanned multiple times, using the MARC metadata
