# Classifying Illustrated Maps

DATASCI 281, Fall 2026 - Cheyenne Tsai, Cierra Jagan, Leona Yung, Siddharthen Sridhar

We classify the geographic scale of maps scanned from fiction published 1800-1928:

| Class | Label | Examples | Images | Distinct maps |
|---|---|---|---|---|
| 0 | Small | estates, buildings, floor plans | 210 (8%) | 172 (9%) |
| 1 | Medium | cities, towns | 553 (21%) | 359 (18%) |
| 2 | Large | regions, countries, continents | 1,859 (71%) | 1,435 (73%) |

"Images" counts every scan in the dataset, including the two the authors say to exclude. "Distinct maps" is what's left after merging duplicate scans of the same map (see [Preprocessing](#1-preprocessingipynb)).

Dataset: "Castles, Battlefields, and Continents" (Bax, Mimno, and Wilkens, 2025) - [data on eCommons](https://ecommons.cornell.edu/items/6d470f13-bac5-4b0e-94f2-4eac93d677cb), [paper](https://doi.org/10.63744/oYbvYsUA743D)

## Setup

```bash
git clone git@github.com:sidsridhar0/281project.git
cd 281project
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

## Pipeline

Run the notebooks in `scripts/` in number order. All of them find the repo root on their own, so they work from any machine.

| Notebook | What it does |
|---|---|
| `0-Cierra_EDA.ipynb` | Image sizes, channels, class balance (optional) |
| `0-Cierra_Duplicate_Check.ipynb` | Finds works scanned more than once, from the MARC metadata (optional) |
| `1-preprocessing.ipynb` | De-duplicates, splits, pads/resizes/denoises -> `data/processed/` and `manifest.csv` |
| `2-augment.ipynb` | Oversamples the minority classes in train |
| `feature_checkpoint.ipynb` | Proof-of-concept features on three sample maps (see [Features](#features)) |

Re-running `1-preprocessing.ipynb` deletes and rebuilds `data/processed/`, including the augmentation, so run `2-augment.ipynb` again afterwards. Preprocessing takes a few minutes, mostly in the denoising step.

### 1-preprocessing.ipynb

1. **Labels:** loads the Gemini `Scale` labels and drops the two maps the authors exclude (2,620 images left).
2. **Works:** normalizes MARC title + author into a work key, so different libraries' scans of the same novel match. 1,555 scanned volumes collapse to 1,024 works.
3. **Duplicates:** finds scans of the same printed map with a perceptual hash (pHash). 276 maps have 2+ scans (930 images; one map has 47). Each map keeps its largest scan as its "kept scan".
   - **One label per map.** Scale is a property of the printed map, not of a scan, so every row of a map gets the same label: the majority vote of Gemini's labels across all its scans. That covers its kept scan, its rescans, and its augmented copies. Ties go to the kept scan's own Gemini label.
   - Gemini disagreed with itself on 91 of the 276 maps. Its original per-scan labels stay in `duplicates.csv` as `label_original`, with `label_conflict` marking those maps.
   - The vote is only as good as the merge: if two different maps were wrongly merged, one gets the other's label. Check merges with the review tools in section 4.
4. **Split:** a group is everything linked by "same work" or "same map". Groups are split 70 / 15 / 15, so a group never spans two splits, and the class mix stays about equal across splits.
5. **Images:** pads each image (no stretching) to the train median aspect ratio, resizes to 673 x 1024, applies light denoising, and saves a PNG.
6. **Rescans:** other scans of train maps come back as extra training rows ("natural augmentation"): up to 8 per map for Small and Medium, none for Large (`MAX_RESCANS_PER_MAP`). The real variation goes where the data is thin instead of widening the imbalance. Rescans carry no augmentation artifacts, so giving them to only some classes doesn't create a shortcut. Val and test keep exactly one scan per map.

| Maps per split | Small | Medium | Large | Total |
|---|---|---|---|---|
| train | 120 | 251 | 1,005 | 1,376 |
| val | 26 | 54 | 215 | 295 |
| test | 26 | 54 | 215 | 295 |

Settings live in the second cell: split fractions, `PHASH_MAX_DIST` (duplicate threshold), `RESCANS_AS_AUGMENTATION`, `MAX_RESCANS_PER_MAP`, `TARGET_HEIGHT`, `ROTATE_LANDSCAPE`, `DENOISE_STRENGTH`. After changing `PHASH_MAX_DIST`, check the merges in section 4: there's a sheet of borderline pairs, and a pager through every merged map, most suspicious first.

### 2-augment.ipynb

- **How many copies:** tops each class up toward the size of the largest one, counting rescans and the largest class's own minimum copies. Copies go first to the maps with the fewest samples, and synthetic copies never take a map past 5 samples. A map with more rescans than that keeps them but gets no synthetic copies.
- **Large gets copies too:** every class, Large included, gets copies for at least 25% of its maps. Otherwise augmentation artifacts would only appear in Small and Medium and become a shortcut for predicting them.
- **Transforms:** all scale-preserving. Flips, a tilt of up to 5 deg, zoom of up to 7%, shift of up to 4%, contrast and brightness. No big zooms, since how zoomed-in a map looks is what we're classifying.
- **Safe to re-run:** the previous augmentation is replaced, never added to.

Section 7 prints the final train composition by class and variant. Expect Medium and Large to end up about even (around 1,256 rows each: Large is its 1,005 maps plus 251 synthetic copies). Small stays near 600, because it has only 120 train maps and few of them were reprinted. Use `class_weight="balanced"` in the classifiers to cover the rest.

### Features

`feature_checkpoint.ipynb` is a proof of concept on three sample maps:
- **HOG:** axis alignment - walls are 0 or 90 deg, coastlines aren't.
- **LBP:** texture entropy - empty rooms vs. hatching and stipple.
- **Hough lines + Harris corners:** the fraction of horizontal/vertical lines, and corner density.

It reads `data/samples/woman_in_the_alcove.jpg`, `colonel_jacques.jpg`, and `treasure_island.jpg`. These aren't in git (`data/` is ignored), so get them from Sid. `feature_checkpoint.pptx` holds the checkpoint slides.

## Data layout

```
data/ecommons/                                         raw download
  final_maps/Bax_LiteratureMaps_final_maps/final_maps/*.jpg    2,622 map images
  Gemini/Bax_LiteratureMaps_Gemini/Gemini_fict-scale_full.csv  labels ("Scale" column: 0/1/2)
  MARC/Bax_LiteratureMaps_MARC/MARC_info_parallel.csv          book metadata (title, author, year)
  CLIP/, non-map_sample/                                       other outputs from the paper

data/processed/                                        written by 1-preprocessing and 2-augment
  train/<class>/            kept scan of each train map
  val/<class>/              kept scans only
  test/<class>/             kept scans only
  train_rescans/<class>/    other scans of train maps
  train_aug/<class>/        synthetic copies (<image_id>__augNN.png)
  manifest.csv              one row per image above - the source of truth for labels and splits
  duplicates.csv            every raw image, its duplicate cluster, and whether it was kept
  dedup_review.png          borderline duplicate pairs
  split_summary.csv         maps per split x class
  preprocess_config.json    preprocessing settings and target size
  augment_config.json       augmentation settings

data/samples/               three sample maps for feature_checkpoint.ipynb (not in git)
```

## Loading the processed data

Load images through `manifest.csv`, not by folder: the training set is spread across `train/`, `train_rescans/`, and `train_aug/`.

```python
from pathlib import Path
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".git").exists())  # repo root
PROC = ROOT / "data" / "processed"

m = pd.read_csv(PROC / "manifest.csv", keep_default_na=False)
m["full_path"] = m["path"].map(lambda p: PROC / p)  # "path" is relative to data/processed/

train = m[m["split"] == "train"]       # all variants - train classifiers on this
val = m[m["split"] == "val"]           # one scan per map
test = m[m["split"] == "test"]         # one scan per map
maps = m[m["variant"] == "original"]   # one row per map - PCA, t-SNE, dataset plots
```

| `variant` | Splits | What it is |
|---|---|---|
| `original` | all | The kept scan of each map - exactly one per map |
| `rescan` | train | Another library's scan of a train map |
| `augmented` | train | A synthetic copy, from `2-augment.ipynb` |

Other useful columns:
- `label` (0/1/2) and `label_name`: the map's majority-vote label, the same on every row of that map
- `source_id`: the map's kept scan, the same for all of that map's rows
- `group_id`: use with `GroupKFold` if you cross-validate inside train
- `fictionality`: real or fictional place, from Gemini
- `n_scans`: how many scans of this map the raw data had
- `label_conflict`: whether those scans got different Gemini labels
- `aug_params`: the transforms applied to an augmented copy

Fit anything learned from data (scalers, PCA) on the train split only.

## Notes

- **Split by group, not by image.** Many novels were scanned more than once (Treasure Island has 23 copies), and collected editions reprint the same maps under other titles. Grouping by work and by duplicate map keeps every copy in one split, which is what prevents train/test leakage. Stratification only balances the class mix; dedup only controls how much each map counts.
- **The labels come from Gemini, not human annotators.** For 91 of the 276 maps with several scans (33%), Gemini gave different scans different labels. We use the majority vote, but this is real label noise and a limitation to discuss. `duplicates.csv` and the manifest's `label_conflict` column flag these maps.
- **Book IDs:** the book ID is the filename prefix before the page number, e.g. `000001112276` in `000001112276_00000151.jpg`.
- **Very large images:** a few fold-out maps are over 100 megapixels. Pillow warns about them as possible "decompression bombs"; the notebooks turn that guard off, since this is a trusted dataset.

## Repo contents

- `scripts/download_dataset.py` - downloads and unzips the dataset
- `scripts/0-Cierra_EDA.ipynb` - image sizes, channels, class balance
- `scripts/0-Cierra_Duplicate_Check.ipynb` - finds works scanned multiple times, using the MARC metadata
- `scripts/1-preprocessing.ipynb` - de-duplication, grouped split, resizing -> `data/processed/`
- `scripts/2-augment.ipynb` - minority-class augmentation of the train split
- `scripts/feature_checkpoint.ipynb` - proof-of-concept HOG, LBP, and line/corner features
- `scripts/feature_checkpoint.pptx` - feature checkpoint slides
