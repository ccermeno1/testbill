# Data manifests

The **images** are not versioned (see `.gitignore`): they come from Roboflow and are rebuilt by
the scripts. What lives here is everything that pins down *which* data was used, so a training
run can be reproduced.

| file | what it is |
|---|---|
| `eurobanknotes_extra_selection.{md,csv,json}` | all 551 photos of the external *Euro Banknote Detection* dataset with their status: 304 included in train (and under what name), 247 excluded with the reason (duplicate, naming the matched photo and distance; close-up; no banknote) |
| `eurobanknotes_extra_exclude.json` | paths of the 142 duplicates found by `find_duplicates.py`, consumed by `convert_yolo_obb.py --exclude` |
| `annotations/*.json` | the generated COCO annotations (4-point polygons), copied from what the pipeline produces |

## `annotations/`

| file | imgs / boxes | used for |
|---|---|---|
| `train.json` | 355 / 556 | the v1 train split |
| `valid.json` | 100 / 143 | validation, fixed across every run |
| `test.json` | 47 / 63 | test, fixed across every run |
| `train_plus_extra.json` | 659 / 1918 | **the recommended training set**: train plus the 304 external photos |
| `train_plus_aug.json` | 1065 / 1668 | train plus 710 offline augmented copies |
| `train_plus_aug_extra.json` | 1369 / 3030 | both of the above, not used so far |
| `eurobanknotes_extra_all.json` | 304 / 1362 | the external selection on its own |
| `billetesprueba_all.json` | 143 / 208 | external test set of real photos |

The `file_name` fields point at `data/banknotes_obb/images/`, with the external ones prefixed
`ebd_`. So once the images are rebuilt with the scripts in the main README, these JSONs can be
copied straight into `data/banknotes_obb/annotations/` without recomputing hashes or filters.
