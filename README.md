# testbank — banknote localization

Localization stage of an inspection pipeline. It crops each banknote out of a
photo and hands it to the stain classifier, which already exists and is out of
this scope.

Architecture: **single-stage OBB detector**. Image in, oriented rectangles out,
one per banknote. The crop is taken with a configurable margin over each box
and rectified by homography.

Two-stage alternatives are discarded and must not be implemented. Classical
segmentation does not separate banknotes that overlap or touch — they share
color and texture, there is no edge between them — and a free quadrilateral
adds nothing because the ground truth is rotated rectangles.

Oriented box and not axis-aligned because banknotes appear fanned or adjacent
at different angles: the aligned boxes of elongated objects at different
angles overlap almost entirely and standard NMS suppresses true detections.
Rotated NMS solves it.

## Caveats

Everything one has to know before believing a number from this project. Each
entry says where it is handled; none is an oversight waiting to be discovered.

### About the data

**The aspect of the banknotes is deformed and is not recovered.** 489 of 502
images already arrive at 416×416 from the export, squashed to a square. A real
banknote is ~1.95:1 and the measured median is 1.45. That is where the 19% of
nearly square boxes that motivates the angle attenuator comes from. →
*[SUSPICION] That 19%…*

**There is leakage between splits in the Roboflow export, and now it can be
fixed.** Nearly identical images — the same shot, two consecutive exposures —
fall in different splits. Measured in color:

| | |
|---|---|
| Pairs crossing | 27 |
| Of those, with `test` on one side | **12** |
| `test` images involved | **10 of 50 (20%)** |

**Correction: this README said 36% for a while, and it was false.** The first
measurement was done with grayscale thumbnails, and in grayscale 18 of the 93
"identical" pairs were banknotes of *different value* with the same stock-photo
framing: an orange 50 and a purple 500 are the same silhouette in gray. Color
separates them (the real duplicate stays at 0.98; the false ones drop to
0.45–0.52). It came to light when stratifying the split by banknote type, which
refused to split a group crossing types. → *Why in color* in
`data/duplicates.py`

A fifth of the test is still contaminated, and `evaluate-test` on the export's
split (`splits/v1`) **will come out inflated**. But the split is no longer
untouchable, and redoing it does not destroy it: each `make-splits` writes a
**new version** (`splits/v2`, `v3`, ...) and every run records which one it used.

```bash
testbank find-duplicates --manifest-out runs/_inspection/dups.json
testbank make-splits --repartition \
    --group-key manifest --group-manifest runs/_inspection/dups.json \
    --stratify-regex '^(\d+|Multiple)_'
```

Ratios and seed are chosen (`--ratios 0.8,0.1,0.1 --seed 1`) and stay in the
manifest next to a digest of the split: same seed, same data, same files byte
for byte — checked on the real export, and with a test.

`--repartition` is the explicit exception to "in adopt mode the export rules":
it pools all samples and splits **by groups** (the two shots of the same photo
go together) and **by strata** (each split receives its share of each banknote
type). Measured on the real export: 355 / 76 / 71, all eight types on each
side, and **zero pairs crossing**. The manifest records it as
`mode: repartition` so nobody confuses it with the original split.

→ `testbank find-duplicates`, `testbank make-splits --repartition`,
`data/duplicates.py`, `data/splits.py`

**64.6% of the annotations are still axis-aligned.** After the re-correction
they are correct boxes — many banknotes are photographed straight — but it
means the advantage of OBB over an aligned box is smaller than the project's
premise suggests. Measured: an aligned detector with perfect localization
reaches mAP50 0.939 against 1.000. The median error of the *lossy* formats is
2 px.

**Contamination has a floor of 0.89 in fans.** Not even a perfect detector goes
below it: if a banknote is partially covered, its box necessarily contains
pixels of the one covering it. → *The two contamination thresholds*

**`check-visibility` can give 0 and mean nothing.** It only compares annotated
quads with each other, so an occlusion caused by an unannotated banknote is
invisible to it. → *How to read `check-visibility`*

**One degenerate annotation remains in Roboflow.** It was deleted locally
(`005_Euro_328`, a stray click), but the next export brings it back.

**The dataset is CC BY 4.0: it requires attribution** wherever the model or
the data are distributed. → `data/datasets.py`

### Assumed approximations

**Training uses an approximate IoU and measuring uses the good one.** The
shapely rotated IoU does not fit in the training loop, so the assignment uses
aligned envelopes and the loss decomposes into shape + rotation. Evaluation
keeps using shapely, so if the approximation allocates the trade-off badly, the
final number gives it away. → `models/assign.py`, `models/losses.py`

**Training uses the clipped truth and measuring the unclipped one.** The `clip`
policy fits the box to the frame; the metrics evaluate against the original
quad. Measured: the 5th percentile of coverage does not move (1.0000 on
validation), only the tail (p1 = 0.936). The crop margin absorbs almost all of
it.

**`clip` returns rectangles, and before it did not.** Until now it pinned each
vertex to [0,1] separately. It is the obvious thing and it is wrong: a
**rotated** rectangle cut against a straight frame gives a trapezoid, not a
smaller rectangle. Measured on the real data, **82 of 679 annotations (12%)
stopped being rectangles**, and the only run recorded until then trained that
way.

**What the change is NOT: a quality improvement.** The initial argument was that
a trapezoid is an unreachable target for a model that predicts
`(cx, cy, w, h, angle)`. It is half true, and the missing half matters:
`quad_to_box` already rectangularized any quad, taking two of its four sides.
So the network never saw a trapezoid — it saw a box silently derived from it.
Measuring the real target against the visible banknote:

| IoU of the target the network sees | median | p05 | min |
|---|---|---|---|
| Old clipping | 0.9828 | 0.8361 | 0.7541 |
| New clipping | 0.9852 | 0.7933 | 0.6371 |

Median delta **+0.0002**: better in 57 cases of 99 and worse in 42, and the
tail is somewhat worse. It is a wash.

**What it is:** consistency. What is written to disk is now what it says it is
— a rectangle —, the implicit conversion of `quad_to_box` stops being hidden,
and `clip` stops being vetoed in formats that only represent rectangles. The
tail cases, banknotes sticking far out of the frame, are precisely the ones
`pad` solves well and `clip` cannot.

Now the extents of the box are shrunk **along its own axes** until the four
corners fit, with the angle intact. Result on the real data:

| | before | now |
|---|---|---|
| Non-rectangles | 82 / 679 | **0 / 679** |
| Outside [0,1] | 0 | **0** |
| Coverage of the visible part | — | median 0.985, p05 0.793, min 0.637 |
| Angle preserved | — | 96 of 99 exact |

The 3 that change angle do so by exactly 90° and all have ratio < 1.1: it is
the side swap of nearly square boxes already documented in § *Angle and nearly
square boxes*, not a geometry error — the rectangle is the same, only which
side is labelled long changes.

Two things came out of measuring and cannot be seen by reading the code, so
they are pinned with regression tests in `tests/test_clip_rectangle.py`:

- **Enveloping "the visible part" does not work.** Cutting a **corner** leaves
  the other three intact, and those are what fix the envelope: the 99
  out-of-frame annotations stayed outside, by up to 23% of the side.
- **Each constraint goes to the axis most aligned with it.** Without that the
  fit is valid and still disastrous: in `Multiple_Euro_154`, a nearly
  horizontal banknote sticking out 0.029 at the top, the constraint `y >= 0`
  was "fixed" by shrinking the **long** side from 0.890 to 0.064. It met
  everything and kept 8% of the banknote.

Practical consequence: the formats that only represent rectangles stopped
needing a veto on `clip`. Those formats (`voc_xml`, `yolox_obb_voc`) were
later removed in the refactor for lack of a consumer, and the veto with them:
`dota` and `coco` accept any quadrilateral.

**The angle attenuator may be unnecessary.** It exists because of the 19% of
nearly square boxes, which is probably an artifact of the resizing. If the
originals are re-uploaded, measure again **before** removing it. →
`enabled=False`

**No letterbox when resizing**, because the aspect was already lost at the
source and adding it now recovers nothing. → `models/data.py`

**With `out_of_bounds=pad` and inference without padding, the numbers measure
the mismatched pipeline.** A `WARNING` appears in the `run.json` and in the
summary. → `pad_at_inference`

### Environment fragilities

**`opencv-python` and `opencv-python-headless` are both installed.**
Ultralytics drags in the former; both occupy the `cv2` namespace and the last
one installed wins. Both versions are recorded in every run so the collision is
visible. → `experiment/provenance.py`

**The RTMDet-R environment breaks on its own if anyone installs anything.**
`numpy<2` is declared by nobody and the resolver bumps it without warning. →
*RTMDet-R environment*

**RTMDet-R uses `mmrotate` from an unpublished development branch.**

**Candidates run in different environments** — torch 2.0 for RTMDet-R, 2.14
for the rest. The `run.json` records the version, but it is still a comparison
across environments two years of PyTorch apart.

**The torch of the main environment is CPU.** It is enough for smoke runs and
for a Mac (MPS); for long baselines on a PC the CUDA torch has to be installed
in that same environment. → *Usage*

### Not tried yet

- **RTMDet-R has trained one smoke epoch on CPU** (§ *RTMDet-R environment*),
  nothing longer. mAP 0 after one epoch, as with every other smoke run.
- **PP-YOLOE-R has never trained.** Inference through the adapter is
  verified; training needs `ppdet/ext_op` compiled, pending on the Mac
  (§ *PP-YOLOE-R environment*).
- **Rotated FCOS R50 has trained one smoke epoch on CPU** (§ *Rotated FCOS
  environment*): mAP50 0.940 [0.900, 0.977] on `valid`, coverage p5 0.689 --
  the first non-zero smoke number, because it starts from a complete DOTA
  detector and only the class layer is new. Boxes still short in the tail
  (coverage), angle right (median 4.5 deg). R18 has not trained.
- **The redone split has not been used for training.** `--repartition` is
  measured on the real export (zero pairs crossing, all eight types on each
  side) but the only materialized version, `splits/v1`, is still the export's.
  Writing a `v2` is a decision: runs on `v1` and `v2` are not comparable, and
  `compare` marks it.
- **`evaluate-test` has not been run.** Deliberate: it would spend an access
  to the sealed set on a model that means nothing.
- **No long baseline has been launched.** Everything measured is a 1- or
  2-epoch smoke run, whose numbers mean nothing.

## Annotation guide

**Approximate oriented rectangle.** Geometric exactness is not pursued. A
crumpled banknote or one with wavy edges is annotated with the rectangle that
reasonably envelops it.

**Visibility threshold: 25%.** A banknote covered by another is annotated only
if at least that percentage is visible. The ones showing a strip remain
unannotated and are background for training purposes.

This rule is really enforced **here, when annotating**. The code cannot enforce
it: if a banknote was not annotated, there is nothing to measure. The only
thing that can be checked is the opposite direction — an annotation that
contradicts it — and that is what `check-visibility` is for, which is a
warning for manual review, never an error.

Read the next section before trusting its output.

### Relative area filter

**In each image the front banknote is kept.** On load, every annotation whose
area is smaller than `min_relative_area` (by default **0.25**) times the area
of the largest annotation of that same image is discarded.

It covers both situations without having to tell them apart:

- In a **fan**, the visible strip of a covered banknote is much smaller than
  the front one, and it drops.
- In a photo of **two or three banknotes together**, all have a similar size
  and all are kept.

Three things worth being clear about:

**It is an approximation to the 25% visibility policy, not a measure of
occlusion.** Relative area and visible fraction are different things: a long,
narrow strip can exceed 25% of the area and be fully covered, and a small but
whole banknote can fall below the threshold with nothing covering it. The
filter and `check-visibility` are complementary, not redundant — the first
removes small fragments, the second flags what is large but covered.

**It is a filter on load, not a deletion.** The annotation files are never
touched. Filtered annotations **stay in the source dataset**, and the report
lists them with image, index and relative area percentage so they can be
reviewed and fixed in Roboflow. Indices are always the position within the
file, never the position after filtering, so they lead to the right annotation.

**It is a parameter, not a constant.** It lives in
`annotation_policy.min_relative_area` and can be raised or lowered without
re-exporting the dataset. `--min-relative-area 0` disables the filter and shows
the annotations as they are in the file.

In the visualization, what is filtered is drawn in **dashed gray** with its
index and percentage, without numbered vertices or anchor arrow: it is there
to be located, not as part of the canonical set.

Measured on the current export (502 images, 693 annotations in
`train`+`valid`), the default filter discards **14 annotations in 10 images**,
2%.

### How to read `check-visibility`

The check measures **geometric overlap, not occlusion**, because the depth
order is not annotated. That A overlaps B does not say which one is on top.
Consequences measured on synthetic data with 3 planted violations among 108
images and 301 annotations:

| | |
|---|---|
| Annotations flagged | 28, in 20 images |
| Real violations | 3 |
| Found by the check | 2 of 3 |

The **false positives** come from not knowing the depth: a heavily overlapped
quad may perfectly well be the top one, which is not covered at all. The line
`Real violations: at most N` bounds this exactly — it walks every possible
depth order with a DP over subsets and returns the maximum number of
annotations that could be covered at once.

The **false negative** has a different cause and no fix in code: if the
covering banknote **is not annotated**, the check cannot see it. It is the same
blind spot of the policy, with the twist that it can hide precisely the
violation it is looking for.

Use the report as an ordered list of candidates, and always look at the
visualization before touching an annotation.

## Coordinates and canonical order

The pivot of every conversion is the **canonical quad**: 4 normalized vertices.
N formats are 2N converters, not N².

**Canonical order.** Clockwise with respect to the centroid, starting at the
vertex that opens the longest side. Tie-break by smallest `x+y`, then smallest
`x`, then smallest `y` — the whole chain is needed because `x+y` ties on
rectangles whose diagonal is perpendicular to `(1,1)`.

"The corner closest to the origin" is not used: it is discontinuous near 45°
and a 2 px jitter rotates the labels by 90°.

**Image aspect.** Normalized coordinates divide `x` by the width and `y` by the
height, which is an **anisotropic** scaling. In that space the longest side,
the angle and the ratio are not the geometric ones: a 2:1 banknote lying in a
16:9 image has normalized ratio 1.13, and in 20:9 it drops below 1 and the
anchor jumps to the short side. That is why every length comparison accepts
the aspect, and the reader receives it from `data/derived/image_sizes.json`.

**Dyadic grid.** Coordinates are snapped to multiples of `2^-30`. It is what
makes `flip(flip(q)) == q` hold **exactly**: in float64 `1 - (1 - 0.1)` gives
`0.09999999999999998`, so `x -> 1-x` is not an involution. On the grid it is,
bit for bit. The introduced error is `2e-6 px` in a 4000 px image. The
annotation files are not modified.

**Tolerant range** `[-0.5, 1.5]`, with a warning outside `[0,1]` but no
failure: there are banknotes crossing the image border and their vertices fall
outside legitimately.

**Unstable anchor warning** if the side ratio drops below 1.1.

## Splits

`splits.py` is the **single door** to the assignment. No other module computes
it. Everything reads from `splits/vN/{train,valid,test}.txt`, never from the
directory tree, so the split stays frozen even if the directory changes.

**Versions.** `splits/` is a container of immutable versions: `v1` is the
export's split as it came from Roboflow, and every `make-splits` writes the
**next** one (`v2`, `v3`, ...) without touching the previous ones. Each
version has its own `manifest.json` with mode, ratios, seed and digest.
Commands read the **latest** by default; `--split-version vN` (or
`data.split_version` in the config) pins one. Whatever gets resolved is written
into the run's `run.json` (`split: {version, mode, digest}`) and shown by
`compare` in a `split` column — two rows on different versions were evaluated
on different images and are not comparable, and the table says so. A version
can only be rewritten in place with `--split-version vN --overwrite`, which is
for redoing one that went wrong, not for replacing history.

```bash
testbank make-splits --repartition ...           # writes splits/v2
testbank train yolox-obb-nano                    # trains on v2 (latest)
testbank train yolox-obb-nano --split-version v1 # trains on the export's split
testbank evaluate-test runs/<run> --reason "..." --split-version v2
```

The test-access log (`runs/test_evaluations.jsonl`) also records the version:
opening the test of `v1` and the test of `v2` are two different sets, and each
spends its own access.

**Adopt mode (default).** If the directory already brings `train/`, `valid/`
and `test/` with `images/` and `labels/` — the structure Roboflow exports —
that is the split. It is neither recomputed nor "improved": only frozen.
Roboflow uses `valid`, not `val`; `val` is accepted as an alias with a note,
and having both at once is an error.

**Create mode.** If that structure does not exist, it is generated with
configurable percentages and a seed recorded in `splits/manifest.json`.

**`--repartition`: the explicit exception.** Ignores the export's split and
splits from scratch as in create mode. It exists because Roboflow's separates
nearly identical shots of the same photo. It is opt-in, and the manifest writes
it as `mode: repartition` with the export's mode next to it, so nobody confuses
this split with the original. If one tries to *adopt* with a group manifest
that proves the export spreads some group, it refuses and the message points
here: the leak is not frozen silently.

**Groups are split, not photos.** With `--group-key manifest` and the JSON
written by `find-duplicates`, the two shots of the same photo fall on the same
side. Consequence: ratios apply to the number of groups, so 0.8 over 441
groups gives 394 photos and not 402. It is the right thing — splitting a group
to make a number add up would reintroduce the leak — but the exact count does
not come out round.

**`--stratify-regex`: representatives of each type on each side.** A regex
with one capture group over the sample name; for the Roboflow export,
`'^(\d+|Multiple)_'` extracts the banknote type. Splitting is done within each
stratum and then merged, so each split receives its proportion of each type.
The command prints the stratum × split table and the manifest stores it. Two
honest limits: with few groups per type the rounding can leave the small split
without one (`round(2 × 0.1) = 0`), and a group crossing types — two "equal"
photos with banknotes of different value — is split anyway and warned about,
because over-grouping is cheap and blocking is not.

**`--ratios` and `--seed`.** `--ratios 0.8,0.1,0.1` (must sum to 1);
`--seed N`. Same seed, same data, same files byte for byte: checked on the real
export and pinned with a test. The manifest stores ratios, seed and a digest of
the split to verify it later.

```bash
testbank find-duplicates --manifest-out runs/_inspection/dups.json
testbank make-splits --repartition --ratios 0.8,0.1,0.1 --seed 1 \
    --group-key manifest --group-manifest runs/_inspection/dups.json \
    --stratify-regex '^(\d+|Multiple)_'
```

`--group-key` accepts `filename-prefix`, `directory`, `manifest` and `none`.
The value `none` asserts that every image is independent, and requires
`--i-confirm-independence` so that it is an explicit assertion and not an
oversight: if several shots of the same physical banknote exist spread across
splits, the metrics come out inflated. The confirmation **is only required
when creating**; in adopt mode we do not choose the split, we only freeze it.

**Integrity on load.** A sample in two splits is a fatal error with a message
that identifies it. With a group key, a spread group as well.

**`splits/duplicate_groups.json` is stale.** It is the group manifest from the
first, grayscale duplicate measurement (the one with 18 false pairs of
different-value banknotes). Do not feed it to `--repartition`: regenerate it
with `find-duplicates`, which now works in color.

**No `extend-splits`.** There was a command to append new samples to an
already materialized split; it was removed in the refactor. If a new export
arrives, either its split is adopted or it is re-split with `--repartition`:
two paths doing different things with the new data was one ambiguity too many.

**Sealed test.** `SplitLoader.load("test")` requires `allow_test=True`. The
runner never passes it. The separate command `evaluate-test` records every
access in `runs/test_evaluations.jsonl`.

**No cross-validation.** There was a `make-folds`; it was removed in the
refactor because cross-validation was discarded and the fold runner was never
written. The near-duplicate manifest now serves `--repartition`.

## Metrics

The two that decide, because they measure whether the crop works:

- **Coverage** — fraction of the real banknote inside the predicted crop with
  margin. A crop that cuts half a stain ruins the downstream classifier.
  Target ≥ 0.98 at the 5th percentile.
- **Contamination** — fraction of the crop belonging to another banknote.
  Background is harmless noise; a piece of the neighbour can bring in a foreign
  stain and cause a false positive. **Two thresholds**, see below.

### The two contamination thresholds

The distribution is **bimodal, not continuous**, so a single threshold would be
useless in both directions at once. Measured with a perfect detector
(prediction = truth) on train+valid of the current export, at margin 0.05:

| scene | n | median | p95 | threshold |
|---|---|---|---|---|
| single banknote | 301 | 0.0000 | **0.0000** | 0.01 |
| fan | 378 | 0.0231 | **0.8906** | 0.92 |

In **single-banknote images** the floor is exactly zero: there is nothing else
in the image that can dirty the crop, so any contamination is a real detector
error. The 0.01 threshold is generous with respect to the floor and strict in
absolute terms, which is what is wanted.

In **fans** not even a perfect detector goes below 0.89, and not for being bad:
if a banknote is partially covered, its box necessarily contains pixels of the
one covering it. It is geometry, not error. The 0.92 threshold leaves ~3 points
of slack, a quarter of the remaining way to 1.0.

**Warning when reading the fan threshold:** between the floor (0.89) and the
ceiling (1.0) there are 11 points, so it discriminates little — it only catches
detectors considerably worse than the perfect one. To compare candidates on
fans look at the **median**, which the perfect detector leaves at 0.023 and has
plenty of room.

An image counts as a fan if it has **more than one banknote present**, also
counting the ones the area filter discarded: a filtered neighbour is still in
the pixels and still dirties.

Both numbers come from the data, so they have to be re-derived when the export
changes: `contamination_floor()` in `metrics/crop.py` recomputes them.

### [PENDING] Polygon masking in the crop

Pending improvement for the crop stage, **outside the current scope**.

Today the crop is the predicted box with margin, so in a fan it inevitably
drags pixels of the front banknote. But at inference all detections of the
image are available, not only one: the **polygons of the other banknotes can
be masked** inside the crop before handing it to the stain classifier.
Contamination would go from being pixels of another banknote — which can bring
in a foreign stain — to being background, which is harmless noise.

It would lower the 0.89 floor and make the fan threshold far more
discriminating. It touches the stage feeding the classifier, which is out of
this scope.

Both depend on the **crop margin**, which is why it lives in the config
(`crop.margin`), is recorded in `metrics.json` and is reported swept over
`{0.00, 0.05, 0.10}`: the margin trades one metric for the other and a single
row hides the trade. Contamination is reported next to that of the
**annotated quad** with the same margin: in a fan the ground truth itself
already contains pieces of the neighbour, and without that floor the model's
error cannot be separated from the irreducible geometry.

Detection: mAP50 and mAP50-95 with rotated IoU via shapely. Angle: error modulo
180°, as `min(|Δ|, 180-|Δ|)`. Vertices: median and p95 distance, in pixels and
as a fraction of the longest side — diagnostic, not a success criterion.

Matching by rotated IoU, threshold 0.5, greedy assignment by descending
confidence. **Undetected banknotes count as misses**, they are not excluded:
excluding them makes a detector that only finds the easy cases look better,
which is the inverted conclusion. Detection rate, error conditioned on
detection, and aggregate are reported separately.

**Intervals.** Every row of the comparison table carries `n` and a bootstrap
interval, without exception. The **resampling unit is the image**, not the
detection: several banknotes in the same image are correlated and resampling
detections narrows the intervals artificially.

The deciding table is computed on the adopted `valid`. 5-fold cross-validation
remains as a check of the winning candidate.

**When interpreting:** because of the visibility policy, in fan images the
model can correctly detect banknotes that are not annotated and they will
count as false positives. If you see precision sunk on those images, look at
the visualizations before concluding the model fails.

## Pretraining: what each candidate loads, and the port that makes it possible on CPU

Until here no own candidate read a checkpoint: the own head trained from
scratch and the DDGRCF port did not exist. Both things have changed.

### The own head loads Megvii's COCO

`yolox_s.pth.tar` — the official YOLOX-S, Apache-2.0, the same file the
buzhidaoshenme fork ships — **fits entirely into our `small` backbone and
neck**. Measured: 354 tensors and 7,066,683 parameters in both, identical
shapes. Only the names change: they wrap the backbone as `backbone.backbone.*`
and call the neck modules `lateral_conv0`, `C3_p4`, `reduce_conv1`…; here they
are `backbone.*` and `neck.lateral_c5`, `neck.p4`, `neck.lateral_c4`… The
mapping is in `models/pretrained.py`, paired by function and checked shape by
shape on load. Their head (80 classes, no angle) is discarded: what is
inherited is *knowing how to see*, not *knowing where the banknote is*.

```bash
testbank train yolox-obb-small --pretrained weights/yolox_s.pth.tar
```

`--pretrained` goes into the config and is frozen in the run: two runs with and
without pretraining are not comparable, and the `config.yaml` has to say so.
Loading the wrong variant (`yolox_s` into `nano`) is an **error**, not a
warning: training "half pretrained" without knowing is worse than training
from scratch. Megvii also publishes `yolox_tiny.pth` and `yolox_nano.pth` in
the GitHub releases — direct download, no Baidu — and **all three load
entirely into their variant**: 354 tensors in `small` and `tiny`, 462 in
`nano` (the *depthwise* convolutions split in two). The three own architectures
are identical to theirs; measured, not assumed. They go in `weights/`, ignored
by git.

Effect over one smoke epoch, for calibration: `classes 0.24` against `1.87`
from scratch, `box 0.57` against `0.86`. It is not a result, it is the signal
that the mapping loads something useful.

### The DDGRCF/YOLOX_OBB port: trains on CPU and on MPS

`yolox-obb-ddgrcf-port` is their network rewritten in pure torch
(`models/ddgrcf.py`), layer by layer following their yaml and **with their
same parameter names**, so their DOTA checkpoint loads with `strict=True`
without any mapping. Verified against their actually built model (with their
operators replaced by a stub, which are only called when training): **426
identical keys and shapes, 8,051,797 parameters**, and with their weights
loaded the output matches theirs — obj, cls and θ exact; decoded boxes to 3e-5
px. That validates both the port and our decoder for their
`(dx, dy, log w, log h)` regression.

A detail that cost one attempt: their yaml puts `n=2` in the head stems, but
their parser applies the depth multiplier, `round(2 × 0.33) = 1`, and **one**
conv without `Sequential` remains. With two, the port had 8.94M parameters and
the keys `model.27.0.*` instead of `model.27.*`.

What replaces their compiled operators:

| Theirs (C++/CUDA) | Here |
|---|---|
| `box_iou_rotated` in SimOTA | `models/overlap.py`: exact polygon IoU in torch, 18.6 ms per image |
| PolyIoU in the loss | The same, differentiable; matches shapely to 2.5e-6 over 2,000 pairs |
| `nms_rotated` | Our `rotated_nms` |
| Their `Trainer` and `DataPrefetcher` (CUDA) | testbank's loop |

The `ddgrcf` recipe reproduces their `get_losses`: PolyIoU ×5 + obj + cls by
IoU + late L1, everything `Σ / num_fg`, SimOTA with `−log IoU`. One deviation,
stated: their late L1 leaves the **angle target at zero** (`get_reg_l1_target`
fills 4 of 5 components), almost certainly an oversight; here the real angle
goes in. And what the port does **not** reproduce: their *dataloader* (mosaic,
mixup, resampling), their optimizer and their EMA.

It is **fit for production** — nothing compiled, no clone — and it is today the
only candidate with DOTA pretraining that can train on an M4. Its weights are
on Baidu Pan (their client is needed): `--pretrained weights/yolox_s_dota1_0.pth`.

### The input went in the wrong format, and with pretraining it showed

The port with the DOTA weights gave **mAP 0.000** after one epoch, worse than
`small` with only COCO. Maximum score 0.004, and every box in the same corner
in every image: the network localized nothing. With backbone *and* regression
head trained on DOTA, that only happens if the image arrives in another format.

And it did: our loop gave **RGB in [0, 1]**; YOLOX — Megvii and DDGRCF alike —
expects **raw BGR in 0–255**, not normalized. Swapped channels and a scale 255
times smaller at the first layer. Training from scratch it does not matter
(BatchNorm absorbs the scale); with pretraining it wrecks it.
`models/data.py:image_to_input` now fixes the YOLOX convention in a single
place, for training and inference.

Same one-epoch smoke run, before and after:

| | RGB [0, 1] | BGR 0–255 |
|---|---|---|
| Port + DOTA weights | 0.000 | **0.105** [0.075–0.153] |

It is not a baseline — it is one epoch — but it is the first time a smoke run
gives a number that is not zero for a reason that is understood.

### What came out when trying it

With the COCO backbone freshly loaded and the head untrained, the network
predicted a vertex at −0.54 normalized and `Quad` rejected it with a
`QuadError`: **the whole evaluation went down**. It would have brought down a
real baseline at the first try. And when the port with DOTA finally produced
detections, the run's visualization went down for the same reason
(`'NoneType' object has no attribute 'points'`): no previous smoke run had
produced such a prediction. Fixed in the two remaining consumers, with a test.

The first fix discarded those predictions, and that was a gift to the metric:
a box the model predicted mostly outside the image is a false positive it
committed. Now the prediction is emitted **without geometry**
(`Prediction.quad = None`), matches nothing and **counts as a false positive at
its score**. It is what the reference frameworks do in practice, which do not
discard. With a regression test on the matching.

## Loss recipes: three networks on the same backbone

The own head can be trained with **three complete loss recipes**, each the one
of a concrete network and without mixing components between them:

```bash
testbank train yolox-obb-nano --loss-recipe own
testbank train yolox-obb-nano --loss-recipe yolox_obb_fork
testbank train yolox-obb-nano --loss-recipe ultralytics_obb
```

| Recipe | Box | Other terms | Assigner | Head |
|---|---|---|---|---|
| `own` | aligned `1 − IoU` | angle (cosine over `sin 2θ, cos 2θ`, attenuated), obj BCE, cls BCE | SimOTA, cost `−log IoU` | direct |
| `yolox_obb_fork` | KLD ×5 | obj BCE, cls BCE with target `one-hot × overlap`, late L1; everything `Σ / num_fg` | SimOTA, cost = KLD | direct |
| `ultralytics_obb` | `1 − ProbIoU` ×7.5 | DFL ×1.5, cls BCE ×0.5 with **soft** target; no objectness; everything `Σ / Σ targets` | TAL (`topk=10, α=0.5, β=6`) | **DFL** (16 bins/side), scalar angle, no obj |

The fourth recipe, `ddgrcf`, belongs to the DDGRCF port (§ *Pretraining*):
`yolox-obb-ddgrcf-port` forces it.

A recipe is not a box loss: it is assigner + targets + terms + normalization +
gains. That is why `ultralytics_obb` **changes the head**: its DFL requires
each distance to be predicted as a distribution, and its class BCE without
objectness needs the soft targets of TAL. Training "their loss" on our direct
regression would have been comparing something else under its name. The recipe
fixes the head (`HeadSpec`), and the head travels **inside the checkpoint**:
loading weights rebuilds the one that produced them.

### Licenses, and what is faithful and what is not

**The fork is Apache-2.0 and was read in full.** Its `get_losses` is
reproduced: same terms, same gains (`reg_weight = 5.0`, `τ = 1.0`), same
normalization, same class target, same SimOTA with the KLD as cost. The own
KLD is numerically anchored against their formula in `tests/test_overlap.py`
(500 random pairs, `atol 1e-5`). Two deviations, both of head: they regress the
angle in degrees directly — with the jump at ±90° — and here
`(sin 2θ, cos 2θ)` is kept; and their late L1 goes over
`(dx, dy, log w, log h)` while here it goes over the four distances, which is
the raw regression of *this* head. With short smoke runs the L1 is on from the
first epoch, just like in the fork when `no_aug_epochs ≥ max_epoch`.

**Ultralytics is AGPL and not a line has been read or copied.** The
*composition* — which terms, which gains, which assigner — comes from its
public documentation. The *formulas* come from the papers: ProbIoU (Llerena et
al., 2021), DFL (Li et al., 2020), TAL (Feng et al., "TOOD", 2021). ProbIoU is
verified against an independent matrix implementation of the Bhattacharyya
distance; the isolation test keeps guaranteeing that nothing outside the
adapter imports `ultralytics`. It is a reproduction of the described recipe,
not a copy: if their code had an undocumented detail, it is not here.

### Two covariance conventions that cannot be mixed

The two papers turn `(w, h)` into variances differently — `w²/4` for KLD,
`w²/12` for ProbIoU — and mixing them changes the numbers without changing the
name. `box_to_gaussian` requires the explicit divisor so it cannot be called
"plainly". → `models/overlap.py`

### What is measured

The three recipes complete the 1-epoch smoke run on the real data, with finite
losses, finite gradients and each reporting **its** terms (the own one has
angle; the fork zero angle and L1; Ultralytics DFL and no objectness). The mAP
after one epoch on CPU is 0 for all three, as it should be: it is a smoke run,
not a result. The three are comparable because they share backbone, data,
split and metric; the only thing that changes is the recipe.

## Angle and nearly square boxes

The own OBB head predicts the angle as **`(sin 2θ, cos 2θ)`**, not as a scalar
in radians. A rectangle rotated θ and another rotated θ+180° are the same
rectangle; regressing θ directly would punish the model for being right —
predicting 179° with truth 1° would give a huge error for 2° of real error.
With the doubled angle, both fall on the same point of the circle and the
ambiguity disappears by construction.

**That solves the 180° periodicity, but not the side swap.** When `w ≈ h`, the
box `(w, h, θ)` and the box `(h, w, θ+90°)` describe the same rectangle and the
encoding sends them to opposite points: two contradictory targets for the same
box.

That is why the angle loss is **attenuated** when the ratio of the true box is
low. If the rectangle is nearly square, the angle barely changes the crop —
the annotation policy already says an approximate rectangle is enough — so it
makes no sense to spend capacity punishing something that is neither well
defined nor alters the result.

Everything is parametrizable in `detector.loss.angle_weight`: `enabled`,
`ratio_threshold`, `min_weight` and `decay` (`linear`, `smoothstep`,
`quadratic`, `step`). It is in the config and not hard-coded **so it can be
measured**: the question "how much does this contribute" is answered by
training with and without, and both runs stay recorded with their config.

The Gaussian representation (GWD/KLD), which would absorb the ambiguity
naturally, was discarded because it moves away from the shapely rotated IoU we
measure with. That traceability weighs more than the elegance of the
formulation.

## Augmentation

Two stages, deliberately separate:

1. **Offline, before training — `testbank make-augmented`.** The geometric
   part: rotation by any angle, a little shear, a very small perspective,
   some translation. Written to disk as `data/augmented/vN/` with labels, a
   manifest and a contact sheet, so it can be **reviewed** before a single
   epoch runs, and **reproduced** from its seed. `testbank train --augmented
   vN` adds the copies to the train split.
2. **On the fly, during training — `--augment`.** Ultralytics' recipe
   (mosaic, scale/translate, HSV, flips), on originals and copies alike.

Both are off by default and both are frozen in the run: `run.json` records
the augmented version (with its digest and recipe; `compare` shows it in the
`aug` column) and `config.yaml` the on-the-fly recipe. "How much does each
add" is answered by rows of the table, not by reasoning.

### Offline: `make-augmented`

```bash
testbank make-augmented                      # -> data/augmented/v1, 3 copies per training photo
testbank make-augmented --copies 5 --seed 7  # another version, never overwrites
testbank train yolox-obb-nano --augmented v1 --augment --name nano_rot   # copies + Ultralytics on top
```

Why this part is offline: the rotation is what decides whether the model
learns the banknotes that are not horizontal or vertical — the training
photos mostly are, and the model was missing the rest — and it is the part
whose geometry can go wrong (a label drifting off its banknote, a banknote
cut or collapsed). On disk it can be looked at: `_contact_sheet.png` draws
24 copies with their boxes, and every copy is a JPEG plus an `obb_yolo`
label like the originals.

| knob (`offline_augment`) | default | what |
|---|---|---|
| `copies` | 3 | copies per training photo; the originals stay |
| `degrees` | 180 | rotation in ±degrees; 180 is every orientation (a banknote is valid in all of them; multiples of 90 would teach four) |
| `shear` | 2 | degrees |
| `perspective` | 0.0002 | on the projective row; 0.0005 is already a lot |
| `translate` | 0.05 | fraction of the side |
| `scale` | 0 | none: scale is the on-the-fly recipe's job |
| `keep_whole` | true | a banknote the draw would push out of the frame is not cut: the draw is zoomed out until it fits, 2 % inside |
| `fill` | border | the canvas that appears takes the photo's own background — the median colour of its border ring, i.e. the table; `gray` is Ultralytics' 114 |
| `seed` | 20260910 | |

Reproducible by construction: each photo gets its own generator from
`sha256(seed, photo id)`, so its copies do not depend on how many photos
came before it or in which order (tested: two versions from the same seed
have the same digest, byte for byte). Versioned like the splits: `v1`,
`v2`, … never overwritten; `manifest.json` records the split version and
digest it was made from, the recipe, the counts and every file.

**Leakage.** Only the train split is read, and `train --augmented` refuses a
version made from another split version or digest: a copy of a photo that is
in `valid` of the split being trained on would be seen in training and the
validation number would be a lie. Measured on the 351 training photos of
`v1`: 702 copies, 1098 boxes from 549, none dropped, no copy without labels.
Non-square photos (13 of 502) are centred on a square canvas of their longer
side, filled with the background; the training resize then distorts
originals and copies alike.

### On the fly: `--augment`

**The recipe is Ultralytics'**, reproduced from its documentation and its
`default.yaml` (AGPL: none of its code has been read). It is what the
`ultralytics-yolo-obb` reference row trains with, so the candidates of this
loop get the same treatment:

| step | default | theirs |
|---|---|---|
| mosaic of 4 images, random center, off for the last `close_mosaic` epochs | 1.0, `close_mosaic` 10 | 1.0, 10 |
| random perspective: scale, translation | ±0.5, ±0.1 | ±0.5, ±0.1 |
| … rotation, shear, perspective (`degrees`, `shear`, `perspective`) | 0, 0, 0 | 0, 0, 0 |
| HSV gains (hue, saturation, value) | 0.015, 0.7, 0.4 | same |
| horizontal / vertical flip | 0.5 / 0.0 | 0.5 / 0.0 |
| box dropped when less than `min_visible` of it survives, or thinner than 2 px | 0.1 | `area_thr` 0.1, `wh_thr` 2 |

Same order as theirs: mosaic → random perspective → HSV → flips. The
geometric knobs, `keep_whole` and `fill` exist here too (same code as the
offline stage) for an experiment that wants them on the fly; by default they
are at Ultralytics' values, so `--augment` on top of `--augmented` does not
rotate twice. Two exact extras, off: vertical flip and `rotations` (turns
by multiples of 90 degrees).

**The quad follows the pixels** (`models/augment.py`, shared by both
stages). Every image operation is applied to the quad with the same 3×3
map, in homogeneous coordinates for the perspective; rotation keeps a
rectangle a rectangle, shear and perspective make it a slight quadrilateral
and the label *is* that quadrilateral. Two things this needed and a test
caught: OpenCV warps in pixel-centre coordinates while the quads live in
pixel-edge coordinates, so the matrix is conjugated by half a pixel or every
box lands 0.5 px off (measured: centroid offset 0.50 → 0.09 px); and the
zoom-out of `keep_whole` has to be centred where the photo's centre lands,
not where its corner does (a large rotation put that outside the frame, the
factor went negative and 20 % of the draws collapsed to a blank tile —
regression test). A box the warp pushes partly out of the frame goes through
`clip_quad` — the same rectangle-preserving clip as the `clip` border
policy — after a coarse clip to the tolerant range; in the mosaic, boxes are
clipped to the canvas before the warp, as Ultralytics clips its labels.
Tested by painting the quad as a mask, warping the mask with the image
operation and comparing with the mask of the warped quad (sub-pixel
rasterization: truncating the corners biases a 20 px box by a tenth of its
IoU); whole boxes must sit on the pixels, cut boxes carry the known cost of
the clip policy.

Deterministic: the dataset owns a `random.Random(seed)` drawn in sampler
order (mosaic partners included); with `num_workers = 0` two runs with the
same seed see the same augmented images (tested). The loop tells the dataset
the epoch (`set_epoch`) so the mosaic stops for the last `close_mosaic`.

Who gets it: the candidates trained by this loop — the three own variants,
the DDGRCF port and Rotated FCOS. RTMDet-R, PP-YOLOE-R and Ultralytics carry
their own inside their pipelines (the offline copies reach them too: they
are just more training files). `valid` and `test` are never augmented.

To pick a subset, a YAML with `--config`; anything not set keeps its default:

```yaml
detector:
  augment: {enabled: true, mosaic: 0.0}                     # perspective + HSV + flips, no mosaic
  augment: {enabled: true, degrees: 30, keep_whole: true, fill: border}   # mild rotation on the fly instead
offline_augment: {copies: 5, degrees: 90, shear: 0}         # what make-augmented writes
```

### [SUSPICION] That 19% is probably an artifact

The attenuator exists because **145 of 762 annotations (19%)** have ratio <
1.1 — the same 19% that triggers the unstable anchor warning. But there are
reasons to believe that number **does not describe the domain, but the
export**:

| | |
|---|---|
| Real ratio of a euro banknote | **~1.95:1** in every denomination |
| Measured median ratio | **1.45** |
| Images at 416×416 | 489 of 502 |

A banknote is 1.95:1 in the world. That the median comes out at 1.45 points to
the deformation of the resizing to a square, not to banknotes appearing
foreshortened. The images arrived already resized at the source, so the
distortion cannot be undone from here.

**If at some point the originals are re-uploaded without resizing**, the ratio
distribution has to be measured again. It is quite likely that the 19%
collapses and the attenuator stops being needed — in which case
`enabled=False` and move on. Measure it before removing it, not the other way
round.

## Discarded in the refactor: the two YOLOX-OBB clones

There were two adapters wrapping cloned and patched foreign repositories. They
were removed in `feature/refactor_code` because what they contributed is now
in-house, and both required CUDA. What was learned is kept, because it cost
measuring.

**`buzhidaoshenme/YOLOX-OBB`** (Apache-2.0, abandoned in 2021). It contributed
its KLD recipe and Megvii's COCO (`yolox_s.pth.tar`, backbone without angle
branch). Today the recipe is `--loss-recipe yolox_obb_fork` on the own head —
numerically anchored against their formula — and the COCO is loaded by
`--pretrained` in the three variants. What it cost to get it running, in case
anyone goes back: `polyiou` is a C++ extension with SWIG without a wheel that
the whole `import yolox` hangs from (it was replaced by shapely: two names,
`VectorDouble` and `iou_poly`); `apex` imported unconditionally; `np.int0` and
`np.bool` removed in NumPy 2.0; their VOC reader subtracts 1 from the
coordinates while their generator writes in base 0; their
`evaluate_detections` returns a fixed `0.0, 0.0`; and their `DataPrefetcher`
is built on `torch.cuda.Stream`. Their README claims 0.712 mAP on DOTA but does
not publish that checkpoint.

**`DDGRCF/YOLOX_OBB`** (Apache-2.0, 2022). It contributed the only published
YOLOX with an OBB head **trained on DOTA** (`YOLOX_s_dota1_0`, 70.82 mAP@0.5,
on Baidu Pan). Today that is `yolox-obb-ddgrcf-port`: their network ported
tensor by tensor and their recipe with the exact IoU in torch
(§ *Pretraining*). The clone required compiling their C++/CUDA operators
(`box_iou_rotated`, `nms_rotated`, `convex`) with MSVC — not available here:
`Microsoft Visual C++ 14.0 or greater is required` — and BboxToolkit, which is
pure Python. The data path (`export dota` → `img_split.py --sizes 1024` →
their `.pkl`, 1 patch = 1 image) was verified end to end. The compiled operator
was *inside* their SimOTA, so it did not admit the shapely trick of the other
fork; that is why the port reimplements the polygon IoU.

## RTMDet-R environment

RTMDet-R is Apache 2.0 and fit for production, but **it does not fit in the
main environment**. MMCV carries compiled operators — rotated NMS, rotated IoU
— linked against the binary ABI of a specific PyTorch version, and from there
comes a chain of constraints that pushes the whole project two years back.

### The chain, and why it ends at torch 2.0

```
mmrotate 1.0.0rc1   ->  mmdet >=3.0.0rc6, <3.2.0
mmdet 3.1.0         ->  mmcv  >=2.0.0rc4, <2.1.0
mmcv 2.0.x          ->  only exists on the torch 2.0 index
```

Each link pushes the next. And **none of these incompatibilities is seen by
the dependency resolver**: they are `assert`s inside each package's
`__init__.py`, which only fire on import.

`cp311-win_amd64` wheels available, measured:

| index | mmcv available |
|---|---|
| `torch2.14` | the index does not exist |
| `torch2.4` | only `manylinux` |
| `torch2.1` | 2.1.0, 2.2.0 — **both above mmdet's cap** |
| **`torch2.0`** | **2.0.0, 2.0.1** — the only ones that work |

And `mmrotate` **is not published on PyPI in its 1.x line**: PyPI only has
0.3.4, which goes with `mmcv-full` 1.x and `mmdet <3`. It has to be installed
from the `dev-1.x` branch on GitHub.

### Verified recipe

```bash
uv venv .venv-rtmdet --python 3.11
VIRTUAL_ENV=.venv-rtmdet uv pip install torch==2.0.0 torchvision==0.15.1   "numpy==1.26.4" packaging "setuptools<81"
VIRTUAL_ENV=.venv-rtmdet uv pip install --only-binary=:all:   --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html   mmcv==2.0.1
VIRTUAL_ENV=.venv-rtmdet uv pip install mmdet==3.1.0   "git+https://github.com/open-mmlab/mmrotate@dev-1.x" "numpy==1.26.4"
```

Checked end to end: the four packages import, `box_iou_rotated` computes
(0.6337 in the test case), `RotatedRTMDetHead` is in the registry, and
Ultralytics still loads YOLO26-obb in that same environment.

**`numpy<2` is the fragile part.** Torch 2.0 predates NumPy 2 and does not
restrict it; `mmcv` and `mmengine` ask for `numpy` without a cap. The resolver
bumps it on its own and `torch.from_numpy` starts failing with *"Numpy is not
available"*. Any later install in that environment can break it again
**silently**: the pin has to be repeated in every `uv pip install`.

### What the first training on CPU needed

RTMDet-R had never trained. The first smoke run on the real data hit three
things, all in `detectors/rtmdet_r.py:build_train_config`, none in the
environment:

1. **`persistent_workers`.** The published config keeps workers alive between
   epochs; we set `num_workers=0` for determinism and MMEngine refuses the
   combination. Set to `False`.
2. **`.png` versus `.jpg`.** `DOTADataset` pairs every `labelTxt` with an image
   of suffix `png` (DOTA's); our export keeps the source `jpg`. Its
   `img_suffix` parameter.
3. **The box loss is CUDA-only.** `RotatedIoULoss` computes its IoU with
   `mmcv.ops.diff_iou_rotated_2d`, which has a CUDA kernel and nothing else:
   the first iteration dies with `implementation for device cpu not found`.
   The assigner's `box_iou_rotated` and the NMS do have CPU code; this is the
   only piece that does not. The adapter registers a twin of that loss --
   same class, same `1 − IoU` in `linear` mode, same weight 2.0 -- whose IoU
   comes from `models/overlap.rotated_iou`, the pure-torch polygon IoU already
   used by the DDGRCF port and matched against shapely to 2.5e-6. The recipe
   does not change; the operator does. MMRotate is built for CUDA and nobody
   needed this kernel on CPU; it is the same wall the two YOLOX-OBB clones
   hit.

Two more things that are not failures but would silently waste the run:

4. **Resolution.** The published pipelines work at DOTA's 1024 px: a
   `Resize` AND a `Pad`, both to 1024. At 1024 one epoch is 44 iterations at
   ~21 s, **15 minutes**. Both steps now follow `detector.image_size`
   (`--image-size`), the same knob as every other candidate; moving only the
   `Resize` shrank the images and padded them back to 1024 with gray, and the
   epoch took the same 20 s per iteration -- measured. At a true 416 it is
   3.9 s per iteration, **~3 minutes per epoch**; 100 epochs is ~5 hours here.
5. **The learning-rate schedule is pinned to their 36 epochs**: a 1000-iteration
   warmup and a cosine from epoch 18 to 36. Left alone, `--epochs 100` would
   sit at the minimum rate from epoch 36 on, and `--epochs 1` never leaves the
   warmup (44 of 1000 iterations: the `lr 1e-5` of the smoke runs). Rescaled
   to `detector.epochs` with the same shape: warmup capped at 10% of the
   iterations, cosine over the second half.

`predict` builds the model from the same one-class, same-resolution config as
training (`build_inference_config`): a one-class checkpoint does not load into
the published 15-class head, and inferring at another resolution would change
every number without saying so.

```bash
.venv-rtmdet\Scripts	estbank.exe train rtmdet-r-tiny --epochs 100 --image-size 416 --name rtmdet_416_e100
```

MMEngine keeps its own log in `_train/work/<timestamp>/`; there is no
`training.json` or `plot-training` for this candidate.

### Not on Apple Silicon

The `mmcv==2.0.1` wheel on that index is **x86_64 only**; there is none for
macOS arm64 (M1–M4), so the recipe above does not resolve on a Mac. Decision:
RTMDet-R runs on the Windows PC, where the recipe is verified end to end,
and everything else (own variants, DDGRCF port, Ultralytics) runs on the Mac
with MPS. Nothing is lost: this environment is CPU-only either way, and the
comparison was already across environments — the `run.json` records platform
and versions, so the row is labelled. Alternatives, not verified: an x86_64
Python under Rosetta (`uv python install cpython-3.11-macos-x86_64-none`,
then the same recipe), or building mmcv from source with `MMCV_WITH_OPS=1`,
which often fails against the torch 2.0.0 headers.

### [DEVIATION] COCO pretraining does not exist for tiny

The specification asked for *"the config with COCO pretraining, not
ImageNet"*. For the `tiny` variant — the one that fits the mobile deployment —
**that config is not published**:

```python
# rotated_rtmdet_tiny-3x-dota.py
checkpoint = '.../cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
```

The only `coco_pretrain` is the `l` variant's, ~52M parameters: ten times the
mobile budget.

**What is used instead**, and why it is better than both options: the
published checkpoints are not backbone pretrainings, they are **rotated
detectors already trained on DOTA**.

| checkpoint | mAP on DOTA |
|---|---|
| `rotated_rtmdet_tiny-3x-dota` | 75.60 |
| `rotated_rtmdet_tiny-3x-dota_ms` | 79.82 |

That is more than "it learned to localize on COCO": it is a model that already
predicts **oriented boxes**, which is exactly our task. Starting there and
fine-tuning on 351 images is a better starting point than any classification
pretraining. The real chain is ImageNet → DOTA, and the destination is what
matters.

### When comparing

A candidate trained here runs with **torch 2.0** while the own candidate and
Ultralytics run with **2.14**. The `run.json` records the version, so the
difference is visible in `compare`, but it is still a comparison across
environments two years of PyTorch apart. It is worth saying when reading the
numbers.

## PP-YOLOE-R environment

`ppyoloe-r-s` is PaddleDetection's PP-YOLOE-R-s, Apache-2.0, trained on DOTA
(73.82 mAP, 8.09M parameters). PaddlePaddle is a **second framework**, not
torch, so it gets its own environment, `.venv-paddle`, and its own adapter
(`detectors/ppyoloe_r.py`, the only module that may import `paddle`/`ppdet`;
a test enforces it). Why try it at all: a third DOTA-pretrained reference,
Apache, with a direct-download checkpoint.

### What runs where

- **Inference needs nothing compiled**: the head's NMS is a core Paddle op.
  Verified on the Windows PC: `predict` through the adapter, DOTA weights,
  5 images at 416 in 6 s, boxes returned as `[class, score, x1..y4]` in
  pixels of the original image.
- **Training needs `ppdet/ext_op`** (`rbox_iou`, `nms_rotated`): the
  task-aligned assigner computes rotated IoU with it. C++ extension built per
  machine with `python setup.py install`; on Windows it needs MSVC
  (`Microsoft Visual C++ 14.0 or greater is required` -- the DDGRCF wall
  again), on macOS clang. The loss (ProbIoU) and everything else are pure
  Paddle. **Not yet built anywhere**: the training smoke is pending on the
  Mac.

### Recipe

```bash
uv venv .venv-paddle --python 3.11
VIRTUAL_ENV=.venv-paddle uv pip install paddlepaddle "setuptools<70" "numpy<2"
cd .. && git clone --depth 1 https://github.com/PaddlePaddle/PaddleDetection.git && cd PaddleDetection
VIRTUAL_ENV=../testbank/.venv-paddle uv pip install -r requirements.txt   # pins numpy<2, opencv<=4.6
VIRTUAL_ENV=../testbank/.venv-paddle uv pip install --no-deps -e .
cd ppdet/ext_op && ../../../testbank/.venv-paddle/bin/python setup.py install   # training only
cd ../../../testbank && VIRTUAL_ENV=.venv-paddle uv pip install -e . "numpy<2" "setuptools<70"
```

The clone lives OUTSIDE the repo and is installed editable: the adapter finds
`configs/` next to `ppdet`. Two fragilities, both seen while setting it up:
`imgaug` (in their requirements) breaks on NumPy 2 and `uv` bumps numpy
unless re-pinned; `setuptools>=70` removed `pkg_resources`, which `ppdet`
imports. Re-pin both after any install in that environment.

Weights: `weights/ppyoloe_r_crn_s_3x_dota.pdparams` (Git LFS; direct download
from the PaddleDetection model zoo otherwise).

### What the adapter changes in the published config

Same surgery as RTMDet-R, in `build_config`: CPU and one class; every reader
`target_size` at `detector.image_size`; `worker_num = 0`; batch and epochs
from our config with the schedule rescaled (theirs is pinned to 36 epochs:
cosine over 44, 1000 warmup iterations); NMS thresholds from
`confidence_threshold`/`nms_iou`; datasets pointed at our `coco` export,
whose `segmentation` carries the oriented quad (`gt_poly`), so this candidate
shares the area filter and the border policy with every other one.

```bash
.venv-paddle/bin/testbank train ppyoloe-r-s --epochs 1 --image-size 416 --name ppyoloe_smoke
```

### Two things it surfaced

- `import testbank.detectors` **required torch**: registering the own
  variants built three networks at import time to count parameters. Now it
  reads a pinned table (`PARAMETER_COUNTS`, checked against the real models
  by a test), and the registry imports in an environment without torch.
- The `coco` exporter now writes the oriented quad as `segmentation` next to
  the envelope `bbox`: what PaddleDetection's own DOTA-to-COCO tool writes.

## Rotated FCOS (oriented-det) environment

`rotated-fcos-r50` and `rotated-fcos-r18` come from
[`oriented-det`](https://github.com/DL4EO/oriented-det) (DL4EO, Apache-2.0,
v0.2, one maintainer): a small torch-only framework for rotated detection.
Why it is here despite the size: it is the **only foreign reference that
meets the project's constraint as is** -- rotated IoU and NMS in pure torch,
nothing compiled -- so it trains on CPU and on a Mac with MPS without any
surgery, and it publishes DOTA checkpoints on Hugging Face
(`dl4eo/oriented-det-pretrained`). Adapter: `detectors/oriented_det.py`, the
only module that may import `oriented_det`.

| candidate | backbone | start | parameters |
|---|---|---|---|
| `rotated-fcos-r50` | ResNet-50 + FPN | `rotated_fcos_dota_le90_3x_riou`, 81.58 mAP50 on DOTA | 36.2M |
| `rotated-fcos-r18` | ResNet-18 + FPN | ImageNet backbone (torchvision); FPN and head from scratch | 19.7M |

R18 is the lightest backbone its constructor accepts; the FCOS head alone is
4.7M, so even that is 23x the own nano. These are references for what a
large model gets out of these data, not deployment candidates. Swapping in a
truly light backbone (MobileNet) would throw away the DOTA weights, which is
the only reason to bring the framework in; if a lighter backbone is ever
wanted, the place is the own candidate.

**Their model, our loop.** `model(images, targets)` returns their losses in
training and `{rboxes, scores, labels}` in inference, and that is all the
adapter uses of them. The dataset (area filter, border policy, `image_size`),
the deterministic loop, validation every `eval_every` epochs with our metrics,
`best.pt`/`last.pt`, `training.json` and `plot-training` are testbank's, like
for the DDGRCF port. Kept from their recipe: SGD (momentum 0.9, weight decay
1e-4, lr 0.0025), warmup capped at 10% of the iterations, step decay x0.1 at
2/3 and 11/12 of the epochs (their 24 and 33 of 36), gradient clipping at 35.
Not reproduced: their flips and their trainer. Inputs go RGB in [0, 1] with
ImageNet mean/std; boxes go in their `le90` convention (long side first,
angle in [-pi/2, pi/2)), converted from ours.

### Recipe

```bash
uv venv .venv-orienteddet --python 3.11
VIRTUAL_ENV=.venv-orienteddet uv pip install --only-binary=:all: oriented-det
VIRTUAL_ENV=.venv-orienteddet uv pip install -e ".[dev]" "numpy<2"
.venv-orienteddet/Scripts/testbank train rotated-fcos-r50 --epochs 1 --image-size 416 --name fcos_smoke
```

`--only-binary` because a transitive dependency (`stringzilla`, via
`albumentations`) has no wheel for every build and wants MSVC otherwise;
`numpy<2` is their pin, and the reason this is not the main environment.
Weights: `weights/rotated_fcos_r50_dota_le90_3x_riou.pth` (Git LFS, 145 MB:
their 289 MB checkpoint with the optimizer state stripped).

## Licenses

No AGPL or GPL code in production. The registry distinguishes **two
independent axes**:

- `production_ready` — license of the **code**. Ultralytics is AGPL-3.0, so it
  stays `False`: performance reference only, isolated behind the common
  interface and in an optional dependency group. A test checks that no module
  outside its adapter imports `ultralytics`.
- `restricted_pretrain` — license of the **pretraining data**. DOTA-v1.0 is
  distributed for academic use only and that reaches the derived weights.
  *Planned, not implemented: today the registry only carries
  `production_ready`, and the DOTA-pretrained runs are told apart by the
  `--pretrained` path frozen in their `config.yaml`.*

The two axes are separated on purpose: the idea is that each fit candidate is
instantiated in two variants, with and without DOTA pretraining, and `compare`
shows them as sibling rows with the column marked. Nothing is discarded up
front.

Datasets carry `license`, `production_ready` and `sources: list[str]` with the
license of each source, because the license of an aggregate does not override
that of its sources. `compare` marks a run as unfit if any dataset in its
chain is.

## Usage: environments and how to launch each candidate

There are **five environments**, not one, and the reason is always the same:
dependencies that do not fit together. Each candidate says which one it needs.

### Main environment — `.venv`

Everything that is not a model, plus the own head and Ultralytics.

```bash
uv venv --python 3.11
uv pip install -e ".[dev,torch]"          # base + torch (CPU)
uv pip install -e ".[ultralytics]"        # optional: the AGPL reference
pytest -q                                 # 493 tests
```

The `torch` it installs is **CPU**. On a Mac with M-series the own loop uses
MPS on its own; on a PC with NVIDIA the CUDA torch from
<https://pytorch.org/get-started/locally/> is needed, installed *in this same
environment*.

| Candidate | Environment | Command |
|---|---|---|
| `yolox-obb-nano` (857k) | main | `testbank train yolox-obb-nano --name nano_base` |
| `yolox-obb-tiny` (4.37M) | main | `testbank train yolox-obb-tiny --name tiny_base` |
| `yolox-obb-small` (7.75M) | main | `testbank train yolox-obb-small --name small_base` |
| `ultralytics-yolo-obb` | main + `[ultralytics]` | `testbank train ultralytics-yolo-obb --name ul_ref` |
| `yolox-obb-ddgrcf-port` (8.05M) | main | `testbank train yolox-obb-ddgrcf-port --pretrained <DOTA.pth>` |
| `rtmdet-r-tiny` | separate `.venv-rtmdet` | see below |
| `ppyoloe-r-s` (8.09M) | separate `.venv-paddle` | `testbank train ppyoloe-r-s --image-size 416` (§ *PP-YOLOE-R environment*) |
| `rotated-fcos-r50` (36.2M) / `-r18` (19.7M) | separate `.venv-orienteddet` | `testbank train rotated-fcos-r50 --image-size 416` (§ *Rotated FCOS environment*) |

`train` trains, evaluates on `valid` with testbank's metrics and leaves the
run in `runs/<date>_<name>/` with the frozen config. **The test is not
touched**: see *Sealed test* in § *Splits*.

**What you see while it trains.** One line per epoch with the learning rate
and every loss term, and every `detector.eval_every` epochs (default **5**,
always on the last one) an evaluation on `valid` with our metrics:

```
epoch 12/100  lr 8.13e-04  total 1.2345  box 0.4102  angle 0.0881  objectness 0.5011  classes 0.2351
  valid @ 15: precision 0.912  recall 0.874  map50 0.712  map50_95 0.498  fitness 0.519  coverage_p5 0.961  coverage_p10 0.973  <- best
```

The validation line reads like the Ultralytics one (P, R, mAP50, mAP50-95,
fitness) plus what decides the crop. `precision` and `recall` are read at
`metrics.report_confidence` (0.25), the operating point, not at the low
inference threshold AP needs — the same figure as `at_decision_confidence` in
`metrics.json`. `coverage_p10` sits beside `p5` because with ~140 truths in
`valid` the 5th percentile falls on the boundary between the undetected
(coverage 0) and the rest, and jumps between 0 and 0.9 with a single missed
banknote; the 10th moves smoothly and tells checkpoints apart. All of it goes
into `training.json` and the plot. During training the evaluation is *light*
(20 bootstrap resamples, so the CI is coarse); the full one runs at the end.

`best.pt` is the checkpoint with the best `detector.selection_metric`.
`fitness` by default, the number Ultralytics picks its `best.pt` by:
`0.1 * mAP50 + 0.9 * mAP50-95`, so the winner is the checkpoint whose boxes
fit tightest and not only the one that finds the most banknotes. `map50` and
`coverage_p5` are the alternatives (coverage saturates at 1.0 early and stops
telling checkpoints apart; the final comparison still reads coverage and
contamination). The note in the run says which epoch won and by what. `last.pt`
the final epoch, and `_train/training.json` is rewritten after **every** epoch
with the loss curve, the validation points and which epoch won — so a run that
dies at hour three still leaves its curve, and you can look at it while it
runs. This is checkpoint *selection*, not early stopping: the cosine schedule
runs whole, because stopping it halfway leaves the learning rate hanging where
it should have decayed. Own loop only; Ultralytics has its own (`patience`) and
RTMDet-R evaluates at the end. With `eval_every: 0` `best.pt` is just the last
epoch, and the run notes say so.

To see the curve, during or after the run:

```bash
testbank plot-training runs/<run>        # -> runs/<run>/viz/training.png
```

Two panels: losses per epoch on the left, validation metrics on the right with
the epoch that produced `best.pt` marked. Needs `matplotlib` (`dev` group).
Own candidates only: Ultralytics and RTMDet-R keep their own logs.

How many epochs: no number has been measured yet. Starting points for 351
images at batch 8 (44 steps per epoch): **100–150 with pretraining** (COCO or
DOTA), **200–300 from scratch**; the validation curve in `training.json` is
what says whether they are too many or too few.

Options valid for all:

```bash
--epochs N                  # overrides the config's; it is recorded
--image-size S              # input side for every candidate; the export is 416x416
--augmented vN              # adds the offline copies of data/augmented/vN (make-augmented) to train; § Augmentation
--augment                   # Ultralytics' on-the-fly recipe (own loop): mosaic, scale/translate, HSV, flips
--out-of-bounds clip|pad|keep   # border policy; clip by default
--loss-recipe own|yolox_obb_fork|ultralytics_obb|ddgrcf   # own head (ddgrcf: the port); § Recipes
--pretrained path.pth       # foreign checkpoint to START from; stays in the config
--weights path.pt           # re-evaluates some weights instead of training
```

And after several runs:

```bash
testbank compare --csv-out runs/table.csv
```

### RTMDet-R — `.venv-rtmdet`, separate

It requires torch 2.0 and a chain of versions that does not coexist with the
main environment. The full, verified recipe is in § *RTMDet-R environment*;
in short:

```bash
uv venv .venv-rtmdet --python 3.11
VIRTUAL_ENV=.venv-rtmdet uv pip install torch==2.0.0 torchvision==0.15.1 "numpy==1.26.4" packaging "setuptools<81"
VIRTUAL_ENV=.venv-rtmdet uv pip install --only-binary=:all: --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html mmcv==2.0.1
VIRTUAL_ENV=.venv-rtmdet uv pip install mmdet==3.1.0 "git+https://github.com/open-mmlab/mmrotate@dev-1.x" "numpy==1.26.4"
VIRTUAL_ENV=.venv-rtmdet uv pip install -e .
.venv-rtmdet/Scripts/testbank train rtmdet-r-tiny --name rtmdet_base
```

Its numbers come from torch 2.0 and the rest from torch 2.14: the table notes it.

### Before the baselines

- **`clip` is the default policy and returns rectangles** — with the original
  angle, inside the frame. `pad` and `keep` remain available by flag.
- What is in `runs/` to date are 1–2 epoch smoke runs. **None is a quality
  measurement.** It is worth emptying it, or naming the baselines so that
  `compare` does not mix them.
- The three own variants, the DDGRCF port and Ultralytics can be launched
  today, and use the GPU if there is one (`cuda` → `mps` → `cpu`;
  `TESTBANK_DEVICE` forces it). RTMDet-R needs its environment.
- **Decide the split.** The only version is `v1`, the export's, with 20% of
  the test contaminated by near-duplicates. `make-splits --repartition` writes
  a `v2` without leaks and stratified (§ *Splits*) and leaves `v1` intact;
  runs record their version and `compare` shows it, so mixing them is visible
  but still meaningless — pick one before the baselines.
- **Then the offline copies.** `make-augmented` is bound to a split version:
  make it after the split is decided (a version made from `v1` is refused by
  a run on `v2`), look at its contact sheet, and run the baselines in pairs —
  with and without `--augmented`, with and without `--augment`.

## Status

**Done.** Canonical quad with the five flip invariants. Strict `obb_yolo`
reader, which accumulates all the errors of a file instead of dying at the
first line. Structure detection and split materialization in both modes,
visibility check, relative area filter, border policy (`clip` / `pad` /
`keep`, with `clip` returning rectangles with the original angle),
near-duplicate detection in color, and opt-in re-splitting by groups and
strata with ratios and seed (`--repartition`). Three format converters
(`obb_yolo`, `dota`, `bbox_coco`) and two exporters (`dota`, `coco`). Metrics
with 101-point AP, rotated IoU via shapely and bootstrap by image. Experiment
engine with the config frozen in every run. Own OBB head on YOLOX in three
variants (857k / 4.37M / 7.75M parameters), deterministic training, angle
attenuator by ratio. Adapters: Ultralytics (reference,
`production_ready=False`), RTMDet-R, the three own variants, and
DDGRCF/YOLOX_OBB **as a pure-torch port** (`yolox-obb-ddgrcf-port`, trains on
CPU/MPS, loads its DOTA weights). Nine registered candidates (PP-YOLOE-R-s in its
own Paddle environment, inference verified, training pending `ext_op`;
Rotated FCOS R50/R18 from oriented-det in `.venv-orienteddet`, pure torch); the
two YOLOX-OBB clones were removed in the refactor (§ *Discarded*). The own head loads
**Megvii's COCO** (`--pretrained`). The own head trains with **three complete
loss recipes** (`--loss-recipe own | yolox_obb_fork | ultralytics_obb`), each
the one of a concrete network, on the same backbone; the port trains with its
own (`ddgrcf`).

**Real data in.** 502 images from Roboflow, 762 annotations. No longer
dependent on `tools/make_synthetic.py`, which is kept for the tests.

**Pending.**

- RTMDet-R baseline: the smoke epoch runs on CPU (~3 min/epoch at 416);
  100 epochs is ~5 hours here.
- Polygon masking for the crop, rectangular today (§ *Caveats*).
- Long baselines and the comparison table. Everything run so far is 1–2 epoch
  smoke runs to verify the plumbing does not leak, **not results**: no
  performance number in this repository is yet a quality measurement of the
  detector.
- The test set remains **sealed**. `evaluate-test` is the only path and leaves
  a record. On the export's split, 20% of the test has a near-duplicate in
  train; `--repartition` brings it to zero (§ *Caveats*).
