# CIFAR-10 with Dense Networks Only

A local, shareable web application for the Deep Learning capstone. It presents the whole 20-part project,
classifying CIFAR-10 with fully connected networks and **no convolution**, in four sections.

## The four tabs

| # | Tab | Covers | What is in it |
|---|-----|--------|---------------|
| 01 | **Before Optimization** | Parts 1 to 8 | The dataset and the committed split, the Part 3 baseline and why its hyperparameters were chosen, the shallow/medium/deep comparison, and the Part 8 overfitting diagnosis with its ruled-out alternatives. |
| 02 | **After Optimization** | Parts 9 to 16 | Learning rate, batch size and optimizer sweeps; the three regularizers; the Batch Normalization failure; the Dropout ablation; the final model's configuration; the sealed test-set result; **the Part 15 re-run** (see below); and the data-augmentation extra experiment. |
| 03 | **Insights & Reflection** | Parts 17 to 19 | The confusion matrix, the largest confusion pairs, easiest and hardest classes, the six findings that shaped the result, the full experiment log, the Part 15 confirmation, and the reflection. |
| 04 | **Deployment & CNN** | Part 20 | What would actually ship and what it is honestly fit for, what flattening costs, a Dense-against-CNN comparison, and this project's own evidence that the ceiling is architectural. |

## Where the numbers come from

Nothing in this app is estimated, rounded up, or typed from memory.

- **`app/curves.ts` is generated, not written.** The Parts 4 to 7 architecture curves are read from `../Results/*.json`.
  Every optimization run's curve is parsed back out of the stdout Keras printed during the notebook run in
  `../Notebooks/9_20_optimization_and_final_analysis.ipynb`. Regenerate it with `npm run curves` after re-running a
  notebook.
- **`app/data.ts` holds the summary figures**, each one traceable to a specific part of a specific notebook. The file's
  header comment names the sources.
- **`mobilenetSamples` is generated too.** `../scripts/sample_predictions.py` runs both saved models over the ten test
  images in `public/samples/` and writes `../Results/sample_predictions.json`. The CNN is the committed
  `../models/mobilenet_cifar10.keras`; the notebook never saved its Dense weights, so the script retrains the Part 15
  recipe from the same seed and reports what that rebuild scores (53.81% test, against the notebook's 54.08%).
- **Charts are inline SVG** drawn from those per-epoch arrays. There is no smoothing and no interpolation, so a kink in
  a curve is a kink in the run.
- **Derived figures are derived in the app**, not hardcoded: the confusion-pair ranking on tab 03 is computed from the
  confusion matrix, and the headline metrics on tab 02 are reduced from the run tables they summarise.

### The Part 15 re-run

Part 12 measured four regularizers and **Dropout won on every metric it reported**, but the first version of Part 15
shipped L2 and left Dropout out entirely. `../scripts/part15_rerun.py` was written to settle that: a 2×2 of Dropout
on/off crossed with L2 on/off, everything else identical, all four with Early Stopping and seed 42 reseeded before
every build so they start from the same weights.

**Its recommendation was adopted.** Part 14 is now a Dropout ablation and Part 15 ships Dropout 0.3 + Early Stopping,
so this experiment no longer reports a correction, it reports a confirmation.

| Configuration | Dropout | L2 | Epochs | Best val | Gap | Test |
|---|---|---|---|---|---|---|
| L2 only | off | 1e-4 | 19 | 51.86% | 0.1137 | 52.12% |
| Dropout only, **shipped** | 0.3 | off | 46 | 53.66% | 0.0408 | 53.51% |
| Dropout + L2, *highest val* | 0.3 | 1e-4 | 46 | 54.40% | 0.0334 | 53.90% |
| Early Stopping only, *control* | off | off | 20 | 51.46% | 0.1366 | 52.54% |

Read one contrast at a time (validation points, 1 s.e. ≈ 0.7):

- **L2 alone vs control: +0.40 pts.** Nothing.
- **Dropout alone vs control: +2.20 pts.** Three standard errors. This is the whole effect.
- **L2 on top of Dropout: +0.74 pts** (37 images). One standard error. Not a result.

*Highest val* and *shipped* are deliberately different rows: Dropout + L2 scores highest, but its lead is inside
noise, and Parts 4 to 7 already rejected extra depth on exactly that reasoning. Applying that rule consistently means
shipping the simpler model, which also has the lowest test loss of the four, at 1.2941.

Selection was made on validation and recorded before the test set was touched; no tuning followed.

#### A reproducibility caveat

Re-running the *identical* shipped configuration with the same seed gave 53.66% val / 53.51% test, against the
notebook's 54.62% / 54.08%. The seed fixes the initial weights and the shuffle order, but not the order floating-point
reductions accumulate across threads nor which kernels XLA fuses, so run-to-run drift is about the same size as the
differences Part 12 was ranking regularizers on. The four re-run configs
are compared against each other, never against the notebook's numbers.

### Two different generalization gaps

The project reports the gap two ways, and they are not interchangeable. Both are kept, and every place a gap appears is
labelled with which one it is:

- `GAP_PEAK` is final train accuracy minus **best** validation accuracy. Used by Parts 4 to 7 and Part 19.
- `GAP_FINAL` is final train accuracy minus **final-epoch** validation accuracy. Used by Parts 12 to 14.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Share on the same Wi-Fi

```bash
npm run share
```

Find your machine's local IP address, then have others open `http://YOUR-LOCAL-IP:3000`. Keep the terminal open while
they use it.

## Validate

```bash
npm run build      # production build
npm test           # build, then check the rendered HTML and the data invariants
npm run typecheck  # tsc --noEmit
npm run lint       # eslint
```

The test suite checks more than "it rendered". It asserts that the confusion matrix is a valid 10 × 10 whose rows each
total 1,000, that its diagonal reproduces both the per-class recalls and the reported 54.08% test accuracy, and that
every generated curve series is length-aligned, so a bad regeneration of `curves.ts` fails the build rather than
quietly drawing the wrong picture. It also re-derives every stated contrast in the Part 15 re-run from the raw
`../Results/part15_rerun.json`, so a claim in the UI cannot drift away from the number the experiment produced.

Two further tests close the gap that let wrong figures in once already. One recomputes every run's best validation
accuracy, best epoch, minimum validation loss, final training accuracy and gap straight from that run's own curve
and compares them with what `data.ts` states. The other re-derives the per-class precision, recall and F1 from the
confusion matrix, the weighted averages from the per-class values, the architecture spread from the three best
validation accuracies, and each augmentation gain from the control. A hand-typed number that does not match the
data it claims to summarise now fails `npm test`.

## Headline results

| Measure | Value |
|---|---|
| Best validation accuracy, any tuning run | 52.78% (optimizer experiment, Adam @ 0.0001) |
| Best validation accuracy, final model | 54.62% (Dropout 0.3 + Early Stopping) |
| Final test accuracy | 54.08% on 10,000 sealed images |
| Weighted test F1 | 53.99% |
| Best score recorded anywhere | 59.12% (flip + shift augmentation) |
| Smallest generalization gap | 4.47% (Dropout, down from 32.11%) |
| Worst run in the project | 44.64% (Batch Normalization, and the highest training accuracy at 99.20%) |
| Part 15 2×2 confirmation | Adding L2 on top of Dropout is worth +0.74 pts, one standard error, so not a result |

During the project proper, the test set was opened exactly once, after every architecture and hyperparameter
decision was final. The Part 15 re-run opened it a second time, after its own selection had been made on
validation and recorded, with no tuning following. Both openings are reported above rather than merged.

## The constraint

No CNN, no Conv2D, no pretrained weights, no transfer learning. Every image is flattened from 32 × 32 × 3 to 3,072 × 1
before it reaches the first layer. `src/models.py` in the parent project enforces this at build time, where `validate_model`
raises on any convolutional, recurrent or embedding layer, and on a `BatchNormalization` placed before `Flatten`, where
it would silently normalise 3 colour channels instead of 3,072 features.

Tab 04 is the argument that this constraint, not the hyperparameters, set the final number.

---

Khalid · Wasan · Building & Developing AI Models Bootcamp, Tuwaiq Academy
