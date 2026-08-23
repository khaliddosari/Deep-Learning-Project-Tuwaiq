// Every figure below is a value this project actually produced.
//
// Sources, in order of preference:
//   * Results/*.json                                    - Parts 4-7 architecture runs
//   * Notebooks/9_20_optimization_and_final_analysis.ipynb - Parts 9-20
//   * Notebooks/03-08_baseline_and_architectures.ipynb     - Parts 3-8
//
// Per-epoch curves live in ./curves.ts, which is generated from those same two
// places by scripts/generate-curves.py. Nothing here is estimated or rounded to
// look better than it was.
//
// Two different generalization gaps appear in the project, and they are NOT the
// same measurement. Both are kept, each labelled where it is shown:
//   GAP_PEAK  = final train accuracy - BEST validation accuracy  (Parts 4-7, 19)
//   GAP_FINAL = final train accuracy - FINAL validation accuracy (Parts 12-14)

export const GAP_PEAK = "final train accuracy minus best validation accuracy";
export const GAP_FINAL = "final train accuracy minus final-epoch validation accuracy";

export const navItems = [
  ["before", "01", "Before Optimization", "Parts 1 to 8"],
  ["after", "02", "After Optimization", "Parts 9 to 16"],
  ["insights", "03", "Insights & Reflection", "Parts 17 to 19"],
  ["deployment", "04", "Deployment & CNN", "Part 20"],
] as const;

export type ViewId = (typeof navItems)[number][0];

// ---------------------------------------------------------------- the dataset

export const classNames = [
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck",
] as const;

export const datasetFacts = [
  { label: "Images", value: "60,000", note: "50,000 train + 10,000 test" },
  { label: "Image shape", value: "32 × 32 × 3", note: "flattened to 3,072 inputs" },
  { label: "Classes", value: "10", note: "6,000 images each, perfectly balanced" },
  { label: "Random-guess floor", value: "10%", note: "the number every model must beat" },
];

export const splitRows = [
  { name: "X_train", shape: "(45000, 32, 32, 3)", role: "training images, pixels scaled to [0, 1]" },
  { name: "X_val", shape: "(5000, 32, 32, 3)", role: "stratified, 500 per class, drives every decision" },
  { name: "X_test", shape: "(10000, 32, 32, 3)", role: "sealed until Part 16, opened exactly once" },
];

// ---------------------------------------------------------------- Part 3

export const baseline = {
  architecture: "1 hidden layer × 512 units",
  params: 1_578_506,
  bestValAccuracy: 0.5222,
  minValLoss: 1.3877,
  finalTrainAccuracy: 0.6746,
  learningRate: 0.0001,
  batchSize: 128,
  epochs: 50,
  optimizer: "Adam",
};

/** Why the baseline's hyperparameters are what they are (Part 3 write-up). */
export const baselineChoices = [
  {
    setting: "512 neurons",
    reason:
      "Chosen by testing. At 32 units the model stalled at 19% accuracy with 31 of 32 hidden units dead. ReLU had collapsed the layer to a single effective neuron. 512 keeps the first hidden layer wide enough that it compresses the 3,072-value input only six-fold.",
  },
  {
    setting: "Learning rate 0.0001",
    reason:
      "The value the Part 9 sweep selects, adopted up front so every architecture comparison runs at the same setting the tuned model uses.",
  },
  {
    setting: "Batch size 128",
    reason:
      "Above the range swept in Part 10. At 45,000 training images it gives 352 weight updates per epoch.",
  },
  {
    setting: "50 epochs",
    reason:
      "Validation loss reaches its minimum near epoch 42. Extending to 100 epochs gained 0.4 points of validation accuracy while training accuracy gained 1.1. That is memorisation, not learning.",
  },
];

// ---------------------------------------------------------------- Parts 4-7

export type Architecture = {
  key: "shallow" | "medium" | "deep";
  name: string;
  layers: string;
  hiddenLayers: number;
  params: number;
  bestValAccuracy: number;
  bestValEpoch: number;
  minValLoss: number;
  minValLossEpoch: number;
  finalTrainAccuracy: number;
  finalValLoss: number;
  /** GAP_PEAK, the definition Parts 4-7 and Part 19 report. */
  gap: number;
  verdict: string;
  selected?: boolean;
};

export const architectures: Architecture[] = [
  {
    key: "shallow", name: "Shallow", layers: "1 × 512", hiddenLayers: 1,
    params: 1_578_506, bestValAccuracy: 0.5222, bestValEpoch: 44,
    minValLoss: 1.3877, minValLossEpoch: 42, finalTrainAccuracy: 0.6746,
    finalValLoss: 1.3896, gap: 0.1524,
    verdict:
      "Underfitting. Validation loss never meaningfully rises: it ends at 1.3896 against a minimum of 1.3877. Not memorising, just too small.",
  },
  {
    key: "medium", name: "Medium", layers: "3 × 512", hiddenLayers: 3,
    params: 2_103_818, bestValAccuracy: 0.5240, bestValEpoch: 30,
    minValLoss: 1.4024, minValLossEpoch: 13, finalTrainAccuracy: 0.8199,
    finalValLoss: 1.7835, gap: 0.2959,
    verdict:
      "Good fit for 13 epochs, then overfitting. Validation loss bottoms at 1.4024 and rises 27% to 1.7835 while training loss keeps falling.",
    selected: true,
  },
  {
    key: "deep", name: "Deep", layers: "5 × 512", hiddenLayers: 5,
    params: 2_629_130, bestValAccuracy: 0.5298, bestValEpoch: 25,
    minValLoss: 1.3968, minValLossEpoch: 12, finalTrainAccuracy: 0.8597,
    finalValLoss: 2.4324, gap: 0.3299,
    verdict:
      "Strongest overfitting. Validation loss bottoms at epoch 12 then rises 74% to 2.4324, the largest rise of the three, despite the highest training accuracy.",
  },
];

/** The spread across all three, in images out of the 5,000-image validation set. */
export const architectureSpread = {
  points: 0.76,
  images: 38,
  validationSize: 5000,
  standardError: 0.7,
};

// ---------------------------------------------------------------- Part 8

export const diagnosis = {
  model: "Medium: 3 hidden layers of 512 neurons, 2,103,818 parameters",
  verdict: "Overfitting, beginning around epoch 13",
  evidence: [
    "Validation loss reaches its minimum of 1.4024 at epoch 13 and rises steadily afterwards: 1.402, 1.412, 1.522, 1.655 and 1.784 at epochs 13, 20, 30, 40 and 50.",
    "Training loss falls across all 50 epochs, from 1.845 to 0.542, and training accuracy climbs from 0.343 to 0.820.",
    "The gap widens from 0.0759 at epoch 13, the lowest-validation-loss epoch, to 0.2993 at epoch 50.",
    "Validation accuracy peaks at 0.5240 on epoch 30 and then falls back to 0.5206.",
  ],
  ruledOut: [
    {
      label: "Not underfitting",
      why: "The model reaches 0.820 training accuracy and its training loss is still falling at epoch 50, so it clearly has the capacity to keep fitting.",
    },
    {
      label: "Not optimization instability",
      why: "The curves are smooth: epoch-to-epoch validation loss varies by only 0.016, with a largest single jump of 0.089.",
    },
  ],
  fix: "Early Stopping on validation loss, with the best weights restored.",
};

// ---------------------------------------------------------------- Part 9

export const learningRateRuns = [
  { value: 0.01, bestVal: 0.3864, bestValEpoch: 44, minValLoss: 1.7300, minValLossEpoch: 44, finalTrain: 0.3718, finalVal: 0.3440, finalValLoss: 1.7814 },
  { value: 0.001, bestVal: 0.4768, bestValEpoch: 12, minValLoss: 1.5077, minValLossEpoch: 6, finalTrain: 0.7597, finalVal: 0.4250, finalValLoss: 2.8688 },
  { value: 0.0001, bestVal: 0.5264, bestValEpoch: 25, minValLoss: 1.3979, minValLossEpoch: 18, finalTrain: 0.8136, finalVal: 0.5116, finalValLoss: 1.7912, selected: true },
];

export const learningRateVerdict =
  "0.0001 wins on both accuracy and loss. At 0.001 the model peaks by epoch 12 then diverges to a validation loss of 2.8688; at 0.01 it learns too aggressively and never exceeds 38.64%.";

// ---------------------------------------------------------------- Part 10

export const batchSizeRuns = [
  { value: 16, bestVal: 0.5058, bestValEpoch: 18, minValLoss: 1.4440, finalTrain: 0.9427, finalVal: 0.4838, finalValLoss: 3.4169 },
  { value: 32, bestVal: 0.5092, bestValEpoch: 14, minValLoss: 1.4249, finalTrain: 0.9196, finalVal: 0.4928, finalValLoss: 2.9443 },
  { value: 64, bestVal: 0.5204, bestValEpoch: 14, minValLoss: 1.4164, finalTrain: 0.8814, finalVal: 0.4986, finalValLoss: 2.4574 },
  { value: 128, bestVal: 0.5252, bestValEpoch: 21, minValLoss: 1.3948, finalTrain: 0.8322, finalVal: 0.4940, finalValLoss: 1.8806, selected: true },
];

export const batchSizeVerdict =
  "Accuracy improved at every step, but the gain from 64 to 128 is only 0.48 points, inside the noise of a 5,000-image validation set, so the advantage over 64 is not clearly established. The clear effect is on overfitting: batch 16 reached 94.27% training accuracy and a validation loss of 3.4169.";

// ---------------------------------------------------------------- Part 11

export const optimizerRuns = [
  { name: "Adam", learningRate: 0.0001, bestVal: 0.5278, bestValEpoch: 31, minValLoss: 1.3932, minValLossEpoch: 14, finalTrain: 0.8152, finalVal: 0.5254, finalValLoss: 1.7265, selected: true },
  { name: "SGD", learningRate: 0.01, bestVal: 0.4928, bestValEpoch: 50, minValLoss: 1.4807, minValLossEpoch: 34, finalTrain: 0.6492, finalVal: 0.4928, finalValLoss: 1.4998 },
];

export const optimizerVerdict =
  "The learning rates differ on purpose: Adam normalises its own step sizes, so matching them would cripple SGD. Adam wins on best accuracy, on minimum validation loss, and on speed: it peaks at epoch 31 while SGD is still climbing at epoch 50. SGD's lower final validation loss reflects how far Adam kept overfitting after its best epoch, not a better model.";

// ---------------------------------------------------------------- Part 12

export type RegularizationRun = {
  name: string;
  detail: string;
  finalTrain: number;
  finalVal: number;
  /** GAP_FINAL, the definition Parts 12-14 report. */
  gap: number;
  finalValLoss: number;
  bestVal: number;
  minValLoss: number;
  selected?: boolean;
};

export const regularizationRuns: RegularizationRun[] = [
  { name: "Baseline", detail: "no regularization", finalTrain: 0.8199, finalVal: 0.4988, gap: 0.3211, finalValLoss: 1.8616, bestVal: 0.5196, minValLoss: 1.3889 },
  { name: "Dropout", detail: "rate 0.3 after each hidden block", finalTrain: 0.5863, finalVal: 0.5416, gap: 0.0447, finalValLoss: 1.3022, bestVal: 0.5416, minValLoss: 1.2991, selected: true },
  { name: "Early Stopping", detail: "monitors validation loss, restores best weights", finalTrain: 0.6259, finalVal: 0.5172, gap: 0.1087, finalValLoss: 1.4264, bestVal: 0.5204, minValLoss: 1.3934 },
  { name: "L2", detail: "strength 0.0001 on every hidden kernel", finalTrain: 0.7983, finalVal: 0.5230, gap: 0.2753, finalValLoss: 1.8350, bestVal: 0.5284, minValLoss: 1.5364 },
];

export const regularizationVerdict =
  "Dropout is the only technique that closes the gap rather than trimming it: 32.11% down to 4.47%, with the lowest final validation loss of the four. L2 barely moves it.";

// ---------------------------------------------------------------- Part 13

export const batchnormRuns = [
  { name: "Without BatchNorm", finalTrain: 0.8100, finalVal: 0.5138, gap: 0.2962, finalValLoss: 1.8096, bestVal: 0.5318, minValLoss: 1.3930, selected: true },
  { name: "With BatchNorm", finalTrain: 0.9920, finalVal: 0.4070, gap: 0.5850, finalValLoss: 4.4448, bestVal: 0.4464, minValLoss: 1.6881 },
];

export const batchnormVerdict =
  "It backfired. Batch Normalization drove training accuracy to 99.20% and validation accuracy down to 40.70%, a 58.50-point gap and a final validation loss of 4.4448. This is the clearest demonstration in the project that high training accuracy is not evidence of a good model, and it is why BatchNorm was excluded from the final build.";

// ---------------------------------------------------------------- Part 14

export const ablationRuns = [
  { name: "With Dropout", finalTrain: 0.5914, finalVal: 0.5440, gap: 0.0474, finalValLoss: 1.3026, bestVal: 0.5462, minValLoss: 1.2971, selected: true },
  { name: "Without Dropout", finalTrain: 0.8180, finalVal: 0.5104, gap: 0.3076, finalValLoss: 1.8046, bestVal: 0.5284, minValLoss: 1.3955 },
];

export const ablationVerdict =
  "Dropout raises the peak and closes the gap at the same time. Final validation accuracy rises from 51.04% to 54.40% and the gap narrows from 30.76% to 4.74%, while best validation accuracy goes from 52.84% to 54.62%, which is 89 images out of 5,000. Both models were reseeded before being built, so they start from identical weights and Dropout is the only variable. This is the evidence that carries Dropout into the final model.";

// ---------------------------------------------------------------- Part 15

export const finalModel = {
  architecture: "3 hidden layers × 512 units",
  params: 2_103_818,
  nonTrainable: 0,
  sizeMB: 8.03,
  epochsRun: 50,
  epochsAllowed: 50,
  settings: [
    { name: "Optimizer", value: "Adam", why: "Part 11: higher best accuracy, lower minimum loss, faster convergence" },
    { name: "Learning rate", value: "0.0001", why: "Part 9: 52.64% against 47.68% and 38.64%" },
    { name: "Batch size", value: "128", why: "Part 10: highest accuracy and lowest minimum validation loss of the four" },
    { name: "Dropout", value: "0.3", why: "Parts 12 and 14: ranked first of four regularizers, and the ablation confirms it in isolation" },
    { name: "Early Stopping", value: "on validation loss", why: "Part 8: recovers the best epoch instead of the last one" },
    { name: "L2", value: "excluded", why: "Part 12: left the generalization gap almost unchanged when tested on its own" },
    { name: "Batch Normalization", value: "excluded", why: "Part 13: dropped best validation accuracy from 53.18% to 44.64%" },
  ],
};

// ---------------------------------------------------------------- Part 16

export const testResults = {
  accuracy: 0.5408,
  loss: 1.2815,
  precision: 0.5421,
  recall: 0.5408,
  f1: 0.5399,
  size: 10_000,
};

export type ClassResult = {
  id: number;
  name: string;
  precision: number;
  recall: number;
  f1: number;
};

export const perClass: ClassResult[] = [
  { id: 0, name: "airplane", precision: 0.6207, recall: 0.607, f1: 0.6138 },
  { id: 1, name: "automobile", precision: 0.6497, recall: 0.662, f1: 0.6558 },
  { id: 2, name: "bird", precision: 0.3944, recall: 0.461, f1: 0.4251 },
  { id: 3, name: "cat", precision: 0.3731, recall: 0.347, f1: 0.3596 },
  { id: 4, name: "deer", precision: 0.4504, recall: 0.499, f1: 0.4734 },
  { id: 5, name: "dog", precision: 0.4922, recall: 0.379, f1: 0.4282 },
  { id: 6, name: "frog", precision: 0.5668, recall: 0.624, f1: 0.5940 },
  { id: 7, name: "horse", precision: 0.6254, recall: 0.571, f1: 0.5970 },
  { id: 8, name: "ship", precision: 0.6445, recall: 0.678, f1: 0.6608 },
  { id: 9, name: "truck", precision: 0.6042, recall: 0.580, f1: 0.5918 },
];

/** Rows are the true class, columns the predicted class. 1,000 test images per row. */
export const confusionMatrix: number[][] = [
  [607, 28, 98, 13, 36, 11, 20, 23, 123, 41],
  [35, 662, 18, 17, 12, 10, 13, 17, 68, 148],
  [58, 13, 461, 76, 153, 59, 89, 51, 19, 21],
  [31, 18, 111, 347, 68, 166, 143, 49, 23, 44],
  [45, 8, 168, 58, 499, 31, 94, 62, 25, 10],
  [18, 9, 117, 217, 82, 379, 74, 64, 22, 18],
  [3, 17, 92, 74, 115, 30, 624, 16, 13, 16],
  [35, 14, 71, 72, 101, 65, 21, 571, 13, 37],
  [110, 68, 17, 30, 27, 9, 5, 11, 678, 45],
  [36, 182, 16, 26, 15, 10, 18, 49, 68, 580],
];

// ---------------------------------------------------------------- Extra experiment

export const augmentationRuns = [
  { name: "No Augmentation", transforms: "none (control)", bestVal: 0.5284, bestValEpoch: 21, minValLoss: 1.3955, finalTrain: 0.9295, gap: 0.4085, change: 0, images: 0 },
  { name: "Flip only", transforms: "random horizontal flip", bestVal: 0.5492, bestValEpoch: 27, minValLoss: 1.3141, finalTrain: 0.8831, gap: 0.3631, change: 2.08, images: 104 },
  { name: "Flip + Shift", transforms: "random horizontal flip + random shift up to 10% (≈3 px)", bestVal: 0.5912, bestValEpoch: 79, minValLoss: 1.2057, finalTrain: 0.6693, gap: 0.0791, change: 6.28, images: 314, selected: true },
];

export const augmentationEpochs = 100;

export const augmentationVerdict =
  "The strongest single result in the project, and it cost no parameters. Augmentation layers sit before Flatten, so they transform the image while it still has spatial structure, and they are active only during fit, so validation and test are always measured on unmodified images.";

// ---------------------------------------------------------------- Part 15, re-run

// The experiment that decided which regularizer the final model ships.
//
// Part 12 measured four regularizers and Dropout won on every metric it
// reported, but the first version of Part 15 shipped L2 and left Dropout out
// entirely. scripts/part15_rerun.py settled it with a 2x2: Dropout on/off
// crossed with L2 on/off, everything else identical, all four with Early
// Stopping and the same seed.
//
// That recommendation was adopted. Part 14 is now a Dropout ablation and
// Part 15 ships Dropout 0.3 + Early Stopping, so this experiment confirms the
// shipped model rather than correcting it.
//
// Results/part15_rerun.json holds the raw output; the per-epoch curves are in
// ./curves.ts under `part15Rerun`.

export type RerunRun = {
  key: "l2" | "dropout" | "both" | "neither";
  label: string;
  l2: number | null;
  dropout: number | null;
  epochsRun: number;
  stoppedEarly: boolean;
  bestVal: number;
  bestValEpoch: number;
  minValLoss: number;
  finalTrain: number;
  /** GAP_PEAK, so it is comparable with the Parts 4-7 numbers. */
  gap: number;
  test: number;
  testLoss: number;
  /** The configuration Part 15 ships. */
  shipped?: boolean;
  /** Highest best-validation accuracy of the four. */
  winner?: boolean;
};

export const rerunRuns: RerunRun[] = [
  { key: "l2", label: "L2 only", l2: 0.0001, dropout: null, epochsRun: 19, stoppedEarly: true, bestVal: 0.5186, bestValEpoch: 16, minValLoss: 1.5338, finalTrain: 0.6323, gap: 0.1137, test: 0.5212, testLoss: 1.4946 },
  { key: "dropout", label: "Dropout only", l2: null, dropout: 0.3, epochsRun: 46, stoppedEarly: true, bestVal: 0.5366, bestValEpoch: 44, minValLoss: 1.3177, finalTrain: 0.5774, gap: 0.0408, test: 0.5351, testLoss: 1.2941, shipped: true },
  { key: "both", label: "Dropout + L2", l2: 0.0001, dropout: 0.3, epochsRun: 46, stoppedEarly: true, bestVal: 0.5440, bestValEpoch: 44, minValLoss: 1.4055, finalTrain: 0.5774, gap: 0.0334, test: 0.5390, testLoss: 1.3843, winner: true },
  { key: "neither", label: "Early Stopping only", l2: null, dropout: null, epochsRun: 20, stoppedEarly: true, bestVal: 0.5146, bestValEpoch: 15, minValLoss: 1.4073, finalTrain: 0.6512, gap: 0.1366, test: 0.5254, testLoss: 1.3653 },
];

/** The 2x2 read one contrast at a time. Deltas are validation points. */
export const rerunContrasts = [
  { question: "Does L2 do anything on its own?", pair: "L2 only vs control", points: 0.4, images: 20, verdict: "no" as const, note: "Inside the noise of a 5,000-image validation set, where one standard error is about 0.7 points." },
  { question: "Does Dropout do anything on its own?", pair: "Dropout only vs control", points: 2.2, images: 110, verdict: "yes" as const, note: "Three standard errors. This is the effect the whole comparison turns on." },
  { question: "Does L2 add anything on top of Dropout?", pair: "Dropout + L2 vs Dropout only", points: 0.74, images: 37, verdict: "no" as const, note: "One standard error. By the same standard Parts 4 to 7 used to reject extra depth, this is not a result." },
];

export const rerunProtocol = {
  shared: "[512, 512, 512] Dense · Adam @ 0.0001 · batch 128 · 100-epoch cap · EarlyStopping(val_loss, patience 5, restore_best_weights) · seed 42 reseeded before every build. Early Stopping fired in all four arms, so the cap never bound",
  selectedOn: "best validation accuracy, the quantity restore_best_weights recovers",
  testDiscipline:
    "The winner was chosen on validation and that decision recorded before the test set was touched. No tuning followed. All four test scores are shown for transparency, not because they were used to choose.",
};

/** Same config, same seed, two runs, 0.57 points apart on test. */
export const reproducibilityNote = {
  notebookVal: 0.5462,
  notebookTest: 0.5408,
  rerunVal: 0.5366,
  rerunTest: 0.5351,
  why:
    "Seeding fixes the initial weights and the shuffle order, but not the order in which floating-point reductions are accumulated across threads, nor the kernels XLA chooses to fuse. Run-to-run drift is therefore about the same size as the differences Part 12 was ranking regularizers on, which is the strongest argument in the project for reading small gaps as noise.",
};

export const rerunVerdict = {
  headline: "Dropout is the entire effect. L2 is not measurable either alone or on top of it.",
  mechanism:
    "Without Dropout, validation loss bottoms at epoch 14 or 15 and Early Stopping fires by epoch 19 or 20, so the model has stopped learning anything that generalises. With Dropout it keeps improving to epoch 44 and runs 46 epochs, and the generalization gap falls from the 0.11 to 0.14 band down to the 0.03 to 0.04 band.",
  recommendation:
    "Ship Dropout 0.3 plus Early Stopping, which is what Part 15 now does. Dropout + L2 scores 0.74 points higher, but that is one standard error, and Parts 4 to 7 already rejected extra depth on exactly that reasoning, so consistency says take the simpler model. It also has the lowest test loss of the four at 1.2941.",
  honest:
    "The notebook adopted this. Part 14 is now a Dropout ablation and Part 15 builds Dropout 0.3 + Early Stopping, reaching 54.62% validation and 54.08% test. This 2×2 is the evidence behind that choice, not a correction still pending.",
};

// ---------------------------------------------------------------- Part 18

export const experimentLog = [
  { name: "Baseline", architecture: "Medium (3×512)", lr: "0.0001", batch: "128", optimizer: "Adam", dropout: "No", batchnorm: "No", l2: "No", earlyStop: "No", bestVal: 0.5196 },
  { name: "Shallow", architecture: "Shallow (1×512)", lr: "-", batch: "-", optimizer: "-", dropout: "No", batchnorm: "No", l2: "No", earlyStop: "No", bestVal: 0.5222 },
  { name: "Medium", architecture: "Medium (3×512)", lr: "-", batch: "-", optimizer: "-", dropout: "No", batchnorm: "No", l2: "No", earlyStop: "No", bestVal: 0.5240 },
  { name: "Deep", architecture: "Deep (5×512)", lr: "-", batch: "-", optimizer: "-", dropout: "No", batchnorm: "No", l2: "No", earlyStop: "No", bestVal: 0.5298 },
  { name: "LR Tuning", architecture: "Medium (3×512)", lr: "0.0001", batch: "128", optimizer: "Adam", dropout: "No", batchnorm: "No", l2: "No", earlyStop: "No", bestVal: 0.5264 },
  { name: "Batch Tuning", architecture: "Medium (3×512)", lr: "0.0001", batch: "128", optimizer: "Adam", dropout: "No", batchnorm: "No", l2: "No", earlyStop: "No", bestVal: 0.5252 },
  { name: "Optimizer (Adam)", architecture: "Medium (3×512)", lr: "0.0001", batch: "128", optimizer: "Adam", dropout: "No", batchnorm: "No", l2: "No", earlyStop: "No", bestVal: 0.5278 },
  { name: "Optimizer (SGD)", architecture: "Medium (3×512)", lr: "0.01", batch: "128", optimizer: "SGD", dropout: "No", batchnorm: "No", l2: "No", earlyStop: "No", bestVal: 0.4928 },
  { name: "Dropout", architecture: "Medium (3×512)", lr: "0.0001", batch: "128", optimizer: "Adam", dropout: "0.3", batchnorm: "No", l2: "No", earlyStop: "No", bestVal: 0.5416 },
  { name: "Early Stopping", architecture: "Medium (3×512)", lr: "0.0001", batch: "128", optimizer: "Adam", dropout: "No", batchnorm: "No", l2: "No", earlyStop: "Yes", bestVal: 0.5204 },
  { name: "L2", architecture: "Medium (3×512)", lr: "0.0001", batch: "128", optimizer: "Adam", dropout: "No", batchnorm: "No", l2: "0.0001", earlyStop: "No", bestVal: 0.5284 },
  { name: "BatchNorm", architecture: "Medium (3×512)", lr: "0.0001", batch: "128", optimizer: "Adam", dropout: "No", batchnorm: "Yes", l2: "No", earlyStop: "No", bestVal: 0.4464 },
  { name: "Final Model", architecture: "Medium (3×512)", lr: "0.0001", batch: "128", optimizer: "Adam", dropout: "0.3", batchnorm: "No", l2: "No", earlyStop: "Yes", bestVal: 0.5462, selected: true },
];

// ---------------------------------------------------------------- Part 19

export const findings = [
  { n: "01", title: "Depth did almost nothing", detail: "0.76 points across 1, 3 and 5 hidden layers, which is 38 images out of 5,000, inside the noise of the validation set." },
  { n: "02", title: "Width is where the parameters live", detail: "The first hidden layer holds 1,573,376 of the Medium model's 2,103,818 parameters: 75% of the network sits between the input and layer one." },
  { n: "03", title: "Every architecture overfit", detail: "Training accuracy reached 82% while validation sat at 52%. The problem was never capacity." },
  { n: "04", title: "Regularization beat architecture", detail: "Dropout closed the gap from 32.11% to 4.47%, more than any change of depth achieved." },
  { n: "05", title: "BatchNorm made it worse", detail: "99.20% training accuracy against 40.70% validation. The highest-scoring model on training data was the worst model in the project." },
  { n: "06", title: "Augmentation won outright", detail: "59.12% best validation accuracy with flip + shift, the highest number recorded anywhere in the project, at zero parameter cost." },
];

export const reflection = [
  { label: "Hardest", text: "Telling a real gain apart from validation-set noise. On 5,000 images the standard error is about 0.7 points, so anything under roughly 1.4 points is not a result." },
  { label: "Best judgement", text: "Selecting on the generalization gap once accuracy had effectively tied. Three architectures within 38 images of each other are not ranked by accuracy." },
  { label: "Next time", text: "Start with augmentation, not with depth. The single largest gain in the project came from more spatial variety, not more parameters." },
];

export const conclusion =
  "The best model is not the model with the highest training accuracy. The final decision was made on validation performance, generalization behaviour, learning curves and controlled experiments, while respecting the project restriction against CNNs and transfer learning.";

// ---------------------------------------------------------------- Part 20 + deployment

export const deploymentCard = {
  what: "The Medium network: 3 × 512 Dense, Adam at 0.0001, batch 128, Dropout 0.3, Early Stopping",
  params: 2_103_818,
  sizeMB: 8.03,
  accuracy: 0.5408,
  inputContract: "One 32 × 32 RGB image, pixels scaled to [0, 1], flattened to 3,072 values by the model's own first layer.",
  outputContract: "A softmax vector of 10 probabilities summing to 1, one per CIFAR-10 class.",
};

export const deploymentGuidance = [
  {
    tone: "yes" as const,
    title: "Ship it as a teaching artefact",
    detail: "The model, the split and the 13 tracked runs behind it are reproducible from a fixed seed. That is what this application demonstrates.",
  },
  {
    tone: "no" as const,
    title: "Do not ship it as a vision product",
    detail: "54.08% test accuracy means roughly one prediction in two is wrong. No downstream decision should depend on a single output from this model.",
  },
  {
    tone: "no" as const,
    title: "Do not trust it on animals",
    detail: "Cat scores 0.36 F1 and dog 0.43. Cat and dog alone account for 383 of the model's errors, the largest confusion pair in the matrix.",
  },
  {
    tone: "next" as const,
    title: "The next version is a CNN",
    detail: "The ceiling here is architectural, not a tuning failure. Every remaining gain identified in this project points at preserving spatial structure.",
  },
];

export const flatteningCosts = [
  {
    question: "What happens to the spatial structure after flattening?",
    answer:
      "It is removed. 32 × 32 × 3 becomes 3,072 × 1. Every pixel value survives, but their arrangement does not, because the model receives a long vector of numbers rather than a structured image.",
  },
  {
    question: "Does a Dense network know which pixels are neighbours?",
    answer:
      "No. Each input pixel is an individual feature wired to the next layer. The network can learn relationships between pixel values, but nothing tells it that one pixel sits directly above, below or beside another.",
  },
  {
    question: "Why does that make image classification harder?",
    answer:
      "Because the patterns that identify objects, such as edges, corners, textures, shapes and the relationships between nearby regions, are local and spatial. A Dense layer has to relearn each of them separately at every position, which costs parameters and invites overfitting.",
  },
  {
    question: "What would an architecture built for images preserve?",
    answer:
      "Spatial structure and local relationships: detect local patterns, keep neighbouring pixels adjacent, learn hierarchical features from edges to shapes to objects, and recognise a pattern regardless of where in the image it appears.",
  },
];

export const cnnComparison = [
  {
    aspect: "Input handling",
    dense: "Flattens 32 × 32 × 3 into 3,072 independent values before the first layer sees anything.",
    cnn: "Keeps the 32 × 32 × 3 grid intact and slides filters across it.",
  },
  {
    aspect: "Spatial awareness",
    dense: "None. Adjacency is not represented anywhere in the model.",
    cnn: "Built in. A filter only ever looks at a local neighbourhood of pixels.",
  },
  {
    aspect: "Translation invariance",
    dense: "None. Shift an object three pixels and almost every one of the 3,072 inputs changes.",
    cnn: "The same filter is applied at every position, so a shifted object is still recognised.",
  },
  {
    aspect: "Parameter use",
    dense: "One weight per pixel position. 1,573,376 parameters between the input and the first hidden layer alone.",
    cnn: "Filters are reused across the whole image, so far fewer weights cover the same field of view.",
  },
  {
    aspect: "Feature hierarchy",
    dense: "Flat. Every hidden layer sees the same undifferentiated 3,072-value vector.",
    cnn: "Layered: edges, then shapes, then objects, each built from the layer beneath it.",
  },
  {
    aspect: "Response to augmentation",
    dense: "A shifted image is a nearly unrelated input vector, so the network must learn the same object a second time.",
    cnn: "Absorbs flips and shifts naturally, because the filter that matched before still matches after.",
  },
];

export const mobilenetBenchmark = {
  name: "MobileNetV2 CNN (ImageNet Pretrained + Augmented Cosine Fine-Tuning)",
  testAccuracy: 0.9124,
  testLoss: 0.7092,
  weightedF1: 0.9120,
  params: 2_593_610,
  epochs: 6,
  gainPoints: 37.16,
  trainingMinutes: 13.6,
  latencyMs: 2.0,
  gradioAppCommand: "python app.py",
  // A gradio.live share URL expires after 72 hours, so none is hard-coded here.
  // app.py prints a fresh one every time it starts.
};

export interface SampleTrial {
  name: string;
  src: string;
  /** Index into the 10,000-image test set, so the picture can be traced back. */
  testIndex: number;
  densePred: string;
  denseConf: number;
  denseCorrect: boolean;
  /** Test-set F1 for this class from the shipped Dense model (Part 16). */
  denseF1: number;
  cnnPred: string;
  cnnConf: number;
  cnnCorrect: boolean;
  /** Test-set F1 for this class from the fine-tuned MobileNetV2. */
  cnnF1: number;
}

/** One test image per class, run through both models.
 *
 *  Generated by scripts/sample_predictions.py into Results/sample_predictions.json.
 *  The CNN is the committed models/mobilenet_cifar10.keras. The Dense weights are
 *  the Part 15 recipe retrained from the same seed, because the notebook never
 *  saved its model; that rebuild scores 53.81% on the test set against the
 *  notebook's 54.08%, which is the same run-to-run drift reproducibilityNote
 *  documents. Confidences are the softmax probability of the predicted class.
 */
export const mobilenetSamples: SampleTrial[] = [
  { name: "airplane", src: "/samples/airplane.png", testIndex: 3, densePred: "ship", denseConf: 27.3, denseCorrect: false, denseF1: 0.6138, cnnPred: "airplane", cnnConf: 84.6, cnnCorrect: true, cnnF1: 0.9278 },
  { name: "automobile", src: "/samples/automobile.png", testIndex: 6, densePred: "dog", denseConf: 48.9, denseCorrect: false, denseF1: 0.6558, cnnPred: "automobile", cnnConf: 87.4, cnnCorrect: true, cnnF1: 0.9607 },
  { name: "bird", src: "/samples/bird.png", testIndex: 25, densePred: "frog", denseConf: 34.8, denseCorrect: false, denseF1: 0.4251, cnnPred: "bird", cnnConf: 85.0, cnnCorrect: true, cnnF1: 0.9102 },
  { name: "cat", src: "/samples/cat.png", testIndex: 0, densePred: "cat", denseConf: 52.2, denseCorrect: true, denseF1: 0.3596, cnnPred: "cat", cnnConf: 86.5, cnnCorrect: true, cnnF1: 0.8219 },
  { name: "deer", src: "/samples/deer.png", testIndex: 22, densePred: "airplane", denseConf: 56.0, denseCorrect: false, denseF1: 0.4734, cnnPred: "deer", cnnConf: 86.8, cnnCorrect: true, cnnF1: 0.9045 },
  { name: "dog", src: "/samples/dog.png", testIndex: 12, densePred: "frog", denseConf: 32.1, denseCorrect: false, denseF1: 0.4282, cnnPred: "dog", cnnConf: 90.7, cnnCorrect: true, cnnF1: 0.86 },
  { name: "frog", src: "/samples/frog.png", testIndex: 4, densePred: "deer", denseConf: 63.2, denseCorrect: false, denseF1: 0.5940, cnnPred: "frog", cnnConf: 89.1, cnnCorrect: true, cnnF1: 0.9084 },
  { name: "horse", src: "/samples/horse.png", testIndex: 13, densePred: "horse", denseConf: 65.9, denseCorrect: true, denseF1: 0.5970, cnnPred: "horse", cnnConf: 96.2, cnnCorrect: true, cnnF1: 0.9326 },
  { name: "ship", src: "/samples/ship.png", testIndex: 1, densePred: "ship", denseConf: 50.1, denseCorrect: true, denseF1: 0.6608, cnnPred: "ship", cnnConf: 85.1, cnnCorrect: true, cnnF1: 0.9502 },
  { name: "truck", src: "/samples/truck.png", testIndex: 11, densePred: "truck", denseConf: 79.6, denseCorrect: true, denseF1: 0.5918, cnnPred: "truck", cnnConf: 91.8, cnnCorrect: true, cnnF1: 0.9438 },
];

/** How the two models did across those ten images. */
export const sampleTrialScore = {
  denseCorrect: 4,
  cnnCorrect: 10,
  total: 10,
  denseRebuildTestAccuracy: 0.5381,
  shippedDenseTestAccuracy: 0.5408,
};

/** The project's own evidence that the ceiling is architectural. */
export const cnnEvidence = [
  {
    label: "Augmentation pointed the same way",
    detail: "Adding spatial variety, flip and shift, bought +6.28 points and closed the gap from 0.4085 to 0.0791. More spatial information helped where more parameters had not.",
  },
  {
    label: "Confusion is spatial",
    detail: "Cat/dog (383 errors) and automobile/truck (330) are classes that differ in shape and local texture, which is exactly the information flattening discards.",
  },
  {
    label: "Tuning had run out",
    detail: "Thirteen tracked runs across depth, learning rate, batch size, optimizer, dropout, L2, early stopping and batch normalization moved the ceiling by a few points at most.",
  },
];

export const finalReflection =
  "This project showed that a fully connected Dense network can perform image classification, but it is not designed to understand images. Flattening a 32 × 32 × 3 image into 3,072 values removes the explicit spatial structure and makes local visual patterns harder to learn. The limitations measured here are a clear motivation for architectures that preserve spatial information, a natural transition to Computer Vision and Convolutional Neural Networks.";

// ---------------------------------------------------------------- credits

export const project = {
  title: "CIFAR-10 with Dense Networks Only",
  question: "How far can a fully connected network see, with no convolution allowed?",
  rule: "No CNN, no Conv2D, no pretrained weights, no transfer learning.",
  team: ["Khalid", "Wasan"],
  course: "Building & Developing AI Models Bootcamp",
  org: "Tuwaiq Academy",
  seed: 42,
};

// ---------------------------------------------------------------- notebook plots

export type NotebookPlot = {
  id: string;
  file: string;
  tab: "before" | "after" | "insights" | "deployment";
  section: string;
  part: string;
  title: string;
  caption: string;
};

export const notebookPlots: readonly NotebookPlot[] = [
  {
    id: "nb1_class_distribution",
    file: "nb1_class_distribution.png",
    tab: "before",
    section: "data_split",
    part: "Part 1",
    title: "Examples per class (Train vs Test)",
    caption: "Balanced distribution of 5,000 training and 1,000 test images for each of the 10 CIFAR-10 classes.",
  },
  {
    id: "nb1_pixel_raw",
    file: "nb1_pixel_raw.png",
    tab: "before",
    section: "data_split",
    part: "Part 1",
    title: "Raw pixel-intensity distribution (2,000 sample images)",
    caption: "Distribution of unnormalized pixel intensities [0, 255] across 2,000 random training images (mean 120.71, std 64.15).",
  },
  {
    id: "nb1_pixel_normalized",
    file: "nb1_pixel_normalized.png",
    tab: "before",
    section: "data_split",
    part: "Part 2",
    title: "Normalized pixel-intensity distribution (2,000 sample images)",
    caption: "Distribution of scaled pixel values [0.0, 1.0] after division by 255.0 (mean 0.47, std 0.25).",
  },
  {
    id: "nb1_random_samples",
    file: "nb1_random_samples.png",
    tab: "before",
    section: "data_split",
    part: "Part 1",
    title: "15 random training images",
    caption: "Random 32×32 pixel images sampled from the training set with their corresponding class labels.",
  },
  {
    id: "nb1_class_examples",
    file: "nb1_class_examples.png",
    tab: "before",
    section: "data_split",
    part: "Part 1",
    title: "One example of each of the 10 classes",
    caption: "First training example from each of the 10 CIFAR-10 classes illustrating visual diversity and low resolution.",
  },
  {
    id: "nb2_baseline_curves",
    file: "nb2_baseline_curves.png",
    tab: "before",
    section: "baseline",
    part: "Part 3",
    title: "Baseline - 1 hidden layer, 128 units",
    caption: "Training vs validation loss and accuracy curves for the initial 1-hidden-layer 128-unit baseline with min val loss marker.",
  },
  {
    id: "nb2_shallow_curves",
    file: "nb2_shallow_curves.png",
    tab: "before",
    section: "architectures",
    part: "Part 5",
    title: "Shallow: 1 hidden layer, 128 units",
    caption: "Loss and accuracy learning curves for the shallow architecture (min val loss at epoch 18).",
  },
  {
    id: "nb2_medium_curves",
    file: "nb2_medium_curves.png",
    tab: "before",
    section: "architectures",
    part: "Part 6",
    title: "Medium: 2 hidden layers, 128 units",
    caption: "Loss and accuracy learning curves for the medium architecture (min val loss at epoch 13).",
  },
  {
    id: "nb2_deep_curves",
    file: "nb2_deep_curves.png",
    tab: "before",
    section: "architectures",
    part: "Part 7",
    title: "Deep: 3 hidden layers, 128 units",
    caption: "Loss and accuracy learning curves for the deep architecture (min val loss at epoch 12).",
  },
  {
    id: "nb2_all_architectures",
    file: "nb2_all_architectures.png",
    tab: "before",
    section: "architectures",
    part: "Part 7",
    title: "Validation loss & accuracy - all architectures",
    caption: "Direct comparison of validation loss and validation accuracy across shallow, medium, and deep networks.",
  },
  {
    id: "nb3_medium_baseline_loss",
    file: "nb3_medium_baseline_loss.png",
    tab: "before",
    section: "diagnosis",
    part: "Part 8",
    title: "Medium Baseline - Loss",
    caption: "Training vs validation loss for the medium baseline showing overfitting after epoch 13.",
  },
  {
    id: "nb3_medium_baseline_accuracy",
    file: "nb3_medium_baseline_accuracy.png",
    tab: "before",
    section: "diagnosis",
    part: "Part 8",
    title: "Medium Baseline - Accuracy",
    caption: "Training vs validation accuracy for the medium baseline showing divergence.",
  },
  {
    id: "nb3_lr_0.01_acc",
    file: "nb3_lr_0.01_acc.png",
    tab: "after",
    section: "learning_rate",
    part: "Part 9",
    title: "Learning Rate = 0.01 - Accuracy",
    caption: "Training and validation accuracy across 50 epochs for LR = 0.01.",
  },
  {
    id: "nb3_lr_0.01_loss",
    file: "nb3_lr_0.01_loss.png",
    tab: "after",
    section: "learning_rate",
    part: "Part 9",
    title: "Learning Rate = 0.01 - Loss",
    caption: "Training and validation loss across 50 epochs for LR = 0.01.",
  },
  {
    id: "nb3_lr_0.001_acc",
    file: "nb3_lr_0.001_acc.png",
    tab: "after",
    section: "learning_rate",
    part: "Part 9",
    title: "Learning Rate = 0.001 - Accuracy",
    caption: "Training and validation accuracy across 50 epochs for LR = 0.001.",
  },
  {
    id: "nb3_lr_0.001_loss",
    file: "nb3_lr_0.001_loss.png",
    tab: "after",
    section: "learning_rate",
    part: "Part 9",
    title: "Learning Rate = 0.001 - Loss",
    caption: "Training and validation loss across 50 epochs for LR = 0.001.",
  },
  {
    id: "nb3_lr_0.0001_acc",
    file: "nb3_lr_0.0001_acc.png",
    tab: "after",
    section: "learning_rate",
    part: "Part 9",
    title: "Learning Rate = 0.0001 - Accuracy",
    caption: "Training and validation accuracy across 50 epochs for LR = 0.0001.",
  },
  {
    id: "nb3_lr_0.0001_loss",
    file: "nb3_lr_0.0001_loss.png",
    tab: "after",
    section: "learning_rate",
    part: "Part 9",
    title: "Learning Rate = 0.0001 - Loss",
    caption: "Training and validation loss across 50 epochs for LR = 0.0001.",
  },
  {
    id: "nb3_batch_16_acc",
    file: "nb3_batch_16_acc.png",
    tab: "after",
    section: "batch_size",
    part: "Part 10",
    title: "Batch Size = 16 - Accuracy",
    caption: "Training and validation accuracy across 50 epochs for Batch Size = 16.",
  },
  {
    id: "nb3_batch_16_loss",
    file: "nb3_batch_16_loss.png",
    tab: "after",
    section: "batch_size",
    part: "Part 10",
    title: "Batch Size = 16 - Loss",
    caption: "Training and validation loss across 50 epochs for Batch Size = 16.",
  },
  {
    id: "nb3_batch_32_acc",
    file: "nb3_batch_32_acc.png",
    tab: "after",
    section: "batch_size",
    part: "Part 10",
    title: "Batch Size = 32 - Accuracy",
    caption: "Training and validation accuracy across 50 epochs for Batch Size = 32.",
  },
  {
    id: "nb3_batch_32_loss",
    file: "nb3_batch_32_loss.png",
    tab: "after",
    section: "batch_size",
    part: "Part 10",
    title: "Batch Size = 32 - Loss",
    caption: "Training and validation loss across 50 epochs for Batch Size = 32.",
  },
  {
    id: "nb3_batch_64_acc",
    file: "nb3_batch_64_acc.png",
    tab: "after",
    section: "batch_size",
    part: "Part 10",
    title: "Batch Size = 64 - Accuracy",
    caption: "Training and validation accuracy across 50 epochs for Batch Size = 64.",
  },
  {
    id: "nb3_batch_64_loss",
    file: "nb3_batch_64_loss.png",
    tab: "after",
    section: "batch_size",
    part: "Part 10",
    title: "Batch Size = 64 - Loss",
    caption: "Training and validation loss across 50 epochs for Batch Size = 64.",
  },
  {
    id: "nb3_batch_128_acc",
    file: "nb3_batch_128_acc.png",
    tab: "after",
    section: "batch_size",
    part: "Part 10",
    title: "Batch Size = 128 - Accuracy",
    caption: "Training and validation accuracy across 50 epochs for Batch Size = 128.",
  },
  {
    id: "nb3_batch_128_loss",
    file: "nb3_batch_128_loss.png",
    tab: "after",
    section: "batch_size",
    part: "Part 10",
    title: "Batch Size = 128 - Loss",
    caption: "Training and validation loss across 50 epochs for Batch Size = 128.",
  },
  {
    id: "nb3_opt_val_acc",
    file: "nb3_opt_val_acc.png",
    tab: "after",
    section: "optimizer",
    part: "Part 11",
    title: "Optimizer Comparison - Validation Accuracy",
    caption: "Validation accuracy comparison between Adam and SGD optimizers.",
  },
  {
    id: "nb3_opt_val_loss",
    file: "nb3_opt_val_loss.png",
    tab: "after",
    section: "optimizer",
    part: "Part 11",
    title: "Optimizer Comparison - Validation Loss",
    caption: "Validation loss comparison between Adam and SGD optimizers.",
  },
  {
    id: "nb3_reg_val_acc",
    file: "nb3_reg_val_acc.png",
    tab: "after",
    section: "regularization",
    part: "Part 12",
    title: "Validation Accuracy - Regularization Experiments",
    caption: "Validation accuracy comparison for Baseline, Dropout, Early Stopping, and L2 regularization.",
  },
  {
    id: "nb3_reg_val_loss",
    file: "nb3_reg_val_loss.png",
    tab: "after",
    section: "regularization",
    part: "Part 12",
    title: "Validation Loss - Regularization Experiments",
    caption: "Validation loss comparison for Baseline, Dropout, Early Stopping, and L2 regularization.",
  },
  {
    id: "nb3_bn_val_acc",
    file: "nb3_bn_val_acc.png",
    tab: "after",
    section: "batchnorm",
    part: "Part 13",
    title: "Validation Accuracy - Batch Normalization",
    caption: "Comparison of validation accuracy with and without Batch Normalization.",
  },
  {
    id: "nb3_bn_val_loss",
    file: "nb3_bn_val_loss.png",
    tab: "after",
    section: "batchnorm",
    part: "Part 13",
    title: "Validation Loss - Batch Normalization",
    caption: "Comparison of validation loss with and without Batch Normalization.",
  },
  {
    id: "nb3_ablation_val_acc",
    file: "nb3_ablation_val_acc.png",
    tab: "after",
    section: "ablation",
    part: "Part 14",
    title: "Ablation Experiment - Validation Accuracy",
    caption: "Validation accuracy comparison between models trained with and without Dropout.",
  },
  {
    id: "nb3_confusion_matrix",
    file: "nb3_confusion_matrix.png",
    tab: "insights",
    section: "confusion_matrix",
    part: "Part 16",
    title: "Final Model - Test Set Confusion Matrix",
    caption: "Heatmap visualization of the 10×10 confusion matrix on the 10,000 sealed test images.",
  },
  {
    id: "nb3_misclassified_images",
    file: "nb3_misclassified_images.png",
    tab: "insights",
    section: "error_analysis",
    part: "Part 17",
    title: "Incorrectly Classified Test Images",
    caption: "10 sample test images misclassified by the final model, showing true vs predicted class labels.",
  },
  {
    id: "nb3_augmentation_curves",
    file: "nb3_augmentation_curves.png",
    tab: "after",
    section: "augmentation",
    part: "Part 20 Extra",
    title: "Validation Accuracy & Loss - Augmentation Settings",
    caption: "Validation accuracy and validation loss curves across 100 epochs for No Augmentation, Flip only, and Flip + Shift.",
  },
  {
    id: "nb3_augmentation_gap",
    file: "nb3_augmentation_gap.png",
    tab: "after",
    section: "augmentation",
    part: "Part 20 Extra",
    title: "Generalization Gap - Augmentation Settings",
    caption: "Generalization gap (Train Accuracy - Validation Accuracy) per epoch for the three data augmentation settings.",
  },
] as const;

export const plotById = Object.fromEntries(
  notebookPlots.map((p) => [p.id, p]),
) as Record<string, NotebookPlot>;

