import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("renders the capstone application with all four sections", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();

  assert.match(html, /<title>CIFAR-10 with Dense Networks Only<\/title>/i);
  for (const tab of ["Before Optimization", "After Optimization", "Insights &amp; Reflection", "Deployment &amp; CNN"]) {
    assert.match(html, new RegExp(tab));
  }
  // The landing view is tab 01, so its headline numbers must be in the markup.
  assert.match(html, /52\.22%/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});

test("ships the numbers the notebooks actually produced", async () => {
  const data = await readFile(new URL("../app/data.ts", import.meta.url), "utf8");

  // Part 16 final test evaluation.
  assert.match(data, /accuracy: 0\.5408/);
  assert.match(data, /loss: 1\.2815/);
  assert.match(data, /f1: 0\.5399/);
  // Parts 4-7 architecture sweep, as committed to Results/*.json.
  assert.match(data, /bestValAccuracy: 0\.5222/);
  assert.match(data, /bestValAccuracy: 0\.5298/);
  // The augmentation high-water mark.
  assert.match(data, /bestVal: 0\.5912/);
});

test("the confusion matrix is a valid 10x10 with 1,000 images per class", async () => {
  const { confusionMatrix, perClass, classNames } = await import(new URL("../app/data.ts", import.meta.url));

  assert.equal(confusionMatrix.length, 10);
  assert.equal(classNames.length, 10);
  assert.equal(perClass.length, 10);
  for (const row of confusionMatrix) {
    assert.equal(row.length, 10);
    assert.equal(row.reduce((a, b) => a + b, 0), 1000);
  }

  // Recall per class must be the diagonal over the row total, which is 1,000.
  for (const cls of perClass) {
    const recall = confusionMatrix[cls.id][cls.id] / 1000;
    assert.ok(
      Math.abs(recall - cls.recall) < 1e-9,
      `${cls.name}: matrix recall ${recall} against reported ${cls.recall}`,
    );
  }

  // Overall accuracy is the trace over the 10,000-image test set.
  const trace = confusionMatrix.reduce((sum, row, i) => sum + row[i], 0);
  assert.equal(trace / 10000, 0.5408);
});

test("every curve series is aligned and every run is accounted for", async () => {
  const { curves } = await import(new URL("../app/curves.ts", import.meta.url));

  const expected = {
    architecture: 3, learningRate: 3, batchSize: 4, optimizer: 2,
    regularization: 4, batchnorm: 2, ablation: 2, finalModel: 1,
    part15Rerun: 4,
  };

  for (const [group, count] of Object.entries(expected)) {
    const runs = curves[group];
    assert.equal(Object.keys(runs).length, count, `${group} run count`);
    for (const [name, series] of Object.entries(runs)) {
      const epochs = series.valAcc.length;
      assert.ok(epochs > 0, `${group}/${name} has no epochs`);
      for (const key of ["acc", "loss", "valLoss"]) {
        assert.equal(series[key].length, epochs, `${group}/${name}.${key} length`);
      }
    }
  }

  // The one notebook run that stopped early, and the fixed 50-epoch budget
  // everywhere else -- Part 15's Early Stopping never fired, so it ran the lot.
  assert.equal(curves.regularization["Early Stopping"].valAcc.length, 17);
  assert.equal(Object.keys(curves.part15Rerun).length, 4);
  assert.equal(curves.finalModel["Final Model"].valAcc.length, 50);
  assert.equal(curves.architecture.medium.valAcc.length, 50);
});

test("the Part 15 re-run matches the raw results it was generated from", async () => {
  const { rerunRuns, rerunContrasts } = await import(new URL("../app/data.ts", import.meta.url));
  const raw = JSON.parse(
    await readFile(new URL("../../Results/part15_rerun.json", import.meta.url), "utf8"),
  );

  assert.equal(rerunRuns.length, 4);
  assert.equal(raw.winner, "both");
  assert.equal(raw.incumbent, "dropout");

  // Every displayed figure must still equal the number the script wrote.
  for (const run of rerunRuns) {
    const source = raw.runs[run.key];
    assert.ok(source, `no raw run for ${run.key}`);
    assert.equal(run.bestVal, Number(source.best_val_accuracy.toFixed(4)), `${run.key} best val`);
    assert.equal(run.test, Number(source.test_accuracy.toFixed(4)), `${run.key} test`);
    assert.equal(run.epochsRun, source.epochs_run, `${run.key} epochs`);
    assert.equal(run.gap, Number(source.gap_peak.toFixed(4)), `${run.key} gap`);
    // The whole comparison rests on all four having converged rather than
    // being cut off by the epoch cap.
    assert.equal(source.stopped_early, true, `${run.key} hit the epoch cap`);
  }

  // Exactly one shipped config and one winner. The shipped row is the one the
  // re-run recommended and Part 15 adopted, so it must be the raw file's
  // incumbent -- and deliberately NOT the top scorer, whose lead is inside noise.
  assert.equal(rerunRuns.filter((r) => r.shipped).length, 1);
  assert.equal(rerunRuns.filter((r) => r.winner).length, 1);
  assert.equal(rerunRuns.find((r) => r.shipped).key, raw.incumbent);
  assert.equal(rerunRuns.find((r) => r.winner).key, raw.winner);
  assert.notEqual(
    rerunRuns.find((r) => r.shipped).key,
    rerunRuns.find((r) => r.winner).key,
  );
  // The headline delta is the winner's lead over what ships.
  const lead =
    rerunRuns.find((r) => r.winner).bestVal - rerunRuns.find((r) => r.shipped).bestVal;
  assert.ok(Math.abs(lead * 100 - raw.delta_points) < 0.005, "delta_points");
  assert.equal(Math.round(lead * raw.validation_size), raw.delta_images);

  // Each stated contrast must be the arithmetic it claims to be.
  const by = Object.fromEntries(rerunRuns.map((r) => [r.key, r.bestVal]));
  const expected = {
    "L2 only vs control": by.l2 - by.neither,
    "Dropout only vs control": by.dropout - by.neither,
    "Dropout + L2 vs Dropout only": by.both - by.dropout,
  };
  for (const contrast of rerunContrasts) {
    const delta = expected[contrast.pair];
    assert.ok(delta !== undefined, `unknown contrast pair ${contrast.pair}`);
    assert.ok(
      Math.abs(delta * 100 - contrast.points) < 0.005,
      `${contrast.pair}: stated ${contrast.points} against computed ${(delta * 100).toFixed(2)}`,
    );
    assert.equal(contrast.images, Math.round(delta * 5000), `${contrast.pair} images`);
    // A contrast is only called a result if it clears roughly two standard
    // errors on a 5,000-image validation set.
    assert.equal(contrast.verdict, Math.abs(contrast.points) > 1.4 ? "yes" : "no", `${contrast.pair} verdict`);
  }
});

// Every summary figure the demo prints for a run is a number that can be
// recomputed from that run's own per-epoch curve. This is the guard that stops
// a hand-typed figure in data.ts drifting away from the run it describes.
test("every run summary matches the curve it was taken from", async () => {
  const data = await import(new URL("../app/data.ts", import.meta.url));
  const { curves } = await import(new URL("../app/curves.ts", import.meta.url));

  // curves.ts is parsed from the 4-decimal figures Keras prints, so a value
  // rounded from a longer decimal can sit one ulp of that display away.
  const TOL = 1.1e-4;
  const argmax = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
  const argmin = (a) => a.reduce((b, v, i) => (v < a[b] ? i : b), 0);

  /** Recomputes a run's headline numbers straight from its curve. */
  const summarise = (series) => {
    const bi = argmax(series.valAcc);
    const li = argmin(series.valLoss);
    return {
      bestVal: series.valAcc[bi],
      bestValEpoch: bi + 1,
      minValLoss: series.valLoss[li],
      minValLossEpoch: li + 1,
      finalTrain: series.acc.at(-1),
      finalVal: series.valAcc.at(-1),
      finalValLoss: series.valLoss.at(-1),
      epochsRun: series.valAcc.length,
      gapPeak: series.acc.at(-1) - Math.max(...series.valAcc),
      gapFinal: series.acc.at(-1) - series.valAcc.at(-1),
    };
  };

  const check = (group, key, row, aliases = {}) => {
    const series = curves[group][key];
    assert.ok(series, `no curve for ${group}/${key}`);
    const want = summarise(series);
    for (const [field, value] of Object.entries(want)) {
      const stated = row[aliases[field] ?? field];
      if (stated === undefined) continue;
      assert.ok(
        Math.abs(stated - value) < TOL,
        `${group}/${key}.${field}: data.ts says ${stated}, curve gives ${value.toFixed(4)}`,
      );
    }
    // Both gap definitions are in use, each labelled where it is shown, so a
    // stated gap only has to be one of the two.
    if (row.gap !== undefined) {
      assert.ok(
        Math.abs(row.gap - want.gapPeak) < TOL || Math.abs(row.gap - want.gapFinal) < TOL,
        `${group}/${key}.gap: data.ts says ${row.gap}, peak is ${want.gapPeak.toFixed(4)}, final is ${want.gapFinal.toFixed(4)}`,
      );
    }
  };

  const ARCH_ALIASES = { bestVal: "bestValAccuracy", finalTrain: "finalTrainAccuracy" };
  for (const arch of data.architectures) check("architecture", arch.key, arch, ARCH_ALIASES);
  for (const run of data.learningRateRuns) check("learningRate", String(run.value), run);
  for (const run of data.batchSizeRuns) check("batchSize", String(run.value), run);
  for (const run of data.optimizerRuns) check("optimizer", run.name, run);
  for (const run of data.regularizationRuns) check("regularization", run.name, run);
  for (const run of data.batchnormRuns) check("batchnorm", run.name, run);
  for (const run of data.ablationRuns) check("ablation", run.name, run);

  // Part 3's baseline is the same run as the shallow architecture.
  check("architecture", "shallow", data.baseline, ARCH_ALIASES);

  // Part 15 ran to the 50-epoch cap, so Early Stopping never truncated it.
  assert.equal(data.finalModel.epochsRun, curves.finalModel["Final Model"].valAcc.length);
  assert.equal(data.finalModel.epochsRun, data.finalModel.epochsAllowed);

  const RERUN_KEYS = {
    l2: "L2 only",
    dropout: "Dropout only (final model)",
    both: "Dropout + L2",
    neither: "Early Stopping only (control)",
  };
  for (const run of data.rerunRuns) check("part15Rerun", RERUN_KEYS[run.key], run);
});

test("every derived headline figure is the arithmetic it claims", async () => {
  const data = await import(new URL("../app/data.ts", import.meta.url));
  const close = (a, b, what) =>
    assert.ok(Math.abs(a - b) < 5e-4, `${what}: stated ${b}, computed ${a.toFixed(4)}`);

  // Per-class precision, recall and F1 all come out of the confusion matrix.
  const cm = data.confusionMatrix;
  for (const cls of data.perClass) {
    const tp = cm[cls.id][cls.id];
    const predicted = cm.reduce((sum, row) => sum + row[cls.id], 0);
    const actual = cm[cls.id].reduce((a, b) => a + b, 0);
    const precision = tp / predicted;
    const recall = tp / actual;
    close(precision, cls.precision, `${cls.name} precision`);
    close(recall, cls.recall, `${cls.name} recall`);
    close((2 * precision * recall) / (precision + recall), cls.f1, `${cls.name} F1`);
  }

  // Classes are balanced at 1,000 test images each, so every weighted average
  // is the plain mean of the ten per-class values.
  const mean = (field) => data.perClass.reduce((s, c) => s + c[field], 0) / 10;
  close(mean("precision"), data.testResults.precision, "weighted precision");
  close(mean("recall"), data.testResults.recall, "weighted recall");
  close(mean("f1"), data.testResults.f1, "weighted F1");

  // The spread Parts 4-7 report, in points and in validation images.
  const accuracies = data.architectures.map((a) => a.bestValAccuracy);
  const spread = Math.max(...accuracies) - Math.min(...accuracies);
  close(spread * 100, data.architectureSpread.points, "architecture spread");
  assert.equal(Math.round(spread * 5000), data.architectureSpread.images);

  // Augmentation gains are measured against the no-augmentation control.
  const control = data.augmentationRuns[0].bestVal;
  for (const run of data.augmentationRuns) {
    close((run.bestVal - control) * 100, run.change, `${run.name} change`);
    assert.equal(Math.round((run.bestVal - control) * 5000), run.images, `${run.name} images`);
  }

  // Part 18 tracks one row per run, and the demo counts them rather than
  // stating a number.
  assert.equal(data.experimentLog.length, 13);
  assert.equal(data.experimentLog.filter((r) => r.selected).length, 1);
});

test("all 37 notebook plots exist and are registered", async () => {
  const { notebookPlots, plotById } = await import(new URL("../app/data.ts", import.meta.url));
  const { stat } = await import("node:fs/promises");

  assert.equal(notebookPlots.length, 37);
  assert.equal(Object.keys(plotById).length, 37);

  for (const plot of notebookPlots) {
    assert.ok(plot.id, `plot has id`);
    assert.ok(plot.file, `plot has file`);
    assert.ok(plot.title, `plot has title`);
    assert.ok(plot.caption, `plot has caption`);
    const filePath = new URL(`../public/plots/${plot.file}`, import.meta.url);
    const fileStat = await stat(filePath);
    assert.ok(fileStat.size > 1000, `${plot.file} is non-empty (${fileStat.size} bytes)`);
  }
});

