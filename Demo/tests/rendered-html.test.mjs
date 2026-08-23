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
