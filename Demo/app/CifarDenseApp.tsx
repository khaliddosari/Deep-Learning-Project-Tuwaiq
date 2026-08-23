"use client";

import { type ReactNode, useEffect, useMemo, useState } from "react";
import {
  ablationRuns,
  ablationVerdict,
  architectures,
  architectureSpread,
  augmentationEpochs,
  augmentationRuns,
  augmentationVerdict,
  baseline,
  baselineChoices,
  batchnormRuns,
  batchnormVerdict,
  batchSizeRuns,
  batchSizeVerdict,
  classNames,
  cnnComparison,
  cnnEvidence,
  conclusion,
  confusionMatrix,
  datasetFacts,
  deploymentCard,
  deploymentGuidance,
  diagnosis,
  experimentLog,
  finalModel,
  finalReflection,
  findings,
  flatteningCosts,
  GAP_FINAL,
  GAP_PEAK,
  learningRateRuns,
  learningRateVerdict,
  navItems,
  optimizerRuns,
  optimizerVerdict,
  perClass,
  project,
  reflection,
  reproducibilityNote,
  rerunContrasts,
  rerunProtocol,
  rerunRuns,
  rerunVerdict,
  regularizationRuns,
  regularizationVerdict,
  splitRows,
  testResults,
  type ViewId,
} from "./data";
import { curves, type Series } from "./curves";

type Tone = "aqua" | "orange" | "purple" | "green" | "red";

/** Series colours, in the order runs are listed. Kept away from the green that
 *  marks a selected run everywhere else in the app. */
const SERIES_COLORS = ["#522cba", "#f4a664", "#2a78d6", "#e34948", "#1baf7a"];

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
const pts = (v: number) => `${(v * 100).toFixed(2)} pts`;

// ---------------------------------------------------------------- primitives

function Dot({ tone = "aqua" }: { tone?: Tone }) {
  return <span className={`dot dot-${tone}`} aria-hidden="true" />;
}

function PageHeading({ eyebrow, title, subtitle }: { eyebrow: string; title: string; subtitle: string }) {
  return (
    <header className="page-heading">
      <p className="eyebrow"><Dot />{eyebrow}</p>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </header>
  );
}

function SectionHeading({ children, note }: { children: ReactNode; note?: string }) {
  return <h2 className="section-heading">{children}{note ? <small>{note}</small> : null}</h2>;
}

function Metric({ value, label, tone = "aqua", best = false }: { value: string; label: string; tone?: Tone; best?: boolean }) {
  return (
    <article className={`metric-card${best ? " metric-best" : ""}`}>
      <Dot tone={best ? "green" : tone} />
      <strong>{value}</strong>
      <span>{label}</span>
    </article>
  );
}

function Insight({ tone = "orange", children }: { tone?: Tone; children: ReactNode }) {
  return <div className="insight-strip"><Dot tone={tone} /><strong>{children}</strong></div>;
}

function StatRows({ rows }: { rows: readonly (readonly [string, string])[] }) {
  return (
    <div className="stat-rows">
      {rows.map(([label, value]) => (
        <div key={label}><span>{label}</span><strong>{value}</strong></div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------- line chart

const CHART_W = 560;
const CHART_H = 250;
const PAD = { top: 14, right: 14, bottom: 30, left: 46 };

type Line = { label: string; values: readonly number[]; color: string; dashed?: boolean };

/** Per-epoch line chart. Every point is a real epoch from a real run — there is
 *  no smoothing and no interpolation, so a kink in a curve is a kink in the run. */
function LineChart({
  lines,
  yFormat = (v: number) => v.toFixed(2),
  caption,
}: {
  lines: Line[];
  yFormat?: (value: number) => string;
  caption: string;
}) {
  const epochs = Math.max(...lines.map((line) => line.values.length));
  const values = lines.flatMap((line) => [...line.values]);
  const rawLo = Math.min(...values);
  const rawHi = Math.max(...values);
  const span = rawHi - rawLo || 1;
  const lo = rawLo - span * 0.08;
  const hi = rawHi + span * 0.08;

  const plotW = CHART_W - PAD.left - PAD.right;
  const plotH = CHART_H - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (epochs > 1 ? (i / (epochs - 1)) * plotW : plotW / 2);
  const y = (v: number) => PAD.top + (1 - (v - lo) / (hi - lo)) * plotH;

  const yTicks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  // Epoch labels every 10, plus epoch 1 and the final epoch of the longest run.
  const xTicks = [0, ...Array.from({ length: Math.floor(epochs / 10) }, (_, i) => (i + 1) * 10 - 1)]
    .filter((i) => i < epochs)
    .filter((i, at, all) => all.indexOf(i) === at);

  return (
    <figure className="chart-figure">
      <svg className="linechart" viewBox={`0 0 ${CHART_W} ${CHART_H}`} role="img" aria-label={caption}>
        {yTicks.map((tick) => (
          <g key={tick}>
            <line className="grid" x1={PAD.left} x2={CHART_W - PAD.right} y1={y(tick)} y2={y(tick)} />
            <text x={PAD.left - 8} y={y(tick) + 3} textAnchor="end">{yFormat(tick)}</text>
          </g>
        ))}
        <line className="axis" x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={CHART_H - PAD.bottom} />
        <line className="axis" x1={PAD.left} x2={CHART_W - PAD.right} y1={CHART_H - PAD.bottom} y2={CHART_H - PAD.bottom} />
        {xTicks.map((i) => (
          <text key={i} x={x(i)} y={CHART_H - PAD.bottom + 14} textAnchor="middle">{i + 1}</text>
        ))}
        <text className="axis-label" x={PAD.left + plotW / 2} y={CHART_H - 2} textAnchor="middle">Epoch</text>
        {lines.map((line) => (
          <path
            key={line.label}
            className={line.dashed ? "dashed" : undefined}
            stroke={line.color}
            d={line.values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ")}
          />
        ))}
      </svg>
      <figcaption className="chart-legend">
        {lines.map((line) => (
          <span key={line.label}>
            <i className={line.dashed ? "dashed" : undefined} style={{ background: line.dashed ? undefined : line.color, color: line.color }} />
            {line.label}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}

function ChartCard({ title, note, children }: { title: string; note: string; children: ReactNode }) {
  return (
    <section className="chart-card">
      <div className="chart-heading"><h2>{title}</h2><span>{note}</span></div>
      {children}
    </section>
  );
}

/** Builds the val-accuracy / val-loss line pair for one experiment group. */
function seriesFrom(group: Record<string, Series>, key: "valAcc" | "valLoss", labelFor?: (name: string) => string): Line[] {
  return Object.entries(group).map(([name, run], i) => ({
    label: labelFor ? labelFor(name) : name,
    values: run[key],
    color: SERIES_COLORS[i % SERIES_COLORS.length],
  }));
}

// ---------------------------------------------------------------- run list

type Run = { label: string; value: number; caption: string; selected?: boolean; tone?: "best" | "warn" | "bad" };

/** Ranked bars for one experiment. `max` is fixed by the caller so bars stay
 *  comparable between cards rather than each card rescaling to its own winner. */
function RunList({ runs, max, format = pct }: { runs: Run[]; max: number; format?: (v: number) => string }) {
  return (
    <div className="run-list">
      {runs.map((run) => (
        <div className={`run-row${run.selected ? " selected" : ""}`} key={run.label}>
          <span>{run.label}{run.selected ? <em>best</em> : null}</span>
          <div className="mini-track" role="img" aria-label={`${run.label}: ${format(run.value)}`}>
            <i
              className={run.selected ? "fill-best" : run.tone ? `fill-${run.tone}` : undefined}
              style={{ width: `${Math.max(2, (run.value / max) * 100)}%` }}
            />
          </div>
          <strong>{format(run.value)}</strong>
          <span className="run-caption">{run.caption}</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------- view 1

function BeforeOptimization() {
  const archLines = (key: "valAcc" | "valLoss"): Line[] =>
    architectures.map((arch, i) => ({
      label: `${arch.name} (${arch.layers})`,
      values: curves.architecture[arch.key][key],
      color: SERIES_COLORS[i % SERIES_COLORS.length],
    }));

  const medium = curves.architecture.medium;

  return (
    <section className="view">
      <div className="hero">
        <div className="hero-orbit orbit-one" aria-hidden="true" />
        <div className="hero-orbit orbit-two" aria-hidden="true" />
        <div className="hero-copy">
          <p className="hero-kicker">DEEP LEARNING CAPSTONE · PARTS 1–8</p>
          <h1>Before<br />optimization</h1>
          <p className="hero-lede">
            One baseline and three depths, all trained on the same split with the same seed. Depth moved best
            validation accuracy by 0.76 points and widened the generalization gap by 17.75. The problem was never
            capacity.
          </p>
        </div>
      </div>

      <div className="metric-grid hero-metrics">
        <Metric value={pct(baseline.bestValAccuracy)} label="baseline validation accuracy · 1 × 512" />
        <Metric value="10%" label="random-guess floor the baseline had to beat" tone="purple" />
        <Metric value={`${architectureSpread.points} pts`} label={`spread across all three depths — ${architectureSpread.images} images`} tone="orange" />
        <Metric value={`${architectures[0].gap.toFixed(2)} → ${architectures[2].gap.toFixed(2)}`} label="generalization gap, shallow to deep" tone="red" />
      </div>

      <div className="rule-note">
        <Dot tone="orange" />
        <div>
          <strong>One rule shapes everything: {project.rule}</strong>
          <p>
            Every image must be flattened before it reaches the first layer — 32 × 32 × 3 becomes 3,072 × 1. That single
            decision sets the ceiling for every experiment that follows.
          </p>
        </div>
      </div>

      <SectionHeading note="Parts 1–2">The data and the split</SectionHeading>
      <div className="metric-grid">
        {datasetFacts.map((fact) => (
          <Metric key={fact.label} value={fact.value} label={`${fact.label} — ${fact.note}`} tone="purple" />
        ))}
      </div>
      <div className="table-card">
        <div className="table-title">
          <h2><Dot />Split once, shared by every experiment</h2>
          <span>seed {project.seed} · indices committed to splits.npz</span>
        </div>
        <div className="setting-list">
          {splitRows.map((row) => (
            <article key={row.name}>
              <b>{row.name}</b>
              <code>{row.shape}</code>
              <span>{row.role}</span>
            </article>
          ))}
        </div>
      </div>
      <p className="footnote">
        The validation set is carved out of the training half only, stratified so every class contributes exactly 500
        images. Nothing from the test set influences any architecture or hyperparameter decision.
      </p>

      <SectionHeading note="Part 3">The baseline</SectionHeading>
      <div className="two-col">
        <article className="soft-card">
          <h2><Dot />{baseline.architecture}</h2>
          <StatRows
            rows={[
              ["Best validation accuracy", pct(baseline.bestValAccuracy)],
              ["Minimum validation loss", baseline.minValLoss.toFixed(4)],
              ["Final training accuracy", pct(baseline.finalTrainAccuracy)],
              ["Trainable parameters", baseline.params.toLocaleString()],
              ["Optimizer", `${baseline.optimizer} @ ${baseline.learningRate}`],
              ["Batch size / epochs", `${baseline.batchSize} / ${baseline.epochs}`],
            ]}
          />
          <p className="footnote">
            About 5× the 10% random-guess floor — and 99.7% of those parameters sit in the very first layer, between the
            3,072 inputs and the 512 hidden units.
          </p>
        </article>
        <div className="table-card flush">
          <div className="table-title"><h2><Dot tone="purple" />Why these hyperparameters</h2></div>
          <div className="qa-list">
            {baselineChoices.map((choice) => (
              <article key={choice.setting}>
                <h3><Dot tone="purple" />{choice.setting}</h3>
                <p>{choice.reason}</p>
              </article>
            ))}
          </div>
        </div>
      </div>

      <SectionHeading note="Parts 4–7 · same data, same seed, only depth changes">Three depths, one width</SectionHeading>
      <div className="arch-grid">
        {architectures.map((arch) => (
          <article className={`arch-card${arch.selected ? " selected" : ""}`} key={arch.key}>
            <header>
              <h3>{arch.name}</h3>
              <code>{arch.layers}</code>
            </header>
            {arch.selected ? <span className="best-badge">Selected for Parts 8–16</span> : null}
            <dl className="arch-stats">
              <div><dt>Best val accuracy</dt><dd>{pct(arch.bestValAccuracy)}</dd></div>
              <div><dt>Gap</dt><dd>{arch.gap.toFixed(4)}</dd></div>
              <div><dt>Final train accuracy</dt><dd>{pct(arch.finalTrainAccuracy)}</dd></div>
              <div><dt>Parameters</dt><dd className="dd-small">{arch.params.toLocaleString()}</dd></div>
            </dl>
            <p>{arch.verdict}</p>
          </article>
        ))}
      </div>
      <p className="footnote">
        Gap here is <code>{GAP_PEAK}</code>, the definition Parts 4–7 report. The regularization experiments in tab 02
        use a different one, and it is labelled there.
      </p>

      <div className="chart-grid">
        <ChartCard title="Validation accuracy by depth" note="50 epochs · Results/*.json">
          <LineChart lines={archLines("valAcc")} yFormat={(v) => `${(v * 100).toFixed(0)}%`} caption="Validation accuracy per epoch for the shallow, medium and deep networks. All three plateau near 52%." />
        </ChartCard>
        <ChartCard title="Validation loss by depth" note="lower is better — and only shallow stays down">
          <LineChart lines={archLines("valLoss")} caption="Validation loss per epoch. Shallow stays flat, medium rises after epoch 13, deep rises hardest after epoch 12." />
        </ChartCard>
      </div>
      <Insight>
        Depth bought overfitting, not accuracy. All three land within {architectureSpread.images} images of each other on
        a {architectureSpread.validationSize.toLocaleString()}-image validation set, where the standard error is about{" "}
        {architectureSpread.standardError} points.
      </Insight>

      <SectionHeading note="Part 8">The diagnosis</SectionHeading>
      <div className="two-col">
        <article className="soft-card">
          <h2><Dot tone="red" />{diagnosis.verdict}</h2>
          <p className="diagnosis-model">{diagnosis.model}</p>
          <ol className="evidence-list">
            {diagnosis.evidence.map((line) => <li key={line}>{line}</li>)}
          </ol>
        </article>
        <div className="stack">
          {diagnosis.ruledOut.map((item) => (
            <article className="soft-card" key={item.label}>
              <h3>{item.label}</h3>
              <p>{item.why}</p>
            </article>
          ))}
          <article className="soft-card card-good">
            <h3>The fix carried into Part 15</h3>
            <p>{diagnosis.fix}</p>
          </article>
        </div>
      </div>

      <div className="chart-grid">
        <ChartCard title="Medium — training against validation accuracy" note="the two curves separate and never rejoin">
          <LineChart
            lines={[
              { label: "Training accuracy", values: medium.acc, color: SERIES_COLORS[0] },
              { label: "Validation accuracy", values: medium.valAcc, color: SERIES_COLORS[1], dashed: true },
            ]}
            yFormat={(v) => `${(v * 100).toFixed(0)}%`}
            caption="Training accuracy climbs to 82% while validation accuracy flattens near 52%."
          />
        </ChartCard>
        <ChartCard title="Medium — training against validation loss" note="validation loss bottoms at epoch 13, then reverses">
          <LineChart
            lines={[
              { label: "Training loss", values: medium.loss, color: SERIES_COLORS[0] },
              { label: "Validation loss", values: medium.valLoss, color: SERIES_COLORS[1], dashed: true },
            ]}
            caption="Training loss falls all 50 epochs while validation loss bottoms at 1.4024 on epoch 13 and rises to 1.7835."
          />
        </ChartCard>
      </div>
      <Insight tone="red">
        The two curves move in opposite directions for 37 of the 50 epochs. Everything after epoch 13 is memorisation.
      </Insight>
    </section>
  );
}

// ---------------------------------------------------------------- view 2

function AfterOptimization() {
  const bestF1 = Math.max(...perClass.map((c) => c.f1));
  const bestTuningRun = learningRateRuns.reduce((a, b) => (b.bestVal > a.bestVal ? b : a));
  const bestGapRun = regularizationRuns.reduce((a, b) => (b.gap < a.gap ? b : a));
  const bestAugmentation = augmentationRuns.reduce((a, b) => (b.bestVal > a.bestVal ? b : a));

  return (
    <section className="view">
      <PageHeading
        eyebrow="Parts 9–16 · optimization and the sealed test set"
        title="After optimization"
        subtitle="Eight controlled experiments, one variable at a time: learning rate, batch size, optimizer, three regularizers, batch normalization and an L2 ablation. Then the test set was opened, exactly once."
      />

      <div className="metric-grid">
        <Metric value={pct(bestTuningRun.bestVal)} label={`best validation accuracy of any tuning run — LR ${bestTuningRun.value}`} tone="purple" />
        <Metric value={pct(testResults.accuracy)} label={`final test accuracy · ${testResults.size.toLocaleString()} sealed images`} best />
        <Metric value={pct(bestGapRun.gap)} label={`smallest generalization gap — ${bestGapRun.name}, down from ${pct(regularizationRuns[0].gap)}`} tone="orange" />
        <Metric value={pct(bestAugmentation.bestVal)} label={`best score recorded anywhere — ${bestAugmentation.name.toLowerCase()} augmentation`} tone="aqua" />
      </div>

      <SectionHeading note="Part 9 · everything else held fixed">Learning rate</SectionHeading>
      <div className="two-col">
        <div className="table-card flush">
          <div className="table-title"><h2><Dot />Three rates, 50 epochs each</h2><span>best validation accuracy</span></div>
          <RunList
            max={0.55}
            runs={learningRateRuns.map((run) => ({
              label: `LR ${run.value}`,
              value: run.bestVal,
              selected: run.selected,
              tone: run.value === 0.01 ? "bad" : "warn",
              caption: `peak at epoch ${run.bestValEpoch} · min val loss ${run.minValLoss.toFixed(4)} · final val loss ${run.finalValLoss.toFixed(4)}`,
            }))}
          />
        </div>
        <article className="soft-card">
          <h2><Dot tone="orange" />What the curves show</h2>
          <p>{learningRateVerdict}</p>
          <p>
            0.0001 learns more slowly but more consistently, reaching its best validation accuracy at epoch 22 and
            holding. It was carried into every experiment that follows.
          </p>
        </article>
      </div>
      <div className="chart-grid">
        <ChartCard title="Validation accuracy by learning rate" note="0.01 never gets off the floor">
          <LineChart lines={seriesFrom(curves.learningRate, "valAcc", (n) => `LR ${n}`)} yFormat={(v) => `${(v * 100).toFixed(0)}%`} caption="Validation accuracy per epoch for learning rates 0.01, 0.001 and 0.0001." />
        </ChartCard>
        <ChartCard title="Validation loss by learning rate" note="0.001 diverges to 3.1259">
          <LineChart lines={seriesFrom(curves.learningRate, "valLoss", (n) => `LR ${n}`)} caption="Validation loss per epoch. The 0.001 run peaks at epoch 12 then diverges sharply." />
        </ChartCard>
      </div>

      <SectionHeading note="Part 10 · at the selected learning rate">Batch size</SectionHeading>
      <div className="two-col">
        <div className="table-card flush">
          <div className="table-title"><h2><Dot />Four batch sizes</h2><span>best validation accuracy</span></div>
          <RunList
            max={0.55}
            runs={batchSizeRuns.map((run) => ({
              label: `Batch ${run.value}`,
              value: run.bestVal,
              selected: run.selected,
              caption: `peak at epoch ${run.bestValEpoch} · final train ${pct(run.finalTrain)} · final val loss ${run.finalValLoss.toFixed(4)}`,
            }))}
          />
        </div>
        <article className="soft-card">
          <h2><Dot tone="orange" />Larger was better, but barely</h2>
          <p>{batchSizeVerdict}</p>
        </article>
      </div>
      <div className="chart-grid">
        <ChartCard title="Validation accuracy by batch size" note="16 / 32 / 64 / 128">
          <LineChart lines={seriesFrom(curves.batchSize, "valAcc", (n) => `Batch ${n}`)} yFormat={(v) => `${(v * 100).toFixed(0)}%`} caption="Validation accuracy per epoch across four batch sizes." />
        </ChartCard>
        <ChartCard title="Validation loss by batch size" note="smaller batches diverge much harder">
          <LineChart lines={seriesFrom(curves.batchSize, "valLoss", (n) => `Batch ${n}`)} caption="Validation loss per epoch. Batch 16 ends at 3.4160 against 1.8418 for batch 128." />
        </ChartCard>
      </div>

      <SectionHeading note="Part 11">Optimizer</SectionHeading>
      <div className="two-col">
        <div className="table-card flush">
          <div className="table-title"><h2><Dot />Adam against SGD</h2><span>best validation accuracy</span></div>
          <RunList
            max={0.55}
            runs={optimizerRuns.map((run) => ({
              label: `${run.name} @ ${run.learningRate}`,
              value: run.bestVal,
              selected: run.selected,
              caption: `peak at epoch ${run.bestValEpoch} · min val loss ${run.minValLoss.toFixed(4)} · final train ${pct(run.finalTrain)}`,
            }))}
          />
        </div>
        <article className="soft-card">
          <h2><Dot tone="orange" />Why the rates differ</h2>
          <p>{optimizerVerdict}</p>
        </article>
      </div>
      <div className="chart-grid">
        <ChartCard title="Validation accuracy — Adam against SGD" note="Adam peaks at 29, SGD is still climbing at 49">
          <LineChart lines={seriesFrom(curves.optimizer, "valAcc")} yFormat={(v) => `${(v * 100).toFixed(0)}%`} caption="Validation accuracy per epoch for Adam and SGD." />
        </ChartCard>
        <ChartCard title="Validation loss — Adam against SGD" note="Adam overfits after its best epoch">
          <LineChart lines={seriesFrom(curves.optimizer, "valLoss")} caption="Validation loss per epoch for Adam and SGD." />
        </ChartCard>
      </div>

      <SectionHeading note="Part 12 · three regularizers against an unregularized baseline">Regularization</SectionHeading>
      <div className="table-card">
        <div className="table-title">
          <h2><Dot />Closing the gap</h2>
          <span>gap = {GAP_FINAL}</span>
        </div>
        <div className="table-scroll">
          <table className="log-table">
            <thead>
              <tr>
                <th>Experiment</th><th>Setting</th><th>Final train</th><th>Final val</th>
                <th>Gap</th><th>Final val loss</th><th>Best val acc.</th>
              </tr>
            </thead>
            <tbody>
              {regularizationRuns.map((run) => (
                <tr key={run.name} className={run.selected ? "selected" : undefined}>
                  <td>{run.name}</td>
                  <td>{run.detail}</td>
                  <td>{pct(run.finalTrain)}</td>
                  <td>{pct(run.finalVal)}</td>
                  <td>{pts(run.gap)}</td>
                  <td>{run.finalValLoss.toFixed(4)}</td>
                  <td>{pct(run.bestVal)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="chart-grid">
        <ChartCard title="Validation accuracy by regularizer" note="Early Stopping halts at epoch 18">
          <LineChart lines={seriesFrom(curves.regularization, "valAcc")} yFormat={(v) => `${(v * 100).toFixed(0)}%`} caption="Validation accuracy per epoch for the baseline, Dropout, Early Stopping and L2." />
        </ChartCard>
        <ChartCard title="Validation loss by regularizer" note="only Dropout keeps it falling">
          <LineChart lines={seriesFrom(curves.regularization, "valLoss")} caption="Validation loss per epoch. Dropout is the only run whose loss does not turn upward." />
        </ChartCard>
      </div>
      <Insight>{regularizationVerdict}</Insight>

      <SectionHeading note="Part 13 · the same network, with and without">Batch Normalization</SectionHeading>
      <div className="two-col">
        <article className="soft-card card-bad">
          <h2><Dot tone="red" />It backfired</h2>
          <p>{batchnormVerdict}</p>
        </article>
        <div className="table-card flush">
          <div className="table-title"><h2><Dot tone="purple" />Side by side</h2><span>gap = {GAP_FINAL}</span></div>
          <div className="setting-list">
            {batchnormRuns.map((run) => (
              <article key={run.name} className="cols-a">
                <b>{run.name}</b>
                <code>train {pct(run.finalTrain)}</code>
                <span>val {pct(run.finalVal)} · gap {pts(run.gap)} · best {pct(run.bestVal)} · final val loss {run.finalValLoss.toFixed(4)}</span>
              </article>
            ))}
          </div>
        </div>
      </div>
      <ChartCard title="Batch Normalization — training against validation accuracy" note="99.20% train, 40.70% validation">
        <LineChart
          lines={[
            { label: "Without BN — training", values: curves.batchnorm["Without BatchNorm"].acc, color: SERIES_COLORS[0] },
            { label: "Without BN — validation", values: curves.batchnorm["Without BatchNorm"].valAcc, color: SERIES_COLORS[0], dashed: true },
            { label: "With BN — training", values: curves.batchnorm["With BatchNorm"].acc, color: SERIES_COLORS[3] },
            { label: "With BN — validation", values: curves.batchnorm["With BatchNorm"].valAcc, color: SERIES_COLORS[3], dashed: true },
          ]}
          yFormat={(v) => `${(v * 100).toFixed(0)}%`}
          caption="With Batch Normalization training accuracy reaches 99.20% while validation accuracy falls to 40.70%."
        />
      </ChartCard>

      <SectionHeading note="Part 14 · one variable removed, everything else identical">Dropout ablation</SectionHeading>
      <div className="two-col">
        <div className="table-card flush">
          <div className="table-title"><h2><Dot />With and without Dropout</h2><span>gap = {GAP_FINAL}</span></div>
          <div className="setting-list">
            {ablationRuns.map((run) => (
              <article key={run.name} className="cols-b">
                <b>{run.name}</b>
                <code>final val {pct(run.finalVal)}</code>
                <span>gap {pts(run.gap)} · best val {pct(run.bestVal)} · min val loss {run.minValLoss.toFixed(4)}</span>
              </article>
            ))}
          </div>
        </div>
        <article className="soft-card">
          <h2><Dot tone="green" />It raises the peak and closes the gap</h2>
          <p>{ablationVerdict}</p>
        </article>
      </div>
      <ChartCard title="Dropout ablation — validation accuracy" note="best val differs by 89 images out of 5,000">
        <LineChart lines={seriesFrom(curves.ablation, "valAcc")} yFormat={(v) => `${(v * 100).toFixed(0)}%`} caption="Validation accuracy per epoch with and without Dropout." />
      </ChartCard>

      <SectionHeading note="Part 15">The final model</SectionHeading>
      <div className="two-col">
        <div className="table-card flush">
          <div className="table-title">
            <h2><Dot tone="green" />{finalModel.architecture}</h2>
            <span>{finalModel.params.toLocaleString()} trainable · {finalModel.nonTrainable} non-trainable</span>
          </div>
          <div className="setting-list">
            {finalModel.settings.map((setting) => (
              <article key={setting.name}>
                <b>{setting.name}</b>
                <code>{setting.value}</code>
                <span>{setting.why}</span>
              </article>
            ))}
          </div>
        </div>
        <div className="stack">
          <ChartCard title="Final model — validation accuracy" note={finalModel.epochsRun < finalModel.epochsAllowed ? `stopped at epoch ${finalModel.epochsRun} of ${finalModel.epochsAllowed}` : `ran all ${finalModel.epochsAllowed} epochs — Early Stopping never fired`}>
            <LineChart
              lines={[
                { label: "Training accuracy", values: curves.finalModel["Final Model"].acc, color: SERIES_COLORS[0] },
                { label: "Validation accuracy", values: curves.finalModel["Final Model"].valAcc, color: SERIES_COLORS[4], dashed: true },
              ]}
              yFormat={(v) => `${(v * 100).toFixed(0)}%`}
              caption="Training and validation accuracy for the final model, which Early Stopping halted at epoch 21."
            />
          </ChartCard>
        </div>
      </div>

      <SectionHeading note="Part 16 · the test set was opened exactly once">Final test results</SectionHeading>
      <div className="metric-grid">
        <Metric value={pct(testResults.accuracy)} label="test accuracy" best />
        <Metric value={pct(testResults.f1)} label="weighted F1-score" tone="purple" />
        <Metric value={testResults.loss.toFixed(4)} label="test loss" tone="orange" />
        <Metric value={testResults.size.toLocaleString()} label="test images, never seen during any decision" tone="aqua" />
      </div>
      <div className="table-card">
        <div className="table-title">
          <h2><Dot />Per-class F1 on the test set</h2>
          <span>weighted precision {pct(testResults.precision)} · recall {pct(testResults.recall)}</span>
        </div>
        <RunList
          max={bestF1}
          runs={[...perClass]
            .sort((a, b) => b.f1 - a.f1)
            .map((c) => ({
              label: `${c.id} · ${c.name}`,
              value: c.f1,
              selected: c.f1 === bestF1,
              tone: c.f1 < 0.45 ? ("bad" as const) : undefined,
              caption: `precision ${pct(c.precision)} · recall ${pct(c.recall)} · 1,000 test images`,
            }))}
        />
      </div>

      <SectionHeading note="Part 15, re-run · a 2×2 that was not in the original submission">Does Dropout belong in the final model?</SectionHeading>
      <div className="rule-note">
        <Dot tone="orange" />
        <div>
          <strong>The experiment that decided which regularizer the final model ships.</strong>
          <p>
            Part 12 measured four regularizers and Dropout won on every metric it reported, but the first version of
            Part 15 shipped L2 and left Dropout out entirely. This re-run crosses Dropout on/off with L2 on/off,
            everything else identical, to settle it — and its recommendation was adopted, so Part 15 now ships
            Dropout 0.3 + Early Stopping.
          </p>
        </div>
      </div>

      <div className="table-card">
        <div className="table-title">
          <h2><Dot tone="purple" />Four configurations, one variable at a time</h2>
          <span>{rerunProtocol.shared}</span>
        </div>
        <div className="table-scroll">
          <table className="log-table">
            <thead>
              <tr>
                <th>Configuration</th><th>Dropout</th><th>L2</th><th>Epochs</th>
                <th>Best val acc.</th><th>Best epoch</th><th>Gap</th><th>Test acc.</th><th>Test loss</th>
              </tr>
            </thead>
            <tbody>
              {rerunRuns.map((run) => (
                <tr key={run.key} className={run.shipped ? "selected" : undefined}>
                  <td>
                    {run.label}
                    {run.shipped ? <em className="tag tag-shipped">shipped</em> : null}
                    {run.winner ? <em className="tag tag-top">highest val</em> : null}
                  </td>
                  <td>{run.dropout ?? "—"}</td>
                  <td>{run.l2 ?? "—"}</td>
                  <td>{run.epochsRun}</td>
                  <td>{pct(run.bestVal)}</td>
                  <td>{run.bestValEpoch}</td>
                  <td>{run.gap.toFixed(4)}</td>
                  <td>{pct(run.test)}</td>
                  <td>{run.testLoss.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="footnote">
        <strong>Highest val and shipped are deliberately different rows.</strong> Dropout + L2 scores highest, but
        its lead over Dropout alone is 0.74 points — one standard error — and Parts 4–7 already rejected extra depth on
        exactly that reasoning. Applying the project&rsquo;s own significance rule consistently means taking the simpler
        model. Every run early-stopped rather than hitting the 100-epoch cap, so all four are converged and the
        comparison is not truncated. Gap is <code>{GAP_PEAK}</code>. {rerunProtocol.testDiscipline}
      </p>

      <div className="two-col">
        <div className="table-card flush">
          <div className="table-title"><h2><Dot />Reading the 2×2, one contrast at a time</h2><span>validation points · 1 s.e. ≈ 0.7</span></div>
          <div className="qa-list">
            {rerunContrasts.map((contrast) => (
              <article key={contrast.question}>
                <h3>
                  <Dot tone={contrast.verdict === "yes" ? "green" : "red"} />
                  {contrast.question}
                </h3>
                <p>
                  <strong>{contrast.pair}: {contrast.points > 0 ? "+" : ""}{contrast.points.toFixed(2)} pts
                  {" "}({contrast.images > 0 ? "+" : ""}{contrast.images} images)</strong> — {contrast.note}
                </p>
              </article>
            ))}
          </div>
        </div>
        <div className="stack">
          <ChartCard title="Validation accuracy — the four configurations" note="the two Dropout runs separate from the two without it">
            <LineChart lines={seriesFrom(curves.part15Rerun, "valAcc")} yFormat={(v) => `${(v * 100).toFixed(0)}%`} caption="Validation accuracy per epoch for L2 only, Dropout only, Dropout + L2 and the Early Stopping control." />
          </ChartCard>
          <article className="soft-card card-good">
            <h3>What ships</h3>
            <p>{rerunVerdict.recommendation}</p>
          </article>
        </div>
      </div>

      <div className="two-col">
        <article className="soft-card">
          <h2><Dot tone="green" />Why Dropout works here</h2>
          <p>{rerunVerdict.mechanism}</p>
        </article>
        <article className="soft-card card-warn">
          <h3>The same config, run twice</h3>
          <p>
            The notebook&rsquo;s Part 15 reached {pct(reproducibilityNote.notebookVal)} validation and{" "}
            {pct(reproducibilityNote.notebookTest)} test. This re-run of the identical configuration, same seed, reached{" "}
            {pct(reproducibilityNote.rerunVal)} and {pct(reproducibilityNote.rerunTest)}.
          </p>
          <p>{reproducibilityNote.why}</p>
        </article>
      </div>
      <Insight tone="green">{rerunVerdict.headline} {rerunVerdict.honest}</Insight>

      <SectionHeading note={`Extra experiment · ${augmentationEpochs} epochs, augmentation applied before Flatten`}>Data augmentation</SectionHeading>
      <div className="table-card">
        <div className="table-title"><h2><Dot tone="green" />More spatial variety, zero extra parameters</h2><span>against a no-augmentation control</span></div>
        <div className="table-scroll">
          <table className="log-table">
            <thead>
              <tr>
                <th>Setting</th><th>Transformations</th><th>Best val acc.</th><th>Best epoch</th>
                <th>Min val loss</th><th>Final train</th><th>Gap</th><th>Change</th>
              </tr>
            </thead>
            <tbody>
              {augmentationRuns.map((run) => (
                <tr key={run.name} className={run.selected ? "selected" : undefined}>
                  <td>{run.name}</td>
                  <td>{run.transforms}</td>
                  <td>{pct(run.bestVal)}</td>
                  <td>{run.bestValEpoch}</td>
                  <td>{run.minValLoss.toFixed(4)}</td>
                  <td>{pct(run.finalTrain)}</td>
                  <td>{run.gap.toFixed(4)}</td>
                  <td>{run.change === 0 ? "control" : `+${run.change.toFixed(2)} pts · +${run.images} images`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="footnote">{augmentationVerdict}</p>
      <Insight tone="green">
        +6.28 points and the gap nearly closed, from 0.4085 to 0.0791 — the strongest single result in the project. A
        flipped car is still a car, but to a Dense network it is an entirely different 3,072-value vector.
      </Insight>
    </section>
  );
}

// ---------------------------------------------------------------- view 3

/** Ranked confusion pairs, derived from the matrix rather than typed by hand. */
function useConfusionPairs() {
  return useMemo(() => {
    const pairs: { a: number; b: number; total: number; detail: string }[] = [];
    for (let a = 0; a < confusionMatrix.length; a += 1) {
      for (let b = a + 1; b < confusionMatrix.length; b += 1) {
        pairs.push({
          a,
          b,
          total: confusionMatrix[a][b] + confusionMatrix[b][a],
          detail: `${confusionMatrix[a][b]} ${classNames[a]}→${classNames[b]} · ${confusionMatrix[b][a]} ${classNames[b]}→${classNames[a]}`,
        });
      }
    }
    return pairs.sort((x, y) => y.total - x.total).slice(0, 6);
  }, []);
}

function ConfusionMatrix() {
  const offDiagonalMax = useMemo(
    () => Math.max(...confusionMatrix.flatMap((row, i) => row.filter((_, j) => i !== j))),
    [],
  );
  const diagonalMax = useMemo(() => Math.max(...confusionMatrix.map((row, i) => row[i])), []);

  return (
    <div className="matrix-wrap">
      <table className="matrix">
        <caption>Rows are the true class, columns the prediction. Each row totals the 1,000 test images of that class.</caption>
        <thead>
          <tr>
            <th scope="col" />
            {classNames.map((name, i) => (
              <th scope="col" key={name}>{i}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {confusionMatrix.map((row, i) => (
            <tr key={classNames[i]}>
              <th scope="row">{i} · {classNames[i]}</th>
              {row.map((value, j) => {
                const correct = i === j;
                const intensity = correct ? value / diagonalMax : value / offDiagonalMax;
                const alpha = 0.05 + intensity * 0.62;
                return (
                  <td
                    key={classNames[j]}
                    className={correct ? "diagonal" : undefined}
                    style={{
                      background: correct ? `rgba(27, 175, 122, ${alpha})` : `rgba(82, 44, 186, ${alpha})`,
                      color: alpha > 0.45 ? "white" : "var(--ink)",
                    }}
                  >
                    {value}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Insights({ navigate }: { navigate: (view: ViewId) => void }) {
  const pairs = useConfusionPairs();
  const sorted = useMemo(() => [...perClass].sort((a, b) => b.f1 - a.f1), []);
  const easiest = sorted.slice(0, 3);
  const hardest = sorted.slice(-3).reverse();

  return (
    <section className="view">
      <PageHeading
        eyebrow="Parts 17–19 · error analysis, tracking and conclusion"
        title="Insights & reflection"
        subtitle="Where the model fails is not random. The errors concentrate between classes that differ by shape and local texture — exactly the information flattening throws away."
      />

      <div className="metric-grid">
        <Metric value={`${pairs[0].total}`} label={`errors between ${classNames[pairs[0].a]} and ${classNames[pairs[0].b]} — the largest pair in the matrix`} tone="red" />
        <Metric value={pct(easiest[0].f1)} label={`best class F1 — ${easiest[0].name}`} best />
        <Metric value={pct(hardest[0].f1)} label={`worst class F1 — ${hardest[0].name}`} tone="orange" />
        <Metric value={`${experimentLog.length}`} label="tracked experiments across the whole project" tone="purple" />
      </div>

      <SectionHeading note="Part 17">Where it fails</SectionHeading>
      <div className="table-card">
        <div className="table-title">
          <h2><Dot tone="red" />Confusion matrix</h2>
          <span>final model · {testResults.size.toLocaleString()} test images</span>
        </div>
        <ConfusionMatrix />
      </div>

      <div className="two-col">
        <div className="table-card flush">
          <div className="table-title"><h2><Dot tone="orange" />Largest confusion pairs</h2><span>both directions summed</span></div>
          <div className="run-list">
            {pairs.map((pair) => (
              <div className="run-row" key={`${pair.a}-${pair.b}`}>
                <span>{classNames[pair.a]} ↔ {classNames[pair.b]}</span>
                <div className="mini-track" role="img" aria-label={`${classNames[pair.a]} and ${classNames[pair.b]}: ${pair.total} errors`}>
                  <i className="fill-bad" style={{ width: `${Math.max(2, (pair.total / pairs[0].total) * 100)}%` }} />
                </div>
                <strong>{pair.total}</strong>
                <span className="run-caption">{pair.detail}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="stack">
          <article className="soft-card">
            <h2><Dot tone="green" />Easiest classes</h2>
            <StatRows rows={easiest.map((c) => [c.name, `${pct(c.f1)} F1`] as const)} />
          </article>
          <article className="soft-card">
            <h2><Dot tone="red" />Hardest classes</h2>
            <StatRows rows={hardest.map((c) => [c.name, `${pct(c.f1)} F1`] as const)} />
          </article>
          <article className="soft-card card-warn">
            <h3>Why these pairs</h3>
            <p>
              Cats and dogs share shapes, poses and colours. At 32 × 32 the resolution removes the detail that separates
              them, and a Dense network sees the image only after it has been flattened, so it never directly represents
              the spatial relationships that remain.
            </p>
          </article>
        </div>
      </div>

      <SectionHeading note="Part 19">What moved the needle</SectionHeading>
      <div className="findings-grid">
        {findings.map((finding) => (
          <article key={finding.n}>
            <span>{finding.n}</span>
            <h3>{finding.title}</h3>
            <p>{finding.detail}</p>
          </article>
        ))}
      </div>
      <Insight>
        The constraint in the brief mattered more to the final number than any hyperparameter chosen after it.
      </Insight>

      <SectionHeading note="Part 18 · every run, one row each">Experiment log</SectionHeading>
      <div className="table-card">
        <div className="table-title">
          <h2><Dot />Tracked experiments</h2>
          <span>best validation accuracy · a dash means the setting was not varied in that run</span>
        </div>
        <div className="table-scroll">
          <table className="log-table">
            <thead>
              <tr>
                <th>Experiment</th><th>Architecture</th><th>LR</th><th>Batch</th><th>Optimizer</th>
                <th>Dropout</th><th>BatchNorm</th><th>L2</th><th>Early stop</th><th>Best val acc.</th>
              </tr>
            </thead>
            <tbody>
              {experimentLog.map((row) => (
                <tr
                  key={row.name}
                  className={row.selected ? "selected" : row.bestVal < 0.46 ? "worst" : undefined}
                >
                  <td>{row.name}</td><td>{row.architecture}</td><td>{row.lr}</td><td>{row.batch}</td>
                  <td>{row.optimizer}</td><td>{row.dropout}</td><td>{row.batchnorm}</td><td>{row.l2}</td>
                  <td>{row.earlyStop}</td><td>{pct(row.bestVal)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="footnote">
        Thirteen tracked runs span 44.64% to 54.62% — 9.98 points, and the bottom of that range is the run with the
        highest training accuracy in the entire project.
      </p>

      <SectionHeading note="Found after submission, then adopted">The regularizer decision, revisited</SectionHeading>
      <div className="two-col">
        <article className="soft-card card-good">
          <h2><Dot tone="green" />Part 15 now ships the regularizer the evidence backs</h2>
          <p>
            Part 12 ranked Dropout first on every metric it measured, but the first version of the final model used L2
            and no Dropout.
          </p>
          <p>
            A 2×2 re-run — Dropout on/off crossed with L2 on/off, everything else held — showed L2 contributes nothing
            measurable either alone (+0.40 pts) or on top of Dropout (+0.74 pts). That finding was adopted: Part 14 is
            now a Dropout ablation and the final model is <strong>Dropout 0.3 + Early Stopping</strong>.
          </p>
          <button className="link-button" onClick={() => navigate("after")}>
            See the full 2×2 on tab 02 →
          </button>
        </article>
        <div className="stack">
          <article className="soft-card">
            <h3>Why it is worth saying out loud</h3>
            <p>
              Part 15&rsquo;s brief is explicit: <em>&ldquo;Do not include a technique simply because it is popular. Every
              important choice in your final model should be supported by evidence from your experiments.&rdquo;</em> The
              first version of the final model failed that test, and the fix is more useful reported than buried.
            </p>
          </article>
          <article className="soft-card card-good">
            <h3>What it changed</h3>
            <p>
              Part 16 now reports {pct(testResults.accuracy)} for the Dropout model, against 52.94% for the L2 model
              built first. Adding L2 back on top would be worth 0.74 points — one standard error — so the simpler
              model stands.
            </p>
          </article>
        </div>
      </div>

      <SectionHeading note="Part 19 · reflection">What we took away</SectionHeading>
      <div className="reflection-grid">
        {reflection.map((item) => (
          <article key={item.label}>
            <span>{item.label}</span>
            <p>{item.text}</p>
          </article>
        ))}
      </div>
      <div className="transition">
        <span>The conclusion</span>
        <p>{conclusion}</p>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- view 4

function Deployment() {
  return (
    <section className="view">
      <PageHeading
        eyebrow="Part 20 · deployment and the road to computer vision"
        title="Deployment & CNN comparison"
        subtitle="What this model is honestly fit for, and why the remaining gap is architectural rather than a tuning failure that one more sweep would have closed."
      />

      <SectionHeading note="What would actually ship">The deployable artefact</SectionHeading>
      <div className="deploy-hero">
        <article className="deploy-card">
          <h2>{deploymentCard.what}</h2>
          <p>
            Early Stopping restored the weights from the best epoch, so the shipped model is the one that scored best on
            validation — not the one that had trained longest.
          </p>
          <div className="contract">
            <div>
              <span>Input contract</span>
              <p>{deploymentCard.inputContract}</p>
            </div>
            <div>
              <span>Output contract</span>
              <p>{deploymentCard.outputContract}</p>
            </div>
          </div>
        </article>
        <div className="stack stack-tight">
          <Metric value={deploymentCard.params.toLocaleString()} label="trainable parameters" tone="purple" />
          <Metric value={`${deploymentCard.sizeMB} MB`} label="model weights, float32" tone="aqua" />
          <Metric value={pct(deploymentCard.accuracy)} label="test accuracy — the number any deployment decision must start from" tone="orange" />
        </div>
      </div>

      <div className="guidance">
        {deploymentGuidance.map((item) => (
          <article className={`tone-${item.tone}`} key={item.title}>
            <b>{item.tone === "yes" ? "✓" : item.tone === "no" ? "✕" : "→"}</b>
            <div>
              <h3>{item.title}</h3>
              <p>{item.detail}</p>
            </div>
          </article>
        ))}
      </div>
      <Insight tone="red">
        At {pct(deploymentCard.accuracy)} on ten balanced classes, the model is roughly five times better than guessing
        and still wrong about half the time. That is a demonstration, not a product.
      </Insight>

      <SectionHeading note="Part 20 · the four questions">What flattening costs</SectionHeading>
      <div className="flow">
        <div>
          <span>The image</span>
          <strong>32 × 32 × 3</strong>
          <small>height, width and three colour channels — structure intact</small>
        </div>
        <div className="arrow" aria-hidden="true">→</div>
        <div>
          <span>After Flatten</span>
          <strong>3,072 × 1</strong>
          <small>every value survives; the arrangement does not</small>
        </div>
      </div>
      <div className="table-card">
        <div className="table-title"><h2><Dot tone="purple" />The reflection questions</h2></div>
        <div className="qa-list">
          {flatteningCosts.map((item) => (
            <article key={item.question}>
              <h3><Dot tone="purple" />{item.question}</h3>
              <p>{item.answer}</p>
            </article>
          ))}
        </div>
      </div>

      <SectionHeading note="Dense against convolutional">The comparison</SectionHeading>
      <div className="table-card">
        <div className="table-title">
          <h2><Dot />What a CNN gives back</h2>
          <span>the right column is what this project was not allowed to use</span>
        </div>
        <div className="table-scroll">
          <table className="versus">
            <colgroup>
              <col className="col-aspect" />
              <col className="col-side" />
              <col className="col-side" />
            </colgroup>
            <thead>
              <tr>
                <th scope="col">Aspect</th>
                <th scope="col">Dense network — what we built</th>
                <th scope="col">Convolutional network — what comes next</th>
              </tr>
            </thead>
            <tbody>
              {cnnComparison.map((row) => (
                <tr key={row.aspect}>
                  <td>{row.aspect}</td>
                  <td>{row.dense}</td>
                  <td className="cnn">{row.cnn}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <SectionHeading note="From this project's own measurements">Why we know the ceiling is architectural</SectionHeading>
      <div className="evidence-grid">
        {cnnEvidence.map((item) => (
          <article key={item.label}>
            <h3><Dot tone="green" />{item.label}</h3>
            <p>{item.detail}</p>
          </article>
        ))}
      </div>

      <div className="transition">
        <span>Final reflection · Part 20</span>
        <p>{finalReflection}</p>
      </div>

      <footer className="team-line">
        {project.team.join(" · ")}
        <span>{project.course} · {project.org}</span>
      </footer>
    </section>
  );
}

// ---------------------------------------------------------------- shell

export default function CifarDenseApp({ initialView = "before" }: { initialView?: ViewId } = {}) {
  const [view, setView] = useState<ViewId>(initialView);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [view]);

  const content = useMemo(() => {
    switch (view) {
      case "after": return <AfterOptimization />;
      case "insights": return <Insights navigate={setView} />;
      case "deployment": return <Deployment />;
      default: return <BeforeOptimization />;
    }
  }, [view]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <button className="brand" onClick={() => setView("before")} aria-label="Back to the first section">
            <span>DNN</span>
            <div>
              <strong>CIFAR-10, Dense Only</strong>
              <small>Deep Learning Capstone</small>
            </div>
          </button>
          <p className="brand-meta">
            <b>{project.team.join(" · ")}</b>
            {project.course} · {project.org}
          </p>
        </div>
        <nav className="tabs" aria-label="Project sections">
          {navItems.map(([id, number, label, parts]) => (
            <button
              key={id}
              className={view === id ? "active" : ""}
              aria-current={view === id ? "page" : undefined}
              onClick={() => setView(id)}
            >
              <span>{number}</span>
              <div><b>{label}</b><small>{parts}</small></div>
            </button>
          ))}
        </nav>
      </header>
      <main>{content}</main>
    </div>
  );
}
