"""Emit Demo/app/curves.ts from the committed Results/*.json and the notebook logs.

Nothing here is typed by hand: the architecture runs come from Results/, and the
optimization runs are parsed back out of the stdout Keras printed during the
9_20 notebook run, so the app can only ever show numbers the project produced.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / 'Notebooks/9_20_optimization_and_final_analysis.ipynb'

EPOCH = re.compile(
    r'accuracy: ([0-9.]+) - loss: ([0-9.]+) - val_accuracy: ([0-9.]+) - val_loss: ([0-9.]+)')

nb = json.loads(NB.read_text(encoding='utf-8'))


def stdout(i):
    return ''.join(''.join(o['text']) for o in nb['cells'][i].get('outputs', []) if 'text' in o)


def rows(text):
    return [[round(float(x), 4) for x in m] for m in EPOCH.findall(text)]


def split_runs(i, pattern):
    parts = re.split(pattern, stdout(i))
    return {parts[j].strip(): rows(parts[j + 1]) for j in range(1, len(parts), 2)}


def to_series(r):
    """[[acc, loss, valAcc, valLoss], ...] -> column-wise series."""
    return {
        'acc':     [x[0] for x in r],
        'loss':    [x[1] for x in r],
        'valAcc':  [x[2] for x in r],
        'valLoss': [x[3] for x in r],
    }


groups = {}

# --- Parts 4-7: read from the committed Results/, not from any notebook log.
arch = {}
for name in ('shallow', 'medium', 'deep'):
    h = json.loads((ROOT / 'Results' / f'{name}.json').read_text(encoding='utf-8'))['history']
    arch[name] = {k: [round(v, 4) for v in h[j]]
                  for k, j in (('acc', 'accuracy'), ('loss', 'loss'),
                               ('valAcc', 'val_accuracy'), ('valLoss', 'val_loss'))}
groups['architecture'] = arch

# --- Parts 9-15: parsed from the notebook's own training logs.
groups['learningRate']   = {k: to_series(v) for k, v in split_runs(15, r'Training with Learning Rate = ([0-9.]+)').items()}
groups['batchSize']      = {k: to_series(v) for k, v in split_runs(22, r'Training with Batch Size = (\d+)').items()}
groups['optimizer']      = {k: to_series(v) for k, v in split_runs(29, r'Training with Optimizer = (\w+)').items()}
groups['regularization'] = {n: to_series(rows(stdout(i)))
                            for n, i in (('Baseline', 35), ('Dropout', 36),
                                         ('Early Stopping', 37), ('L2', 38))}
groups['batchnorm']      = {n: to_series(rows(stdout(i)))
                            for n, i in (('Without BatchNorm', 45), ('With BatchNorm', 46))}
groups['ablation']       = {n: to_series(rows(stdout(i)))
                            for n, i in (('With Dropout', 52), ('Without Dropout', 53))}
groups['finalModel']     = {'Final Model': to_series(rows(stdout(58)))}

# --- Part 15 re-run: the four-way regularizer comparison from
# scripts/part15_rerun.py, which asked whether Dropout belonged in the final
# model. Read from Results/, like the architecture runs.
rerun_path = ROOT / 'Results' / 'part15_rerun.json'
if rerun_path.exists():
    rerun = json.loads(rerun_path.read_text(encoding='utf-8'))
    groups['part15Rerun'] = {
        run['label']: {
            'acc':     [round(v, 4) for v in run['history']['accuracy']],
            'loss':    [round(v, 4) for v in run['history']['loss']],
            'valAcc':  [round(v, 4) for v in run['history']['val_accuracy']],
            'valLoss': [round(v, 4) for v in run['history']['val_loss']],
        }
        for run in rerun['runs'].values()
    }
else:
    print(f'note: {rerun_path} missing, skipping the Part 15 re-run curves')

body = json.dumps(groups, separators=(',', ':'))

out = ROOT / 'Demo/app/curves.ts'
out.write_text(
    "// GENERATED FILE - do not edit by hand.\n"
    "//\n"
    "// Per-epoch training curves for every run in the project. The Parts 4-7\n"
    "// architecture runs are read from Results/*.json; every optimization run is\n"
    "// parsed back out of the stdout Keras printed in\n"
    "// Notebooks/9_20_optimization_and_final_analysis.ipynb. Regenerate with\n"
    "// `npm run curves` after re-running a notebook.\n"
    "//\n"
    "// The Part 12 Early Stopping run has 17 epochs because it stopped early.\n"
    "// Every other notebook run is a full 50; the Part 15 re-run arms stop\n"
    "// between 19 and 46 epochs.\n"
    "\n"
    "/** One training run: four aligned per-epoch series. */\n"
    "export type Series = {\n"
    "  acc: readonly number[];\n"
    "  loss: readonly number[];\n"
    "  valAcc: readonly number[];\n"
    "  valLoss: readonly number[];\n"
    "};\n"
    "\n"
    "/** The runs inside one experiment, keyed by the label the notebook gave them. */\n"
    "export type RunGroup = Record<string, Series>;\n"
    "\n"
    "export const curves: {\n"
    + "".join(f"  {g}: RunGroup;\n" for g in groups)
    + "} = " + body + ";\n",
    encoding="utf-8",
)

for g, runs in groups.items():
    print(g, {k: len(v['valAcc']) for k, v in runs.items()})
print('bytes:', out.stat().st_size)
