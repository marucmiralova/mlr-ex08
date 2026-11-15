# Builds the Task 4 table from the runs written by task4_variants.py.
#
#   python task4_table.py                      # the table for REPORT.md
#   python task4_table.py --per-seed           # every seed on its own line

import glob
import json
import sys
from collections import defaultdict

import numpy as np

ORDER = ["baseline", "aug", "aug+BN", "aug+color", "aug+GAP", "aug+dropout",
         "aug+AdamW", "aug+smooth", "aug+bestep", "aug+cos40", "scratch-best",
         "mob-head", "mob-head+aug", "mob-full-plain", "mob-full", "mob-full+color",
         "mob-full+cos30", "mob-best",
         "mob-best+hist", "scratch+hist"]


def load(pattern="task4_results/*.jsonl"):
    runs = defaultdict(list)
    changes = {}
    for file in sorted(glob.glob(pattern)):
        for line in open(file, encoding="utf-8"):
            row = json.loads(line)
            runs[row["name"]].append(row)
            changes[row["name"]] = row["change"]
    for name in runs:
        runs[name].sort(key=lambda r: r["seed"])
    return runs, changes


def main():
    runs, changes = load()
    per_seed = "--per-seed" in sys.argv

    if per_seed:
        print("| variant | seed | validation | camera-like | last year | train loss |")
        print("| --- | --- | --- | --- | --- | --- |")
        for name in ORDER:
            for row in runs.get(name, []):
                print(f"| {name} | {row['seed']} | {row['val']:.3f} | {row['camera']:.3f} "
                      f"| {row['hist']:.3f} | {row['train_loss']:.3f} |")
        return

    print("| variant | what it changes | training | validation | camera-like | last year |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name in ORDER:
        if name not in runs:
            continue
        rows = runs[name]
        cells = []
        for column in ["val", "camera", "hist"]:
            values = np.array([r[column] for r in rows])
            if len(values) == 1:
                cells.append(f"{values[0]:.3f}")
            else:
                cells.append(f"{values.mean():.3f} +/- {values.std(ddof=1):.3f}")
        trainings = sorted({f"{r['n_train']} imgs" for r in rows})
        epochs = sorted({r["epochs"] for r in rows})
        print(f"| {name} | {changes[name]} | {'+'.join(trainings)}, {epochs[-1]} ep "
              f"| {' | '.join(cells)} |")

    print()
    print("spread over seeds (validation accuracy, in percent):")
    for name in ORDER:
        rows = runs.get(name, [])
        if len(rows) > 1:
            values = 100 * np.array([r["val"] for r in rows])
            print(f"  {name:16s} {', '.join(f'{v:.1f}' for v in values)}"
                  f"   range {values.max() - values.min():.1f} pp")
    print(f"  one validation image = {100 / 62:.2f} pp")

    print()
    print("seconds per run:", {n: [r["seconds"] for r in runs[n]] for n in ORDER if n in runs})


if __name__ == "__main__":
    main()
