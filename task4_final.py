# Task 4 — the model we would put on the phone.
#
# Trains the winning variant again with the same seed as in the table, so that we
# hold the trained network in hand, and then
#   * prints the confusion matrix on our validation images and on last year's,
#   * measures it once more, averaging each image with its mirror,
#   * exports it to ONNX exactly as in 2e, with the preprocessing in the metadata,
#   * writes the two figures of the report.
#
#   python task4_final.py
#
# Nothing here changes the comparison: WINNER is one line of the VARIANTS table.

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")                      # figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import onnx
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay

import task4_variants as T

WINNER = "mob-best"
SEED = 0
device = T.device

# the colours and ink of the report figures (one light surface, three hues)
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": "#dcdad5", "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
    "grid.color": "#e8e6e1", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def load_rows(pattern="task4_results/*.jsonl"):
    rows = []
    for file in sorted(glob.glob(pattern)):
        for line in open(file, encoding="utf-8"):
            rows.append(json.loads(line))
    return rows


def settings_of(name):
    for variant_name, change, settings in T.VARIANTS:
        if variant_name == name:
            return change, settings
    raise KeyError(name)


def accuracy_mirrored(model, loader):
    # a mirrored leaf is still the same species, so the logits of an image and of
    # its mirror are averaged before the argmax: the same network, a bit more careful
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images) + model(images.flip(-1))
            correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
    return correct / len(loader.dataset)


def epoch_curve(name, seed, files):
    pattern = re.compile(r"\[(\S+) seed (\d+)\] epoch\s+(\d+) loss \S+ val (\S+)")
    epochs, values = [], []
    for file in files:
        for match in pattern.finditer(open(file, encoding="utf-8").read()):
            if match.group(1) == name and match.group(2) == str(seed):
                epochs.append(int(match.group(3)))
                values.append(100 * float(match.group(4)))
    return epochs, values


def confusion_figure(model):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, (loader, dataset, title) in zip(
            axes, [(T.t4_val_loader, T.t4_val_ds, "our 62 validation photographs"),
                   (T.t4_hist_val_loader, T.t4_hist_val_ds, "last year's 120 photographs")]):
        images = torch.cat([batch for batch, _ in loader]).to(device)
        labels = torch.cat([targets for _, targets in loader])
        with torch.no_grad():
            predictions = model(images).argmax(dim=1).cpu()
        ConfusionMatrixDisplay.from_predictions(
            labels.numpy(), predictions.numpy(), display_labels=dataset.classes, ax=ax,
            colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(f"{title}\n{100 * (predictions == labels).float().mean():.1f} % correct")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.grid(False)
    fig.suptitle(f"The model for the phone ({WINNER}): which species get mixed up", color=INK)
    fig.tight_layout()
    fig.savefig("task4_confusion.png", dpi=200, facecolor=SURFACE)
    print("wrote task4_confusion.png")


def results_figure(rows):
    order = ["baseline", "aug", "mob-head", "mob-full+color", "mob-best"]
    sets = [("our validation", "val", BLUE), ("camera-like", "camera", ORANGE),
            ("last year's", "hist", AQUA)]

    fig, (bars, lines) = plt.subplots(1, 2, figsize=(10.6, 4.0), width_ratios=[1.2, 1])

    x = np.arange(len(order))
    for i, (label, column, color) in enumerate(sets):
        means, spreads = [], []
        for name in order:
            values = [100 * r[column] for r in rows if r["name"] == name]
            means.append(np.mean(values))
            spreads.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
        bars.bar(x + (i - 1) * 0.27, means, width=0.25, color=color, label=label,
                 yerr=spreads, error_kw={"lw": 1, "ecolor": INK_2, "capsize": 2})
    # one accuracy label per bar only for the model we choose; the table has the rest
    for i, (_, column, _) in enumerate(sets):
        value = [100 * r[column] for r in rows if r["name"] == "mob-best"]
        value = np.mean(value)
        bars.text(x[-1] + (i - 1) * 0.27, value + 2.5, f"{value:.0f}", ha="center",
                  fontsize=8.5, color=INK)
    bars.set_xticks(x, ["baseline", "+ aug", "frozen\nhead", "fine-tuned\n+ colour",
                        "our model"], fontsize=8.5)
    bars.set_ylim(0, 112)
    bars.set_yticks([0, 25, 50, 75, 100])
    bars.set_ylabel("accuracy in %")
    bars.set_title("One variant, three test sets\n(mean of three seeds; the last three use\n"
                   "the pretrained MobileNet, the first two LeafNet)")
    bars.legend(loc="upper left", fontsize=8.5)
    bars.grid(axis="x", visible=False)

    files = ["task4_logs_scratch.txt", "task4_logs_transfer.txt"]
    for name, color in [("baseline", BLUE), ("aug", ORANGE), ("mob-best", AQUA)]:
        epochs, values = epoch_curve(name, 0, files)
        lines.plot(epochs, values, color=color, lw=2, marker="o", ms=3.5,
                   markevery=max(1, len(epochs) // 6))
        lines.text(epochs[-1] + 0.8, values[-1], name, color=color, fontsize=8.5,
                   va="center")
    lines.set_xlabel("epoch")
    lines.set_ylabel("validation accuracy in %")
    lines.set_xlim(1, 35)
    lines.set_ylim(20, 100)
    lines.set_yticks([20, 40, 60, 80, 100])
    lines.set_title("Every epoch, on our 62 validation images\n(seed 0; one image is 1.6 %)")

    fig.tight_layout()
    fig.savefig("task4_figure.png", dpi=200, facecolor=SURFACE)
    print("wrote task4_figure.png")


def main():
    change, settings = settings_of(WINNER)
    print(f"retraining {WINNER} (seed {SEED}): {change}")
    row, model = T.run_variant(name=WINNER, change=change, seed=SEED, **settings)
    print(json.dumps(row, indent=1))

    model.eval().to(device)
    for loader, title in [(T.t4_val_loader, "validation  "), (T.t4_camera_loader, "camera-like"),
                          (T.t4_hist_val_loader, "last year's ")]:
        print(f"{title} plain {T.accuracy(model, loader):.3f}  "
              f"mirrored {accuracy_mirrored(model, loader):.3f}")

    confusion_figure(model)
    results_figure(load_rows())

    # --- the export of 2e, with the chosen model -----------------------------
    model.eval().cpu()
    torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), "leaf_model_best.onnx",
                      input_names=["image"], output_names=["logits"], external_data=False)
    model_info = {"classes": T.t4_train_plain.classes, "input_size": 96, "mean": T.mean, "std": T.std}
    onnx_model = onnx.load("leaf_model_best.onnx")
    onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
    onnx.save(onnx_model, "leaf_model_best.onnx")
    print(f"leaf_model_best.onnx: {os.path.getsize('leaf_model_best.onnx') / 1e6:.1f} MB")

    x = T.transform(Image.open("check_image.jpg").convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(x).softmax(dim=1)[0]
    check_expected = dict(zip(T.t4_train_plain.classes, probabilities.tolist()))
    print("check_image.jpg:", check_expected)
    with open("check_expected_best.json", "w") as f:
        json.dump(check_expected, f, indent=1)
    with open("model_info_best.json", "w") as f:
        json.dump(model_info, f)


if __name__ == "__main__":
    main()
