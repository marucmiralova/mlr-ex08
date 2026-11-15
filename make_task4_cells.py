# Inserts the Task 4 cells into ex08_student.ipynb.
#
# The code of every cell is sliced out of task4_variants.py, so the notebook and
# the script that produced the numbers hold exactly the same lines. Run with
#
#   python make_task4_cells.py
#
# (the notebook is rewritten in place; the old copy is kept as ex08_student.ipynb.bak)

import json
import shutil

NOTEBOOK = "ex08_student.ipynb"
SOURCE = "task4_variants.py"
BACKUP = "ex08_student.ipynb.bak"


def slice_source(start_marker, end_marker):
    """Lines of task4_variants.py from start_marker up to (not including) end_marker."""
    lines = open(SOURCE, encoding="utf-8").read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(start_marker))
    end = next(i for i, line in enumerate(lines) if i > start and line.startswith(end_marker))
    return "\n".join(lines[start:end]).rstrip() + "\n"


# --- the code cells, sliced out of the script --------------------------------

DATA_BODY = slice_source("# --- the datasets of Task 4", "# --- the training loop")
MODEL_BODY = slice_source("# --- the two changes of LeafNet itself", "# --- the datasets of Task 4")
LOOP_BODY = slice_source("def fixed_seed(seed):", "VARIANTS = [")
VARIANTS_BODY = slice_source("VARIANTS = [", "def main():").rstrip()

DATA_CELL = ("# The one import the notebook does not have yet.\n"
             "from torch.utils.data import ConcatDataset\n\n"
             + DATA_BODY)

RUN_CELL = (
    VARIANTS_BODY + "\n\n"
    "# Every variant, three seeds. On a T4 in Colab the whole table takes a few\n"
    "# minutes, on a laptop CPU tens of minutes.\n"
    "rows = []\n"
    "for name, change, settings in VARIANTS:\n"
    "    for seed in [0, 1, 2]:\n"
    "        row, model = run_variant(name=name, change=change, seed=seed, **settings)\n"
    "        rows.append(row)\n"
    "        print(f\"{name:16s} seed {seed}  validation {row['val']:.3f}  \"\n"
    "              f\"camera-like {row['camera']:.3f}  last year {row['hist']:.3f}\")\n"
    "with open('task4_results.jsonl', 'w') as f:\n"
    "    for row in rows:\n"
    "        f.write(json.dumps(row) + chr(10))\n"
)

TABLE_CELL = (
    "# The table of the report: the mean and the spread over the three seeds.\n"
    "def table(rows):\n"
    "    print('| variant | what it changes | validation | camera-like | last year |')\n"
    "    print('| --- | --- | --- | --- | --- |')\n"
    "    for name, change, _ in VARIANTS:\n"
    "        here = [r for r in rows if r['name'] == name]\n"
    "        if not here:\n"
    "            continue\n"
    "        cells = []\n"
    "        for column in ['val', 'camera', 'hist']:\n"
    "            values = np.array([r[column] for r in here])\n"
    "            cells.append(f'{values[0]:.3f}' if len(values) == 1 else\n"
    "                         f'{values.mean():.3f} +/- {values.std(ddof=1):.3f}')\n"
    "        print('| ' + name + ' | ' + change + ' | ' + ' | '.join(cells) + ' |')\n"
    "\n"
    "table(rows)\n"
    "print()\n"
    "print('the same variant with three seeds, validation accuracy in percent:')\n"
    "for name, _, _ in VARIANTS:\n"
    "    here = [100 * r['val'] for r in rows if r['name'] == name]\n"
    "    if len(here) > 1:\n"
    "        print(f'  {name:16s} ' + ', '.join(f'{v:.1f}' for v in here))\n"
    "print(f'one validation image is worth {100 / len(t4_val_ds):.2f} %')\n"
)

EXPORT_CELL = (
    "# The model we would put on the phone, trained once more with the seed of the\n"
    "# table, and exported exactly as in 2e.\n"
    "CHOSEN = \"mob-best\"\n"
    "name, change, settings = [v for v in VARIANTS if v[0] == CHOSEN][0]\n"
    "row, model = run_variant(name=name, change=change, seed=0, **settings)\n"
    "print(f\"{CHOSEN}: validation {row['val']:.3f}  camera-like {row['camera']:.3f}  \"\n"
    "      f\"last year {row['hist']:.3f}\")\n"
    "\n"
    "model.eval().cpu()\n"
    "torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), 'leaf_model_best.onnx',\n"
    "                  input_names=['image'], output_names=['logits'], external_data=False)\n"
    "chosen_info = {'classes': t4_train_plain.classes, 'input_size': 96, 'mean': mean, 'std': std}\n"
    "onnx_model = onnx.load('leaf_model_best.onnx')\n"
    "onnx_model.metadata_props.add(key='model_info', value=json.dumps(chosen_info))\n"
    "onnx.save(onnx_model, 'leaf_model_best.onnx')\n"
    "print(f\"leaf_model_best.onnx: {os.path.getsize('leaf_model_best.onnx') / 1e6:.1f} MB\")\n"
    "\n"
    "# the same check image and its new probabilities, to compare with your page\n"
    "x = transform(Image.open('check_image.jpg').convert('RGB')).unsqueeze(0)\n"
    "with torch.no_grad():\n"
    "    probabilities = model(x).softmax(dim=1)[0]\n"
    "print(dict(zip(t4_train_plain.classes, probabilities.tolist())))\n"
    "Image.open(t4_val_ds.samples[0][0]).save('check_image_best.jpg', quality=95)\n"
)


def markdown(text):
    return {"cell_type": "markdown", "metadata": {},
            "id": "task4md" + str(abs(hash(text)) % 10**8),
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "id": "task4code" + str(abs(hash(text)) % 10**8),
            "source": text.splitlines(keepends=True)}


def main():
    shutil.copy(NOTEBOOK, BACKUP)
    notebook = json.load(open(NOTEBOOK, encoding="utf-8"))
    sources = ["".join(c["source"]) for c in notebook["cells"]]
    if any("run_variant" in s for s in sources):
        raise SystemExit("the notebook already holds the Task 4 cells -- restore "
                         + BACKUP + " or delete them first")

    # insert after the "Task 4 - A better classifier" instructions, before "## Summary"
    at = next(i for i, text in enumerate(sources) if text.startswith("## Summary"))

    cells = [
        markdown(
            "### 4a. How every variant is measured\n"
            "\n"
            "Three things make the comparison fair: the **same training images and a fixed\n"
            "seed** for every variant, **one change at a time** against the same baseline,\n"
            "and **the trained network measured on three sets** instead of one number:\n"
            "\n"
            "- **validation** — the 62 photographs of our group that nothing trained on;\n"
            "- **camera-like** — those same 62 images once through a random crop, a rotation\n"
            "  and a colour change, drawn with a fixed seed and kept as tensors: a fixed set\n"
            "  of harder images, closer to what the phone camera shows;\n"
            "- **last year's** — the 120 validation photographs of the archive ZIP: other\n"
            "  students, other phones, nobody of our group behind the camera.\n"
            "\n"
            "The last two are what the phone really has to do. The first is what we tune\n"
            "against, so we expect it to flatter the network more than the other two do.\n"
            "\n"
            "The datasets of Task 4 are named `t4_...` so that they do not quietly replace\n"
            "the `train_ds` and `val_ds` of Tasks 2 and 3.\n"),
        code(DATA_CELL),
        markdown(
            "Two changes of the network itself, each one line different from `LeafNet`, and\n"
            "the pretrained MobileNet of Task 2 as a function, so that a variant can ask for\n"
            "it with the backbone frozen or not.\n"),
        code(MODEL_BODY),
        markdown(
            "### 4b. The loop of 3c, with one argument per change\n"
            "\n"
            "`run_variant` is the training loop of 3c with nothing added to it but an\n"
            "argument for every change we want to measure: `n_epochs`, `optimizer` and\n"
            "`weight_decay`, `label_smoothing`, `scheduler=\"cosine\"`, `keep_best` (the\n"
            "weights of the best epoch, as in 1g) and `freeze_backbone`. A variant is\n"
            "therefore one line in the table of variants, not another copy of the loop.\n"
            "\n"
            "Read the frozen case carefully: `model.features.eval()` keeps the batch\n"
            "normalization of the frozen part on the statistics it came with. Without that\n"
            "line the \"frozen\" features would still drift with our 244 photographs, and the\n"
            "row would not be the experiment we think it is.\n"),
        code(LOOP_BODY),
        markdown(
            "Each row below is the baseline with **one** thing changed; the last rows\n"
            "combine what helped and add last year's photographs. Everything is trained with\n"
            "three seeds, because one run on 62 validation images is noisy — and the noise,\n"
            "not only the accuracy, is a result we need for the report.\n"
            "\n"
            "Two runs with the same seed in the same notebook reproduce exactly, but the same\n"
            "seed on another machine, or with another number of compute threads, moves a row\n"
            "by a couple of points. Expect the table you rebuild here to differ from the one\n"
            "in `REPORT.md` a little — that difference *is* the noise we keep quoting.\n"),
        code(RUN_CELL),
        markdown("The table of the report, with the spread over the seeds.\n"),
        code(TABLE_CELL),
        markdown(
            "### 4c. The model for the phone\n"
            "\n"
            "The chosen line of the table, trained once more with its seed and exported as\n"
            "in 2e. The architecture and the preprocessing are those of Task 2, so the file\n"
            "is the same size and runs just as fast in the browser — only the weights are\n"
            "better. To try it in the course camera app, take `leaf_model_best.onnx` under\n"
            "the name `leaf_model.onnx`.\n"),
        code(EXPORT_CELL),
    ]
    notebook["cells"][at:at] = cells
    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"inserted {len(cells)} cells at position {at} of {NOTEBOOK}")


if __name__ == "__main__":
    main()
