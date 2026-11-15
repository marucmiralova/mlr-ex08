# Task 4 — what each change brings.
#
# This is the Task 3 pipeline, cell by cell, with the variants of Task 4 next to
# it. Every variant changes ONE thing in it (a transform, one layer of LeafNet,
# the optimizer, the number of epochs, the model or the training folder) and is
# trained on the same images with a fixed seed. The same trained network is then
# measured on three sets: the group validation photographs, the camera-like set
# built from them, and last year's photographs.
#
# The same code lives in the notebook as the cells of Task 4; running this file
# writes every run as one line of JSON into task4_results/results.jsonl.
#
#   python task4_variants.py                 # every variant, one seed
#   python task4_variants.py --seeds 0 1 2   # repeated rows
#   python task4_variants.py --only aug mob-full

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(int(os.environ.get("NTHREADS", "6")))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


class LeafDataset(Dataset):                                    # Task 3a, unchanged
    def __init__(self, folder, transform):
        self.transform = transform
        folder_path = Path(folder)
        self.classes = sorted([d.name for d in folder_path.iterdir() if d.is_dir()])
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            for path in sorted((folder_path / class_name).glob("*")):
                if path.is_file():
                    self.samples.append((str(path), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, class_idx = self.samples[i]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, class_idx


class LeafNet(nn.Module):                                      # Task 3b, unchanged
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class LeafNetBN(LeafNet):                                      # Task 3d, unchanged
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# --- the two changes of LeafNet itself: one line each, copied from LeafNet -----

class LeafNetGAP(LeafNet):
    # one change: global average pooling (64 x 6 x 6 -> 64 numbers) instead of
    # flattening (2304 numbers), so the first linear layer is 36 x smaller.
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.mean(dim=[2, 3])
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class LeafNetDrop(LeafNet):
    # one change: dropout between the two linear layers
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(self.drop(x))


class LeafNetCombined(LeafNet):
    # the three changes of LeafNet together: batch normalization after every
    # convolution, dropout before the output layer and global average pooling
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 64)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.mean(dim=[2, 3])
        x = torch.relu(self.fc1(x))
        return self.fc2(self.drop(x))


# --- the datasets of Task 4 -------------------------------------------------

# geometric augmentation exactly as in Task 3d
t4_transform_aug = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# the same augmentation with one new line: colour augmentation
t4_transform_aug_color = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

t4_train_plain = LeafDataset("data/leaves/train", transform)
t4_train_aug = LeafDataset("data/leaves/train", t4_transform_aug)
t4_train_aug_color = LeafDataset("data/leaves/train", t4_transform_aug_color)
t4_val_ds = LeafDataset("data/leaves/val", transform)
t4_val_loader = DataLoader(t4_val_ds, batch_size=32)
t4_hist_train = LeafDataset("data/leaves_hist/train", t4_transform_aug_color)
t4_hist_val_ds = LeafDataset("data/leaves_hist/val", transform)
t4_hist_val_loader = DataLoader(t4_hist_val_ds, batch_size=32)

# last year's photographs added to our own training images
t4_train_more = ConcatDataset([t4_train_aug_color, t4_hist_train])

# The camera-like validation set: every validation image once through a random
# crop, rotation and colour change, drawn with a fixed seed and kept as tensors,
# so that every variant is measured on exactly the same 62 harder images.
t4_camera_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
torch.manual_seed(0)
_t4_camera_ds = LeafDataset("data/leaves/val", t4_camera_transform)
t4_camera_images = torch.stack([_t4_camera_ds[i][0] for i in range(len(_t4_camera_ds))])
t4_camera_labels = torch.tensor([_t4_camera_ds[i][1] for i in range(len(_t4_camera_ds))])
t4_camera_loader = DataLoader(TensorDataset(t4_camera_images, t4_camera_labels), batch_size=32)


# --- the training loop of Task 3c, with one argument per change --------------

def fixed_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def accuracy(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def make_mobilenet(freeze_backbone=False):
    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    if freeze_backbone:                       # only the new last layer is trained
        for p in model.features.parameters():
            p.requires_grad = False
    return model


def run_variant(name, change, make_model, train_dataset, n_epochs=15, lr=1e-3,
                optimizer="Adam", weight_decay=0.0, label_smoothing=0.0,
                scheduler=None, keep_best=False, freeze_backbone=False, seed=0):
    fixed_seed(seed)
    model = make_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
                 if scheduler == "cosine" else None)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))

    start = time.perf_counter()
    best_val, best_state = -1.0, None
    for epoch in range(n_epochs):
        model.train()
        if freeze_backbone:            # the frozen part keeps its batch-norm statistics
            model.features.eval()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
        if scheduler is not None:
            scheduler.step()
        val_acc = accuracy(model, t4_val_loader)
        if keep_best and val_acc > best_val:
            best_val, best_state = val_acc, copy.deepcopy(model.state_dict())
        print(f"  [{name} seed {seed}] epoch {epoch + 1:2d} "
              f"loss {loss_sum / len(train_dataset):.4f} val {val_acc:.3f}", flush=True)
    if keep_best and best_state is not None:
        model.load_state_dict(best_state)

    row = {"name": name, "change": change, "seed": seed, "epochs": n_epochs,
           "n_train": len(train_dataset), "seconds": round(time.perf_counter() - start, 1),
           "train_loss": loss_sum / len(train_dataset),
           "val": accuracy(model, t4_val_loader),
           "camera": accuracy(model, t4_camera_loader),
           "hist": accuracy(model, t4_hist_val_loader),
           "model_parameters": sum(p.numel() for p in params)}
    return row, model        # the trained network, so that the last cell can export it


VARIANTS = [
    # name, what one change it adds, and the arguments of run_variant
    ("baseline", "Task 3c: LeafNet, no augmentation", dict(
        make_model=LeafNet, train_dataset=t4_train_plain)),
    ("aug", "+ geometric augmentation (Task 3d)", dict(
        make_model=LeafNet, train_dataset=t4_train_aug)),
    ("aug+BN", "aug + batch normalization (Task 3d)", dict(
        make_model=LeafNetBN, train_dataset=t4_train_aug)),
    ("aug+color", "aug + ColorJitter", dict(
        make_model=LeafNet, train_dataset=t4_train_aug_color)),
    ("aug+GAP", "aug + global average pooling instead of flatten", dict(
        make_model=LeafNetGAP, train_dataset=t4_train_aug)),
    ("aug+dropout", "aug + dropout 0.3 before the output layer", dict(
        make_model=LeafNetDrop, train_dataset=t4_train_aug)),
    ("aug+AdamW", "aug + AdamW with weight decay 1e-4", dict(
        make_model=LeafNet, train_dataset=t4_train_aug,
        optimizer="AdamW", weight_decay=1e-4)),
    ("aug+smooth", "aug + label smoothing 0.1", dict(
        make_model=LeafNet, train_dataset=t4_train_aug, label_smoothing=0.1)),
    ("aug+bestep", "aug + keep the weights of the best epoch", dict(
        make_model=LeafNet, train_dataset=t4_train_aug, keep_best=True)),
    ("aug+cos40", "aug + cosine schedule, 40 epochs", dict(
        make_model=LeafNet, train_dataset=t4_train_aug, n_epochs=40, scheduler="cosine")),
    ("scratch-best", "combined: BN + dropout + GAP + colour aug + AdamW + smoothing "
                     "+ cosine 40 epochs + best epoch", dict(
        make_model=LeafNetCombined, train_dataset=t4_train_aug_color, n_epochs=40,
        optimizer="AdamW", weight_decay=1e-4, label_smoothing=0.1,
        scheduler="cosine", keep_best=True)),
    ("mob-head", "Task 2 network pretrained, only the new last layer trained", dict(
        make_model=lambda: make_mobilenet(freeze_backbone=True), train_dataset=t4_train_plain, freeze_backbone=True)),
    ("mob-head+aug", "frozen backbone, but with the geometric augmentation", dict(
        make_model=lambda: make_mobilenet(freeze_backbone=True),
        train_dataset=t4_train_aug, freeze_backbone=True)),
    ("mob-full-plain", "whole network fine-tuned, no augmentation", dict(
        make_model=make_mobilenet, train_dataset=t4_train_plain)),
    ("mob-full", "Task 2: whole pretrained network fine-tuned + geometric aug", dict(
        make_model=make_mobilenet, train_dataset=t4_train_aug)),
    ("mob-full+color", "fine-tuned + ColorJitter on top of the geometric augmentation", dict(
        make_model=make_mobilenet, train_dataset=t4_train_aug_color)),
    ("mob-full+cos30", "fine-tuned + cosine schedule, 30 epochs", dict(
        make_model=make_mobilenet, train_dataset=t4_train_aug,
        n_epochs=30, scheduler="cosine")),
    ("mob-best", "fine-tuned + colour aug + AdamW + smoothing + cosine 30 epochs "
                 "+ best epoch", dict(
        make_model=make_mobilenet, train_dataset=t4_train_aug_color, n_epochs=30,
        optimizer="AdamW", weight_decay=1e-4, label_smoothing=0.1,
        scheduler="cosine", keep_best=True)),
    ("mob-best+hist", "the best network + last year's photographs added to training", dict(
        make_model=make_mobilenet, train_dataset=t4_train_more, n_epochs=30,
        optimizer="AdamW", weight_decay=1e-4, label_smoothing=0.1,
        scheduler="cosine", keep_best=True)),
    ("scratch+hist", "the best from-scratch network + last year's photographs", dict(
        make_model=LeafNetCombined, train_dataset=t4_train_more, n_epochs=40,
        optimizer="AdamW", weight_decay=1e-4, label_smoothing=0.1,
        scheduler="cosine", keep_best=True)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="names of variants to run")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--out", default="task4_results/results.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as results:
        for name, change, settings in VARIANTS:
            if args.only and name not in args.only:
                continue
            for seed in args.seeds:
                row, _ = run_variant(name=name, change=change, seed=seed, **settings)
                print(json.dumps(row), flush=True)
                results.write(json.dumps(row) + "\n")
                results.flush()


if __name__ == "__main__":
    main()
