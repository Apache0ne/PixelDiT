"""Paired source/target datasets for PixelDiT2 image editing."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset

from .data import _clean_filename, _save_fn, center_crop_arr


def _load_rgb(path, image_size):
    image = PIL.Image.open(path).convert("RGB")
    return center_crop_arr(image, image_size)


def _tensor(image):
    raw = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    return (raw - 0.5) / 0.5, raw


class PairedEditDataset(Dataset):
    """JSONL pairs: {source, target, target_class}."""

    def __init__(
        self,
        manifest: str,
        root: str = ".",
        image_size: int = 256,
        random_flip: bool = True,
    ):
        self.root = Path(root)
        self.image_size = int(image_size)
        self.random_flip = bool(random_flip)
        with open(manifest, "r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]
        if not self.items:
            raise ValueError(f"Empty edit manifest: {manifest}")
        for i, item in enumerate(self.items[:100]):
            for key in ("source", "target", "target_class"):
                if key not in item:
                    raise KeyError(f"Manifest item {i} missing {key!r}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        source = _load_rgb(self.root / item["source"], self.image_size)
        target = _load_rgb(self.root / item["target"], self.image_size)
        if self.random_flip and random.random() < 0.5:
            source = source.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            target = target.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        source_norm, source_raw = _tensor(source)
        target_norm, target_raw = _tensor(target)
        target_class = int(item["target_class"])
        metadata = {
            "source_image": source_norm,
            "source_raw_image": source_raw,
            "raw_image": target_raw,
            "class": target_class,
            "source_class": int(item.get("source_class", -1)),
        }
        return target_norm, target_class, metadata


class EditPredictionDataset(Dataset):
    """JSONL inference items: {source, target_class, seed?, filename?, t_start?}."""

    def __init__(
        self,
        manifest: str,
        root: str = ".",
        image_size: int = 256,
    ):
        self.root = Path(root)
        self.image_size = int(image_size)
        with open(manifest, "r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]
        if not self.items:
            raise ValueError(f"Empty edit prediction manifest: {manifest}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        source = _load_rgb(self.root / item["source"], self.image_size)
        source_norm, _ = _tensor(source)
        target_class = int(item["target_class"])
        seed = int(item.get("seed", 1234 + index))
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(
            (3, self.image_size, self.image_size),
            generator=generator,
        )
        filename = item.get(
            "filename",
            f"{_clean_filename(Path(item['source']).stem)}_to_{target_class}_{seed}",
        )
        metadata = {
            "source_image": source_norm,
            "filename": filename,
            "seed": seed,
            "condition": target_class,
            "t_start": float(item.get("t_start", 0.70)),
            "source_strength": float(item.get("source_strength", 1.0)),
            "save_fn": _save_fn,
        }
        return noise, target_class, metadata
