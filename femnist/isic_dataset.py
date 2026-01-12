"""Dataset helpers for Fed-ISIC2019."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import requests


TRAIN_TEST_SPLIT_URL = (
    "https://raw.githubusercontent.com/owkin/FLamby/main/"
    "flamby/datasets/fed_isic2019/dataset_creation_scripts/train_test_split"
)


def ensure_split_csv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(TRAIN_TEST_SPLIT_URL, timeout=30)
    resp.raise_for_status()
    path.write_text(resp.text, encoding="utf-8")


class Isic2019Dataset(Dataset):
    """ISIC2019 dataset using the fixed FLamby train/test split."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        center: Optional[int] = None,
        pooled: bool = False,
        preprocessed_dir: str = "ISIC_2019_Training_Input_preprocessed",
        split_csv: Optional[str | Path] = None,
        transform=None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.center = center
        self.pooled = pooled
        self.transform = transform

        if split_csv is None:
            split_csv = self.data_root / "train_test_split"
            ensure_split_csv(Path(split_csv))
        self.split_csv = Path(split_csv)

        df = pd.read_csv(self.split_csv)
        if self.pooled:
            df = df.query("fold == @split").reset_index(drop=True)
        else:
            if center is None:
                raise ValueError("center must be set when pooled=False")
            key = f"{split}_{center}"
            df = df.query("fold2 == @key").reset_index(drop=True)

        images = df["image"].tolist()
        self.targets = df["target"].astype(int).tolist()

        self.input_dir = self.data_root / preprocessed_dir
        if not self.input_dir.is_dir():
            raise FileNotFoundError(
                f"Preprocessed directory not found: {self.input_dir}. "
                "Run isic_preprocess.py first."
            )

        self.image_paths = [self.input_dir / f"{name}.jpg" for name in images]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        target = self.targets[idx]
        if self.transform is not None:
            transformed = self.transform(image=np.asarray(image))
            image = transformed["image"] if isinstance(transformed, dict) else transformed
        return image, target
