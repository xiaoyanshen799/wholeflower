"""Fed-IXI (IXI Tiny) dataset utilities for local training/simulation."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DEFAULT_TARGET_SHAPE = (48, 60, 48)
CENTER_LABELS = {"Guys": 0, "HH": 1, "IOP": 2}
SUBJECT_RE = re.compile(r"IXI(\d{3})-([A-Za-z]+)-")


@dataclass(frozen=True)
class IXISubject:
    patient_id: int
    center: str
    split: str
    image_path: Path
    label_path: Path


def _resolve_ixi_sample_dir(root: Path) -> Path:
    if root.name == "IXI_sample":
        return root
    direct = root / "IXI_sample"
    if direct.is_dir():
        return direct
    for child in root.iterdir():
        if not child.is_dir():
            continue
        candidate = child / "IXI_sample"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"IXI_sample directory not found under: {root}")


def _load_metadata(metadata_path: Path) -> dict[int, str]:
    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        mapping: dict[int, str] = {}
        for row in reader:
            pid = int(row["Patient ID"])
            mapping[pid] = row["Split"].strip().lower()
    return mapping


def _parse_subject(subject_name: str) -> Optional[Tuple[int, str]]:
    match = SUBJECT_RE.search(subject_name)
    if not match:
        return None
    pid = int(match.group(1))
    center = match.group(2)
    return pid, center


def _first_nii(path: Path) -> Optional[Path]:
    files = sorted(path.glob("*.nii.gz"))
    if not files:
        return None
    return files[0]


def _iter_subjects(sample_dir: Path, metadata: dict[int, str]) -> Iterable[IXISubject]:
    for subject_dir in sorted(sample_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        parsed = _parse_subject(subject_dir.name)
        if not parsed:
            continue
        pid, center = parsed
        split = metadata.get(pid)
        if split is None:
            continue
        image_dir = subject_dir / "T1"
        label_dir = subject_dir / "label"
        image_path = _first_nii(image_dir) if image_dir.is_dir() else None
        label_path = _first_nii(label_dir) if label_dir.is_dir() else None
        if image_path is None or label_path is None:
            continue
        yield IXISubject(
            patient_id=pid,
            center=center,
            split=split,
            image_path=image_path,
            label_path=label_path,
        )


def _normalize_center(center: Optional[str | int]) -> Optional[str]:
    if center is None:
        return None
    if isinstance(center, int):
        for name, idx in CENTER_LABELS.items():
            if idx == center:
                return name
        raise ValueError(f"Unknown center id: {center}")
    center = center.strip()
    if center in CENTER_LABELS:
        return center
    if center.isdigit():
        return _normalize_center(int(center))
    raise ValueError(f"Unknown center name: {center}")


def _resize_volume(
    volume: torch.Tensor,
    target_shape: Tuple[int, int, int],
    mode: str,
) -> torch.Tensor:
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {tuple(volume.shape)}")
    vol = volume.unsqueeze(0).unsqueeze(0)
    kwargs = {}
    if mode != "nearest":
        kwargs["align_corners"] = False
    vol = F.interpolate(vol, size=target_shape, mode=mode, **kwargs)
    return vol.squeeze(0).squeeze(0)


class FedIXITinyDataset(Dataset):
    """Load IXI Tiny data with train/test splits and optional center filtering."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        center: Optional[str | int] = None,
        metadata_path: Optional[str | Path] = None,
        target_shape: Tuple[int, int, int] = DEFAULT_TARGET_SHAPE,
        normalize: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split.strip().lower()
        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.center = _normalize_center(center)
        self.target_shape = target_shape
        self.normalize = normalize

        if metadata_path is None:
            metadata_path = Path(__file__).with_name("ixi_metadata_tiny.csv")
        metadata = _load_metadata(Path(metadata_path))
        sample_dir = _resolve_ixi_sample_dir(self.root)

        subjects = list(_iter_subjects(sample_dir, metadata))
        self.samples: List[IXISubject] = []
        for subject in subjects:
            if subject.split != self.split:
                continue
            if self.center and subject.center != self.center:
                continue
            self.samples.append(subject)

        if not self.samples:
            raise RuntimeError(
                "No IXI samples found. Check dataset path, split, and center filter."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_nifti(self, path: Path) -> torch.Tensor:
        data = nib.load(path).get_fdata().astype("float32")
        return torch.from_numpy(data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = self._load_nifti(sample.image_path)
        label = self._load_nifti(sample.label_path)

        image = _resize_volume(image, self.target_shape, mode="trilinear")
        label = _resize_volume(label, self.target_shape, mode="nearest")

        if self.normalize:
            mean = image.mean()
            std = image.std()
            if std > 0:
                image = (image - mean) / (std + 1e-8)

        label = (label > 0.5).float()
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)
        return image, label
