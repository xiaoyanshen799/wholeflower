#!/usr/bin/env python3
"""Download and prepare the Fed-ISIC2019 dataset artifacts."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


ISIC_INPUT_URL = (
    "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
)
ISIC_METADATA_URL = (
    "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv"
)
ISIC_GROUNDTRUTH_URL = (
    "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
)
HAM_METADATA_URL = (
    "https://raw.githubusercontent.com/owkin/FLamby/main/"
    "flamby/datasets/fed_isic2019/HAM10000_metadata"
)

LICENSE_URL = "https://challenge.isic-archive.com/data/"
EXPECTED_IMAGE_COUNT = 23247


def _confirm_license(assume_yes: bool) -> None:
    if assume_yes:
        return
    prompt = (
        "ISIC2019/HAM10000 data are under CC BY-NC 4.0.\n"
        f"See: {LICENSE_URL}\n"
        "Do you accept the license terms? [y/N]: "
    )
    try:
        accepted = input(prompt).strip().lower() in {"y", "yes"}
    except EOFError:
        accepted = False
    if not accepted:
        print("License not accepted. Aborting download.")
        sys.exit(2)


def _download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"Already exists: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        use_tqdm = tqdm is not None and total > 0
        progress = (
            tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024)
            if use_tqdm
            else None
        )
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                if progress:
                    progress.update(len(chunk))
        if progress:
            progress.close()

    tmp_path.rename(dest)
    print(f"Downloaded: {dest}")


def _extract_zip(zip_path: Path, output_dir: Path, force: bool) -> None:
    target_dir = output_dir / "ISIC_2019_Training_Input"
    if target_dir.exists() and not force:
        print(f"Already extracted: {target_dir}")
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)

    print(f"Extracting {zip_path} to {output_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def _prepare_metadata(
    metadata_csv: Path,
    groundtruth_csv: Path,
    ham_metadata_csv: Path,
    images_dir: Path,
    output_csv: Path,
) -> None:
    meta = pd.read_csv(metadata_csv)
    gt = pd.read_csv(groundtruth_csv)
    ham = pd.read_csv(ham_metadata_csv)

    ham = ham.rename(columns={"image_id": "image"})
    ham = ham.drop(
        ["age", "sex", "localization", "lesion_id", "dx", "dx_type"],
        axis=1,
    )

    # Remove samples without datacenter
    drop_indices = meta.index[meta["lesion_id"].isna()].tolist()
    for idx in drop_indices:
        image = meta.loc[idx, "image"]
        image_path = images_dir / f"{image}.jpg"
        if image_path.exists():
            image_path.unlink()
    meta = meta.drop(drop_indices)
    gt = gt.drop(drop_indices)

    meta["dataset"] = meta["lesion_id"].str[:4]

    result = pd.merge(meta, ham, how="left", on="image")
    result["dataset"] = result["dataset_x"] + result["dataset_y"].astype(str)
    result = result.drop(["dataset_x", "dataset_y", "lesion_id"], axis=1)

    output_csv.write_text(result.to_csv(index=False), encoding="utf-8")
    metadata_csv.write_text(meta.to_csv(index=False), encoding="utf-8")
    groundtruth_csv.write_text(gt.to_csv(index=False), encoding="utf-8")

    print("Datacenters")
    print(result["dataset"].value_counts())
    print("Metadata rows", meta.shape[0])
    print("GroundTruth rows", gt.shape[0])
    print("MetadataFL rows", result.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ISIC2019 data and generate Fed-ISIC2019 metadata."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to store raw images, metadata, and processed CSVs",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Skip interactive license confirmation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-extract even if files exist",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip after extraction",
    )
    args = parser.parse_args()

    _confirm_license(args.accept_license)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "ISIC_2019_Training_Input.zip"
    metadata_csv = output_dir / "ISIC_2019_Training_Metadata.csv"
    groundtruth_csv = output_dir / "ISIC_2019_Training_GroundTruth.csv"
    ham_metadata_csv = output_dir / "HAM10000_metadata"
    metadata_fl_csv = output_dir / "ISIC_2019_Training_Metadata_FL.csv"

    _download(ISIC_INPUT_URL, zip_path, args.force)
    _extract_zip(zip_path, output_dir, args.force)
    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()

    _download(ISIC_METADATA_URL, metadata_csv, args.force)
    _download(ISIC_GROUNDTRUTH_URL, groundtruth_csv, args.force)
    _download(HAM_METADATA_URL, ham_metadata_csv, args.force)

    images_dir = output_dir / "ISIC_2019_Training_Input"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    _prepare_metadata(
        metadata_csv=metadata_csv,
        groundtruth_csv=groundtruth_csv,
        ham_metadata_csv=ham_metadata_csv,
        images_dir=images_dir,
        output_csv=metadata_fl_csv,
    )

    image_count = len(list(images_dir.glob("*.jpg")))
    if image_count == EXPECTED_IMAGE_COUNT:
        print("Download OK")
    else:
        print(f"Warning: expected {EXPECTED_IMAGE_COUNT} images, found {image_count}")


if __name__ == "__main__":
    main()
