#!/usr/bin/env python3
"""Download and extract the Fed-IXI (IXI Tiny) dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


DATASET_URL = (
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1"
    ".amazonaws.com/7kd5wj7v7p-3.zip"
)
FOLDER_NAME = "7kd5wj7v7p-3"
LICENSE_URL = "https://brain-development.org/ixi-dataset/"


def _confirm_license(assume_yes: bool) -> None:
    if assume_yes:
        return
    prompt = (
        "The IXI dataset is released under CC BY-SA 3.0.\n"
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
        print(f"Archive already exists: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        use_tqdm = tqdm is not None and total > 0
        if use_tqdm:
            progress = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024)
        else:
            progress = None
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


def _extract(zip_path: Path, output_dir: Path, force: bool) -> None:
    target_dir = output_dir / FOLDER_NAME
    if target_dir.exists() and not force:
        print(f"Extracted dataset already exists: {target_dir}")
        return

    if target_dir.exists() and force:
        shutil.rmtree(target_dir)

    print(f"Extracting {zip_path} to {output_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Fed-IXI (IXI Tiny) dataset used by FedCompass."
    )
    parser.add_argument(
        "--output-dir",
        default="data/fed_ixi",
        help="Directory to store the downloaded dataset",
    )
    parser.add_argument(
        "--url",
        default=DATASET_URL,
        help="Override the dataset download URL",
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
    args = parser.parse_args()

    _confirm_license(args.accept_license)

    output_dir = Path(args.output_dir).resolve()
    archive_name = Path(args.url).name
    archive_path = output_dir / archive_name

    _download(args.url, archive_path, args.force)
    _extract(archive_path, output_dir, args.force)

    expected = output_dir / FOLDER_NAME / "IXI_sample"
    if expected.exists():
        print(f"Dataset ready at: {expected}")
    else:
        print("Warning: expected IXI_sample folder not found after extraction.")


if __name__ == "__main__":
    main()
