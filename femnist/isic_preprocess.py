#!/usr/bin/env python3
"""Preprocess ISIC2019 images (resize + optional color constancy)."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def color_constancy(img: np.ndarray, power: int = 6, gamma: float | None = None) -> np.ndarray:
    """Apply color constancy transform (FLamby/ISIC style)."""
    img_dtype = img.dtype
    if gamma is not None and gamma > 0:
        if cv2 is None:
            img = img.astype("float32")
            img = np.power(img / 255.0, 1.0 / gamma) * 255.0
            img = img.astype(img_dtype)
        else:
            img = img.astype("uint8")
            look_up = np.ones((256, 1), dtype="uint8") * 0
            for i in range(256):
                look_up[i][0] = 255 * pow(i / 255, 1 / gamma)
            img = cv2.LUT(img, look_up)

    img = img.astype("float32")
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1.0 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)
    img = np.clip(img, 0, 255)
    return img.astype(img_dtype)


def resize_and_save(
    input_path: Path,
    output_dir: Path,
    short_edge: int,
    apply_cc: bool,
    gamma: float | None,
) -> None:
    img = Image.open(input_path).convert("RGB")
    old_w, old_h = img.size
    ratio = float(short_edge) / min(old_w, old_h)
    new_size = (int(old_w * ratio), int(old_h * ratio))
    img = img.resize(new_size, resample=Image.BILINEAR)

    if apply_cc:
        img = Image.fromarray(color_constancy(np.asarray(img), gamma=gamma))

    output_path = output_dir / input_path.name
    img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ISIC2019 images.")
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset root containing ISIC_2019_Training_Input",
    )
    parser.add_argument(
        "--input-dir",
        default="ISIC_2019_Training_Input",
        help="Input image directory (relative to data-root)",
    )
    parser.add_argument(
        "--output-dir",
        default="ISIC_2019_Training_Input_preprocessed",
        help="Output directory for preprocessed images (relative to data-root)",
    )
    parser.add_argument("--size", type=int, default=224, help="Short edge size")
    parser.add_argument(
        "--no-color-constancy",
        action="store_true",
        help="Disable color constancy",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Optional gamma correction before color constancy",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of threads for preprocessing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images for quick tests",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    input_dir = data_root / args.input_dir
    output_dir = data_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(input_dir.glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]

    apply_cc = not args.no_color_constancy

    if args.workers <= 1:
        for img_path in tqdm(images) if tqdm else images:
            resize_and_save(img_path, output_dir, args.size, apply_cc, args.gamma)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    resize_and_save,
                    img_path,
                    output_dir,
                    args.size,
                    apply_cc,
                    args.gamma,
                )
                for img_path in images
            ]
            iterator = as_completed(futures)
            if tqdm:
                iterator = tqdm(iterator, total=len(futures))
            for future in iterator:
                future.result()

    print(f"Saved {len(images)} preprocessed images to {output_dir}")


if __name__ == "__main__":
    main()
