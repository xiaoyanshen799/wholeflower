"""Augmentations matching FLamby Fed-ISIC2019 defaults."""

from __future__ import annotations

import random


def build_isic_transform(train: bool):
    try:
        try:
            import albucore.utils as albucore_utils
            if not hasattr(albucore_utils, "preserve_channel_dim"):
                def _preserve_channel_dim(func):
                    def wrapper(*args, **kwargs):
                        return func(*args, **kwargs)
                    return wrapper
                albucore_utils.preserve_channel_dim = _preserve_channel_dim
            if not hasattr(albucore_utils, "contiguous"):
                def _contiguous(img):
                    return img
                albucore_utils.contiguous = _contiguous
        except Exception:
            pass
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Albumentations is required for ISIC transforms. "
            "Install it via requirements_ixi.txt."
        ) from exc

    size = 200
    if train:
        return A.Compose(
            [
                A.RandomScale(0.07),
                A.Rotate(50),
                A.RandomBrightnessContrast(0.15, 0.1),
                A.Flip(p=0.5),
                A.Affine(shear=0.1),
                A.RandomCrop(size, size),
                A.CoarseDropout(random.randint(1, 8), 16, 16),
                A.Normalize(always_apply=True),
                ToTensorV2(),
            ]
        )

    return A.Compose(
        [
            A.CenterCrop(size, size),
            A.Normalize(always_apply=True),
            ToTensorV2(),
        ]
    )
