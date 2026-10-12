"""Visual Representation Engine using DINOv2 Vision Foundation Model.

Extracts deep 384-dimensional visual embeddings capturing:
- Model-specific rendering aesthetic & lighting habits
- Color grading, brushwork, and diffusion micro-contrast
- Robust against JPEG compression, downscaling, and format transcoding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image

from .preprocessor import center_crop_rgb

# Standard ImageNet normalization coefficients
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VisualEngine:
    """Stateful engine for deep visual embedding extraction via DINOv2."""

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        device: str | None = None,
        img_size: int = 224,
        crop_size: int = 512,
    ):
        self.model_name = model_name
        self.img_size = img_size
        self.crop_size = crop_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load DINOv2 vision transformer
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # Standard Vision Transformer transform
        self.transform = T.Compose(
            [
                T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _to_pil(self, img_input: str | Path | Image.Image | np.ndarray) -> Image.Image:
        """Convert any supported image input to RGB PIL Image."""
        if isinstance(img_input, (str, Path)):
            with Image.open(img_input) as img:
                return img.convert("RGB")
        elif isinstance(img_input, Image.Image):
            return img_input.convert("RGB")
        elif isinstance(img_input, np.ndarray):
            if img_input.dtype != np.uint8:
                if img_input.max() <= 1.0:
                    img_input = (img_input * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img_input = img_input.clip(0, 255).astype(np.uint8)
            if img_input.ndim == 2:
                img_input = np.stack([img_input] * 3, axis=-1)
            elif img_input.shape[2] > 3:
                img_input = img_input[:, :, :3]
            return Image.fromarray(img_input, mode="RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")

    @torch.no_grad()
    def extract(self, img_input: str | Path | Image.Image | np.ndarray) -> np.ndarray:
        """Extract L2-normalized 384-dimensional visual embedding for a single image.

        Returns:
            np.ndarray of shape (384,), dtype float32.
        """
        pil_img = self._to_pil(img_input)
        pil_img = Image.fromarray(center_crop_rgb(np.asarray(pil_img), self.crop_size))
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        features = self.model(tensor)  # (1, 384)

        # L2 normalize
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        features = features / (norm + 1e-12)

        return features.cpu().squeeze(0).numpy().astype(np.float32)

    @torch.no_grad()
    def extract_batch(
        self, images: list[str | Path | Image.Image | np.ndarray], batch_size: int = 32
    ) -> np.ndarray:
        """Extract embeddings for a batch of images with chunking.

        Returns:
            np.ndarray of shape (N, 384), dtype float32.
        """
        if not images:
            return np.empty((0, 384), dtype=np.float32)

        all_features = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            tensors = [
                self.transform(
                    Image.fromarray(
                        center_crop_rgb(np.asarray(self._to_pil(img_in)), self.crop_size)
                    )
                )
                for img_in in chunk
            ]
            batch_tensor = torch.stack(tensors).to(self.device)
            feats = self.model(batch_tensor)
            norm = torch.norm(feats, p=2, dim=-1, keepdim=True)
            feats = feats / (norm + 1e-12)
            all_features.append(feats.cpu().numpy().astype(np.float32))

        return np.concatenate(all_features, axis=0)


# Module-level convenience function
_GLOBAL_VISUAL_ENGINE = None


def extract_visual_features(image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    """Convenience function using a shared VisualEngine instance."""
    global _GLOBAL_VISUAL_ENGINE
    if _GLOBAL_VISUAL_ENGINE is None:
        _GLOBAL_VISUAL_ENGINE = VisualEngine()
    return _GLOBAL_VISUAL_ENGINE.extract(image)
