"""
Frozen CLIP ViT-B/16 (OpenCLIP, LAION-2B) image encoder -- the backbone
trinetra_v1_export's demo_head.pt was trained on top of. Trimmed from that
project's src/backbone/{base.py,clip_backbone.py} down to the one backbone
predictor.py actually uses (DINOv2 and the other Experiment-0 scaffolding
there were never part of the shipped demo head).

The backbone is used strictly as a fixed feature extractor: frozen weights,
eval() mode, every forward pass under torch.no_grad(). predictor.py trains
nothing here -- it only loads the linear head from demo_head.pt.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torchvision.transforms import Normalize


class CLIPBackbone:
    """Frozen OpenCLIP ViT-B/16 image tower -- visual tower only, text tower
    discarded. Embeddings are the projected 512-d CLIP embedding space,
    taken before any L2 normalization (the linear head's caller normalizes
    separately, matching how demo_head.pt was trained -- see predictor.py)."""

    def __init__(self, checkpoint: str = "laion2b_s34b_b88k", arch: str = "ViT-B-16",
                 device: str | None = None, precision: str = "fp16") -> None:
        import open_clip

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = precision
        self.checkpoint = checkpoint
        self.arch = arch

        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=checkpoint)
        visual = model.visual

        # Read normalization from open_clip's own transform rather than
        # hard-coding OpenAI's constants -- LAION-2B checkpoints don't all
        # use them.
        norm = next(t for t in preprocess.transforms if isinstance(t, Normalize))
        self.input_size = visual.image_size[0] if isinstance(visual.image_size, (tuple, list)) else visual.image_size

        with torch.no_grad():
            embed_dim = visual(torch.zeros(1, 3, self.input_size, self.input_size)).shape[-1]

        visual.eval()
        visual.requires_grad_(False)
        self.model = visual.to(self.device)
        self.mean = torch.tensor(norm.mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(norm.std, device=self.device).view(1, 3, 1, 1)
        self.embed_dim = embed_dim

    def _autocast(self):
        use = self.precision == "fp16" and self.device.type == "cuda"
        return torch.autocast("cuda", dtype=torch.float16, enabled=use)

    def _to_batch(self, crop) -> torch.Tensor:
        """uint8 RGB (H,W,3) or (N,H,W,3) -> normalized float NCHW tensor,
        resized to the model's native input size if it isn't already."""
        a = np.asarray(crop) if not isinstance(crop, Sequence) else np.stack(crop)
        if a.ndim == 3:
            a = a[None, ...]
        if a.ndim != 4 or a.shape[-1] != 3:
            raise ValueError(f"expected uint8 RGB (H,W,3) or (N,H,W,3), got {a.shape}")

        x = torch.from_numpy(np.ascontiguousarray(a)).to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)

        if x.shape[-2:] != (self.input_size, self.input_size):
            x = torch.nn.functional.interpolate(
                x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False,
            )
        return (x - self.mean) / self.std

    @torch.no_grad()
    def encode_image(self, crop) -> np.ndarray:
        """Returns float32 (embed_dim,) for a single HxWx3 crop, or
        (N, embed_dim) for a batch."""
        single = isinstance(crop, np.ndarray) and crop.ndim == 3
        x = self._to_batch(crop)
        with self._autocast():
            feats = self.model(x)
        out = feats.float().cpu().numpy()
        return out[0] if single else out
