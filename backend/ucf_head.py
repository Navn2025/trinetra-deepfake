"""
The two small classifier-head modules UCFDetector wraps around its Xception
backbone (DeepfakeBench/training/detectors/ucf_detector.py: Conv2d1x1, Head),
copied verbatim rather than imported.

Importing UCFDetector itself means importing the `detectors` package
(detectors/__init__.py), which eagerly imports every detector -- including
VideoMAE/CLIP/X-CLIP/TimeSformer -- pulling in transformers/timm/einops/
fvcore just to run inference with this one. predictor.py already avoids that
for the Xception backbone by importing networks/xception.py directly; this
does the same for the two head modules on top of it.

Only the "shared" (encoder_f -> block_sha -> head_sha) path is reproduced,
since that's the only one UCFDetector.forward(inference=True) actually reads
(pred_dict['cls']); the "specific" path (block_spe/head_spe) and the
Conditional_UNet reconstruction branch (con_gan, encoder_c) exist purely for
training-time regularization and contribute nothing to the inference output.
"""
import torch.nn as nn


class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_f, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_f, 1, 1),
        )

    def forward(self, x):
        return self.conv2d(x)


class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_f, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, out_f),
        )

    def forward(self, x):
        bs = x.size(0)
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat
