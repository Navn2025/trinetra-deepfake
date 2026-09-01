"""SLS classifier with XLS-R backbone for audio deepfake detection.

Architecture:
  XLS-R 300M (wav2vec 2.0) → Selective Layer Summarization (SLS) →
  BatchNorm → MaxPool → FC → SELU → FC → LogSoftmax

Reference:
  Q. Zhang, S. Wen, T. Hu, "Audio Deepfake Detection with XLS-R and SLS
  classifier," Proc. ACM Multimedia 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq


class SSLModel(nn.Module):
    def __init__(self, device, cp_path='xlsr2_300m.pt'):
        super(SSLModel, self).__init__()
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        layerresult = self.model(input_tmp, mask=False, features_only=True)['layer_results']
        return emb, layerresult


def getAttenF(layerResult):
    """Selective Layer Summarization: attention-weighted feature aggregation.

    For each of the 24 transformer layers, computes an attention weight via
    adaptive average pooling, then applies element-wise multiplication to
    produce the final aggregated feature.
    """
    poollayerResult = []
    fullf = []
    for layer in layerResult:
        layery = layer[0].transpose(0, 1).transpose(1, 2)  # (x,z)  x(201,b,1024) -> (b,201,1024) -> (b,1024,201)
        layery = F.adaptive_avg_pool1d(layery, 1)  # (b,1024,1)
        layery = layery.transpose(1, 2)  # (b,1,1024)
        poollayerResult.append(layery)

        x = layer[0].transpose(0, 1)
        x = x.view(x.size(0), -1, x.size(1), x.size(2))
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)
    fullfeature = torch.cat(fullf, dim=1)
    return layery, fullfeature


class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        xlsr_path = getattr(args, 'xlsr_model', 'xlsr2_300m.pt')
        self.ssl_model = SSLModel(self.device, cp_path=xlsr_path)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        # 22847 = floor(1024/3) * floor(201/3) = 341 * 67
        # where 1024 = XLS-R hidden dim, 201 = time frames for 64600 samples
        # at 16kHz with conv feature extractor stride, after MaxPool2d(3,3)
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        y0, fullfeature = getAttenF(layerResult)
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1)
        x = self.first_bn(fullfeature)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc3(x)
        x = self.selu(x)
        output = self.logsoftmax(x)

        return output
