import itertools
from typing import Iterator

import torch
import torch.nn as nn

MODEL_REGISTRY: dict = {}


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class DefaultCrackNet(nn.Module):
    def __init__(self):
        super().__init__()
        import timm

        try:
            self.encoder = timm.create_model(
                'resnet18', pretrained=True, features_only=True,
                out_indices=(0, 1, 2, 3, 4),
            )
        except Exception:
            print("[CrackMark] Warning: ImageNet weights unavailable, using random init")
            self.encoder = timm.create_model(
                'resnet18', pretrained=False, features_only=True,
                out_indices=(0, 1, 2, 3, 4),
            )

        try:
            ch = [info['num_chs'] for info in self.encoder.feature_info]
        except (TypeError, KeyError):
            ch = [64, 64, 128, 256, 512]

        self.dec4 = _DecoderBlock(ch[4], ch[3], 256)
        self.dec3 = _DecoderBlock(256,   ch[2], 128)
        self.dec2 = _DecoderBlock(128,   ch[1], 64)
        self.dec1 = _DecoderBlock(64,    ch[0], 64)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        x = self.dec4(f[4], f[3])
        x = self.dec3(x, f[2])
        x = self.dec2(x, f[1])
        x = self.dec1(x, f[0])
        x = self.dec0(x)
        return self.head(x)

    def encoder_params(self) -> Iterator:
        return self.encoder.parameters()

    def decoder_params(self) -> Iterator:
        return itertools.chain(
            self.dec4.parameters(), self.dec3.parameters(),
            self.dec2.parameters(), self.dec1.parameters(),
            self.dec0.parameters(), self.head.parameters(),
        )


MODEL_REGISTRY['default'] = DefaultCrackNet


def build_model(name: str = 'default') -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()


def load_model(model: nn.Module, ckpt_path, device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = {
            (k[len('model.'):] if k.startswith('model.') else k): v
            for k, v in ckpt['state_dict'].items()
        }
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format in {ckpt_path}")

    result = model.load_state_dict(sd, strict=False)

    n_loaded = len(sd) - len(result.unexpected_keys)
    print(f"[CrackMark] Loaded {n_loaded}/{len(sd)} keys from {ckpt_path}")
    if result.missing_keys:
        s = result.missing_keys[:3]
        print(f"[CrackMark]   missing ({len(result.missing_keys)}): {s}{'...' if len(result.missing_keys) > 3 else ''}")
    if result.unexpected_keys:
        s = result.unexpected_keys[:3]
        print(f"[CrackMark]   skipped ({len(result.unexpected_keys)}): {s}{'...' if len(result.unexpected_keys) > 3 else ''}")

    return model
