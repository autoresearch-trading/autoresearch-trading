# tape/model.py
"""TapeEncoder — dilated CNN with RF=253; spec §Architecture.

Channel layout follows the spec exactly:
    17 -> 64 (dilation 1)
       -> 128 (dilation 2)
       -> 128 (dilation 4)   + residual
       -> 128 (dilation 8)   + residual
       -> 128 (dilation 16)  + residual
       -> 128 (dilation 32)  + residual

Total receptive field: 1 + sum_k (k-1) * d_k where k=5, dilations={1,2,4,8,16,32}
                     = 1 + 4*(1+2+4+8+16+32) = 1 + 4*63 = 253.

Global embedding = concat[GlobalAvgPool(per_pos), per_pos[:, -1, :]] -> 256-dim.

The channel multiplier scales every Conv1d output channel count except the
final 17 -> first hidden, which always emits at least 32 to keep the input
projection meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EncoderConfig:
    in_channels: int = 17
    base_channels: int = 64  # first conv output before mult
    hidden_channels: int = 128  # middle conv output before mult
    channel_mult: float = 1.0
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    dropout_p: float = 0.1


def _scaled(c: int, mult: float, *, floor: int = 32) -> int:
    return max(floor, int(round(c * mult)))


class _ConvBlock(nn.Module):
    """Conv1d + LayerNorm + ReLU (+ Dropout) (+ residual)."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: int,
        dilation: int,
        *,
        dropout_p: float,
        residual: bool,
    ) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_c, out_c, kernel_size=kernel, dilation=dilation, padding=pad
        )
        # LayerNorm over channels at each position.
        self.norm = nn.LayerNorm(out_c)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.residual = residual and (in_c == out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.conv(x)
        # LayerNorm wants channels last.
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.drop(h)
        if self.residual:
            h = h + x
        return h


class TapeEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_bn = nn.BatchNorm1d(cfg.in_channels)

        c0 = _scaled(cfg.base_channels, cfg.channel_mult)
        ch = _scaled(cfg.hidden_channels, cfg.channel_mult)

        blocks: list[nn.Module] = []
        in_c = cfg.in_channels
        for i, d in enumerate(cfg.dilations):
            out_c = c0 if i == 0 else ch
            blocks.append(
                _ConvBlock(
                    in_c=in_c,
                    out_c=out_c,
                    kernel=cfg.kernel_size,
                    dilation=d,
                    dropout_p=cfg.dropout_p if i < 2 else 0.0,
                    residual=(i >= 2),  # last 4 blocks residual per spec
                )
            )
            in_c = out_c
        self.blocks = nn.Sequential(*blocks)
        self._per_pos_dim = ch

    @property
    def per_position_dim(self) -> int:
        return self._per_pos_dim

    @property
    def global_dim(self) -> int:
        return 2 * self._per_pos_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C) -> (B, C, T) for Conv1d / BN1d
        h = x.transpose(1, 2)
        h = self.input_bn(h)
        h = self.blocks(h)  # (B, C', T)
        per_pos = h.transpose(1, 2)  # (B, T, C')
        avg = per_pos.mean(dim=1)
        last = per_pos[:, -1, :]
        global_emb = torch.cat([avg, last], dim=-1)
        return per_pos, global_emb


class MEMDecoder(nn.Module):
    """Per-position linear decoder: per_pos_dim -> 17 raw feature channels."""

    def __init__(self, per_position_dim: int, n_features: int = 17) -> None:
        super().__init__()
        self.linear = nn.Linear(per_position_dim, n_features)

    def forward(self, per_pos: torch.Tensor) -> torch.Tensor:
        return self.linear(per_pos)


class ProjectionHead(nn.Module):
    """Linear -> ReLU -> Linear -> L2-normalize."""

    def __init__(self, in_dim: int = 256, hidden: int = 256, out: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return z / (z.norm(dim=-1, keepdim=True) + 1e-12)
