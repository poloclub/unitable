import torch
from torch import nn, Tensor
import os
import math
from functools import partial
from collections import OrderedDict
import torch.nn.functional as F



class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_layers: int,
    ):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_layers

        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers**2)
        self.id_path = (
            nn.Conv2d(self.n_in, self.n_out, 1)
            if self.n_in != self.n_out
            else nn.Identity()
        )

        self.res_path = nn.Sequential(
            OrderedDict(
                [
                    ("relu_1", nn.ReLU()),
                    ("conv_1", nn.Conv2d(self.n_in, self.n_hid, 3, padding=1)),
                    ("relu_2", nn.ReLU()),
                    ("conv_2", nn.Conv2d(self.n_hid, self.n_hid, 3, padding=1)),
                    ("relu_3", nn.ReLU()),
                    ("conv_3", nn.Conv2d(self.n_hid, self.n_hid, 3, padding=1)),
                    ("relu_4", nn.ReLU()),
                    ("conv_4", nn.Conv2d(self.n_hid, self.n_out, 1)),
                ]
            )
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                nn.init.normal_(m.weight, std=1 / math.sqrt(fan_in))
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self.id_path(x) + self.post_gain * self.res_path(x)
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        group_count: int = 4,
        n_hid: int = 256,
        n_blk_per_group: int = 2,
        input_channels: int = 3,
        vocab_size: int = 8192,
    ) -> None:
        super().__init__()

        self.group_count = group_count
        self.n_hid = n_hid
        self.n_blk_per_group = n_blk_per_group
        self.input_channels = input_channels
        self.vocab_size = vocab_size

        blk_range = range(self.n_blk_per_group)
        n_layers = self.group_count * self.n_blk_per_group

        make_blk = partial(EncoderBlock, n_layers=n_layers)

        self.blocks = nn.Sequential(
            OrderedDict(
                [
                    (
                        "input",
                        nn.Conv2d(self.input_channels, 1 * self.n_hid, 7, padding=3),
                    ),
                    (
                        "group_1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(1 * self.n_hid, 1 * self.n_hid),
                                        )
                                        for i in blk_range
                                    ],
                                    ("pool", nn.MaxPool2d(kernel_size=2)),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                1 * self.n_hid
                                                if i == 0
                                                else 2 * self.n_hid,
                                                2 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    ("pool", nn.MaxPool2d(kernel_size=2)),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                2 * self.n_hid
                                                if i == 0
                                                else 4 * self.n_hid,
                                                4 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    ("pool", nn.MaxPool2d(kernel_size=2)),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_4",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                4 * self.n_hid
                                                if i == 0
                                                else 8 * self.n_hid,
                                                8 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                ]
                            )
                        ),
                    ),
                    (
                        "output",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("relu", nn.ReLU()),
                                    (
                                        "conv",
                                        nn.Conv2d(8 * self.n_hid, self.vocab_size, 1),
                                    ),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 4:
            raise ValueError(f"input shape {x.shape} is not 4d")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"input has {x.shape[1]} channels but model built for {self.input_channels}"
            )
        if x.dtype != torch.float32:
            raise ValueError("input must have dtype torch.float32")

        return self.blocks(x)


class DecoderBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_layers: int) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_layers

        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers**2)
        self.id_path = (
            nn.Conv2d(self.n_in, self.n_out, 1)
            if self.n_in != self.n_out
            else nn.Identity()
        )

        self.res_path = nn.Sequential(
            OrderedDict(
                [
                    ("relu_1", nn.ReLU()),
                    ("conv_1", nn.Conv2d(self.n_in, self.n_hid, 1)),
                    ("relu_2", nn.ReLU()),
                    ("conv_2", nn.Conv2d(self.n_hid, self.n_hid, 3, padding=1)),
                    ("relu_3", nn.ReLU()),
                    ("conv_3", nn.Conv2d(self.n_hid, self.n_hid, 3, padding=1)),
                    ("relu_4", nn.ReLU()),
                    ("conv_4", nn.Conv2d(self.n_hid, self.n_out, 3, padding=1)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.id_path(x) + self.post_gain * self.res_path(x)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        group_count: int = 4,
        n_init: int = 128,
        n_hid: int = 256,
        n_blk_per_group: int = 2,
        output_channels: int = 3,
        vocab_size: int = 8192,
    ) -> None:
        super().__init__()

        self.group_count = group_count
        self.n_init = n_init
        self.n_hid = n_hid
        self.n_blk_per_group = n_blk_per_group
        self.output_channels = output_channels
        self.vocab_size = vocab_size

        blk_range = range(self.n_blk_per_group)
        n_layers = self.group_count * self.n_blk_per_group
        make_blk = partial(DecoderBlock, n_layers=n_layers)

        self.blocks = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Conv2d(self.vocab_size, self.n_init, 1)),
                    (
                        "group_1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                self.n_init
                                                if i == 0
                                                else 8 * self.n_hid,
                                                8 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    (
                                        "upsample",
                                        nn.Upsample(scale_factor=2, mode="nearest"),
                                    ),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                8 * self.n_hid
                                                if i == 0
                                                else 4 * self.n_hid,
                                                4 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    (
                                        "upsample",
                                        nn.Upsample(scale_factor=2, mode="nearest"),
                                    ),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                4 * self.n_hid
                                                if i == 0
                                                else 2 * self.n_hid,
                                                2 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    (
                                        "upsample",
                                        nn.Upsample(scale_factor=2, mode="nearest"),
                                    ),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_4",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                2 * self.n_hid
                                                if i == 0
                                                else 1 * self.n_hid,
                                                1 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                ]
                            )
                        ),
                    ),
                    (
                        "output",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("relu", nn.ReLU()),
                                    (
                                        "conv",
                                        nn.Conv2d(
                                            1 * self.n_hid, 2 * self.output_channels, 1
                                        ),
                                    ),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 4:
            raise ValueError(f"input shape {x.shape} is not 4d")
        if x.shape[1] != self.vocab_size:
            raise ValueError(
                f"input has {x.shape[1]} channels but model built for {self.vocab_size}"
            )
        if x.dtype != torch.float32:
            raise ValueError("input must have dtype torch.float32")

        return self.blocks(x)


if __name__ == "__main__":

    input = torch.rand(1, 3, 256, 256)
    encoder = Encoder()
    decoder = Decoder()

    # forward
    z_logits = encoder(input)
    z = torch.argmax(z_logits, dim=1)
    print(z)
    z = F.one_hot(z, num_classes=encoder.vocab_size).permute(0, 3, 1, 2).float()

    # print(encoder)
    # print(decoder)
    # print(output.shape)