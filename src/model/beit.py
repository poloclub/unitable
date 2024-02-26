import math
import torch
from torch import nn, Tensor
from functools import partial

from src.model.components import ImgLinearBackbone, PositionEmbedding, Encoder


class BeitEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,  # embed_dim
        backbone: nn.Module,
        max_seq_len: int,  # for positional embedding
        codebook_tokens: int,
        dropout: float,
        encoder: Encoder,
        norm_layer: nn.Module,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.init_std = init_std

        self.backbone = backbone
        self.pos_embed = PositionEmbedding(
            max_seq_len=max_seq_len, d_model=d_model, dropout=dropout
        )

        self.encoder = encoder
        self.norm = norm_layer(d_model)
        self.generator = nn.Linear(d_model, codebook_tokens)

        self.trunc_normal = partial(
            nn.init.trunc_normal_, std=init_std, a=-init_std, b=init_std
        )
        self.apply(self._init_weights)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            self.trunc_normal(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            self.trunc_normal(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, PositionEmbedding):
            self.trunc_normal(m.embedding.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def forward(
        self, x: Tensor, bool_masked_pos: Tensor, return_all_tokens: bool = False
    ):
        x = self.backbone(x)
        B, S, E = x.shape
        assert E == self.d_model

        mask_token = self.mask_token.expand(B, S, -1)

        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = self.pos_embed(x)

        x = self.encoder(x)
        x = self.norm(x)

        if return_all_tokens:
            return self.generator(x)
        else:
            return self.generator(x[bool_masked_pos])
