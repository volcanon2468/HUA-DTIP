import torch
import torch.nn as nn


class HierarchicalFusion(nn.Module):
    def __init__(self, micro_dim: int = 256, meso_dim: int = 512,
                 macro_dim: int = 128, proj_dim: int = 256, output_dim: int = 512):
        super().__init__()
        self.proj_micro = nn.Sequential(nn.Linear(micro_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.proj_meso  = nn.Sequential(nn.Linear(meso_dim, proj_dim),  nn.LayerNorm(proj_dim))
        self.proj_macro = nn.Sequential(nn.Linear(macro_dim, proj_dim), nn.LayerNorm(proj_dim))

        self.macro_to_meso_attn = nn.MultiheadAttention(proj_dim, num_heads=4, batch_first=True)
        self.meso_to_micro_attn = nn.MultiheadAttention(proj_dim, num_heads=4, batch_first=True)

        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim * 3),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(proj_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, z_micro: torch.Tensor, z_meso: torch.Tensor, z_macro: torch.Tensor) -> torch.Tensor:
        p_micro = self.proj_micro(z_micro)
        p_meso  = self.proj_meso(z_meso)
        p_macro = self.proj_macro(z_macro)

        meso_ctx, _ = self.macro_to_meso_attn(
            p_meso.unsqueeze(1), p_macro.unsqueeze(1), p_macro.unsqueeze(1)
        )
        meso_ctx = (p_meso + meso_ctx.squeeze(1))

        micro_ctx, _ = self.meso_to_micro_attn(
            p_micro.unsqueeze(1), meso_ctx.unsqueeze(1), meso_ctx.unsqueeze(1)
        )
        micro_ctx = (p_micro + micro_ctx.squeeze(1))

        concat = torch.cat([micro_ctx, meso_ctx, p_macro], dim=-1)
        gated  = self.gate(concat) * concat
        return self.output_proj(gated)
