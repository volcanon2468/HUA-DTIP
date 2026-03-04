import torch
import torch.nn as nn


class MesoScaleModel(nn.Module):
    def __init__(self, input_dim: int = 512, transformer_heads: int = 8,
                 transformer_layers: int = 4, ff_dim: int = 1024,
                 sequence_len: int = 7, output_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dow_embed = nn.Embedding(7, input_dim)
        self.pos_embed = nn.Embedding(sequence_len, input_dim)
        self.input_norm = nn.LayerNorm(input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=transformer_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(input_dim)

        recency = [0.05, 0.07, 0.10, 0.12, 0.16, 0.20, 0.30]
        self.register_buffer("recency_weights", torch.tensor(recency, dtype=torch.float32))

        self.next_day_head    = nn.Linear(output_dim, input_dim)
        self.capacity_head    = nn.Linear(output_dim, 1)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, dow: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos)

        if dow is not None:
            dow_emb = self.dow_embed(dow)
            x = x + dow_emb + pos_emb
        else:
            x = x + pos_emb

        x = self.input_norm(x)
        x = self.transformer(x)
        x = self.norm(x)

        w = self.recency_weights[:T].unsqueeze(0).unsqueeze(-1)
        z_meso = (w * x).sum(dim=1)
        return z_meso

    def predict(self, x: torch.Tensor, dow: torch.Tensor = None):
        z = self.forward(x, dow)
        return self.next_day_head(z), self.capacity_head(z)
