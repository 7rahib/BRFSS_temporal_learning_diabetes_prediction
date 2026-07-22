"""
model.py
--------
FT-Transformer (Feature Tokenizer + Transformer) for binary diabetes
classification on tabular BRFSS data.

Reference: Gorishniy et al. (2021) — "Revisiting Deep Learning Models
for Tabular Data" (the original FT-Transformer paper).

WHY FT-TRANSFORMER:
    Every BRFSS feature (BMI, Age, Cholesterol, ...) is turned into its
    own embedding ("tokenized"), and self-attention lets the model learn
    which *combinations* of features matter for predicting diabetes —
    e.g. how BMI interacts with Age — instead of treating every feature
    as independent, the way a plain MLP does.

    EWC still works unchanged: FT-Transformer is a standard nn.Module
    with differentiable weights, so the same Fisher-information penalty
    from ewc.py applies with no modifications needed there.

ARCHITECTURE:
    1. Feature Tokenizer  — each of the `input_size` numerical features
       gets its own learned (weight, bias), turning one row into a
       sequence of `input_size` embedding vectors of size `embed_dim`.
    2. [CLS] token — one extra learnable token is prepended to the
       sequence, the same idea as BERT's [CLS] token.
    3. Transformer encoder — standard multi-head self-attention layers
       let every feature-token attend to every other feature-token.
    4. Classification head — the [CLS] token's final representation is
       passed through a small MLP head that outputs a diabetes
       probability (0 to 1).

NOTE ON FEATURES:
    All BRFSS columns used here are already numeric (binary, ordinal,
    or continuous) and are standard-scaled in dataset.py, so every
    feature is tokenized as a numerical feature. This keeps the model
    simple — no separate categorical-embedding path is needed.

CONFIGURABLE PARAMETERS (all set from main.py — no need to edit this file):
    embed_dim       — size of each feature's token vector. Bigger = more
                       capacity per feature, but more parameters and slower.
    n_heads         — number of attention heads. Must evenly divide embed_dim.
    n_layers        — number of stacked Transformer encoder layers (depth).
    dropout         — dropout applied in attention, feed-forward, and head.
    ffn_dim         — hidden size of the feed-forward block inside each
                       encoder layer. Defaults to 4 x embed_dim if left as
                       None (the standard Transformer ratio).
    activation      — activation used in the feed-forward block: "gelu" or
                       "relu".
    head_hidden_dim — hidden layer size of the final classification head.
                       Defaults to embed_dim // 2 if left as None.
"""

import torch
import torch.nn as nn


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for tabular binary classification."""

    def __init__(self, input_size, embed_dim=64, n_heads=4, n_layers=2, dropout=0.1,
                 ffn_dim=None, activation="gelu", head_hidden_dim=None):
        super().__init__()

        # Sensible defaults that scale with embed_dim, but can be overridden
        # directly for finer control (see CONFIGURABLE PARAMETERS above).
        ffn_dim         = ffn_dim         if ffn_dim         is not None else embed_dim * 4
        head_hidden_dim = head_hidden_dim if head_hidden_dim is not None else embed_dim // 2

        # --- Feature Tokenizer ---
        # One learned (weight, bias) pair per input feature. For feature i:
        #   token_i = x_i * weight_i + bias_i        (a vector of size embed_dim)
        self.feature_weight = nn.Parameter(torch.randn(input_size, embed_dim) * 0.02)
        self.feature_bias   = nn.Parameter(torch.zeros(input_size, embed_dim))

        # Learnable [CLS] token, prepended to the feature-token sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # --- Transformer encoder ---
        # Self-attention across feature tokens (not across rows/patients)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Classification head ---
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, input_size)
        batch_size = x.size(0)

        # Tokenize every feature -> (batch, input_size, embed_dim)
        tokens = x.unsqueeze(-1) * self.feature_weight + self.feature_bias

        # Prepend [CLS] -> (batch, input_size + 1, embed_dim)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Let every feature token attend to every other feature token
        encoded = self.transformer(tokens)

        # Classify using the [CLS] token's output representation
        cls_out = self.norm(encoded[:, 0, :])
        return self.head(cls_out)


def init_model(input_size, embed_dim=64, n_heads=4, n_layers=2, dropout=0.1,
               ffn_dim=None, activation="gelu", head_hidden_dim=None):
    """Create and return a freshly initialised FTTransformer."""
    model = FTTransformer(input_size, embed_dim, n_heads, n_layers, dropout,
                          ffn_dim, activation, head_hidden_dim)
    resolved_ffn  = ffn_dim         if ffn_dim         is not None else embed_dim * 4
    resolved_head = head_hidden_dim if head_hidden_dim is not None else embed_dim // 2
    print(f"\nModel ready — FT-Transformer | {input_size} input features")
    print(f"  Embed dim: {embed_dim} | Heads: {n_heads} | Layers: {n_layers} | Dropout: {dropout}")
    print(f"  FFN dim: {resolved_ffn} | Activation: {activation} | Head hidden dim: {resolved_head}")
    return model
