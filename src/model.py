"""
model.py
--------
Transformer-based model for diabetes classification.

WHY TRANSFORMER INSTEAD OF MLP:
    MLP treats all 32 features independently — it cannot learn which
    features interact with each other. For example, the relationship
    between BMI and BloodPressure as a combined predictor of diabetes
    is invisible to an MLP unless explicitly engineered.

    A Transformer uses a self-attention mechanism that learns which
    features to pay attention to in the context of all other features.
    This is critical for tabular health data where combinations of
    risk factors (e.g. high BMI + high blood pressure + low exercise)
    are more predictive than any single feature alone.

    For temporal continual learning specifically, Transformers learn
    richer, more separable representations. This means the Fisher
    Information Matrix can more precisely identify which weights encode
    knowledge about specific years, making EWC protection more effective.

ARCHITECTURE:
    Each feature is treated as a token. A linear embedding layer maps
    each feature value to an embedding vector. Multi-head self-attention
    then learns relationships between all feature pairs. The output is
    pooled and passed through a classification head.

    input: (batch, n_features)  → one scalar per feature
    embed: (batch, n_features, embed_dim)  → one vector per feature
    attention: (batch, n_features, embed_dim)  → context-aware vectors
    pool: (batch, embed_dim)  → single vector summarising all features
    classify: (batch, 1)  → probability of diabetes
"""

import torch
import torch.nn as nn
import math


class TabularTransformer(nn.Module):
    """
    Transformer encoder for tabular diabetes classification.

    Each input feature becomes a token. Self-attention learns which
    features are most relevant given the other features present.

    Args:
        n_features  : number of input features (e.g. 32)
        embed_dim   : size of each feature embedding (default 64)
        n_heads     : number of attention heads (default 4)
        n_layers    : number of Transformer encoder layers (default 2)
        dropout     : dropout rate (default 0.1)
    """

    def __init__(self, n_features, embed_dim=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()

        self.n_features = n_features
        self.embed_dim  = embed_dim

        # Each feature scalar is projected to an embed_dim vector
        # This is the equivalent of a word embedding but for feature values
        self.feature_embedding = nn.Linear(1, embed_dim)

        # Learnable positional encoding — tells the model which position
        # (i.e. which feature slot) each token occupies
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, n_features, embed_dim)
        )

        # Transformer encoder — the core attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,    # (batch, seq, features) format
            norm_first=True,     # Pre-norm: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head: pool across feature dimension then classify
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialise weights for stable training."""
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        x: (batch_size, n_features) — one row per patient

        Steps:
        1. Reshape each feature into a 1-element sequence
        2. Embed each feature to embed_dim dimensions
        3. Add positional encoding
        4. Pass through Transformer encoder
        5. Average pool across feature dimension
        6. Classify
        """
        # (batch, n_features) → (batch, n_features, 1)
        x = x.unsqueeze(-1)

        # Embed each feature: (batch, n_features, embed_dim)
        x = self.feature_embedding(x)

        # Add positional encoding
        x = x + self.positional_encoding

        # Self-attention across features: (batch, n_features, embed_dim)
        x = self.transformer(x)

        # Average pool across features: (batch, embed_dim)
        x = x.mean(dim=1)

        # Classify: (batch, 1)
        return self.classifier(x)


def init_model(n_features, embed_dim=64, n_heads=4, n_layers=2, dropout=0.1):
    """Create and return a freshly initialised TabularTransformer."""
    model = TabularTransformer(
        n_features=n_features,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTransformer model ready")
    print(f"  Input features : {n_features}")
    print(f"  Embed dim      : {embed_dim}")
    print(f"  Attention heads: {n_heads}")
    print(f"  Encoder layers : {n_layers}")
    print(f"  Parameters     : {n_params:,}")
    return model
