"""
ViT-Internal FiLM + Aggregation Token Compressor (CogVLA-inspired)

Architecture:
  - Aggregation Tokens: Learnable tokens concatenated to history image sequences
    inside the ViT. After self-attention, these tokens attend to all image tokens
    and become a compressed representation. (~CogVLA design)
  - FiLM Conditioning: Applied at every ViT block between attention and FFN (same as CogVLA).
    Formula: x = x * (1 + gamma) + beta, where gamma/beta are projected from
    the instruction embedding. At init, gamma=beta=0 -> identity transform.
  - Projection: Aggregation tokens (vit_dim) -> LLM dim via LayerNorm + Linear.

Designed for Qwen3-VL-2B:
  vit_dim = 1024 (ViT hidden_size)
  llm_dim = 2048 (LLM hidden_size / ViT out_hidden_size)
  n_aggr = 64 (number of aggregation tokens per history image, CogVLA uses 64)
  n_film_layers = 24 (all ViT blocks, same as CogVLA)
  spatial_merge_size = 2 (PatchMerger 2x2 merge)

Total params: ~109M (24 FiLM layers + aggr proj + 3 deepstack aggr projections)
"""

import torch
import torch.nn as nn
import math


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    Transforms: x = x * (1 + scale(instr)) + shift(instr)
    
    At initialization, scale and shift outputs are zero,
    so the layer acts as identity. This avoids the bf16 precision
    trap (values near 1.0 have low precision; values near 0.0 have high precision).
    """

    def __init__(self, vit_dim: int, cond_dim: int):
        super().__init__()
        self.scale = nn.Linear(cond_dim, vit_dim)
        self.shift = nn.Linear(cond_dim, vit_dim)

        # Zero init: identity at start
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, vit_dim] — ViT hidden states (packed 1D)
            cond: [cond_dim] — instruction embedding (single vector, broadcast)
        Returns:
            [seq_len, vit_dim]
        """
        gamma = self.scale(cond)  # [vit_dim]
        beta = self.shift(cond)   # [vit_dim]
        return x * (1.0 + gamma) + beta


class AggrTokenCompressor(nn.Module):
    """
    Aggregation Token Compressor with ViT-internal FiLM conditioning.
    
    This module holds:
      1. aggr_tokens: Learnable aggregation tokens [1, n_aggr, vit_dim]
      2. film_layers: FiLM layers applied at designated ViT blocks
      3. aggr_proj: Projects extracted aggr tokens from vit_dim to llm_dim
      4. aggr_ds_proj_list: Projects aggr tokens at deepstack layers to llm_dim
    
    The actual injection into the ViT forward is done by the wrapper
    (compressor_wrapper_film_vit.py) via monkey-patching.
    """

    def __init__(
        self,
        vit_dim: int = 1024,
        llm_dim: int = 2048,
        n_aggr: int = 64,
        n_film_layers: int = 24,
        share_film: bool = False,
        spatial_merge_size: int = 2,
        n_deepstack_layers: int = 3,
    ):
        super().__init__()
        self.vit_dim = vit_dim
        self.llm_dim = llm_dim
        self.n_aggr = n_aggr
        self.n_film_layers = n_film_layers
        self.share_film = share_film
        self.spatial_merge_size = spatial_merge_size

        # Merge unit: PatchMerger groups merge_size^2 adjacent tokens
        self.merge_unit = spatial_merge_size ** 2  # 4 for Qwen3-VL

        # ---- Aggregation tokens ----
        # Small random init, will be learned via ViT self-attention
        self.aggr_tokens = nn.Parameter(
            torch.randn(1, n_aggr, vit_dim) * 0.02
        )

        # ---- FiLM layers ----
        # cond_dim = llm_dim because instruction embedding is in LLM space
        if share_film:
            single_film = FiLMLayer(vit_dim, llm_dim)
            self.film_layers = nn.ModuleList([single_film] * n_film_layers)
        else:
            self.film_layers = nn.ModuleList([
                FiLMLayer(vit_dim, llm_dim) for _ in range(n_film_layers)
            ])

        # ---- Aggregation token projection ----
        # After ViT: aggr tokens are in vit_dim space.
        # We skip PatchMerger for aggr tokens (no spatial structure to merge).
        # Instead, project directly: vit_dim -> llm_dim
        self.aggr_proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, llm_dim),
        )

        # Initialize projection
        nn.init.xavier_uniform_(self.aggr_proj[1].weight)
        nn.init.zeros_(self.aggr_proj[1].bias)

        # ---- DeepStack projections for aggregation tokens ----
        # At deepstack ViT layers (e.g. 5, 11, 17), aggr tokens carry
        # useful intermediate representations. Project them to llm_dim
        # so history images get real deepstack features (not zeros).
        self.n_deepstack_layers = n_deepstack_layers
        self.aggr_ds_proj_list = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(vit_dim),
                nn.Linear(vit_dim, llm_dim),
            )
            for _ in range(n_deepstack_layers)
        ])
        for proj in self.aggr_ds_proj_list:
            nn.init.xavier_uniform_(proj[1].weight)
            nn.init.zeros_(proj[1].bias)

    def project_aggr_tokens(self, aggr_hidden: torch.Tensor) -> torch.Tensor:
        """
        Project extracted aggregation tokens from vit_dim to llm_dim.
        
        Args:
            aggr_hidden: [n_aggr_total, vit_dim] — extracted aggr tokens (flat)
        Returns:
            [n_aggr_total, llm_dim]
        """
        return self.aggr_proj(aggr_hidden)

    def project_aggr_deepstack(self, aggr_hidden: torch.Tensor, ds_idx: int) -> torch.Tensor:
        """
        Project aggr tokens at a deepstack layer from vit_dim to llm_dim.

        Args:
            aggr_hidden: [n_aggr_total, vit_dim] — aggr tokens at deepstack layer
            ds_idx: deepstack layer index (0, 1, 2)
        Returns:
            [n_aggr_total, llm_dim]
        """
        return self.aggr_ds_proj_list[ds_idx](aggr_hidden)

    def get_film_layer(self, film_idx: int) -> FiLMLayer:
        """Get the FiLM layer for a given index (0-based within film layers)."""
        return self.film_layers[film_idx]

    def dummy_forward(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        """
        Dummy forward for ZeRO-2 compatibility.
        When no history images exist, we still need all parameters to
        participate in the computation graph.
        Returns a zero scalar.
        """
        # Touch aggr_tokens
        dummy = self.aggr_tokens.sum() * 0.0
        # Touch all film layers
        dummy_cond = torch.zeros(
            self.llm_dim, device=ref_tensor.device, dtype=ref_tensor.dtype
        )
        dummy_x = torch.zeros(
            1, self.vit_dim, device=ref_tensor.device, dtype=ref_tensor.dtype
        )
        for film in self.film_layers:
            dummy = dummy + film(dummy_x, dummy_cond).sum() * 0.0
        # Touch projection
        dummy_aggr = torch.zeros(
            1, self.vit_dim, device=ref_tensor.device, dtype=ref_tensor.dtype
        )
        dummy = dummy + self.aggr_proj(dummy_aggr).sum() * 0.0
        # Touch deepstack projections
        for ds_proj in self.aggr_ds_proj_list:
            dummy = dummy + ds_proj(dummy_aggr).sum() * 0.0
        return dummy

    def extra_repr(self) -> str:
        return (
            f"n_aggr={self.n_aggr}, vit_dim={self.vit_dim}, llm_dim={self.llm_dim}, "
            f"n_film_layers={self.n_film_layers}, share_film={self.share_film}, "
            f"merge_unit={self.merge_unit}"
        )
