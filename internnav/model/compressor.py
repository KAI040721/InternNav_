"""
Instruction-Conditioned Visual Token Compressor for VLN

Bottleneck Cross-Attention Compressor:
  - Projects frame tokens (2048) down to d_bottleneck (512) for efficient cross-attention
  - Uses FiLM conditioning from instruction embeddings
  - Compresses 144 tokens/frame → n_queries tokens/frame (default 16)
  - Shared weights: same compressor applied to primary + 3 deepstack feature sets

Architecture:
  instr_emb [2048] → FiLM(γ,β) [d_bn]
  queries [n_q, d_bn] → Q_cond = γ⊙Q + β
  frame_tokens [144, 2048] → proj_in [144, d_bn]
  CrossAttn(Q_cond, KV=proj_in) × n_layers → [n_q, d_bn]
  proj_out [n_q, d_bn] → [n_q, 2048]

Total params: ~10.5M (d_bottleneck=512, n_queries=16, n_layers=2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BottleneckCompressor(nn.Module):
    """
    Bottleneck Cross-Attention Compressor with FiLM conditioning.

    Compresses per-frame visual tokens from T tokens to n_queries tokens,
    conditioned on the navigation instruction embedding via FiLM.

    Designed for Qwen3-VL-2B:
      - d_model = 2048 (merger output / LLM hidden dim)
      - T = 144 tokens/frame (384×384, patch_size=16, merge 2×2)
      - DeepStack: 3 sets of features at ViT layers [5, 11, 17]

    The same compressor instance is reused for primary and all deepstack
    feature sets to keep parameter count low (~10.5M).
    """

    def __init__(
        self,
        d_model: int = 2048,
        d_bottleneck: int = 512,
        n_queries: int = 16,
        n_heads: int = 8,
        n_layers: int = 2,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_bottleneck = d_bottleneck
        self.n_queries = n_queries
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Learnable query tokens (in bottleneck space)
        self.queries = nn.Parameter(torch.randn(n_queries, d_bottleneck) * 0.02)

        # FiLM conditioning: instruction → γ, β for query modulation
        self.film_gamma = nn.Linear(d_model, d_bottleneck)
        self.film_beta = nn.Linear(d_model, d_bottleneck)

        # Projection: d_model → d_bottleneck (for frame tokens)
        self.proj_in = nn.Linear(d_model, d_bottleneck)

        # Cross-attention layers (in bottleneck space)
        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                d_model=d_bottleneck,
                n_heads=n_heads,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Projection back: d_bottleneck → d_model
        self.proj_out = nn.Linear(d_bottleneck, d_model)
        self.final_norm = nn.LayerNorm(d_bottleneck)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following common practices."""
        # FiLM: gamma → 1, beta → 0 (identity initialization)
        nn.init.ones_(self.film_gamma.weight.data[:, 0])
        nn.init.zeros_(self.film_gamma.weight.data[:, 1:])
        nn.init.ones_(self.film_gamma.bias.data)
        nn.init.zeros_(self.film_beta.weight.data)
        nn.init.zeros_(self.film_beta.bias.data)

        # Projections
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def compress_frames(self, frame_tokens: torch.Tensor, instr_emb: torch.Tensor) -> torch.Tensor:
        """
        Compress a batch of per-frame tokens.

        Args:
            frame_tokens: [N_frames, T, d_model] — visual tokens per frame
                          T=144 for Qwen3-VL-2B (384×384)
            instr_emb:    [d_model] — mean-pooled instruction embedding (single sample)

        Returns:
            compressed:   [N_frames, n_queries, d_model]
        """
        N, T, D = frame_tokens.shape

        # Project frame tokens to bottleneck
        kv = self.proj_in(frame_tokens)  # [N, T, d_bn]

        # FiLM-conditioned queries
        gamma = self.film_gamma(instr_emb)  # [d_bn]
        beta = self.film_beta(instr_emb)    # [d_bn]
        Q = self.queries.unsqueeze(0).expand(N, -1, -1)  # [N, n_q, d_bn]
        Q_cond = gamma.unsqueeze(0).unsqueeze(0) * Q + beta.unsqueeze(0).unsqueeze(0)  # [N, n_q, d_bn]

        # Cross-attention layers
        h = Q_cond
        for layer in self.layers:
            h = layer(h, kv)  # [N, n_q, d_bn]

        h = self.final_norm(h)

        # Project back to d_model
        compressed = self.proj_out(h)  # [N, n_q, d_model]
        return compressed

    def forward(
        self,
        image_embeds_list: list,
        deepstack_embeds_list: list,
        num_history: int,
        instr_emb: torch.Tensor,
    ):
        """
        Compress history frames' primary + deepstack embeddings.
        Current frames (after num_history) are passed through unchanged.

        Args:
            image_embeds_list: list of N tensors, each [T_i, d_model]
                               First num_history are history, rest are current/birdseye
            deepstack_embeds_list: list of 3 tensors, each [total_T, d_model]
                                   where total_T = sum(T_i for all N images)
            num_history: number of history frames to compress
            instr_emb: [d_model] — instruction embedding

        Returns:
            new_image_embeds: [total_compressed_T, d_model]
            new_deepstack_embeds: list of 3 tensors, each [total_compressed_T, d_model]
                                  or None if input is None
        """
        tokens_per_image = [e.shape[0] for e in image_embeds_list]
        N_total = len(image_embeds_list)

        # --- Primary embeddings ---
        # Compress history frames
        if num_history > 0:
            history_tokens = torch.stack(
                [image_embeds_list[i] for i in range(num_history)]
            )  # [num_history, T, d_model]
            compressed_primary = self.compress_frames(history_tokens, instr_emb)
            # [num_history, n_queries, d_model] → [num_history * n_queries, d_model]
            compressed_primary = compressed_primary.reshape(-1, self.d_model)
        else:
            compressed_primary = torch.zeros(0, self.d_model,
                                              device=image_embeds_list[0].device,
                                              dtype=image_embeds_list[0].dtype)

        # Keep current frames unchanged
        current_parts = [image_embeds_list[i] for i in range(num_history, N_total)]
        if current_parts:
            current_primary = torch.cat(current_parts, dim=0)
        else:
            current_primary = torch.zeros(0, self.d_model,
                                           device=image_embeds_list[0].device,
                                           dtype=image_embeds_list[0].dtype)

        new_image_embeds = torch.cat([compressed_primary, current_primary], dim=0)

        # --- DeepStack embeddings ---
        if deepstack_embeds_list is None or len(deepstack_embeds_list) == 0:
            return new_image_embeds, None

        new_deepstack = []
        for ds_idx, ds_embeds in enumerate(deepstack_embeds_list):
            # ds_embeds: [total_T, d_model] — all images concatenated
            # Split into per-image based on tokens_per_image
            ds_per_image = torch.split(ds_embeds, tokens_per_image)

            # Compress history frames' deepstack
            if num_history > 0:
                ds_history = torch.stack(
                    [ds_per_image[i] for i in range(num_history)]
                )  # [num_history, T, d_model]
                ds_compressed = self.compress_frames(ds_history, instr_emb)
                ds_compressed = ds_compressed.reshape(-1, self.d_model)
            else:
                ds_compressed = torch.zeros(0, self.d_model,
                                             device=ds_embeds.device,
                                             dtype=ds_embeds.dtype)

            # Keep current frames' deepstack unchanged
            ds_current_parts = [ds_per_image[i] for i in range(num_history, N_total)]
            if ds_current_parts:
                ds_current = torch.cat(ds_current_parts, dim=0)
            else:
                ds_current = torch.zeros(0, self.d_model,
                                          device=ds_embeds.device,
                                          dtype=ds_embeds.dtype)

            new_deepstack.append(torch.cat([ds_compressed, ds_current], dim=0))

        return new_image_embeds, new_deepstack


class CrossAttentionBlock(nn.Module):
    """
    Pre-LN Cross-Attention + FFN block.
    Q attends to KV (no self-attention, since Q is only n_queries tokens).
    """

    def __init__(self, d_model: int, n_heads: int, ffn_expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Linear(d_model * ffn_expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q:  [B, n_q, d_model]
            kv: [B, T, d_model]
        Returns:
            out: [B, n_q, d_model]
        """
        # Cross-attention with Pre-LN
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_normed, kv_normed, kv_normed)
        q = q + attn_out

        # FFN with Pre-LN
        q = q + self.ffn(self.norm_ffn(q))
        return q
