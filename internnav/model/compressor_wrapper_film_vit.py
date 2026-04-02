"""
ViT-Internal FiLM + Aggregation Token Wrapper for Qwen3VLForConditionalGeneration

Monkey-patches:
  1. Qwen3VLVisionModel.forward — inject aggr tokens into history image sequences,
     apply FiLM conditioning at designated ViT blocks, extract aggr tokens after
     all blocks and project to LLM dim.
  2. Qwen3VLModel.forward (inner model) — replace history image embeddings with
     compressed aggr tokens, handle DeepStack, compute RoPE with compressed grid.
  3. Qwen3VLForConditionalGeneration.forward (outer model) — intercept compressor
     kwargs (is_history_image, image_grid_thw_rope).

Design:
  - History images: patch tokens + appended aggr tokens go through ViT together.
    After all blocks, aggr tokens are extracted and projected (aggr_proj).
    The original patch tokens for history images are discarded.
    History deepstack = zeros (no spatial structure to merge).
  - Current images: unchanged path through ViT + PatchMerger + DeepStack.
  - FiLM is applied to ALL tokens (history + current) at designated blocks.
    At init FiLM is identity, so current images are unaffected.

Qwen3-VL ViT specifics:
  - Packed 1D sequence: all images concatenated, tracked by cu_seqlens
  - Flash Attention 2 with cu_seqlens (variable-length)
  - RoPE (cos, sin) positional embeddings per token
  - PatchMerger: 2x2 spatial merge, Linear(4*1024 -> 4096) -> GELU -> Linear(4096 -> 2048)
  - DeepStack at layers [5, 11, 17] with separate PatchMergers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import functools
from typing import Optional
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast

from internnav.model.compressor_film_vit import AggrTokenCompressor


# ============================================================
# Utility: Instruction Embedding Extraction
# ============================================================

def get_instruction_embedding(model, input_ids, inputs_embeds):
    """
    Extract instruction embedding by mean-pooling text tokens.
    Returns: [B, d_model]
    """
    config = model.config
    special_ids = {config.image_token_id, config.video_token_id}
    if hasattr(config, 'vision_start_token_id'):
        special_ids.add(config.vision_start_token_id)
    if hasattr(config, 'vision_end_token_id'):
        special_ids.add(config.vision_end_token_id)

    text_mask = torch.ones_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        text_mask = text_mask & (input_ids != sid)
    if hasattr(config, 'pad_token_id') and config.pad_token_id is not None:
        text_mask = text_mask & (input_ids != config.pad_token_id)

    instr_emb = []
    for b in range(input_ids.shape[0]):
        mask_b = text_mask[b]
        if mask_b.sum() > 0:
            emb_b = inputs_embeds[b, mask_b].mean(dim=0)
        else:
            emb_b = inputs_embeds[b].mean(dim=0)
        instr_emb.append(emb_b)
    return torch.stack(instr_emb)


# ============================================================
# Patched ViT Forward
# ============================================================

def make_patched_vit_forward(original_vit_forward, compressor, outer_model_ref):
    """
    Create a patched forward for Qwen3VLVisionModel that:
      1. Appends aggr tokens to each history image's token sequence
      2. Applies FiLM conditioning at designated ViT blocks
      3. After all blocks: extracts aggr tokens for history, runs normal merger for current
    
    Returns:
      image_embeds: [total_merged_tokens, llm_dim] — current images have normal merged tokens,
                    history images have n_aggr projected tokens
      deepstack_features: list of [total_merged_tokens, llm_dim] — history portions are zeros
      aggr_info: dict with metadata for downstream token replacement
    """

    def patched_vit_forward(
        self_vit,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ):
        # Retrieve compressor context from outer model
        is_history_image = getattr(outer_model_ref, '_compressor_is_history', None)
        # Don't clear yet - inner forward also needs it

        n_images = grid_thw.shape[0]
        if is_history_image is None:
            is_history_image = torch.zeros(n_images, dtype=torch.bool)
        is_history_image = is_history_image.bool()

        has_history = is_history_image.any()
        n_aggr = compressor.n_aggr
        vit_dim = compressor.vit_dim
        n_film_layers = compressor.n_film_layers
        n_blocks = len(self_vit.blocks)
        # FiLM applies to the last n_film_layers blocks
        film_start_block = n_blocks - n_film_layers

        # ---- Step 1: Patch embed + positional embedding (same as original) ----
        hidden_states = self_vit.patch_embed(hidden_states)
        pos_embeds = self_vit.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # Compute RoPE
        rotary_pos_emb = self_vit.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Compute cu_seqlens for original tokens
        cu_seqlens_orig = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens_orig = F.pad(cu_seqlens_orig, (1, 0), value=0)

        # ---- Step 2: Inject aggr tokens for history images ----
        if has_history:
            # Get instruction embedding for FiLM conditioning
            instr_emb_cached = getattr(outer_model_ref, '_compressor_instr_emb', None)
            if instr_emb_cached is not None:
                instr_emb_single = instr_emb_cached
            else:
                # Fallback: zero (shouldn't happen in normal flow)
                instr_emb_single = torch.zeros(
                    compressor.llm_dim,
                    device=hidden_states.device, dtype=hidden_states.dtype
                )

            # Compute per-image token counts (before merge)
            tokens_per_image = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()

            # Build new hidden_states with aggr tokens inserted
            # Also build new cu_seqlens and position_embeddings
            new_parts = []
            new_cos_parts = []
            new_sin_parts = []
            new_seqlens = []
            cos_emb, sin_emb = position_embeddings

            # Prepare aggr tokens (shared across history images)
            aggr = compressor.aggr_tokens[0].to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )  # [n_aggr, vit_dim]
            is_hist_list = is_history_image.tolist() if torch.is_tensor(is_history_image) else list(is_history_image)

            offset = 0
            for img_idx in range(n_images):
                n_tokens = tokens_per_image[img_idx]
                img_tokens = hidden_states[offset:offset + n_tokens]  # [T_i, vit_dim]
                img_cos = cos_emb[offset:offset + n_tokens]
                img_sin = sin_emb[offset:offset + n_tokens]

                if is_hist_list[img_idx]:
                    new_parts.append(img_tokens)
                    new_parts.append(aggr)
                    new_cos_parts.append(img_cos)
                    new_sin_parts.append(img_sin)
                    # Aggr tokens use per-image average RoPE (Option C):
                    # average the raw rotary angles of all patch tokens in this image,
                    # then compute cos/sin. This gives aggr tokens a "center of image"
                    # position, enabling meaningful relative-position attention with patches.
                    # Note: cos_emb/sin_emb are cos(θ)/sin(θ) of the raw angles,
                    # and mean(cos(θ_i)) ≈ cos(mean(θ_i)) for nearby angles.
                    aggr_cos_img = img_cos.mean(dim=0, keepdim=True).expand(n_aggr, -1)
                    aggr_sin_img = img_sin.mean(dim=0, keepdim=True).expand(n_aggr, -1)
                    new_cos_parts.append(aggr_cos_img)
                    new_sin_parts.append(aggr_sin_img)
                    new_seqlens.append(n_tokens + n_aggr)
                else:
                    new_parts.append(img_tokens)
                    new_cos_parts.append(img_cos)
                    new_sin_parts.append(img_sin)
                    new_seqlens.append(n_tokens)

                offset += n_tokens

            hidden_states = torch.cat(new_parts, dim=0)
            new_cos = torch.cat(new_cos_parts, dim=0)
            new_sin = torch.cat(new_sin_parts, dim=0)
            position_embeddings = (new_cos, new_sin)

            cu_seqlens = torch.tensor(
                new_seqlens, device=grid_thw.device, dtype=torch.int32
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        else:
            cu_seqlens = cu_seqlens_orig

        # ---- Step 3: Run ViT blocks with FiLM ----
        deepstack_feature_lists = []
        aggr_deepstack_features = []  # aggr token features at deepstack layers
        film_idx = 0

        # Build boolean mask for history tokens (including aggr tokens).
        # FiLM should ONLY modulate history image tokens, not current frame tokens.
        # CogVLA processes one image at a time so this isn't an issue there,
        # but our packed ViT forward has history + current in the same sequence.
        if has_history:
            _hist_mask_parts = []
            _offset = 0
            for _img_idx in range(n_images):
                _n_tok = tokens_per_image[_img_idx]
                if is_hist_list[_img_idx]:
                    # History image patches + aggr tokens: apply FiLM
                    _hist_mask_parts.append(torch.ones(_n_tok + n_aggr, dtype=torch.bool, device=hidden_states.device))
                else:
                    # Current image patches: do NOT apply FiLM
                    _hist_mask_parts.append(torch.zeros(_n_tok, dtype=torch.bool, device=hidden_states.device))
            history_token_mask = torch.cat(_hist_mask_parts)  # [total_seq_len]

        for layer_num, blk in enumerate(self_vit.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # Apply FiLM after the block (between blocks), ONLY to history tokens
            if has_history and layer_num >= film_start_block:
                film_layer = compressor.get_film_layer(film_idx)
                # FiLM: x_hist = x_hist * (1 + gamma) + beta, x_curr unchanged
                filmed_all = film_layer(hidden_states, instr_emb_single)
                hidden_states = torch.where(
                    history_token_mask.unsqueeze(-1),  # [seq, 1]
                    filmed_all,
                    hidden_states,
                )
                film_idx += 1

            # DeepStack: extract intermediate features
            if layer_num in self_vit.deepstack_visual_indexes:
                ds_idx = self_vit.deepstack_visual_indexes.index(layer_num)
                if has_history:
                    # Remove aggr tokens before deepstack merger (merger expects spatial structure)
                    ds_hidden = _extract_image_tokens_only(
                        hidden_states, tokens_per_image, is_history_image, n_aggr
                    )
                    deepstack_feature = self_vit.deepstack_merger_list[ds_idx](ds_hidden)
                    # Extract aggr tokens at this deepstack layer for history images
                    ds_aggr_indices = []
                    _offset = 0
                    for _img_idx in range(n_images):
                        _n_tok = tokens_per_image[_img_idx]
                        if is_hist_list[_img_idx]:
                            _aggr_start = _offset + _n_tok
                            ds_aggr_indices.append(torch.arange(
                                _aggr_start, _aggr_start + n_aggr,
                                device=hidden_states.device
                            ))
                            _offset = _aggr_start + n_aggr
                        else:
                            _offset += _n_tok
                    if ds_aggr_indices:
                        all_ds_aggr_idx = torch.cat(ds_aggr_indices)
                        all_ds_aggr_hidden = hidden_states[all_ds_aggr_idx]
                        all_ds_aggr_proj = compressor.project_aggr_deepstack(
                            all_ds_aggr_hidden, ds_idx
                        )
                        aggr_deepstack_features.append(all_ds_aggr_proj)
                    else:
                        aggr_deepstack_features.append(None)
                else:
                    deepstack_feature = self_vit.deepstack_merger_list[ds_idx](hidden_states)
                    aggr_deepstack_features.append(None)
                deepstack_feature_lists.append(deepstack_feature)

        # ---- Step 4: Extract results (batched for performance) ----
        if has_history:
            tokens_per_image_list = tokens_per_image  # before-merge counts
            is_hist_list = is_history_image.tolist() if torch.is_tensor(is_history_image) else list(is_history_image)
            merge_size_sq = compressor.merge_unit  # 4

            # Collect indices for aggr tokens and current-image tokens in one pass
            aggr_indices = []
            current_indices = []
            offset = 0
            for img_idx in range(n_images):
                n_tokens = tokens_per_image_list[img_idx]
                if is_hist_list[img_idx]:
                    aggr_start = offset + n_tokens
                    aggr_indices.append(torch.arange(aggr_start, aggr_start + n_aggr, device=hidden_states.device))
                    offset = aggr_start + n_aggr
                else:
                    current_indices.append(torch.arange(offset, offset + n_tokens, device=hidden_states.device))
                    offset += n_tokens

            # Batch project all aggr tokens at once
            if aggr_indices:
                all_aggr_idx = torch.cat(aggr_indices)
                all_aggr_hidden = hidden_states[all_aggr_idx]  # [n_hist * n_aggr, vit_dim]
                all_aggr_projected = compressor.project_aggr_tokens(all_aggr_hidden)  # [n_hist * n_aggr, llm_dim]

            # Batch merge all current tokens at once
            if current_indices:
                all_current_idx = torch.cat(current_indices)
                all_current_tokens = hidden_states[all_current_idx]  # [total_current_tokens, vit_dim]
                all_current_merged = self_vit.merger(all_current_tokens)  # [total_current_tokens/4, llm_dim]

            # Reassemble in order: interleave history (aggr) and current (merged) parts
            result_parts = []
            aggr_offset = 0
            current_offset = 0
            # Compute per-current-image merged token counts for splitting
            current_merged_counts = []
            for img_idx in range(n_images):
                if not is_hist_list[img_idx]:
                    current_merged_counts.append(tokens_per_image_list[img_idx] // merge_size_sq)
            if current_indices:
                current_splits = torch.split(all_current_merged, current_merged_counts)
            current_split_idx = 0

            for img_idx in range(n_images):
                if is_hist_list[img_idx]:
                    result_parts.append(all_aggr_projected[aggr_offset:aggr_offset + n_aggr])
                    aggr_offset += n_aggr
                else:
                    result_parts.append(current_splits[current_split_idx])
                    current_split_idx += 1

            image_embeds = torch.cat(result_parts, dim=0)

            # Fix deepstack: replace history merged features with projected aggr features
            if len(deepstack_feature_lists) > 0:
                merged_counts = [t // merge_size_sq for t in tokens_per_image_list]
                fixed_deepstack = []
                for ds_layer_idx, ds_feat in enumerate(deepstack_feature_lists):
                    ds_per_image = torch.split(ds_feat, merged_counts)
                    ds_fixed_parts = []
                    aggr_ds_offset = 0
                    for img_idx in range(n_images):
                        if is_hist_list[img_idx]:
                            # Use real projected aggr tokens instead of zeros
                            ds_aggr_proj = aggr_deepstack_features[ds_layer_idx]
                            if ds_aggr_proj is not None:
                                ds_fixed_parts.append(
                                    ds_aggr_proj[aggr_ds_offset:aggr_ds_offset + n_aggr]
                                )
                            else:
                                ds_fixed_parts.append(
                                    torch.zeros(
                                        n_aggr, ds_per_image[img_idx].shape[-1],
                                        device=ds_feat.device, dtype=ds_feat.dtype
                                    )
                                )
                            aggr_ds_offset += n_aggr
                        else:
                            ds_fixed_parts.append(ds_per_image[img_idx])
                    fixed_deepstack.append(torch.cat(ds_fixed_parts, dim=0))
                deepstack_feature_lists = fixed_deepstack

            # Add dummy forward for ZeRO-2 (ensure all params in graph)
            dummy_zero = compressor.dummy_forward(image_embeds)
            image_embeds = image_embeds + dummy_zero
        else:
            # No history: standard path
            image_embeds = self_vit.merger(hidden_states)
            dummy_zero = compressor.dummy_forward(image_embeds)
            image_embeds = image_embeds + dummy_zero

        return image_embeds, deepstack_feature_lists

    def _extract_image_tokens_only(hidden_states, tokens_per_image, is_history_image, n_aggr):
        """
        Remove aggr tokens from packed hidden_states, keeping only real image tokens.
        Used for deepstack merger which needs spatial token layout.
        """
        is_hist = is_history_image.tolist() if torch.is_tensor(is_history_image) else list(is_history_image)
        # Build index of all non-aggr token positions
        n_images = len(tokens_per_image)
        keep_indices = []
        offset = 0
        for img_idx in range(n_images):
            n_tokens = tokens_per_image[img_idx]
            keep_indices.append(torch.arange(offset, offset + n_tokens, device=hidden_states.device))
            if is_hist[img_idx]:
                offset += n_tokens + n_aggr
            else:
                offset += n_tokens
        all_keep = torch.cat(keep_indices)
        return hidden_states[all_keep]

    # Attach helper as attribute of the function
    patched_vit_forward._extract_image_tokens_only = _extract_image_tokens_only

    return patched_vit_forward


# ============================================================
# Patched Inner Model Forward (Qwen3VLModel)
# ============================================================

def make_inner_forward(original_inner_forward, compressor, outer_model_ref):
    """
    Replace Qwen3VLModel.forward with compressor-aware version.
    Nearly identical to bottleneck wrapper's inner forward, but:
    - ViT already returns compressed embeddings for history images
    - No separate compress step needed here
    - Just handle RoPE grid and scatter into inputs_embeds
    """

    def compressor_forward(
        self_model,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        is_history_image = getattr(outer_model_ref, '_compressor_is_history', None)
        image_grid_thw_rope = getattr(outer_model_ref, '_compressor_grid_thw_rope', None)
        # Clear after reading
        outer_model_ref._compressor_is_history = None
        outer_model_ref._compressor_grid_thw_rope = None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self_model.get_input_embeddings()(input_ids)

        visual_pos_masks = None
        deepstack_visual_embeds = None

        if pixel_values is not None:
            # Compute instruction embedding BEFORE ViT forward
            # (ViT forward needs it for FiLM conditioning)
            with torch.no_grad():
                instr_emb = get_instruction_embedding(
                    outer_model_ref, input_ids, inputs_embeds.detach()
                )
            instr_emb_single = instr_emb.mean(dim=0)  # [llm_dim]
            # Store for ViT forward to pick up
            outer_model_ref._compressor_instr_emb = instr_emb_single
            # Restore is_history for ViT forward (we cleared it above)
            outer_model_ref._compressor_is_history = is_history_image

            # Run patched ViT (returns already-compressed history embeddings)
            pixel_values_typed = pixel_values.type(self_model.visual.dtype)
            image_embeds, deepstack_features = self_model.visual(
                pixel_values_typed, grid_thw=image_grid_thw
            )

            # Clean up
            outer_model_ref._compressor_instr_emb = None
            outer_model_ref._compressor_is_history = None

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            # Scatter into inputs_embeds
            image_mask, _ = self_model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # DeepStack handling
            if deepstack_features is not None and len(deepstack_features) > 0:
                visual_pos_masks = image_mask[..., 0]
                deepstack_visual_embeds = deepstack_features
            else:
                visual_pos_masks = image_mask[..., 0]
                deepstack_visual_embeds = None

        # Handle videos (pass-through)
        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self_model.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds_cat = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self_model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds_cat,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)

            if visual_pos_masks is not None:
                video_mask_1d = video_mask[..., 0]
                visual_pos_masks = visual_pos_masks | video_mask_1d
                if deepstack_visual_embeds is not None:
                    new_ds = []
                    for img_e, vid_e in zip(deepstack_visual_embeds, deepstack_video_embeds):
                        e = img_e.new_zeros(visual_pos_masks.sum(), img_e.shape[-1])
                        img_joint = image_mask[..., 0][visual_pos_masks]
                        vid_joint = video_mask_1d[visual_pos_masks]
                        e[img_joint] = img_e
                        e[vid_joint] = vid_e
                        new_ds.append(e)
                    deepstack_visual_embeds = new_ds
            else:
                visual_pos_masks = video_mask[..., 0]
                deepstack_visual_embeds = deepstack_video_embeds

        # Use compressed grid_thw for RoPE
        rope_grid_thw = (
            image_grid_thw_rope
            if image_grid_thw_rope is not None
            else image_grid_thw
        )

        if position_ids is None:
            from transformers.utils import is_torchdynamo_compiling

            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask.get("full_attention", None)
            )
            if (
                attention_mask_tensor is not None
                and attention_mask_tensor.ndim == 4
            ):
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = (
                        attention_mask_tensor
                        / torch.finfo(attention_mask_tensor.dtype).min
                    )
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self_model.rope_deltas is None:
                position_ids, rope_deltas = self_model.get_rope_index(
                    input_ids,
                    rope_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self_model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self_model.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self_model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self_model.rope_deltas,
        )

    return compressor_forward


# ============================================================
# Patched Outer Forward (Qwen3VLForConditionalGeneration)
# ============================================================

def make_outer_forward(original_outer_forward):
    """Intercept compressor kwargs and store on self."""
    @functools.wraps(original_outer_forward)
    def outer_forward_wrapper(self, *args, **kwargs):
        if "is_history_image" in kwargs:
            self._compressor_is_history = kwargs.pop("is_history_image")
        if "image_grid_thw_rope" in kwargs:
            self._compressor_grid_thw_rope = kwargs.pop("image_grid_thw_rope")
        return original_outer_forward(self, *args, **kwargs)
    return outer_forward_wrapper


# ============================================================
# Main Entry Point
# ============================================================

def attach_compressor_film_vit(model, compressor_config=None):
    """
    Attach AggrTokenCompressor to a Qwen3VLForConditionalGeneration model.
    
    Args:
        model: Qwen3VLForConditionalGeneration instance
        compressor_config: dict with keys:
            n_aggr (int): number of aggregation tokens, default 16
            n_film_layers (int): number of ViT blocks to apply FiLM, default 24
            share_film (bool): share FiLM weights across blocks, default False
    """
    if compressor_config is None:
        compressor_config = {}

    # Get ViT config
    vit = model.model.visual
    vit_dim = vit.blocks[0].norm1.normalized_shape[0]  # 1024
    llm_dim = vit.merger.linear_fc2.out_features  # 2048
    spatial_merge_size = vit.spatial_merge_size  # 2

    n_aggr = compressor_config.get('n_aggr', 16)
    n_film_layers = compressor_config.get('n_film_layers', 24)
    share_film = compressor_config.get('share_film', False)

    comp = AggrTokenCompressor(
        vit_dim=vit_dim,
        llm_dim=llm_dim,
        n_aggr=n_aggr,
        n_film_layers=n_film_layers,
        share_film=share_film,
        spatial_merge_size=spatial_merge_size,
    )

    ref_param = next(model.parameters())
    comp = comp.to(device=ref_param.device, dtype=ref_param.dtype)

    model.compressor = comp

    # Initialize instance attributes
    model._compressor_is_history = None
    model._compressor_grid_thw_rope = None
    model._compressor_instr_emb = None

    # ---- Patch outer forward ----
    original_outer_forward = model.__class__.forward
    patched_outer = make_outer_forward(original_outer_forward)
    model.__class__.forward = patched_outer

    # ---- Patch ViT forward ----
    vit._original_forward = vit.forward
    patched_vit = make_patched_vit_forward(vit.forward, comp, model)
    vit.forward = types.MethodType(patched_vit, vit)

    # ---- Patch inner forward ----
    inner_model = model.model
    inner_model._original_forward = inner_model.forward
    new_inner_forward = make_inner_forward(inner_model.forward, comp, model)
    inner_model.forward = types.MethodType(new_inner_forward, inner_model)

    return model
