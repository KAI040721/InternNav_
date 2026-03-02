"""
Compressor Wrapper for Qwen3VLForConditionalGeneration

Monkey-patches BOTH the outer ForConditionalGeneration.forward AND the inner
Qwen3VLModel.forward to inject compressor logic.

Strategy: Store compressor args on model instance attributes before they get
lost in decorator/wrapper chains.

Key design for distributed training (ZeRO-2):
  ALL samples go through the same code path. When there are no history images,
  a dummy compressor forward is executed so that compressor parameters always
  participate in the computation graph and trigger identical allreduce hooks
  on every rank, preventing NCCL deadlocks.

Batch>1 support:
  Uses a boolean mask `is_history_image` (one entry per image across the whole
  batch) to identify which images need compression, regardless of how many
  samples are in the batch.

Stage 1a Strategy:
  - Freeze ALL original params (ViT + LLM)
  - Train ONLY Compressor (~10.5M params)
  - History frames: primary tokens compressed 144→16, deepstack set to zeros
  - Current frame + birdseye: unchanged (full 144 tokens + deepstack)
"""

import torch
import torch.nn as nn
import types
import functools
from typing import Optional, Union, List
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast

from internnav.model.compressor import BottleneckCompressor


def get_instruction_embedding(model, input_ids, inputs_embeds):
    """
    Extract instruction embedding by mean-pooling text tokens
    (excluding special tokens and image_pad tokens).
    Returns: [B, d_model]
    """
    config = model.config
    special_ids = {
        config.image_token_id,   # 151655
        config.video_token_id,   # 151656
    }
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


def _dummy_compressor_forward(compressor, ref_tensor):
    """
    Run a dummy forward through compressor so its parameters are in the
    computation graph (required for ZeRO-2 allreduce consistency).
    Returns a zero scalar that can be added to any tensor without changing it.
    """
    dummy_input = torch.zeros(
        1, 1, compressor.d_model,
        device=ref_tensor.device, dtype=ref_tensor.dtype,
    )
    dummy_instr = torch.zeros(
        compressor.d_model,
        device=ref_tensor.device, dtype=ref_tensor.dtype,
    )
    dummy_out = compressor.compress_frames(dummy_input, dummy_instr)
    return dummy_out.sum() * 0.0


def make_outer_forward(original_outer_forward):
    """
    Wrap Qwen3VLForConditionalGeneration.forward to intercept
    compressor-specific kwargs and store them on self before calling
    the original forward (which eventually calls self.model()).
    """
    @functools.wraps(original_outer_forward)
    def outer_forward_wrapper(self, *args, **kwargs):
        self._compressor_is_history = kwargs.pop("is_history_image", None)
        self._compressor_grid_thw_rope = kwargs.pop("image_grid_thw_rope", None)
        return original_outer_forward(self, *args, **kwargs)
    return outer_forward_wrapper


def make_inner_forward(original_inner_forward, compressor, outer_model_ref):
    """
    Replace Qwen3VLModel.forward with compressor-aware version.

    UNIFIED PATH: all samples go through the same code structure.
    When no history images exist, a dummy compressor forward ensures
    parameters are always in the computation graph.
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
        # Retrieve compressor args from outer model instance
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

        # ================================================================
        # UNIFIED IMAGE PATH — always runs the same set of nn.Module params
        # ================================================================
        if pixel_values is not None:
            # Step 1: Run ViT on all images
            pixel_values_typed = pixel_values.type(self_model.visual.dtype)
            all_image_embeds, all_deepstack = self_model.visual(
                pixel_values_typed, grid_thw=image_grid_thw
            )

            # Split per image
            split_sizes = (
                image_grid_thw.prod(-1) // self_model.visual.spatial_merge_size ** 2
            ).tolist()
            image_embeds_list = list(torch.split(all_image_embeds, split_sizes))
            n_total_images = len(image_embeds_list)

            # Build is_history mask if not provided
            if is_history_image is None:
                is_history_image = torch.zeros(n_total_images, dtype=torch.bool)
            is_history_image = is_history_image.bool()
            assert len(is_history_image) == n_total_images, (
                f"is_history_image length {len(is_history_image)} != "
                f"n_total_images {n_total_images}"
            )

            n_hist = int(is_history_image.sum().item())

            # Step 2: Get instruction embedding for FiLM conditioning
            with torch.no_grad():
                instr_emb = get_instruction_embedding(
                    outer_model_ref, input_ids, inputs_embeds.detach()
                )
            # Mean across batch (instruction should be similar within batch)
            instr_emb_single = instr_emb.mean(dim=0)  # [d_model]

            # Step 3: Compress or dummy-forward
            if n_hist > 0:
                # Gather history embeddings and compress
                history_tokens = torch.stack(
                    [image_embeds_list[i] for i in range(n_total_images) if is_history_image[i]]
                )  # [n_hist, T, d_model]
                compressed_history = compressor.compress_frames(
                    history_tokens, instr_emb_single
                )  # [n_hist, n_queries, d_model]
            else:
                # Dummy forward: ensures compressor params are in compute graph
                compressed_history = None

            dummy_zero = _dummy_compressor_forward(compressor, all_image_embeds)

            # Step 4: Build final flat image embeddings in original order
            result_parts = []
            for i in range(n_total_images):
                if is_history_image[i]:
                    # Find which history index this is
                    hist_idx = int(is_history_image[:i].sum().item())
                    result_parts.append(
                        compressed_history[hist_idx].reshape(-1, compressed_history.shape[-1])
                    )
                else:
                    result_parts.append(image_embeds_list[i])

            image_embeds_final = torch.cat(result_parts, dim=0)
            image_embeds_final = image_embeds_final.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            # Always add dummy_zero to keep compressor in compute graph
            image_embeds_final = image_embeds_final + dummy_zero

            # Step 5: Scatter into inputs_embeds
            image_mask, _ = self_model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds_final,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_embeds_final
            )

            # Step 6: DeepStack handling
            if all_deepstack is not None and len(all_deepstack) > 0:
                new_deepstack = []
                for ds in all_deepstack:
                    ds_per_image = list(torch.split(ds, split_sizes))
                    ds_parts = []
                    for i in range(n_total_images):
                        if is_history_image[i]:
                            # Zero deepstack for history (Stage 1a)
                            ds_parts.append(
                                ds.new_zeros(compressor.n_queries, ds.shape[-1])
                            )
                        else:
                            ds_parts.append(ds_per_image[i])
                    new_deepstack.append(torch.cat(ds_parts, dim=0))

                visual_pos_masks = image_mask[..., 0]
                deepstack_visual_embeds = new_deepstack
            else:
                visual_pos_masks = image_mask[..., 0]
                deepstack_visual_embeds = None

        # Handle videos (pass-through, not relevant for VLN)
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
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask, video_embeds_cat
            )

            if visual_pos_masks is not None:
                video_mask_1d = video_mask[..., 0]
                visual_pos_masks = visual_pos_masks | video_mask_1d
                if deepstack_visual_embeds is not None:
                    new_ds = []
                    for img_e, vid_e in zip(
                        deepstack_visual_embeds, deepstack_video_embeds
                    ):
                        e = img_e.new_zeros(
                            visual_pos_masks.sum(), img_e.shape[-1]
                        )
                        img_joint = image_mask[..., 0][visual_pos_masks]
                        vid_joint = video_mask_1d[visual_pos_masks]
                        e[img_joint] = img_e
                        e[vid_joint] = vid_e
                        new_ds.append(e)
                    deepstack_visual_embeds = new_ds
            else:
                visual_pos_masks = video_mask[..., 0]
                deepstack_visual_embeds = deepstack_video_embeds

        # Use compressed grid_thw for RoPE position encoding
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
                    attention_mask_tensor = (
                        1.0 - attention_mask_tensor
                    ).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (
                    inputs_embeds is not None
                    and inputs_embeds.shape[1] != 1
                )
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (
                    past_key_values is None
                    or past_key_values.get_seq_length() == 0
                )
            )
            if (
                prefill_compiled_stage or prefill_noncompiled_stage
            ) or self_model.rope_deltas is None:
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
                    (
                        cache_position[0] + self_model.rope_deltas
                    ).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(
                    seq_length, device=inputs_embeds.device
                )
                position_ids = position_ids.view(1, -1).expand(
                    batch_size, -1
                )
                if cache_position is not None:
                    delta = delta.repeat_interleave(
                        batch_size // delta.shape[0], dim=0
                    )
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


def attach_compressor(model, compressor_config=None):
    """
    Attach a BottleneckCompressor to a Qwen3VLForConditionalGeneration model.
    """
    if compressor_config is None:
        compressor_config = {}

    d_model = compressor_config.get('d_model', 2048)
    d_bottleneck = compressor_config.get('d_bottleneck', 512)
    n_queries = compressor_config.get('n_queries', 16)
    n_heads = compressor_config.get('n_heads', 8)
    n_layers = compressor_config.get('n_layers', 2)

    comp = BottleneckCompressor(
        d_model=d_model,
        d_bottleneck=d_bottleneck,
        n_queries=n_queries,
        n_heads=n_heads,
        n_layers=n_layers,
    )

    ref_param = next(model.parameters())
    comp = comp.to(device=ref_param.device, dtype=ref_param.dtype)

    model.compressor = comp

    model._compressor_is_history = None
    model._compressor_grid_thw_rope = None

    # Patch outer forward
    original_outer_forward = model.__class__.forward
    patched_outer = make_outer_forward(original_outer_forward)
    model.__class__.forward = patched_outer

    # Patch inner forward
    inner_model = model.model
    inner_model._original_forward = inner_model.forward
    new_inner_forward = make_inner_forward(inner_model.forward, comp, model)
    inner_model.forward = types.MethodType(new_inner_forward, inner_model)

    return model


def apply_compressor_stage1a(model, compressor_config=None):
    """
    Stage 1a: Freeze everything, train only Compressor.
    """
    for param in model.parameters():
        param.requires_grad = False

    model = attach_compressor(model, compressor_config)

    for param in model.compressor.parameters():
        param.requires_grad = True

    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(
            _make_inputs_require_grad
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if rank == 0:
        print("=" * 80)
        print("Compressor Stage 1a Configuration:")
        print("=" * 80)
        print(f"Total params: {total_params / 1e6:.1f}M")
        print(
            f"Trainable params: {trainable_params / 1e6:.2f}M "
            f"({trainable_params / total_params * 100:.2f}%)"
        )
        comp_params = sum(p.numel() for p in model.compressor.parameters())
        print(f"Compressor params: {comp_params / 1e6:.2f}M")
        print(f"  d_model: {model.compressor.d_model}")
        print(f"  d_bottleneck: {model.compressor.d_bottleneck}")
        print(f"  n_queries: {model.compressor.n_queries}")
        print(f"  n_heads: {model.compressor.n_heads}")
        print(f"  n_layers: {model.compressor.n_layers}")
        print("")
        print("Freeze Strategy:")
        print("  ViT (visual):        FROZEN")
        print("  Merger:              FROZEN")
        print("  DeepStack Mergers:   FROZEN")
        print("  LLM:                 FROZEN")
        print("  Compressor:          TRAINABLE")
        print("  History DeepStack:   Set to ZEROS (disabled)")
        print("=" * 80)

    return model
