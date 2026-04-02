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

Stage 2 Strategy:
  - Load Compressor weights from Stage 1a checkpoint
  - Train: Compressor (base lr) + Merger last projection (mm_projector_lr) +
           DeepStack merger last projection + merger-related norms
  - Freeze: ViT backbone, LLM, lm_head, embed_tokens
  - History deepstack: zeros (same as Stage 1a, method X)
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
        # Pop compressor kwargs if passed explicitly (training path).
        # If not in kwargs (inference path), preserve any value already set
        # on the model instance by the caller (e.g. evaluator sets
        # self.model._compressor_is_history before calling generate()).
        if "is_history_image" in kwargs:
            self._compressor_is_history = kwargs.pop("is_history_image")
        if "image_grid_thw_rope" in kwargs:
            self._compressor_grid_thw_rope = kwargs.pop("image_grid_thw_rope")
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

            # Step 6: DeepStack handling — 历史帧 deepstack 也走 FiLM 压缩
            if all_deepstack is not None and len(all_deepstack) > 0:
                new_deepstack = []
                for ds in all_deepstack:
                    ds_per_image = list(torch.split(ds, split_sizes))
                    ds_parts = []
                    if n_hist > 0:
                        # 收集历史帧的 deepstack tokens 并压缩
                        ds_history = torch.stack(
                            [ds_per_image[i] for i in range(n_total_images) if is_history_image[i]]
                        )  # [n_hist, T, d_model]
                        ds_compressed = compressor.compress_frames(
                            ds_history, instr_emb_single
                        )  # [n_hist, n_queries, d_model]
                    hist_counter = 0
                    for i in range(n_total_images):
                        if is_history_image[i]:
                            ds_parts.append(
                                ds_compressed[hist_counter].reshape(-1, ds_compressed.shape[-1])
                            )
                            hist_counter += 1
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


def apply_compressor_stage2(model, compressor_config=None, stage1a_checkpoint=None):
    """
    Stage 2: Fine-tune Compressor + Merger last projection layers.
    
    Load Compressor weights from Stage 1a checkpoint, then:
      - Freeze: ViT backbone, LLM, lm_head, embed_tokens
      - Train:  Compressor (base lr=5e-4)
                merger.linear_fc2 + bias (mm_projector_lr=1e-5)
                deepstack_merger_list[*].linear_fc2 + bias (mm_projector_lr=1e-5)
                merger.norm + deepstack norms (mm_projector_lr=1e-5)
      - History deepstack: zeros (same as Stage 1a)
    
    The create_optimizer in qwenvl_base.py routes:
      - params with "merger" in name → mm_projector_lr group
      - other params (compressor) → base learning_rate group
    """
    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Step 2: Attach compressor (same as Stage 1a)
    model = attach_compressor(model, compressor_config)
    
    # Step 3: Load Compressor weights from Stage 1a checkpoint
    if stage1a_checkpoint is not None:
        import os
        from safetensors import safe_open
        
        ckpt_path = os.path.join(stage1a_checkpoint, "model.safetensors")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Stage 1a checkpoint not found: {ckpt_path}"
            )
        
        comp_state = {}
        with safe_open(ckpt_path, framework="pt") as f:
            for key in f.keys():
                if key.startswith("compressor."):
                    comp_state[key.replace("compressor.", "")] = f.get_tensor(key)
        
        missing, unexpected = model.compressor.load_state_dict(comp_state, strict=False)
        
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        if rank == 0:
            print(f"[Stage 2] Loaded Compressor from: {ckpt_path}")
            print(f"  Loaded {len(comp_state)} tensors")
            if missing:
                print(f"  WARNING missing keys: {missing}")
            if unexpected:
                print(f"  WARNING unexpected keys: {unexpected}")
    else:
        raise ValueError("stage1a_checkpoint is required for Stage 2")
    
    # Step 4: Unfreeze Compressor (all params)
    for param in model.compressor.parameters():
        param.requires_grad = True
    
    # Step 5: Unfreeze Merger last projection + norms
    # merger.linear_fc2 (the last projection: 4096 → 2048)
    for name, param in model.model.visual.merger.named_parameters():
        if "linear_fc2" in name:
            param.requires_grad = True
        if "norm" in name.lower():
            param.requires_grad = True
    
    # deepstack_merger_list[*].linear_fc2 + norms
    for name, param in model.model.visual.deepstack_merger_list.named_parameters():
        if "linear_fc2" in name:
            param.requires_grad = True
        if "norm" in name.lower():
            param.requires_grad = True
    
    # Step 6: Enable input require grads (needed for gradient flow)
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(
            _make_inputs_require_grad
        )
    
    # Step 7: Print configuration
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    
    if rank == 0:
        # Detailed stats
        comp_params = sum(p.numel() for p in model.compressor.parameters() if p.requires_grad)
        merger_params = sum(
            p.numel() for n, p in model.model.visual.merger.named_parameters()
            if p.requires_grad
        )
        ds_params = sum(
            p.numel() for n, p in model.model.visual.deepstack_merger_list.named_parameters()
            if p.requires_grad
        )
        
        print("=" * 80)
        print("Compressor Stage 2 Configuration:")
        print("=" * 80)
        print(f"Total params: {total_params / 1e6:.1f}M")
        print(
            f"Trainable params: {trainable_params / 1e6:.2f}M "
            f"({trainable_params / total_params * 100:.2f}%)"
        )
        print("")
        print("Trainable Components:")
        print(f"  Compressor:                    {comp_params / 1e6:.2f}M  (base lr)")
        print(f"  merger.linear_fc2 + norm:      {merger_params / 1e6:.2f}M  (mm_projector_lr)")
        print(f"  deepstack linear_fc2 + norms:  {ds_params / 1e6:.2f}M  (mm_projector_lr)")
        print("")
        print("Freeze Strategy:")
        print("  ViT backbone:        FROZEN")
        print("  merger.linear_fc1:   FROZEN")
        print("  merger.linear_fc2:   TRAINABLE (mm_projector_lr)")
        print("  merger.norm:         TRAINABLE (mm_projector_lr)")
        print("  deepstack fc1:       FROZEN")
        print("  deepstack fc2:       TRAINABLE (mm_projector_lr)")
        print("  deepstack norms:     TRAINABLE (mm_projector_lr)")
        print("  LLM:                 FROZEN")
        print("  lm_head:             FROZEN")
        print("  embed_tokens:        FROZEN")
        print("  Compressor:          TRAINABLE (base lr)")
        print("  History DeepStack:   ZEROS (same as Stage 1a)")
        print("=" * 80)
        
        # List all trainable params for verification
        print("")
        print("All trainable parameters:")
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"  {n}: {list(p.shape)} ({p.numel()/1e3:.1f}K)")
        print("")
    
    return model


def apply_compressor_stage3b(model, compressor_config=None, stage1a_checkpoint=None,
                              lora_r=32, lora_alpha=64, lora_dropout=0.05):
    """
    Stage 3b (方案B): LLM LoRA + Compressor, freeze ViT/Merger.
    
    Goal: Verify whether LLM can learn to utilize compressed 16-token
    history representations. This directly addresses the bottleneck
    identified in Stage 2 (unfreezing Merger did not reduce loss).
    
    Pipeline:
      1. Freeze all params
      2. Attach & load Compressor from Stage 1a checkpoint
      3. Apply LLM-only LoRA (no ViT LoRA, no modules_to_save)
      4. Unfreeze Compressor
      → Trainable: Compressor (~10.5M, base lr) + LLM LoRA (~14M, base lr)
      → Frozen: ViT, Merger, DeepStack, embed_tokens, lm_head
      → History deepstack: zeros (same as Stage 1a)
    
    Order matters: attach_compressor BEFORE peft wrapping, because
    compressor patches model.model.forward (inner model), and PEFT
    wraps model.forward (outer model).
    """
    from peft import LoraConfig, get_peft_model, TaskType
    
    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Step 2: Attach compressor and patch forward (BEFORE PEFT wrapping)
    model = attach_compressor(model, compressor_config)
    
    # Step 3: Load Compressor weights from Stage 1a
    if stage1a_checkpoint is None:
        raise ValueError("stage1a_checkpoint is required for Stage 3b")
    
    import os
    from safetensors import safe_open
    
    ckpt_path = os.path.join(stage1a_checkpoint, "model.safetensors")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Stage 1a checkpoint not found: {ckpt_path}")
    
    comp_state = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for key in f.keys():
            if key.startswith("compressor."):
                comp_state[key.replace("compressor.", "")] = f.get_tensor(key)
    
    missing, unexpected = model.compressor.load_state_dict(comp_state, strict=False)
    
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    if rank == 0:
        print(f"[Stage 3b] Loaded Compressor from: {ckpt_path}")
        print(f"  Loaded {len(comp_state)} tensors")
        if missing:
            print(f"  WARNING missing keys: {missing}")
        if unexpected:
            print(f"  WARNING unexpected keys: {unexpected}")
    
    # Step 4: Apply LLM-only LoRA
    # Only target LLM attention + MLP linear layers
    # Do NOT add ViT modules (qkv, proj) — ViT stays frozen
    # Do NOT add modules_to_save (merger, embed_tokens, lm_head) — all frozen
    # Target LLM attention + MLP projections, exclude ViT modules
    # ViT also has gate_proj/up_proj/down_proj in visual.blocks.*.mlp
    # Use exclude_modules to prevent LoRA being applied to ViT
    llm_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=llm_target_modules,
        exclude_modules=["visual.*"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Enable input require grads before PEFT (PEFT needs it)
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(
            _make_inputs_require_grad
        )
    
    model = get_peft_model(model, lora_config)
    
    # Step 5: Unfreeze Compressor
    # After PEFT wrapping, compressor is at model.base_model.model.compressor
    # but model.compressor still works via __getattr__
    for name, param in model.named_parameters():
        if "compressor" in name:
            param.requires_grad = True
    
    # Step 6: Print configuration
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    if rank == 0:
        # Count by component
        comp_params = 0
        lora_params = 0
        other_params = 0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "compressor" in n:
                comp_params += p.numel()
            elif "lora" in n.lower():
                lora_params += p.numel()
            else:
                other_params += p.numel()
        
        print("=" * 80)
        print("Compressor Stage 3b Configuration (方案B):")
        print("=" * 80)
        print(f"Total params: {total_params / 1e6:.1f}M")
        print(f"Trainable params: {trainable_params / 1e6:.2f}M "
              f"({trainable_params / total_params * 100:.2f}%)")
        print("")
        print("Trainable Components:")
        print(f"  Compressor:          {comp_params / 1e6:.2f}M  (base lr)")
        print(f"  LLM LoRA (r={lora_r}):    {lora_params / 1e6:.2f}M  (base lr)")
        if other_params > 0:
            print(f"  Other:               {other_params / 1e6:.2f}M")
        print("")
        print(f"LoRA Config:")
        print(f"  rank={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"  targets: {llm_target_modules}")
        print(f"  exclude: ['visual.*']")
        print(f"  bias: none")
        print("")
        print("Freeze Strategy:")
        print("  ViT backbone:        FROZEN (no LoRA)")
        print("  Merger:              FROZEN")
        print("  DeepStack Mergers:   FROZEN")
        print("  LLM layers:          LoRA adapters only")
        print("  embed_tokens:        FROZEN")
        print("  lm_head:             FROZEN")
        print("  Compressor:          TRAINABLE (from Stage 1a)")
        print("  History DeepStack:   ZEROS (same as Stage 1a)")
        print("=" * 80)
        
        # List all trainable params
        print("")
        print("All trainable parameters:")
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"  {n}: {list(p.shape)} ({p.numel()/1e3:.1f}K)")
        print("")
    
    return model
