# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import os
import pathlib
import sys
from pathlib import Path
from typing import Dict

import torch
import transformers
from torchvision.transforms import v2

# Override default NCCL timeout to avoid timeouts during gradient accumulation
# with variable-length sequences (default 600s is too short)
from datetime import timedelta as _timedelta
try:
    import torch.distributed.constants as _dist_constants
    _dist_constants.default_pg_nccl_timeout = _timedelta(seconds=7200)
except Exception:
    pass
try:
    from accelerate.utils.dataclasses import InitProcessGroupKwargs as _IPGK
    _IPGK.__dataclass_fields__["timeout"].default = _timedelta(seconds=7200)
except Exception:
    pass

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl_base import replace_qwen2_vl_attention_class
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,

    Trainer,
)

from internnav.dataset.internvla_n1_lerobot_dataset import make_supervised_data_module
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.compressor_wrapper import apply_compressor_stage1a
from internnav.trainer.internvla_n1_argument import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)

# LoRA相关导入
from peft import LoraConfig, get_peft_model, TaskType


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg



def apply_lora_to_qwen3vl(model, model_args):
    """
    对Qwen3VL模型应用针对VLN任务优化的微调策略 (v4):
    
    设计原则:
      VLN任务需要模型在室内3D扫描场景中理解空间布局、物体语义和方向关系。
      预训练ViT在自然图片上训练，与室内全景场景存在域差距，
      但ViT的底层特征(边缘、纹理)仍然通用，因此用LoRA轻量适配。
      Merger/Deepstack是视觉→语言的桥梁，对任务高度敏感，需全参微调。
      LLM需要学习导航决策的推理模式，用LoRA高效适配。
      Embedding层需解冻以适应导航特有的token分布（坐标、动作词等）。
    
    微调策略:
      ┌─────────────────────────┬──────────────────────┐
      │ 模块                    │ 微调方式             │
      ├─────────────────────────┼──────────────────────┤
      │ ViT patch_embed (Conv3d)│ 冻结                 │
      │ ViT pos_embed           │ 冻结                 │
      │ ViT blocks (attn)       │ LoRA                 │
      │ ViT blocks (mlp)        │ 冻结                 │
      │ ViT norms               │ 全参微调             │
      │ Merger (primary)        │ 全参微调             │
      │ Deepstack Mergers (x3)  │ 全参微调             │
      │ LLM embed_tokens        │ 全参微调             │
      │ LLM layers (attn+mlp)   │ LoRA                 │
      │ LLM norms (RMSNorm)     │ 全参微调             │
      │ lm_head                 │ 全参微调(与embed共享) │
      └─────────────────────────┴──────────────────────┘
    """
    # Step 1: 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # Step 2: 构建 LoRA target modules
    # LLM 线性层
    llm_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
    # ViT attention 线性层 (tune_mm_vision=True 时)
    vision_modules = []
    if getattr(model_args, 'tune_mm_vision', False):
        vision_modules = ["qkv", "proj"]  # ViT attention
        # 注意: 不加 linear_fc1/linear_fc2 避免匹配到 merger
        # ViT MLP 保持冻结，保留通用视觉特征
    
    target_modules = list(set(llm_modules + vision_modules))
    
    lora_bias = getattr(model_args, 'lora_bias', 'none')
    
    # Step 3: modules_to_save = 全参微调的模块
    # "merger" 同时匹配 visual.merger 和 visual.deepstack_merger_list 中的 merger
    # "embed_tokens" 和 "lm_head" 解冻 embedding 和输出头
    modules_to_save = ["merger", "embed_tokens", "lm_head"]
    
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        lora_dropout=model_args.lora_dropout,
        bias=lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Step 4: 应用 LoRA
    model = get_peft_model(model, lora_config)
    
    # Step 5: 解冻 Deepstack Mergers (全参微调)
    # PEFT modules_to_save 的 "merger" 只匹配了 visual.merger，
    # deepstack_merger_list 中的子模块需要手动解冻
    for name, param in model.named_parameters():
        if "deepstack_merger_list" in name:
            param.requires_grad = True
    
    # Step 6: 解冻所有 norm 层 (ViT LayerNorm + LLM RMSNorm)
    for name, param in model.named_parameters():
        if "norm" in name.lower() or "layernorm" in name.lower():
            param.requires_grad = True
    
    # Step 7: 打印可训练参数信息
    model.print_trainable_parameters()
    
    # Step 8: 详细日志
    if torch.distributed.get_rank() == 0:
        print("")
        print("=" * 80)
        print("VLN-Optimized LoRA Configuration (v4):")
        print("=" * 80)
        print(f"LoRA Rank: {model_args.lora_r}")
        print(f"LoRA Alpha: {model_args.lora_alpha}")
        print(f"LoRA Dropout: {model_args.lora_dropout}")
        print(f"LoRA Bias: {lora_bias}")
        print(f"LoRA Target Modules: {target_modules}")
        print(f"Full Fine-tune Modules (modules_to_save): {modules_to_save}")
        print("")
        print("Training Status by Component:")
        print("  [Vision Tower]")
        print(f"    - patch_embed (Conv3d):     FROZEN")
        print(f"    - pos_embed (Embedding):    FROZEN")
        print(f"    - blocks.attn (qkv+proj):   LoRA r={model_args.lora_r}")
        print(f"    - blocks.mlp (fc1+fc2):     FROZEN")
        print(f"    - blocks.norm (LayerNorm):  FULL fine-tune")
        print("  [Visual Projector]")
        print(f"    - merger (primary):         FULL fine-tune")
        print(f"    - deepstack_mergers (x3):   FULL fine-tune")
        print("  [Language Model]")
        print(f"    - embed_tokens:             FULL fine-tune")
        print(f"    - layers.attn (q/k/v/o):    LoRA r={model_args.lora_r}")
        print(f"    - layers.mlp (gate/up/down): LoRA r={model_args.lora_r}")
        print(f"    - layers.norm (RMSNorm):    FULL fine-tune")
        print(f"    - lm_head:                  FULL fine-tune (tied with embed_tokens)")
        print("=" * 80)
        
        # 按模块统计可训练参数
        stats = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "visual" in n and "merger" not in n and "deepstack" not in n:
                cat = "ViT (LoRA + Norms)"
            elif "merger" in n or "deepstack" in n:
                cat = "Merger + Deepstack (Full)"
            elif "embed_tokens" in n:
                cat = "Embed Tokens (Full)"
            elif "lm_head" in n:
                cat = "LM Head (Full)"
            elif "lora" in n.lower():
                cat = "LLM LoRA adapters"
            elif "norm" in n.lower():
                cat = "LLM Norms (Full)"
            else:
                cat = "Other"
            stats[cat] = stats.get(cat, 0) + p.numel()
        
        print("")
        print("Trainable Parameters by Component:")
        total_trainable = sum(stats.values())
        for cat, count in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count/1e6:.2f}M ({count/total_trainable*100:.1f}%)")
        print(f"  ─────────────────────────────────")
        print(f"  Total Trainable: {total_trainable/1e6:.2f}M")
        print("")
    
    return model

def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = False

    if 'nextdit' in model_args.system1:
        modules = [
            'action_encoder',
            'action_decoder',
            'traj_dit',
            'cond_projector',
            'memory_encoder',
            'rgb_resampler',
            'rgb_model',
        ]
        for n, p in model.model.named_parameters():
            if any(k in n for k in modules):
                p.requires_grad = True
        model.model.latent_queries.requires_grad = True
    elif 'navdp' in model_args.system1:
        for n, p in model.model.navdp.named_parameters():
            if "rgb_model" not in n:
                p.requires_grad = True
        model.model.latent_queries.requires_grad = True


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if data_args.data_augmentation:
        data_args.transform_train = v2.Compose(
            [
                v2.ToImage(),
                v2.ColorJitter(brightness=0.2, saturation=0.2),
                v2.RandomPosterize(bits=4),
                v2.RandomAdjustSharpness(sharpness_factor=1.5),
                v2.RandomAutocontrast(),
                v2.ToPILImage(),
                v2.Resize((data_args.resize_h, data_args.resize_w)),
            ]
        )
    else:
        data_args.transform_train = v2.Resize((data_args.resize_h, data_args.resize_w))

    # 检查是否使用LoRA
    use_lora = getattr(model_args, 'use_lora', False)

    if 'internvla-n1-system2' in model_args.model_name_or_path.lower():
        model = InternVLAN1ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "internvla-n1"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen3vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if data_args.model_type == "internvla-n1":
        model.get_model().initialize_vision_modules(model_args=model_args)
    
    # Compressor模式
    use_compressor = getattr(model_args, 'use_compressor', False)
    
    # 应用LoRA或全参微调
    if use_compressor and data_args.model_type == "qwen3vl":
        print("=" * 50)
        print(f"Using Compressor Stage {model_args.compressor_stage}")
        print("=" * 50)
        compressor_config = {
            'd_model': getattr(model.config, 'hidden_size', model.config.text_config.hidden_size),
            'd_bottleneck': model_args.compressor_d_bottleneck,
            'n_queries': model_args.compressor_n_queries,
            'n_heads': model_args.compressor_n_heads,
            'n_layers': model_args.compressor_n_layers,
        }
        model = apply_compressor_stage1a(model, compressor_config)
        # Pass compressor settings to data_args
        data_args.use_compressor = True
        data_args.compressor_n_queries = model_args.compressor_n_queries
    elif use_lora and data_args.model_type == "qwen3vl":
        print("=" * 50)
        print("Using LoRA for attention layers, full fine-tuning for MLP/Merger/Norms")
        print("=" * 50)
        model = apply_lora_to_qwen3vl(model, model_args)
    else:
        set_model(model_args, model)

    # 关闭视觉层的梯度检查点以节省计算和加速训练
    if hasattr(model, 'visual') and hasattr(model.visual, 'gradient_checkpointing'):
        model.visual.gradient_checkpointing = False
        if torch.distributed.get_rank() == 0:
            print("=" * 50)
            print("Vision Tower Gradient Checkpointing: DISABLED")
            print("=" * 50)
    if torch.distributed.get_rank() == 0:
        if hasattr(model, 'visual') and hasattr(model.visual, 'print_trainable_parameters'):
            model.visual.print_trainable_parameters()
        if hasattr(model, 'model') and hasattr(model.model, 'print_trainable_parameters'):
            model.model.print_trainable_parameters()

    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)  # noqa: F821
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
    from tabulate import tabulate

    if trainer.is_world_process_zero():
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            stat.append([i, n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
