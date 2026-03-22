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
from internnav.trainer.internvla_n1_argument import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)

# LoRA相关导入
from peft import LoraConfig, get_peft_model, TaskType

# Compressor相关导入
from internnav.model.compressor_wrapper import attach_compressor


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
    对Qwen3VL模型应用Pure LoRA微调策略:
    - 冻结所有基础参数，只训练 LoRA adapters
    - tune_mm_vision=True 时 ViT 也加 LoRA (qkv, proj, fc1, fc2)
    - Merger/Projector 通过 fc1, fc2 的 LoRA 间接微调
    - Norms: 全部冻结
    - 可训练参数: ~39.7M (1.83%)
    """
    # 先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 从参数获取target_modules配置
    lora_target_modules_str = getattr(model_args, 'lora_target_modules', 
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    
    # 解析target_modules
    if lora_target_modules_str == "all-linear":
        # all-linear: 只包含LLM的线性层
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    else:
        target_modules = [m.strip() for m in lora_target_modules_str.split(",")]
    
    # 如果需要训练视觉塔，添加视觉层模块（含 fc1, fc2 用于覆盖 Merger）
    if getattr(model_args, 'tune_mm_vision', False):
        vision_modules = ["qkv", "proj", "fc1", "fc2"]
        target_modules = list(set(target_modules + vision_modules))
    
    # 获取bias配置
    lora_bias = getattr(model_args, 'lora_bias', 'none')
    
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=model_args.lora_dropout,
        bias=lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 应用LoRA（纯 LoRA，不解冻任何基础参数）
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    # 打印详细的参数训练状态
    if torch.distributed.get_rank() == 0:
        print("")
        print("=" * 80)
        print("LoRA Configuration Summary (Pure LoRA):")
        print("=" * 80)
        print(f"LoRA Rank: {model_args.lora_r}")
        print(f"LoRA Alpha: {model_args.lora_alpha}")
        print(f"LoRA Dropout: {model_args.lora_dropout}")
        print(f"LoRA Bias: {lora_bias}")
        print(f"Target Modules: {target_modules}")
        print("")
        print("Training Status:")
        print(f"  - Vision Tower LoRA: {'Enabled' if getattr(model_args, 'tune_mm_vision', False) else 'FROZEN'}")
        print("  - LLM Attention: LoRA (q/k/v/o_proj)")
        print("  - LLM MLP: LoRA (gate/up/down_proj)")
        print("  - All Norms: FROZEN (LoRA only)")
        print("  - Merger/Projector: LoRA (fc1, fc2)")
        print("=" * 80)
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
    
    # Compressor: 先 attach (monkey-patch forward)，再 apply LoRA
    use_compressor = getattr(model_args, 'use_compressor', False)
    if use_compressor and data_args.model_type == "qwen3vl":
        compressor_config = {
            'd_model': 2048,  # Qwen3-VL-2B LLM hidden_size
            'd_bottleneck': model_args.compressor_d_bottleneck,
            'n_queries': model_args.compressor_n_queries,
            'n_heads': model_args.compressor_n_heads,
            'n_layers': model_args.compressor_n_layers,
        }
        model = attach_compressor(model, compressor_config)
        if torch.distributed.get_rank() == 0:
            comp_params = sum(p.numel() for p in model.compressor.parameters())
            print("=" * 50)
            print(f"Compressor attached: {comp_params/1e6:.2f}M params (randomly initialized)")
            print(f"  d_bottleneck={model_args.compressor_d_bottleneck}, n_queries={model_args.compressor_n_queries}")
            print("=" * 50)
        # Pass compressor settings to data_args for dataset
        data_args.use_compressor = True
        data_args.compressor_n_queries = model_args.compressor_n_queries

    # 应用LoRA或全参微调
    if use_lora and data_args.model_type == "qwen3vl":
        print("=" * 50)
        print("Using Pure LoRA - All base layers frozen, only LoRA adapters trained")
        if use_compressor:
            print("+ Compressor (FiLM) jointly trained")
        print("=" * 50)
        model = apply_lora_to_qwen3vl(model, model_args)
        # LoRA冻结了所有base参数，需要重新unfreeze compressor
        if use_compressor:
            for name, param in model.named_parameters():
                if "compressor" in name:
                    param.requires_grad = True
            if torch.distributed.get_rank() == 0:
                comp_trainable = sum(p.numel() for n, p in model.named_parameters() if "compressor" in n and p.requires_grad)
                print(f"[Compressor] Re-unfrozen after LoRA: {comp_trainable/1e6:.2f}M trainable")
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

    # 保存 Compressor 权重 (PEFT save_pretrained 不包含它们)
    if getattr(model_args, 'use_compressor', False):
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            base_model = trainer.model
            # 穿透 DeepSpeed / PEFT wrappers
            if hasattr(base_model, 'module'):
                base_model = base_model.module
            if hasattr(base_model, 'base_model') and hasattr(base_model.base_model, 'model'):
                base_model = base_model.base_model.model
            if hasattr(base_model, 'compressor'):
                from safetensors.torch import save_file
                comp_state = {k: v.cpu() for k, v in base_model.compressor.state_dict().items()}
                save_path = os.path.join(training_args.output_dir, 'compressor_film.safetensors')
                save_file(comp_state, save_path)
                print(f"[Compressor] Saved {len(comp_state)} tensors to {save_path}")
            else:
                print("[Compressor] WARNING: model has no compressor attribute, skipping save")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
