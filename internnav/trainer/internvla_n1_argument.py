from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    
    # LoRA相关参数
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA for attention layers"})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of LoRA target modules"}
    )
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias type: none, all, lora_only"})

    system1: Optional[str] = field(default='nextdit')
    n_query: int = field(default=4)

    # Compressor相关参数
    use_compressor: bool = field(default=False, metadata={"help": "Whether to use FiLM compressor for history frame compression"})
    compressor_type: str = field(default="film_vit", metadata={"help": "Compressor type: 'bottleneck' (old cross-attention) or 'film_vit' (ViT-internal FiLM + aggr tokens)"})
    compressor_d_bottleneck: int = field(default=512, metadata={"help": "[bottleneck] Compressor bottleneck dimension"})
    compressor_n_queries: int = field(default=64, metadata={"help": "Number of aggregation tokens per history image (CogVLA uses 64)"})
    compressor_n_heads: int = field(default=8, metadata={"help": "[bottleneck] Number of attention heads in compressor"})
    compressor_n_layers: int = field(default=2, metadata={"help": "[bottleneck] Number of cross-attention layers in compressor"})
    # film_vit specific
    compressor_n_film_layers: int = field(default=24, metadata={"help": "[film_vit] Number of ViT blocks to apply FiLM (24=all blocks, same as CogVLA)"})
    compressor_share_film: bool = field(default=False, metadata={"help": "[film_vit] Share FiLM weights across blocks (saves params)"})

    # LFP (Latent Future Prediction) — LLM-side visual token routing (CogVLA-inspired)
    use_lfp: bool = field(default=False, metadata={"help": "Enable LFP token routing in LLM decoder layers"})
    lfp_type: str = field(default="shiftedcos_decay_0.85_0.15", metadata={"help": "LFP layer selection/decay type"})
    lfp_average_factor: float = field(default=0.5, metadata={"help": "LFP base compression ratio (0.5 = keep 50% visual tokens)"})
    lfp_enable_film: bool = field(default=True, metadata={"help": "Use FiLM-conditioned router (same as CogVLA, text modulates vision token routing)"})


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)

    vln_dataset_use: str = field(default="")
    iion_dataset_use: str = field(default="")
    sample_step: int = field(default=4)
    num_history: Optional[int] = field(default=8)
    predict_step_num: Optional[int] = field(default=32)
    pixel_goal_only: Optional[bool] = field(default=False)
    data_augmentation: Optional[bool] = field(default=False)
    transform_train: Optional[str] = field(default=None)
    resize_h: Optional[int] = field(default=384)
    resize_w: Optional[int] = field(default=384)
    num_future_steps: Optional[int] = field(default=4)
    max_dialog_turns: Optional[int] = field(default=6)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
