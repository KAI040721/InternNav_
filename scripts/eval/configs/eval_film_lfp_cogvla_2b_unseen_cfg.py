"""
评估配置文件 - FiLM-ViT + LFP CogVLA 2B 模型
在 R2R val_unseen 上评估
LoRA + Compressor (FiLM-ViT, n_aggr=64) + LFP Router
"""
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "system2",
            "model_path": "checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50",
            "base_model_path": "/data/houdekai/models/Qwen3-VL-2B-Instruct",
            "use_lora": True,
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
            # Compressor 配置 (FiLM-ViT, n_aggr=64)
            "use_compressor": True,
            "compressor_type": "film_vit",
            "compressor_checkpoint": "checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50/compressor_film.safetensors",
            "compressor_n_queries": 64,
            "compressor_n_film_layers": 24,
            "compressor_share_film": False,
            # LFP Router 配置
            "use_lfp": True,
            "lfp_checkpoint": "checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50/lfp_router.safetensors",
            "lfp_type": "shiftedcos_decay_0.85_0.15",
            "lfp_average_factor": 0.5,
            "lfp_enable_film": True,
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_r2r_unseen.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/eval_film_lfp_cogvla_2b_unseen",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2345",
        "dist_url": "env://",
    },
)
