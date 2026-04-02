"""
评估配置文件 - FiLM Joint Training 模型
基于 eval_official_s2_cfg.py，添加 LoRA + Compressor 支持
"""
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "system2",
            "model_path": "checkpoints/FiLM-Joint-2B-AllLoRA-r32-R2XR50",
            "base_model_path": "/data/houdekai/models/Qwen3-VL-2B-Instruct",
            "use_lora": True,
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
            # Compressor 配置
            "use_compressor": True,
            "compressor_checkpoint": "checkpoints/FiLM-Joint-2B-AllLoRA-r32-R2XR50/compressor_film.safetensors",
            "compressor_d_model": 2048,
            "compressor_d_bottleneck": 512,
            "compressor_n_queries": 16,
            "compressor_n_heads": 8,
            "compressor_n_layers": 2,
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
        "output_path": "./logs/habitat/eval_film_joint_s2",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2333",
        "dist_url": "env://",
    },
)
