"""
评估配置文件 - Compressor Stage 3b: Qwen3-2B LoRA + Compressor
GPU 7 上运行
"""
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "system2",
            "model_path": "checkpoints/Compressor-3b-Qwen3-2B-R2R-RxR",
            "base_model_path": "/data/houdekai/models/Qwen3-VL-2B-Instruct",
            "use_lora": True,
            "use_compressor": True,
            "compressor_checkpoint": "checkpoints/Compressor-3b-Qwen3-2B-R2R-RxR/compressor_stage3b.safetensors",
            "compressor_d_model": 2048,
            "compressor_d_bottleneck": 512,
            "compressor_n_queries": 16,
            "compressor_n_heads": 8,
            "compressor_n_layers": 2,
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/eval_compressor_3b_2b",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2341",
        "dist_url": "env://",
    },
)
