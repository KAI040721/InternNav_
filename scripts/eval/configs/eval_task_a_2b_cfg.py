"""
评估配置文件 - Task A: Qwen3-2B LoRA v4 baseline (无 Compressor)
GPU 6 上运行
"""
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "system2",
            "model_path": "checkpoints/InternVLA-N1-System2-Qwen3-2B-VLN-LoRA-v4",
            "base_model_path": "/data/houdekai/models/Qwen3-VL-2B-Instruct",
            "use_lora": True,
            "use_compressor": False,
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
        "output_path": "./logs/habitat/eval_task_a_2b",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2340",
        "dist_url": "env://",
    },
)
