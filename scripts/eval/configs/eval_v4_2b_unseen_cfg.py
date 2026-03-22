"""
评估配置文件 - Qwen3-2B LoRA v4 (Task A baseline, 无 Compressor)
在 GPU 4 上运行
"""
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "system2",  # inference mode: dual_system or system2
            "model_path": "checkpoints/InternVLA-N1-System2-Qwen3-2B-VLN-LoRA-v4",  # LoRA adapter路径
            "base_model_path": "/data/houdekai/models/Qwen3-VL-2B-Instruct",  # 基座模型路径
            "use_lora": True,  # 使用LoRA adapter
            "num_history": 8,  # 与训练时一致 (train_v4_2b 使用 num_history=8)
            "resize_w": 384,  # image resize width
            "resize_h": 384,  # image resize height
            "max_new_tokens": 1024,  # maximum number of tokens for generation
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/eval_v4_2b_unseen",  # output directory for logs/results
        "save_video": False,  # whether to save videos
        "epoch": 0,  # epoch number for logging
        "max_steps_per_episode": 500,  # maximum steps per episode
        # distributed settings
        "port": "2341",  # 使用不同端口避免冲突
        "dist_url": "env://",  # url for distributed setup
    },
)
