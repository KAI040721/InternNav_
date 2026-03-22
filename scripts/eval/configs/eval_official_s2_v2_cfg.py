"""
评估配置文件 - 官方 InternVLA-N1-System2 模型（第二次评估，结果保存到独立目录）
在 GPU 5 上运行
"""
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "system2",
            "model_path": "checkpoints/InternVLA-N1-System2",
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
        "output_path": "./logs/habitat/eval_official_s2_v2",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2342",
        "dist_url": "env://",
    },
)
