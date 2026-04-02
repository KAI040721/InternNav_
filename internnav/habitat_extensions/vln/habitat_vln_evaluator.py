import argparse
import json
import os
import sys
from enum import IntEnum

sys.path.append('./src/diffusion-policy')
import copy
import itertools
import random
import re
from collections import OrderedDict

import cv2
import habitat
import numpy as np
import math

def _shrink_history_image_tokens(inputs, is_hist, n_queries, image_token_id=151655):
    """
    将 input_ids 中历史帧对应的 <image_pad> 占位符从原始数量缩减到 n_queries 个。
    训练时 dataset 在构建 input_ids 前就把历史帧的 grid_thw 替换成 n_queries，
    评估时 processor 按原始尺寸生成 input_ids，需要手动对齐。

    Args:
        inputs: processor 返回的 BatchFeature（包含 input_ids / attention_mask）
        is_hist: BoolTensor [n_images]，True 表示该帧是历史帧
        n_queries: 压缩后每帧的 token 数（默认 16）
        image_token_id: <|image_pad|> 的 token id（Qwen3-VL 默认 151655）
    Returns:
        修改后的 inputs（input_ids / attention_mask 已就地替换）
    """
    input_ids = inputs["input_ids"][0].tolist()  # [seq_len]

    # 找出每段 image token 的起止位置（连续 image_token_id 构成一段）
    segments = []  # list of (start, end) exclusive
    i = 0
    while i < len(input_ids):
        if input_ids[i] == image_token_id:
            j = i
            while j < len(input_ids) and input_ids[j] == image_token_id:
                j += 1
            segments.append((i, j))
            i = j
        else:
            i += 1

    assert len(segments) == len(is_hist), (
        f"image segments {len(segments)} != is_hist {len(is_hist)}"
    )

    # 从后往前修改，避免索引偏移
    new_ids = list(input_ids)
    for seg_idx in reversed(range(len(segments))):
        if not is_hist[seg_idx]:
            continue
        start, end = segments[seg_idx]
        orig_len = end - start
        if orig_len <= n_queries:
            continue  # 已经 <= n_queries，无需修改
        # 保留前 n_queries 个，删掉多余的
        del new_ids[start + n_queries : end]

    new_ids_tensor = torch.tensor([new_ids], dtype=inputs["input_ids"].dtype,
                                   device=inputs["input_ids"].device)
    inputs["input_ids"] = new_ids_tensor

    # attention_mask 同步截断到新长度
    new_len = new_ids_tensor.shape[1]
    if "attention_mask" in inputs:
        attn = inputs["attention_mask"]
        if attn.shape[1] >= new_len:
            inputs["attention_mask"] = attn[:, :new_len]
        else:
            # 极少数情况：pad 到新长度
            pad = torch.ones(1, new_len - attn.shape[1],
                             dtype=attn.dtype, device=attn.device)
            inputs["attention_mask"] = torch.cat([attn, pad], dim=1)

    return inputs

import quaternion
import torch
from peft import PeftModel
import tqdm
from depth_camera_filtering import filter_depth
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_baselines.config.default import get_config as get_habitat_config
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator
from internnav.habitat_extensions.vln.utils import (
    get_axis_align_matrix,
    get_intrinsic_matrix,
    pixel_to_gps,
    preprocess_depth_image_v2,
    xyz_yaw_pitch_to_tf_matrix,
)
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions

# Import for Habitat registry side effects — do not remove
import internnav.habitat_extensions.vln.measures  # noqa: F401 # isort: skip


DEFAULT_IMAGE_TOKEN = "<image>"

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


@Evaluator.register('habitat_vln')
class HabitatVLNEvaluator(DistributedEvaluator):
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)
        self.save_video = args.save_video
        self.epoch = args.epoch
        self.max_steps_per_episode = args.max_steps_per_episode
        self.output_path = args.output_path
        self.num_episodes = getattr(args, "num_episodes", None)

        # create habitat config
        self.config_path = cfg.env.env_settings['config_path']
        self.config = get_habitat_config(self.config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
        cfg.env.env_settings['habitat_config'] = self.config
        cfg.env.env_settings['output_path'] = self.output_path

        # init agent and env
        super().__init__(cfg, init_agent=False)

        # ------------------------------------- model ------------------------------------------
        self.model_args = argparse.Namespace(**cfg.agent.model_settings)

        # 对于LoRA模型，从基座模型路径加载processor
        use_lora = getattr(self.model_args, 'use_lora', False)
        processor_path = getattr(self.model_args, 'base_model_path', self.model_args.model_path) if use_lora else self.model_args.model_path
        processor = AutoProcessor.from_pretrained(processor_path)
        # 兼容Qwen3 (AutoProcessor直接返回tokenizer) 和 Qwen2.5_VL (有.tokenizer属性)
        if hasattr(processor, 'tokenizer'):
            processor.tokenizer.padding_side = 'left'
        else:
            processor.padding_side = 'left'

        device = torch.device(f"cuda:{self.local_rank}")
        if self.model_args.mode == 'dual_system':
            model = InternVLAN1ForCausalLM.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
        elif self.model_args.mode == 'system2':
            # 支持LoRA模型加载
            use_lora = getattr(self.model_args, 'use_lora', False)
            if use_lora:
                base_model_path = getattr(self.model_args, 'base_model_path', None)
                if base_model_path is None:
                    raise ValueError("use_lora=True requires base_model_path to be set")
                print(f"Loading LoRA model: base={base_model_path}, adapter={self.model_args.model_path}")
                # 判断基座模型类型 (Qwen3-VL vs Qwen2.5-VL)
                from transformers import AutoConfig
                base_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
                model_type = getattr(base_config, 'model_type', '')
                print(f"Base model type: {model_type}")
                if 'qwen3' in model_type.lower():
                    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                        base_model_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        device_map={"": device},
                    )
                else:
                    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        base_model_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        device_map={"": device},
                    )
                # 加载LoRA adapter
                model = PeftModel.from_pretrained(base_model, self.model_args.model_path)
                model = model.merge_and_unload()  # 合并LoRA权重以获得更好的推理性能
                print("LoRA weights merged successfully")

                # ---- Compressor 加载 ----
                use_compressor = getattr(self.model_args, 'use_compressor', False)
                if use_compressor:
                    from safetensors.torch import load_file as safe_load_file
                    compressor_type = getattr(self.model_args, 'compressor_type', 'film_vit')

                    if compressor_type == 'film_vit':
                        from internnav.model.compressor_wrapper_film_vit import attach_compressor_film_vit
                        compressor_config = {
                            'n_aggr': getattr(self.model_args, 'compressor_n_queries', 16),
                            'n_film_layers': getattr(self.model_args, 'compressor_n_film_layers', 24),
                            'share_film': getattr(self.model_args, 'compressor_share_film', False),
                        }
                        model = attach_compressor_film_vit(model, compressor_config)
                    else:
                        from internnav.model.compressor_wrapper import attach_compressor
                        compressor_config = {
                            'd_model': getattr(self.model_args, 'compressor_d_model', 2048),
                            'd_bottleneck': getattr(self.model_args, 'compressor_d_bottleneck', 512),
                            'n_queries': getattr(self.model_args, 'compressor_n_queries', 16),
                            'n_heads': getattr(self.model_args, 'compressor_n_heads', 8),
                            'n_layers': getattr(self.model_args, 'compressor_n_layers', 2),
                        }
                        model = attach_compressor(model, compressor_config)

                    ckpt_path = self.model_args.compressor_checkpoint
                    state_dict = safe_load_file(ckpt_path)
                    model.compressor.load_state_dict(state_dict)
                    print(f"Compressor [{compressor_type}] weights loaded from {ckpt_path}")

                # ---- LFP Router 加载 ----
                use_lfp = getattr(self.model_args, 'use_lfp', False)
                if use_lfp:
                    from safetensors.torch import load_file as safe_load_file
                    from internnav.model.lfp_qwen3vl import attach_lfp

                    lfp_ckpt_path = self.model_args.lfp_checkpoint
                    lfp_state = safe_load_file(lfp_ckpt_path)

                    # 从 checkpoint 自动推断 router 覆盖的 decoder 层，保证评估结构与训练完全一致。
                    lfp_layer_pattern = re.compile(r"^model\.language_model\.layers\.(\d+)\.router\.")
                    lfp_layers_from_ckpt = sorted(
                        {
                            int(m.group(1))
                            for k in lfp_state.keys()
                            for m in [lfp_layer_pattern.match(k)]
                            if m is not None
                        }
                    )

                    lfp_config = {
                        'lfp_type': getattr(self.model_args, 'lfp_type', 'shiftedcos_decay_0.85_0.15'),
                        'lfp_average_factor': getattr(self.model_args, 'lfp_average_factor', 0.5),
                        'lfp_enable_film': getattr(self.model_args, 'lfp_enable_film', True),
                    }
                    if lfp_layers_from_ckpt:
                        lfp_config['lfp_target_layers_override'] = lfp_layers_from_ckpt
                        print(f"[LFP] Using checkpoint-defined target layers: {lfp_layers_from_ckpt}")

                    model = attach_lfp(model, lfp_config)

                    # 将保存的 router 权重加载到模型对应参数中
                    model_params = dict(model.named_parameters())
                    loaded_count = 0
                    for key, value in lfp_state.items():
                        if key in model_params:
                            model_params[key].data.copy_(value.to(model_params[key].dtype))
                            loaded_count += 1
                        else:
                            print(f"[LFP] WARNING: key {key} not found in model")
                    print(f"LFP Router weights loaded: {loaded_count}/{len(lfp_state)} tensors from {lfp_ckpt_path}")
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_args.model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map={"": device},
                )
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        model.eval()
        self.device = device

        # 保存 compressor 状态供推理时使用
        self.use_compressor = getattr(self.model_args, 'use_compressor', False)
        self.compressor_n_queries = getattr(self.model_args, 'compressor_n_queries', 16)

        self.model = model
        self.processor = processor
        # 统一的tokenizer引用，兼容Qwen3和Qwen2.5_VL
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        # refactor: this part used in three places
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]
        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.num_history = self.model_args.num_history

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

    def _build_content_list_from_text(self, text_with_placeholders, images):
        """
        Convert a text string containing <image> placeholders into a
        content list for Qwen3-VL processor.apply_chat_template().

        Preserves whitespace (spaces, \n) around <image> tokens exactly
        as they appear in the input string, matching the training tokenization.
        """
        import re
        parts = re.split(r'(<image>)', text_with_placeholders)
        content = []
        img_idx = 0
        for part in parts:
            if part == '<image>':
                content.append({"type": "image", "image": images[img_idx]})
                img_idx += 1
            elif part:  # keep non-empty text parts WITH their whitespace
                content.append({"type": "text", "text": part})
        return content

    def eval_action(self):
        """
        Run local episodes on this rank.

        Returns dict[str, Tensor] on GPU (1D tensors of same length).
        """
        # Old behavior was something like:
        # sucs, spls, oss, nes, ep_num = self.eval_action(self.rank)
        # Now just implement the actual eval here and return dict.

        if self.model_args.mode == 'dual_system':
            sucs, spls, oss, nes, ndtws = self._run_eval_dual_system()
        elif self.model_args.mode == 'system2':
            sucs, spls, oss, nes, ndtws = self._run_eval_system2()
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        result = {
            "sucs": sucs,  # shape [N_local]
            "spls": spls,  # shape [N_local]
            "oss": oss,  # shape [N_local]
            "nes": nes,  # shape [N_local]
        }

        if ndtws is not None:
            result["ndtws"] = ndtws  # shape [N_local]
        return result

    def calc_metrics(self, global_metrics: dict) -> dict:
        """
        global_metrics["sucs"] etc. are global 1-D CPU tensors with all episodes.
        """
        sucs_all = global_metrics["sucs"]
        spls_all = global_metrics["spls"]
        oss_all = global_metrics["oss"]
        nes_all = global_metrics["nes"]

        # avoid /0 if no episodes
        denom = max(len(sucs_all), 1)

        # clean NaN in spls, treat as 0.0
        torch.nan_to_num(spls_all, nan=0.0, posinf=0.0, neginf=0.0, out=spls_all)

        # clean inf in nes, only fiinite nes are counted
        nes_finite_mask = torch.isfinite(nes_all)
        nes_all = nes_all[nes_finite_mask]

        result_all = {
            "sucs_all": float(sucs_all.mean().item()) if denom > 0 else 0.0,
            "spls_all": float(spls_all.mean().item()) if denom > 0 else 0.0,
            "oss_all": float(oss_all.mean().item()) if denom > 0 else 0.0,
            "nes_all": float(nes_all.mean().item()) if denom > 0 else 0.0,
            # "length" will be filled by base class
        }

        if "ndtws" in global_metrics:
            ndtws_all = global_metrics["ndtws"]
            result_all["ndtws_all"] = float(ndtws_all.mean().item()) if denom > 0 else 0.0

        return result_all

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def resume_from_output_path(self) -> None:
        sucs, spls, oss, nes, ndtw = [], [], [], [], []
        if self.rank != 0:
            return sucs, spls, oss, nes, ndtw

        # resume from previous results
        if os.path.exists(os.path.join(self.output_path, 'progress.json')):
            with open(os.path.join(self.output_path, 'progress.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    sucs.append(res['success'])
                    spls.append(res['spl'])
                    oss.append(res['os'])
                    nes.append(res['ne'])
                    if 'ndtw' in res:
                        ndtw.append(res['ndtw'])
        return sucs, spls, oss, nes, ndtw

    def _run_eval_dual_system(self) -> tuple:
        self.model.eval()

        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        # Limit episodes if num_episodes is specified
        if self.num_episodes is not None:
            self.env.episodes = self.env.episodes[:self.num_episodes]

        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            print("episode start", episode_instruction)

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            action = None
            messages = []
            local_actions = []

            done = False
            flag = False
            pixel_goal = None

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                    down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)
                    down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)

                    look_down_image = Image.fromarray(down_observations["rgb"]).convert('RGB')
                    depth = down_observations["depth"]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0

                    self.env.step(action_code.LOOKUP)
                    self.env.step(action_code.LOOKUP)

                if len(action_seq) == 0 and pixel_goal is None:
                    if action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
                        input_img_id = -1
                    else:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True,
                            past_key_values=None,
                            return_dict_in_generate=True,
                        ).sequences

                    llm_outputs = self.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]

                        pixel_goal = [int(coord[1]), int(coord[0])]

                        # look down --> horizontal
                        self.env.step(action_code.LOOKUP)
                        self.env.step(action_code.LOOKUP)

                        local_actions = []
                        pixel_values = inputs.pixel_values
                        image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                        with torch.no_grad():
                            traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                        # prepocess align with navdp
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                        pix_goal_image = copy.copy(image_dp)
                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                        pix_goal_depth = copy.copy(depth_dp)
                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)

                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]

                        action = local_actions[0]
                        if action == action_code.STOP:
                            pixel_goal = None
                            output_ids = None
                            action = action_code.LEFT
                            observations, _, done, _ = self.env.step(action)
                            step_id += 1
                            messages = []
                            continue
                        print('predicted goal', pixel_goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif pixel_goal is not None:
                    if len(local_actions) == 0:
                        # navdp
                        local_actions = []
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255

                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)

                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)

                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]
                        print("local_actions", local_actions)
                        action = local_actions.pop(0)
                    else:
                        action = local_actions.pop(0)

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                    if action == action_code.STOP:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                else:
                    action = 0

                info = self.env.get_metrics()

                if info['top_down_map'] is not None and self.save_video:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if pixel_goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}"
            )

            # Write per-episode progress.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
            }
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']

            # save current progress
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")

            # save video
            if self.save_video and metrics['success'] == 1.0:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
        )

    def _run_eval_system2(self) -> tuple:
        self.model.eval()

        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            print("episode start", episode_instruction)

            agent_state = self.env._env.sim.get_agent_state()
            rotation = agent_state.rotation
            translation = agent_state.position
            rotation_matrix = quaternion.as_rotation_matrix(rotation)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            agent = ShortestPathFollower(self.env._env.sim, 0.25, False)

            intrinsic_matrix = get_intrinsic_matrix(
                self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
            )

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            initial_height = self.env._env.sim.get_agent_state().position[1]

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            goal = None
            action = None
            messages = []

            done = False
            flag = False

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                camera_yaw = observations["compass"][0]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                agent_state = self.env._env.sim.get_agent_state()
                height = agent_state.position[1] - initial_height  # Habitat GPS makes west negative, so flip y
                camera_position = np.array([x, -y, self._camera_height + height])
                tf_camera_to_episodic = (
                    xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30)) @ get_axis_align_matrix()
                )

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                if len(action_seq) == 0 and goal is None:
                    if action == action_code.LOOKDOWN:
                        # last action is look down — second turn of pixel-goal dialogue
                        input_images += [look_down_image]
                        # Build content list for the lookdown turn
                        # Training format: "{conjunction}<image>."
                        conj = random.choice(self.conjunctions)
                        user_content = self._build_content_list_from_text(
                            f" {conj}<image>.", [look_down_image]
                        )
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}
                        )
                        messages.append({'role': 'user', 'content': user_content})
                        input_img_id = -1
                    else:
                        # Build first-turn user message — must match training exactly
                        instruction = episode.instruction.instruction_text
                        # Training template: '...to <instruction>. Where...' -- must add '.' after instruction
                        base_prompt = (
                            f"You are an autonomous navigation assistant. "
                            f"Your task is to {instruction}. "
                            f"Where should you go next to stay on track? "
                            f"Please output the next waypoint's coordinates in the image. "
                            f"Please output STOP when you have successfully completed the task."
                        )

                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            # Training format: " These are your historical observations: <image>\n<image>\n...."
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            base_prompt += f" These are your historical observations: {placeholder}."

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                        # Training format: " {conjunction}<image>."
                        conj = random.choice(self.conjunctions)
                        base_prompt += f" {conj}<image>."

                        user_content = self._build_content_list_from_text(
                            base_prompt, input_images
                        )

                        # Fresh conversation with system message (matching training)
                        messages = [
                            {'role': 'system', 'content': 'You are a helpful assistant.'},
                        ]
                        messages.append({'role': 'user', 'content': user_content})

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    # ---- Compressor: 在调用 generate 前直接设置模型属性 ----
                    # 注意: 不能通过 generate(**kwargs) 传自定义字段，
                    # 否则 _validate_model_kwargs 会因为参数不在 forward 签名中而 raise ValueError。
                    # 正确方式: 直接写入 model._compressor_* 属性，
                    # compressor_forward 会从这里读取并在读后清除。
                    if self.use_compressor and "image_grid_thw" in inputs:
                        n_images = inputs["image_grid_thw"].shape[0]
                        if input_img_id == -1:
                            # look_down 分支: 只有1张图, 不是历史帧
                            is_hist = torch.zeros(n_images, dtype=torch.bool)
                        else:
                            # 正常分支: 前 len(history_id) 张是历史, 最后1张是当前
                            n_hist = len(history_id)
                            is_hist = torch.zeros(n_images, dtype=torch.bool)
                            is_hist[:n_hist] = True

                        # 构造 image_grid_thw_rope: 压缩后的历史帧用 [1, sq, sq]
                        # 与训练时 dataset 保持一致: sq = sqrt(n_queries) = 4
                        # get_rope_index 内部会做 h//merge_size → 4//2=2, 生成 2×2=4 个 visual pos
                        # 剩余 12 个 image_pad 按文本 token 方式编号 (训练时也是如此)
                        sq = int(math.sqrt(self.compressor_n_queries))  # 16 -> 4
                        grid_thw_rope = inputs["image_grid_thw"].clone()
                        for i in range(n_images):
                            if is_hist[i]:
                                grid_thw_rope[i] = torch.tensor([1, sq, sq], dtype=grid_thw_rope.dtype)

                        # 直接写入模型属性 (outer_forward_wrapper 从这里读取)
                        self.model._compressor_is_history = is_hist
                        self.model._compressor_grid_thw_rope = grid_thw_rope

                        # 缩减 input_ids 中历史帧的 image token 占位符数量
                        # processor 按原始尺寸生成 144 个占位符，但 compressor 只输出 n_queries 个
                        # 必须对齐，否则 get_placeholder_mask 会 raise ValueError
                        inputs = _shrink_history_image_tokens(
                            inputs, is_hist, self.compressor_n_queries
                        )

                    # DEBUG: 打印前3步的 token 数量信息
                    if step_id < 30 and self.use_compressor and "image_grid_thw" in inputs:
                        n_img_tokens = (inputs["input_ids"] == 151655).sum().item()
                        n_images_now = inputs["image_grid_thw"].shape[0]
                        print(f"[DEBUG] step_id={step_id} n_images={n_images_now} "
                              f"n_img_tokens_in_ids={n_img_tokens} "
                              f"is_hist={is_hist.tolist() if 'is_hist' in dir() else 'N/A'} "
                              f"grid_thw={inputs['image_grid_thw'].tolist()} "
                              f"input_ids_len={inputs['input_ids'].shape[1]}")

                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True,
                            past_key_values=None,
                            return_dict_in_generate=True,
                        ).sequences

                    llm_outputs = self.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]

                        pixel_goal = [int(coord[1]), int(coord[0])]

                        # look down --> horizontal
                        self.env.step(action_code.LOOKUP)
                        self.env.step(action_code.LOOKUP)

                        goal = pixel_to_gps(pixel_goal, depth / 1000, intrinsic_matrix, tf_camera_to_episodic)

                        goal = (transformation_matrix @ np.array([-goal[1], 0, -goal[0], 1]))[:3]

                        if not self.env._env.sim.pathfinder.is_navigable(np.array(goal)):
                            goal = np.array(self.env._env.sim.pathfinder.snap_point(np.array(goal)))

                        action = agent.get_next_action(goal)
                        if action == action_code.STOP:
                            goal = None
                            output_ids = None
                            action = action_code.LEFT  # random action to avoid deadlock
                            observations, _, done, _ = self.env.step(action)
                            step_id += 1
                            messages = []
                            continue
                        print('predicted goal', pixel_goal, goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif goal is not None:
                    action = agent.get_next_action(goal)
                    action = action.detach().cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
                    action = action[0] if hasattr(action, "__len__") else action

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                    if action == action_code.STOP:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                else:
                    action = 0

                info = self.env.get_metrics()

                if info['top_down_map'] is not None and self.save_video:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}"
            )

            # Write per-episode result.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
            }
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']

            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")
            if self.save_video and metrics['success'] == 1.0:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
        )
