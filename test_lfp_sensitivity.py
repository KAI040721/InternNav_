"""
快速诊断：测试模型是否对视觉输入/指令文本存在响应
使用与 evaluator 完全相同的 content-list 格式（正确的 Qwen3VL 多模态格式）
"""
import sys, os, re, random, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, '/data/houdekai/InternNav_')

from transformers import AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None
from peft import PeftModel
from safetensors.torch import load_file as safe_load_file
from internnav.model.utils.vln_utils import split_and_clean

BASE_MODEL      = '/data/houdekai/models/Qwen3-VL-2B-Instruct'
CKPT_DIR        = '/data/houdekai/InternNav_/checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50-v3'
LORA_PATH       = CKPT_DIR
COMPRESSOR_CKPT = CKPT_DIR + '/compressor_film.safetensors'
LFP_CKPT        = CKPT_DIR + '/lfp_router.safetensors'
DEVICE          = torch.device('cuda:0')

print("=== 加载 processor ===")
processor = AutoProcessor.from_pretrained(BASE_MODEL)
processor.tokenizer.padding_side = 'left'

print("=== 加载基座模型 ===")
base_config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
model_type = getattr(base_config, 'model_type', '')
if Qwen3VLForConditionalGeneration and 'qwen3' in model_type.lower():
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map={"": DEVICE})
else:
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map={"": DEVICE})

print("=== 合并 LoRA ===")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.merge_and_unload()

print("=== 加载 Compressor ===")
from internnav.model.compressor_wrapper_film_vit import attach_compressor_film_vit
model = attach_compressor_film_vit(model, {'n_aggr': 64, 'n_film_layers': 24, 'share_film': False})
compressor_state = safe_load_file(COMPRESSOR_CKPT)
model.compressor.load_state_dict(compressor_state)

print("=== 加载 LFP Router ===")
from internnav.model.lfp_qwen3vl import attach_lfp
lfp_state = safe_load_file(LFP_CKPT)
lfp_layer_pattern = re.compile(r"^model\.language_model\.layers\.(\d+)\.router\.")
lfp_layers_from_ckpt = sorted({int(m.group(1)) for k in lfp_state for m in [lfp_layer_pattern.match(k)] if m})
lfp_config = {
    'lfp_type': 'shiftedcos_decay_0.85_0.15',
    'lfp_average_factor': 0.5,
    'lfp_enable_film': True,
    'lfp_target_layers_override': lfp_layers_from_ckpt,
}
model = attach_lfp(model, lfp_config)
model_params = dict(model.named_parameters())
for k, v in lfp_state.items():
    if k in model_params:
        model_params[k].data.copy_(v.to(model_params[k].dtype))

model.eval()
print("=== 模型加载完毕 ===\n")


# ── 使用与 evaluator 完全相同的 content-list 格式 ─────────────────────────────
def make_content(instruction, pil_images):
    """
    构建 Qwen3VL 多模态 content list
    pil_images[-1] = 当前帧，pil_images[:-1] = 历史帧（如有）
    """
    n_hist = len(pil_images) - 1
    base = (
        f"You are an autonomous navigation assistant. "
        f"Your task is to {instruction}. "
        f"Where should you go next to stay on track? "
        f"Please output the next waypoint's coordinates in the image. "
        f"Please output STOP when you have successfully completed the task."
    )
    if n_hist > 0:
        placeholder = ("<image>\n") * n_hist
        base += f" These are your historical observations: {placeholder}."
    base += " Now<image>."

    parts = split_and_clean(base)
    content = []
    img_idx = 0
    for part in parts:
        if part == "<image>":
            content.append({"type": "image", "image": pil_images[img_idx]})
            img_idx += 1
        else:
            content.append({"type": "text", "text": part})
    return content


def run_inference(instruction, image_arrays):
    pil_images = [Image.fromarray(img.astype(np.uint8)).convert('RGB').resize((384, 384), Image.BILINEAR)
                  for img in image_arrays]
    content = make_content(instruction, pil_images)
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=pil_images, return_tensors='pt', padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != 'token_type_ids'}
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=64, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=True, past_key_values=None,
            return_dict_in_generate=True,
        ).sequences
    n_in = inputs['input_ids'].shape[1]
    return processor.tokenizer.decode(gen_ids[0][n_in:], skip_special_tokens=True)


def get_first_token_probs(instruction, image_array):
    pil_img = Image.fromarray(image_array.astype(np.uint8)).convert('RGB').resize((384, 384), Image.BILINEAR)
    content = make_content(instruction, [pil_img])
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_img], return_tensors='pt', padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != 'token_type_ids'}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    return F.softmax(logits.float(), dim=-1)


# ── 测试图像 ───────────────────────────────────────────────────────────────────
np.random.seed(42)
black_img   = np.zeros((480, 640, 3), dtype=np.uint8)
white_img   = np.ones((480, 640, 3), dtype=np.uint8) * 255
random_img1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
random_img2 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
corridor    = np.zeros((480, 640, 3), dtype=np.uint8); corridor[:, 220:420] = 180

instr_fwd   = "Walk straight ahead and enter the room directly in front of you."
instr_left  = "Turn left and walk down the hallway to the far end."
instr_right = "Turn right toward the kitchen and stop at the counter."
instr_stop  = "Stop here, you have reached the destination."
instr_orig  = "Walk into the living room and keep walking straight past the living room."

print("=" * 65)
print("测试A: 相同图像(黑色) + 不同指令")
print("  若输出全相同 → 模型忽略语言输入 = mode collapse")
print("=" * 65)
results_a = {}
for name, instr in [("直行", instr_fwd), ("左转", instr_left), ("右转", instr_right),
                     ("停止", instr_stop), ("原始", instr_orig)]:
    out = run_inference(instr, [black_img])
    results_a[name] = out
    print(f"  [{name}] => '{out}'")
lang_invariant = len(set(results_a.values())) == 1
print(f"\n  结论: {'所有输出相同，忽略语言输入' if lang_invariant else '不同指令有不同输出，语言有效'}")

print()
print("=" * 65)
print("测试B: 相同指令 + 不同图像")
print("  若输出全相同 → 模型忽略视觉输入 = mode collapse")
print("=" * 65)
results_b = {}
for name, img in [("黑色", black_img), ("白色", white_img),
                   ("随机1", random_img1), ("随机2", random_img2), ("走廊", corridor)]:
    out = run_inference(instr_orig, [img])
    results_b[name] = out
    print(f"  [{name}] => '{out}'")
vis_invariant = len(set(results_b.values())) == 1
print(f"\n  结论: {'所有输出相同，忽略视觉输入' if vis_invariant else '不同图像有不同输出，视觉有效'}")

print()
print("=" * 65)
print("测试C: 第一个生成token的概率分布")
print("=" * 65)
probs = get_first_token_probs(instr_orig, black_img)
topk = torch.topk(probs, 15)
print("Top-15 tokens (黑色图 + 原始指令):")
for prob, idx in zip(topk.values, topk.indices):
    tok = processor.tokenizer.decode([idx.item()])
    print(f"  id={idx.item()} '{tok}': {prob.item()*100:.3f}%")
print()
for tok in ['↑', '←', '→', '↓', 'STOP', ' STOP']:
    ids = processor.tokenizer.encode(tok, add_special_tokens=False)
    if ids:
        p = probs[ids[0]].item()
        print(f"  动作 '{tok}' (id={ids[0]}): {p*100:.4f}%")

print("\n=== 诊断完成 ===")
