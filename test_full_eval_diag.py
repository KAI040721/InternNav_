"""
Diagnostic test: reproduce the exact eval flow with compressor + LFP routing active.
Compare model output with and without compressor forward path.
"""
import sys, os
sys.path.insert(0, '/data/houdekai/InternNav_')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, AutoConfig
from peft import PeftModel

# ---- Paths ----
BASE = "/data/houdekai/models/Qwen3-VL-2B-Instruct"
CKPT = "/data/houdekai/InternNav_/checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50"
IMG_DIR = "/data/houdekai/InternNav_/data/InternNav-N1/episode_000000"

# ---- Load a real training image ----
import glob
imgs = sorted(glob.glob(os.path.join(IMG_DIR, "step_*.jpg")))
img_path = imgs[10] if len(imgs) > 10 else imgs[0]
image = Image.open(img_path).convert('RGB').resize((384, 384))
print(f"Using image: {img_path}")

# ---- Processor ----
processor = AutoProcessor.from_pretrained(BASE)
tokenizer = processor.tokenizer

# ---- Build prompt (matching eval exactly) ----
instruction = "Walk into the living room and keep walking straight past the living room."
conjunction = "you can see "
base_prompt = (
    f"You are an autonomous navigation assistant. "
    f"Your task is to {instruction} "
    f"Where should you go next to stay on track? "
    f"Please output the next waypoint's coordinates in the image. "
    f"Please output STOP when you have successfully completed the task."
    f" {conjunction}<image>."
)

# Build messages
import re
parts = re.split(r'(<image>)', base_prompt)
content = []
img_idx = 0
images_list = [image]
for part in parts:
    if part == '<image>':
        content.append({"type": "image", "image": images_list[img_idx]})
        img_idx += 1
    elif part:
        content.append({"type": "text", "text": part})

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': content},
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=images_list, return_tensors="pt")

print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Image grid thw: {inputs.get('image_grid_thw', 'N/A')}")
n_img_tokens = (inputs["input_ids"] == 151655).sum().item()
print(f"Image pad tokens: {n_img_tokens}")

# ============================================================
# Test A: LoRA only (no compressor, no LFP) - baseline
# ============================================================
print("\n" + "="*60)
print("Test A: Qwen3VL + LoRA (no compressor, no LFP)")
print("="*60)

model_a = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cuda"
)
model_a = PeftModel.from_pretrained(model_a, CKPT)
model_a = model_a.merge_and_unload()
model_a.eval()

inputs_a = {k: v.to(model_a.device) for k, v in inputs.items()}
with torch.no_grad():
    out_a = model_a.generate(
        **inputs_a, max_new_tokens=32, do_sample=False,
        use_cache=True, past_key_values=None,
        return_dict_in_generate=True,
    ).sequences
text_a = tokenizer.decode(out_a[0][inputs_a['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"Output A: '{text_a}'")

del model_a
torch.cuda.empty_cache()

# ============================================================
# Test B: LoRA + Compressor + LFP (full eval flow)
# ============================================================
print("\n" + "="*60)
print("Test B: Qwen3VL + LoRA + Compressor + LFP (full eval)")
print("="*60)

model_b = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cuda"
)
model_b = PeftModel.from_pretrained(model_b, CKPT)
model_b = model_b.merge_and_unload()

# Attach compressor
from internnav.model.compressor_wrapper_film_vit import attach_compressor_film_vit
compressor_config = {
    'n_aggr': 64,
    'n_film_layers': 24,
    'share_film': False,
}
attach_compressor_film_vit(model_b, compressor_config)
from safetensors.torch import load_file
comp_weights = load_file(os.path.join(CKPT, "compressor_film.safetensors"))
model_b._compressor.load_state_dict(comp_weights)
print(f"Compressor loaded: {len(comp_weights)} tensors")

# Attach LFP
from internnav.model.lfp_qwen3vl import attach_lfp
lfp_config = {
    'lfp_type': 'shiftedcos_decay_0.85_0.15',
    'lfp_average_factor': 0.5,
    'lfp_enable_film': True,
}
attach_lfp(model_b, lfp_config)

# Load LFP weights
from internnav.model.lfp_qwen3vl import load_lfp_router_weights
lfp_path = os.path.join(CKPT, "lfp_router.safetensors")
load_lfp_router_weights(model_b, lfp_path)
print("LFP loaded")

model_b.eval()
model_b = model_b.to("cuda")

# Set compressor attributes (matching eval code)
inputs_b = {k: v.to(model_b.device) for k, v in inputs.items()}
n_images = inputs_b["image_grid_thw"].shape[0]
is_hist = torch.zeros(n_images, dtype=torch.bool)  # no history on step 0

import math
sq = int(math.sqrt(64))  # compressor_n_queries=64, sq=8
grid_thw_rope = inputs_b["image_grid_thw"].clone()
# No history images, so no modifications to grid_thw_rope

model_b._compressor_is_history = is_hist
model_b._compressor_grid_thw_rope = grid_thw_rope

print(f"is_hist: {is_hist.tolist()}")
print(f"grid_thw_rope: {grid_thw_rope.tolist()}")

with torch.no_grad():
    out_b = model_b.generate(
        **inputs_b, max_new_tokens=32, do_sample=False,
        use_cache=True, past_key_values=None,
        return_dict_in_generate=True,
    ).sequences
text_b = tokenizer.decode(out_b[0][inputs_b['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"Output B: '{text_b}'")

# ============================================================
# Test C: Same as B but WITHOUT setting compressor attributes
# ============================================================
print("\n" + "="*60)
print("Test C: Same model as B but no compressor context set")
print("="*60)

# Clear any leftover compressor attributes
model_b._compressor_is_history = None
model_b._compressor_grid_thw_rope = None

inputs_c = {k: v.to(model_b.device) for k, v in inputs.items()}
with torch.no_grad():
    out_c = model_b.generate(
        **inputs_c, max_new_tokens=32, do_sample=False,
        use_cache=True, past_key_values=None,
        return_dict_in_generate=True,
    ).sequences
text_c = tokenizer.decode(out_c[0][inputs_c['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"Output C: '{text_c}'")

print("\n" + "="*60)
print("Summary:")
print(f"  A (LoRA only):           '{text_a}'")
print(f"  B (LoRA+Comp+LFP):      '{text_b}'")
print(f"  C (same, no comp ctx):   '{text_c}'")
print("="*60)
