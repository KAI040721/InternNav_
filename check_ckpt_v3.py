"""
Analyze v3 checkpoint-1500 to verify the bugs are fixed.
Key checks:
  1. FiLM MLP weights: must be non-zero (Bug 1 fixed)
  2. LFP router: must have learned values (Bug 7 fixed → better learning)
  3. Layer 27 router: must NOT exist (Bug 2 fixed)
  4. Aggr tokens: must have moved from init (std should grow)
  5. Loss curve in log: check convergence
"""
import torch, os, glob, json, math
from safetensors.torch import load_file

ckpt_dir = "/data/houdekai/InternNav_/checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50-v3"
ckpt_path = os.path.join(ckpt_dir, "checkpoint-1500")

# ---- 1. Check compressor weights ----
print("=" * 60)
print("Compressor weights (film.safetensors)")
print("=" * 60)

comp_path = os.path.join(ckpt_dir, "compressor_film.safetensors")
if os.path.exists(comp_path):
    comp = load_file(comp_path)
    # aggr_tokens
    if "aggr_tokens" in comp:
        t = comp["aggr_tokens"]
        print(f"aggr_tokens:     std={t.std():.6f}, mean={t.mean():.6f}, absmax={t.abs().max():.6f}")
        print(f"  (init was std~0.020; if changed → compressor learning)")
    # FiLM scale/shift
    for li in [0, 10, 23]:
        for m in ["scale", "shift"]:
            k = f"film_layers.{li}.{m}.weight"
            if k in comp:
                v = comp[k]
                print(f"film_layers[{li}].{m}.weight: std={v.std():.6f}, absmax={v.abs().max():.6f}")
    # aggr_proj
    if "aggr_proj.1.weight" in comp:
        v = comp["aggr_proj.1.weight"]
        print(f"aggr_proj.1.weight: std={v.std():.6f}, absmax={v.abs().max():.6f}")
else:
    print(f"  Not found at {comp_path}, checking DeepSpeed shards...")

# ---- 2. Check LFP router weights ----
print()
print("=" * 60)
print("LFP Router weights (lfp_router.safetensors)")
print("=" * 60)

lfp_path = os.path.join(ckpt_dir, "lfp_router.safetensors")
if os.path.exists(lfp_path):
    lfp = load_file(lfp_path)
    # Check layer 27 should NOT exist
    layer27_keys = [k for k in lfp if "layers.27" in k and "router" in k]
    print(f"Layer 27 router keys: {layer27_keys}")
    if not layer27_keys:
        print("  ✓ Bug 2 confirmed: Layer 27 NOT in LFP targets")
    else:
        print("  ✗ Bug 2 NOT FIXED: Layer 27 router still present!")

    # Check a few layers
    layer_stats = {}
    for k, v in lfp.items():
        # extract layer index
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i+1 < len(parts):
                try:
                    li = int(parts[i+1])
                    if li not in layer_stats:
                        layer_stats[li] = {"absmax": 0}
                    layer_stats[li]["absmax"] = max(layer_stats[li]["absmax"], v.abs().max().item())
                except:
                    pass

    print(f"\nRouter absmax per layer (higher = more learned):")
    for li in sorted(layer_stats.keys()):
        bar = "█" * min(int(layer_stats[li]["absmax"] * 100), 40)
        print(f"  L{li:2d}: absmax={layer_stats[li]['absmax']:.6f}  {bar}")
else:
    print(f"  Not found at {lfp_path}")
    # Try to load from DeepSpeed checkpoint
    ds_ckpt = os.path.join(ckpt_path, "global_step1500")
    if os.path.exists(ds_ckpt):
        print(f"  Found DeepSpeed checkpoint, using partial analysis")

# ---- 3. Check training curve from log ----
print()
print("=" * 60)
print("Training curve (last 20 + first 10 steps from log)")
print("=" * 60)
log_path = "/data/houdekai/InternNav_/logs/train_v3_20260402_164947.log"
import json, re
losses = []
with open(log_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("{'loss':"):
            try:
                d = eval(line)
                losses.append((d.get("loss"), d.get("grad_norm"), d.get("epoch", 0)))
            except:
                pass

print(f"Total steps logged: {len(losses)}")
print(f"\nFirst 10 steps:")
for loss, gn, ep in losses[:10]:
    print(f"  loss={loss:.4f}  grad_norm={gn:.3f}  epoch={ep:.3f}")
print(f"\nLast 10 steps:")
for loss, gn, ep in losses[-10:]:
    print(f"  loss={loss:.4f}  grad_norm={gn:.3f}  epoch={ep:.3f}")
if losses:
    first_loss = losses[0][0]
    last_loss = losses[-1][0]
    print(f"\nLoss: {first_loss:.4f} → {last_loss:.4f} ({(first_loss-last_loss)/first_loss*100:.1f}% reduction)")
    max_gn = max(g for _, g, _ in losses)
    avg_gn = sum(g for _, g, _ in losses) / len(losses)
    print(f"Grad norm: max={max_gn:.3f}, avg={avg_gn:.3f}")
