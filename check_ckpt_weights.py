import torch
import numpy as np
import sys, re

ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/houdekai/InternNav_/checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50-v2/checkpoint-1500"

step = ckpt_dir.rstrip('/').split('-')[-1]
model_path = f"{ckpt_dir}/global_step{step}/mp_rank_00_model_states.pt"
print(f"Loading {model_path} ...")
state = torch.load(model_path, map_location='cpu', weights_only=False)

module = state.get('module', state)
all_keys = list(module.keys())
print(f"Total parameter keys: {len(all_keys)}")

# ============ Compressor weights ============
comp_keys = sorted([k for k in all_keys if 'compressor' in k])
print(f"\n{'='*60}")
print(f"COMPRESSOR PARAMETERS ({len(comp_keys)} tensors)")
print(f"{'='*60}")

for k in comp_keys:
    t = module[k].float()
    print(f"  {k}")
    print(f"    shape={list(t.shape)}, mean={t.mean():.6f}, std={t.std():.6f}, absmax={t.abs().max():.6f}, absmean={t.abs().mean():.6f}")

aggr_key = [k for k in comp_keys if 'aggr_tokens' in k]
if aggr_key:
    aggr = module[aggr_key[0]].float()
    print(f"\n  >> aggr_tokens: std={aggr.std():.6f} (init ~0.02)")

film_scale_keys = [k for k in comp_keys if 'film_layers' in k and 'scale.weight' in k]
film_shift_keys = [k for k in comp_keys if 'film_layers' in k and 'shift.weight' in k]
if film_scale_keys:
    all_scale = torch.cat([module[k].float().flatten() for k in film_scale_keys])
    all_shift = torch.cat([module[k].float().flatten() for k in film_shift_keys])
    print(f"  >> FiLM scale: mean={all_scale.mean():.8f}, std={all_scale.std():.8f}, absmax={all_scale.abs().max():.8f}")
    print(f"  >> FiLM shift: mean={all_shift.mean():.8f}, std={all_shift.std():.8f}, absmax={all_shift.abs().max():.8f}")

# ============ LFP Router weights ============
router_keys = sorted([k for k in all_keys if '.router.' in k])
print(f"\n{'='*60}")
print(f"LFP ROUTER PARAMETERS ({len(router_keys)} tensors)")
print(f"{'='*60}")

layer_stats = {}
for k in router_keys:
    t = module[k].float()
    m = re.search(r'layers\.(\d+)', k)
    if m:
        layer_idx = int(m.group(1))
        if layer_idx not in layer_stats:
            layer_stats[layer_idx] = {}
        short_name = k.split(f'layers.{layer_idx}.')[-1]
        layer_stats[layer_idx][short_name] = {
            'shape': list(t.shape), 'mean': t.mean().item(),
            'std': t.std().item(), 'absmax': t.abs().max().item(),
            'absmean': t.abs().mean().item(),
        }

for layer_idx in sorted(layer_stats.keys()):
    params = layer_stats[layer_idx]
    total_absmean = np.mean([v['absmean'] for v in params.values()])
    print(f"\n  Layer {layer_idx} ({len(params)} params, avg_absmean={total_absmean:.8f}):")
    for pname, stats in sorted(params.items()):
        print(f"    {pname}: std={stats['std']:.8f}, absmax={stats['absmax']:.8f}")

# Layer 27 detail
if 27 in layer_stats:
    print(f"\n{'='*60}")
    print(f"LAYER 27 DETAILED (ratio=0.15)")
    print(f"{'='*60}")
    for pname, stats in sorted(layer_stats[27].items()):
        print(f"  {pname}: shape={stats['shape']}, mean={stats['mean']:.8f}, std={stats['std']:.8f}, absmax={stats['absmax']:.8f}")
    for k in router_keys:
        if 'layers.27' in k and 'router.router.weight' in k:
            wt = module[k].float()
            print(f"\n  router.weight row[0] (DROP): top5={wt[0].abs().topk(5).values.tolist()}")
            print(f"  router.weight row[1] (KEEP): top5={wt[1].abs().topk(5).values.tolist()}")
            print(f"  If ≈0: 50/50 KEEP/DROP → no routing")

# Summary
print(f"\n{'='*60}")
print("DIAGNOSIS SUMMARY")
print(f"{'='*60}")
if comp_keys:
    comp_absmean = np.mean([module[k].float().abs().mean().item() for k in comp_keys])
    print(f"Compressor avg |w|: {comp_absmean:.8f}")
if router_keys:
    router_absmean = np.mean([module[k].float().abs().mean().item() for k in router_keys])
    print(f"Router avg |w|: {router_absmean:.8f}")
