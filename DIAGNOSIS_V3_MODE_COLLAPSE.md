# V3 模式坍塌诊断报告

## 之前诊断（已证伪）

**错误诊断**：梯度饥饿 — 新模块（Compressor 109M + LFP 176M = 285M参数）没有学到东西。

**证伪证据**：
- 权重对比（checkpoint-3000 vs 最终）：LoRA变化2.8%、Compressor变化1.7%、LFP变化0.4% — **所有模块都在学习**
- Loss从4.45收敛到0.28 — 学习充分
- `grad_norm_base=0.0`是DeepSpeed ZeRO-2的日志问题（`param.grad`在step后为None），并非真正的零梯度

## 正确诊断：LFP Router 在当前帧进行了错误的Token Dropping

### 核心问题

LFP Router设计目的是对**历史帧**的视觉tokens进行优先级排序/裁剪，但实现中对**所有**视觉tokens（包括当前帧）进行了routing操作。

在 step_id=0（无历史帧）时，模型只有当前帧的 144 个视觉 tokens。LFP Router 在各层的 keep_ratio 为：
- Layer 7: keep 84.3% → 丢弃23个token
- Layer 15: keep 41.3% → 丢弃84个token  
- **Layer 26: keep 15.0% → 仅保留21个，丢弃123个token（85%的当前帧信息被丢弃！）**

这摧毁了 pixel_goal 预测所需的视觉定位信息，模型退化为基于文本先验的固定模式（转弯动作序列）。

### 关键证据

| 模型版本 | step_id=0输出 | LFP状态 | 成功率 |
|---------|-------------|---------|-------|
| verge3 (2B纯LoRA) | ↓ + 坐标 ✓ | 无 | ~50% |
| verge1 (8B纯LoRA) | ↓ + 坐标 ✓ | 无 | ~60% |
| V1 nolfp (compressor, 无LFP推理) | ↓ + 坐标 ✓ | 推理关闭 | 低 |
| V2 (compressor + LFP) | →→→→ ✗ | 开启 | 0% |
| **V3 (compressor + LFP)** | **←←←← ✗** | **开启** | **0%** |

**关键对比**：
- 同一V1模型，开LFP vs 关LFP：关LFP时pixel goal正常，开LFP时mode collapse
- 纯LoRA使用完全相同的训练数据（pixel_goal_only=False），表现正常
- 所有带LFP推理的版本（V2、V3）都出现mode collapse

### 模式坍塌的具体表现

V3 在所有 episode（50个）的行为完全一致：
```
step_id=0: ←←←← (无历史帧，compressor/LFP不应生效)
step_id=1-3: 执行←
step_id=4: ← (此时有4帧历史)
step_id=5: → 或 STOP
step_id=6: STOP
```
不同场景、不同指令 → 完全相同的行为序列，ne值也完全一致。

### 代码层面的bug

`compressor_wrapper_film_vit.py` 中：
```python
# 修改前（BUG）：visual_pos_masks 标记了所有视觉tokens（历史+当前）
visual_pos_masks = image_mask[..., 0]  # 包含历史和当前帧的tokens
# ... 传给LFP，LFP对所有tokens进行routing/dropping
```

`lfp_qwen3vl.py` 中 LFP Layer forward：
```python
visual_pos_mask = getattr(self._text_model_ref, '_lfp_visual_pos_mask', None)
# visual_pos_mask 包含当前帧tokens → 当前帧被routing/dropping
num_visual_tokens = visual_pos_mask[0].sum().item()  # 144 at step_id=0
visual_kept_length = int(num_visual_tokens * self.router_factor)  # Layer 26: 21
# → 丢弃 123/144 = 85% 的当前帧视觉信息！
```

## 修复方案

### 已实施的修改

**文件**: `internnav/model/compressor_wrapper_film_vit.py`

在调用 `language_model()` 前，创建一个仅标记**历史帧**视觉tokens的mask：
- 有历史帧时：`lfp_visual_pos_masks` 仅标记前 `n_history × n_aggr` 个视觉token位置
- 无历史帧时：`lfp_visual_pos_masks = None` → LFP变为no-op

**效果**：
- step_id=0（无历史）：LFP完全跳过，当前帧144个tokens全部保留
- step_id>0（有历史）：LFP仅对历史帧压缩tokens（64个/帧）进行routing，当前帧tokens不受影响

### 需要重新训练

由于修改影响了训练行为（LFP现在只路由历史tokens），现有v3 checkpoint在新代码下的训练行为会改变。建议：

1. **快速验证**：用v3 checkpoint + 修复后的代码运行eval，检查step_id=0是否恢复pixel_goal输出
2. **正式训练**：用v4脚本重新训练（代码已自动生效）

### 次要问题（后续跟进）

即使没有LFP，V1 nolfp模型在step_id>0时也出现→←→←震荡。这说明compressor本身在处理历史帧时也存在问题，但这是一个独立于LFP routing的次要问题。

## 训练数据分布参考

对20个episodes采样统计：
- pixel_goal: 43.3%
- turn: 19.6%  
- stop (×5): 37.0%

纯LoRA在这个数据分布下表现正常 — 数据分布不是mode collapse的原因。
