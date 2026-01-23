# 微调参数对比：v2 (8B) vs v3 (4B)

## 概述

本文档详细对比了 InternVLA-N1-System2 的两个训练版本：
- **v2**: 使用 Qwen3-VL-8B-Instruct 基础模型
- **v3**: 使用 Qwen3-VL-4B-Instruct 基础模型（当前运行中）

---

## 1. 模型架构差异

| 项目 | v2 (8B) | v3 (4B) |
|------|---------|---------|
| **基础模型** | Qwen3-VL-8B-Instruct | Qwen3-VL-4B-Instruct |
| **模型大小** | ~8B 参数 | ~4B 参数 |
| **可训练参数** | ~70.8M (LoRA) | ~70.8M (LoRA) |
| **可训练比例** | 1.57% | 1.57% |

---

## 2. 训练数据差异 ⚠️

### v2 (8B)
```
数据集: r2r_125cm_0_30, rxr_125cm_0_30
- R2R: ~14k episodes
- RxR: ~24k episodes
总计: ~38k episodes
```

### v3 (4B) ⚠️ 重要变化
```
数据集: r2r_125cm_0_30, rxr_125cm_0_30, scalevln_125cm_0_30%50
- R2R: ~14k episodes
- RxR: ~24k episodes  
- ScaleVLN (50% 采样): ~725k episodes (原始 1.45M)
总计: ~763k episodes (20倍于v2!)
```

**关键差异**:
- ✅ v3 加入了大规模 ScaleVLN 数据集（50%采样）
- ✅ 数据量增加约 **20 倍**
- ⚠️ 训练时间显著增长

---

## 3. Batch Size 配置差异

| 参数 | v2 (8B) | v3 (4B) | 差异 |
|------|---------|---------|------|
| **per_device_train_batch_size** | 8 | 12 | +50% |
| **gradient_accumulation_steps** | 4 | 3 | -25% |
| **全局 Batch Size** | 128 | 144 | +12.5% |

**计算**:
- v2: `8 × 4 × 4 GPUs = 128`
- v3: `12 × 3 × 4 GPUs = 144`

**原因**:
- v3 使用 4B 模型，显存占用更小，可以增大 per_device batch size
- 但为了保持全局 batch size 接近，减少了梯度累积步数

---

## 4. 学习率与优化器配置

| 参数 | v2 (8B) | v3 (4B) | 状态 |
|------|---------|---------|------|
| **learning_rate** | 2e-4 | 2e-4 | ✅ 相同 |
| **lr_scheduler_type** | cosine | cosine | ✅ 相同 |
| **warmup_ratio** | 0.03 | 0.03 | ✅ 相同 |
| **weight_decay** | 0.01 | 0.01 | ✅ 相同 |
| **max_grad_norm** | 1.0 | 1.0 | ✅ 相同 |

**结论**: 学习率和优化器配置完全一致，保证了训练的可比性。

---

## 5. LoRA 配置

| 参数 | v2 (8B) | v3 (4B) | 状态 |
|------|---------|---------|------|
| **use_lora** | True | True | ✅ 相同 |
| **lora_r** | 32 | 32 | ✅ 相同 |
| **lora_alpha** | 64 | 64 | ✅ 相同 |
| **lora_dropout** | 0.05 | 0.05 | ✅ 相同 |
| **target_modules** | q/k/v/o_proj, gate/up/down_proj, fc1/fc2 | q/k/v/o_proj, gate/up/down_proj, fc1/fc2 | ✅ 相同 |

**目标模块**:
- Vision Tower: LoRA 启用
- LLM Attention: q_proj, k_proj, v_proj, o_proj
- LLM MLP: gate_proj, up_proj, down_proj
- Projector/Merger: fc1, fc2
- LayerNorm: **冻结** (不使用 LoRA)

---

## 6. 数据加载优化 ⚠️

| 参数 | v2 (8B) | v3 (4B) | 差异 |
|------|---------|---------|------|
| **dataloader_num_workers** | 16 | 16 | ✅ 相同 |
| **dataloader_persistent_workers** | **False** | **True** | ⚠️ 不同 |
| **dataloader_pin_memory** | True | True | ✅ 相同 |
| **dataloader_prefetch_factor** | N/A | 2 | ⚠️ 新增 |

**v3 的优化**:
1. ✅ `persistent_workers=True`: 保持 worker 进程活跃，减少进程创建开销
2. ✅ `prefetch_factor=2`: 预取 2 个 batch，提高 GPU 利用率

---

## 7. 梯度检查点 (Gradient Checkpointing)

| 组件 | v2 (8B) | v3 (4B) | 状态 |
|------|---------|---------|------|
| **gradient_checkpointing** | True | True | ✅ 相同 |
| **Vision Tower GC** | 自动关闭 | 自动关闭 | ✅ 相同 |
| **LLM GC** | True | True | ✅ 相同 |

**说明**: 
- v3 在代码中自动关闭 Vision Tower 的梯度检查点以提高速度
- LLM 保留梯度检查点以避免 OOM

---

## 8. 训练规模与时间

| 参数 | v2 (8B) | v3 (4B) | 对比 |
|------|---------|---------|------|
| **num_train_epochs** | 2 | 2 | ✅ 相同 |
| **save_steps** | 500 | 500 | ✅ 相同 |
| **数据集大小** | ~38k episodes | ~763k episodes | **+1900%** |
| **total_steps** | ~11,620 | ~20,402 | +76% |
| **训练速度** | ~3-4s/it | ~6.11s/it | -43% |
| **预计总时间** | ~13 小时 | ~32 小时 | +146% |

**训练时间计算**:
- v2: 11,620 steps × 3.5s = 11.3 小时
- v3: 20,402 steps × 6.11s = 34.6 小时

---

## 9. 分布式训练修复 (v3 特有)

v3 版本修复了分布式训练中的 GPU 利用率问题：

### 修复内容:
1. ✅ **数据采样随机种子固定**
   - `internnav/dataset/internvla_n1_lerobot_dataset.py`:
     - 第 320 行: `random.shuffle()` → 使用固定种子
     - 第 1359 行: `np.random.shuffle()` → 使用固定种子
   
2. ✅ **vlln_lerobot_dataset.py 随机采样修复**
   - 使用 `rng = random.Random(42)` 固定种子

3. ✅ **lengths() 方法优化**
   - 优先使用预计算的 `length` 字段（轨迹长度）
   - 回退到 `num_tokens` 或动态计算

### 问题现象 (已修复):
- ❌ 某张 GPU 利用率突然变为 0%，持续 30-60 秒
- ❌ 训练速度不稳定，频繁卡顿

### 修复后效果:
- ✅ GPU 利用率稳定在 99-100%
- ✅ 训练速度稳定在 6.11s/it
- ✅ 不再出现长时间 GPU 0% 的情况

---

## 10. 性能对比

### GPU 利用率
| 版本 | 平均利用率 | 稳定性 |
|------|-----------|--------|
| v2 (8B) | 95-99% | 偶尔卡顿 |
| v3 (4B) | **99-100%** | **非常稳定** ✅ |

### 吞吐量
| 版本 | samples/s | samples/s/GPU |
|------|-----------|---------------|
| v2 (8B) | ~41.1 | ~10.3 |
| v3 (4B) | **23.6** | **5.9** |

**说明**: v3 吞吐量较低的原因:
1. 数据集更大（ScaleVLN）
2. 50% 随机采样增加开销
3. persistent_workers 和 prefetch_factor 已优化

---

## 11. 关键差异总结

### ⚠️ 主要差异

| 项目 | v2 (8B) | v3 (4B) | 影响 |
|------|---------|---------|------|
| **模型大小** | 8B | 4B | 速度↑ 容量↓ |
| **数据集** | R2R+RxR | R2R+RxR+ScaleVLN(50%) | 数据量↑20倍 |
| **per_device_batch** | 8 | 12 | 显存利用↑ |
| **grad_accum** | 4 | 3 | 更新频率↑ |
| **global_batch** | 128 | 144 | +12.5% |
| **persistent_workers** | False | True | 效率↑ |
| **prefetch_factor** | 无 | 2 | GPU利用率↑ |
| **训练时间** | ~13h | ~32h | +146% |

### ✅ 保持一致的配置

- Learning rate: 2e-4
- LoRA config: r=32, alpha=64
- Optimizer: AdamW
- LR scheduler: Cosine
- Epochs: 2
- Gradient clipping: 1.0

---

## 12. 建议与展望

### v3 (4B) 的优势:
1. ✅ 模型更小，推理速度更快
2. ✅ 数据集更大（加入 ScaleVLN），泛化能力可能更强
3. ✅ 分布式训练稳定性显著提升
4. ✅ 数据加载优化（persistent_workers, prefetch）

### v3 的挑战:
1. ⚠️ 训练时间显著增长（~32h vs ~13h）
2. ⚠️ 模型容量较小，学习能力可能弱于 8B
3. ⚠️ 50% ScaleVLN 采样是否足够？

### 下一步实验建议:
1. 完成 v3 训练，评估效果
2. 如果效果好，考虑：
   - 增加 ScaleVLN 采样率（50% → 100%）
   - 或在 8B 模型上也加入 ScaleVLN

---

## 13. 代码修改清单

### 修改的文件:
1. `internnav/dataset/internvla_n1_lerobot_dataset.py`
   - Line 320: 固定 random.shuffle 种子
   - Line 335-350: 优化 lengths() 方法
   - Line 1359: 固定 np.random.shuffle 种子

2. `internnav/dataset/vlln_lerobot_dataset.py`
   - 修复 random.sample 使用固定种子

3. `scripts/train/qwenvl_train/train_system2_v3_4b_optimized_lora.sh`
   - 新增训练脚本（4B 模型）

4. `scripts/train/qwenvl_train/train_system2_v3_4b_resume.sh`
   - 新增恢复训练脚本（从 checkpoint-1500 继续）

### Git 分支:
- 分支名: `fix-distributed-training-v3-4b`
- Commit message: "Fix distributed training GPU utilization issues"

---

**文档生成时间**: 2026-01-23  
**当前训练状态**: v3 (4B) 从 checkpoint-1500 恢复，epoch 0.15，loss ~0.67
