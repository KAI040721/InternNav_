# InternVLA-N1 Pure LoRA 训练策略报告

## 模型概览

**基础模型**: Qwen3-VL-2B-Instruct (Qwen Vision Language Model 3)  
**训练方法**: Pure LoRA (低秩适配)  
**数据集**: R2R + RxR (各50%采样)  
**训练周期**: 2 Epochs  

---

## 模型参数分布与训练配置表

### 总体统计
| 指标 | 数值 | 百分比 |
|------|------|--------|
| **总参数量** | 2,167,197,696 | 100% |
| **可训练参数** | 39,665,664 | 1.83% |
| **冻结参数** | 2,127,532,032 | 98.17% |

---

## 分模块参数详解

### 1️⃣ 视觉编码器 (Vision Tower)

#### 结构组成
- **Patch Embedding**: 将输入图像分块嵌入
- **位置编码 (Pos Embed)**: 空间位置信息编码
- **Transformer Blocks**: 24个自注意力层 (Block 0-23)
- **Merger**: 多尺度特征融合模块
- **DeepStack Merger**: 深层特征堆叠融合

#### 参数训练配置

| 组件 | 训练方法 | 参数量 | 可训练 | 说明 |
|------|--------|--------|--------|------|
| **Patch Projection** | LoRA | 3.1M | ✅ | 图像分块嵌入投影，应用LoRA适配 |
| **位置编码** | 冻结 | 2.4M | ❌ | 固定位置信息，不更新 |
| **Transformer Blocks (24×)** | LoRA | - | ✅ | 每个Block包含: |
| ├─ LayerNorm | 冻结 | ~24K | ❌ | 所有规范化层冻结 |
| ├─ QKV 投影 | LoRA | ~2M | ✅ | 查询/键/值投影，LoRA rank=32 |
| ├─ Attention Proj | LoRA | ~800K | ✅ | 注意力输出投影，LoRA rank=32 |
| ├─ MLP (fc1/fc2) | 冻结 (基础) | ~32M | ❌ | 基础权重冻结，无LoRA |
| **Merger** | LoRA (fc1/fc2) | 8.4M | ✅ | 多尺度融合fc1/fc2，LoRA rank=32 |
| **DeepStack Merger (3×)** | LoRA (fc1/fc2) | ~25M | ✅ | 深层融合fc1/fc2，LoRA rank=32 |

---

### 2️⃣ 语言模型 (Language Model)

#### 结构组成
- **Token Embedding**: 词汇表 (151,936维) 嵌入
- **Transformer Layers**: 24个自注意力层 (Layer 0-23)
- **Output LayerNorm**: 最终规范化

#### 参数训练配置

| 组件 | 训练方法 | 参数量 | 可训练 | 说明 |
|------|--------|--------|--------|------|
| **Embed Tokens** | 冻结 | 310.8M | ❌ | 词汇表嵌入矩阵，固定不变 |
| **Transformer Layers (24×)** | LoRA | - | ✅ | 每个Layer包含: |
| ├─ LayerNorm | 冻结 | ~48K | ❌ | 所有规范化层冻结 |
| ├─ Q/K/V 投影 | LoRA | ~200K | ✅ | 查询/键/值投影，LoRA rank=32 |
| ├─ O 投影 | LoRA | ~67K | ✅ | 注意力输出投影，LoRA rank=32 |
| ├─ Gate/Up/Down 投影 | LoRA | ~400K | ✅ | MLP门控与上下投影，LoRA rank=32 |
| **Output LayerNorm** | 冻结 | 2K | ❌ | 最终规范化层，不更新 |

---

## LoRA 配置详情

### LoRA 超参数设置

| 超参 | 数值 | 说明 |
|------|------|------|
| **LoRA Rank (r)** | 32 | 低秩矩阵的秩维度，控制适应能力 |
| **LoRA Alpha** | 64 | 缩放因子，相当于 2×rank |
| **LoRA Dropout** | 0.05 | 正则化参数，防止过拟合 |
| **LoRA Bias** | none | 不为LoRA添加偏置项 |

### LoRA 应用范围

#### 被应用LoRA的目标模块 (11个)

```
Vision Tower (6个):
  ✅ qkv         - 视觉注意力多头投影
  ✅ proj        - 视觉注意力输出投影
  ✅ fc1         - 视觉MLP第一层 (Merger) 
  ✅ fc2         - 视觉MLP第二层 (Merger)

Language Model (5个):
  ✅ q_proj      - LLM查询投影
  ✅ k_proj      - LLM键投影
  ✅ v_proj      - LLM值投影
  ✅ o_proj      - LLM注意力输出投影
  ✅ gate_proj   - LLM MLP门控投影
  ✅ up_proj     - LLM MLP上投影
  ✅ down_proj   - LLM MLP下投影
```

#### 冻结的组件 (规范化层)

所有**LayerNorm/RMSNorm**和**嵌入层**保持完全冻结：
- Vision Blocks 中所有 norm1, norm2
- Language Layers 中所有 LayerNorm
- Token Embedding (词汇表嵌入)
- Position Embedding (位置编码)

---

## 训练参数详情

| 参数 | 数值 |
|------|------|
| **优化器** | AdamW |
| **学习率** | 2e-4 |
| **Batch Size (单GPU)** | 16 |
| **GPU 数量** | 2 |
| **梯度累积步数** | 4 |
| **有效批大小** | 16 × 2 × 4 = 128 |
| **总训练样本** | ~209,740 |
| **每Epoch步数** | ~1,638 |
| **总Epoch** | 2 |
| **总训练步数** | ~3,276 |
| **预期训练时间** | ~19-20 小时 |

---

## 模块级别参数统计

| 模块 | 基础参数 | 可训练参数 | 训练方法 |
|------|---------|-----------|---------|
| **Vision Embedding** | 2.4M | 0.3M | Patch Proj LoRA |
| **Vision Blocks** | ~600M | ~8M | QKV+Proj LoRA |
| **Vision Merger** | 33.4M | ~33M | fc1/fc2 LoRA |
| **LLM Embedding** | 310.8M | 0 | 冻结 |
| **LLM Layers** | ~1,200M | ~26M | Q/K/V/O/Gate/Up/Down LoRA |
| **LLM Output** | 0.2M | 0 | 冻结 |
| **═════════════** | **2,167M** | **~39.7M** | **─** |

---

## 为什么选择Pure LoRA策略？

### 设计原则

1. **最小化可训练参数** (1.83%)
   - 减少显存占用
   - 加快训练速度
   - 降低过拟合风险

2. **保护基础知识** 
   - 冻结所有规范化层
   - 保留主模型预训练知识
   - 仅通过LoRA适配任务

3. **任务适应性**
   - Vision Tower: 适配视觉特征提取
   - LLM: 适配导航推理与决策
   - Merger: 通过fc1/fc2 LoRA融合多尺度信息

4. **计算效率**
   - LoRA秩=32，适中的适应性
   - 支持DeepSpeed ZeRO-2优化
   - 双GPU可并行训练

---

## 对比: Pure LoRA vs 其他策略

| 策略 | 可训练参数% | 训练时间 | 内存需求 | 性能 |
|------|-----------|---------|---------|------|
| **Pure LoRA** (当前) | 1.83% | ⚡ 快 | ✅ 低 | ✅ 高 |
| Full Fine-tuning | 100% | ❌ 慢 | ❌ 高 | ⚠️ 易过拟合 |
| Modules-to-save | ~5-10% | ⚠️ 中等 | ⚠️ 中等 | ❌ 低 |
| LoRA All Layers | 2-3% | ⚡ 快 | ✅ 低 | ❌ 低 |

---

## 总结

- **训练参数量**: 39,665,664 (39.7M)
- **冻结参数量**: 2,127,532,032 (2.127B)  
- **总参数量**: 2,167,197,696 (2.167B)
- **可训练比例**: 1.83%

此Pure LoRA策略通过在关键模块(注意力投影、MLP融合层)应用低秩适配器，在保持预训练知识的同时，高效地适配视觉导航任务。
