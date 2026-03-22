# InternNav SFT训练项目文档总结

## 📚 项目文档体系

本项目共包含**3份详细的技术文档**，涵盖了从原始数据处理到最终模型训练的完整流程。

---

## 文档清单

### 1. 📋 `TRAINING_STRATEGY_REPORT.md` (6.0 KB, 189行)

**目标**: 介绍Pure LoRA训练策略及模型参数分布

**核心内容:**
- ✅ 模型参数总体统计 (2.167B总参数 → 39.7M可训练)
- ✅ Vision Tower (视觉编码器) 参数配置详解
- ✅ Language Model (语言模型) 参数配置详解  
- ✅ LoRA超参数设置 (rank=32, alpha=64, dropout=0.05)
- ✅ 11个目标模块的LoRA应用范围
- ✅ 训练参数详情 (batch=128, lr=2e-4, 2 epochs)
- ✅ Pure LoRA vs 其他策略的对比分析

**适用场景:**
- 中期报告的"训练方法"章节
- 学术论文中的模型设计部分
- 代码审查时的参数验证

**关键数据点:**
```
总参数: 2,167,197,696
可训练: 39,665,664 (1.83%)
冻结: 2,127,532,032 (98.17%)

LoRA目标模块(11个):
- Vision: qkv, proj, fc1, fc2
- LLM: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

---

### 2. 📊 `DATA_TRANSFORMATION_GUIDE.md` (27 KB, 678行)

**目标**: 详细说明原始R2R数据到训练样本的完整转换流程

**核心内容:**

#### 第一部分: 原始R2R数据结构 (61个场景, 104,870个episode)
- LeRobot数据目录布局
- 元数据格式 (episodes.jsonl)
- 低维数据格式 (Parquet列式)
- 视频数据多视点结构
- 原始数据关键特征表

#### 第二部分: 转换过程详解 (7个步骤)
```
Step 1: 配置加载 → R2R_125CM_0_30等配置
Step 2: 原始数据读取 → JSON/JSONL加载+采样率
Step 3: 视频帧提取 → Decord解码→均匀采样(4-8帧)
Step 4: 视觉特征处理 → 预处理→张量化→token网格
Step 5: 对话格式化 → 指令→多轮对话结构
Step 6: Token化与标签 → IGNORE_INDEX掩码标注
Step 7: 最终样本组装 → 完整dict返回
```

#### 第三部分: 数据结构对比
- 原始LeRobot格式 vs 训练样本格式
- 详细的字段映射表
- 数据量与内存估算

#### 第四部分: 完整数据流可视化
- ASCII艺术流程图
- 从原始→模型前向传播的完整管道

#### 第五部分: 配置示例
- 训练数据配置 (R2R+RxR混合采样)
- 视频处理参数
- 损失函数掩码设置

#### 第六部分: 常见问题与深入理解
- Q1: 为什么采样视频帧?
- Q2: 为什么使用IGNORE_INDEX=-100?
- Q3: 多视点如何选择?
- Q4: 不同长度episode的处理?
- Q5: 为什么固定seed=42?

**适用场景:**
- 详细的技术文档和代码审查
- 新成员的数据处理入门教材
- 论文的数据处理部分
- 复现实验的参考指南

**关键数据点:**
```
原始数据: 500GB (104,870 episodes × 5MB)
训练样本: 12KB/样本 (纯参考)
数据集总量: ~238,324个训练样本 (R2R全量+RxR50%)
```

---

### 3. 🎯 `PIXEL_GOAL_DETAILED_GUIDE.md` (29 KB, 833行)

**目标**: 深入介绍像素目标点坐标的生成、处理和应用

**核心内容:**

#### 第一部分: 像素目标点的物理意义
- 什么是像素目标点
- 为什么需要像素目标点
- 数据生成流程概览

#### 第二部分: 原始LeRobot格式的目标点数据
- 数据存储位置和Parquet结构
- 目标点数据的具体含义:
  - `goal.{setting}`: 像素坐标 [x, y]
  - `relative_goal_frame_id`: 路径长度(帧数)
- 特殊标记 [-1, -1] 的含义
- 多视点目标点的区别 (125cm_0deg vs 125cm_30deg)

#### 第三部分: 目标点数据的处理与转换
- 原始数据加载 (Parquet→DataFrame)
- 数据验证与质量检查
- 使用场景分类 (像素目标/转向/停止)

#### 第四部分: 数据转换示例
- 具体的Episode转换案例 (ID=42)
- 原始LeRobot数据 vs 转换后训练样本的详细对比

**示例表格:**
```
Frame | Action | goal.125cm_0deg | relative_goal_id
─────┼────────┼─────────────────┼──────────────────
  0  │   1    │  [300, 250]     │      12
  1  │   1    │  [305, 252]     │      11
  ...
  8  │   1    │  [-1, -1]       │      -1  ← 特殊标记
```

#### 第五部分: 像素目标在模型中的应用
- 从原始坐标到模型输入
- Token化与监督学习
- 多任务学习框架 (像素目标/转向/停止)

#### 第六部分: 常见问题与深入理解
- Q1: 为什么需要多视点目标点?
- Q2: [-1, -1] 标记如何处理?
- Q3: 坐标值的有效范围?
- Q4: 如何从像素坐标恢复3D世界坐标?

#### 第七部分: 完整处理流程总结
- ASCII艺术管道图 (7个步骤)
- 代码参考 (完整的数据加载流程)

**适用场景:**
- 中期报告的"数据处理"章节重点
- 视觉导航任务的细节理解
- 像素坐标相关的bug修复
- 模型输出解释

**关键数据点:**
```
图像分辨率: 512×512
坐标范围: x ∈ [0, 512), y ∈ [0, 512)
特殊标记: [-1, -1] (无有效目标)
样本类型: 3种 (像素目标/转向/停止)
路径长度: >= 3帧才作为有效样本
```

---

## 📈 文档间的关系

```
TRAINING_STRATEGY_REPORT.md
   ↑
   │ 定义训练方法
   │
   ├─→ DATA_TRANSFORMATION_GUIDE.md
   │        ↓
   │        数据处理流程
   │        ↓
   │   PIXEL_GOAL_DETAILED_GUIDE.md
   │        ↓
   │        像素目标细节
   │
   └─→ 模型架构 (vision + language)
        ↓
      Pure LoRA 微调
        ↓
      监督学习训练
```

---

## 🎓 推荐阅读顺序

### 方案A: 快速了解 (30分钟)
1. `TRAINING_STRATEGY_REPORT.md` - 整体概览 (5分钟)
2. `DATA_TRANSFORMATION_GUIDE.md` - 第一~四部分 (15分钟)
3. `PIXEL_GOAL_DETAILED_GUIDE.md` - 第一~二部分 (10分钟)

### 方案B: 深入学习 (2小时)
1. 完整阅读 `TRAINING_STRATEGY_REPORT.md` (20分钟)
2. 完整阅读 `DATA_TRANSFORMATION_GUIDE.md` (60分钟)
3. 完整阅读 `PIXEL_GOAL_DETAILED_GUIDE.md` (40分钟)

### 方案C: 实现和调试 (按需阅读)
1. 数据加载问题 → `DATA_TRANSFORMATION_GUIDE.md` 第二~三部分
2. 像素目标问题 → `PIXEL_GOAL_DETAILED_GUIDE.md` 第三~四部分
3. 模型配置问题 → `TRAINING_STRATEGY_REPORT.md` 全文

---

## 📋 各文档的关键数据表

### TRAINING_STRATEGY_REPORT.md
| 指标 | 数值 |
|------|------|
| 总参数 | 2,167,197,696 |
| 可训练参数 | 39,665,664 |
| 可训练比例 | 1.83% |
| LoRA秩 | 32 |
| LoRA Alpha | 64 |
| 目标模块数 | 11 |
| 学习率 | 2e-4 |
| 有效批大小 | 128 |

### DATA_TRANSFORMATION_GUIDE.md
| 指标 | 数值 |
|------|------|
| 总场景数 | 61 |
| 总Episode数 | 104,870 (R2R) |
| 总帧数 | ~1,500,000+ |
| 平均Episode长度 | 14-50帧 |
| 动作空间 | 5个离散动作 |
| 采样帧数范围 | 4-8帧 |
| 预期训练样本 | ~238,324 |
| 原始数据量 | ~500GB |

### PIXEL_GOAL_DETAILED_GUIDE.md
| 指标 | 数值 |
|------|------|
| 图像分辨率 | 512×512 |
| 坐标有效范围 | [0, 512)×[0, 512) |
| 视点配置数 | 3+种 |
| 样本类型数 | 3种 (像素目标/转向/停止) |
| 最小路径长度 | >= 3帧 |
| 特殊标记值 | [-1, -1] |

---

## 🔧 使用建议

### 对于中期报告写作:
```
第1章: 模型设计
  → 引用 TRAINING_STRATEGY_REPORT.md 的表格和图表

第2章: 数据处理
  → 引用 DATA_TRANSFORMATION_GUIDE.md 第一~四部分
  → 引用 PIXEL_GOAL_DETAILED_GUIDE.md 第一~二部分

第3章: 实现细节
  → 引用 DATA_TRANSFORMATION_GUIDE.md 第二部分
  → 引用 PIXEL_GOAL_DETAILED_GUIDE.md 第三~五部分

第4章: 实验结果
  → 参考所有三份文档的总结部分
```

### 对于代码审查:
```
审查trainer.py
  → 检查 TRAINING_STRATEGY_REPORT.md 的LoRA配置

审查dataset.py
  → 检查 DATA_TRANSFORMATION_GUIDE.md 的转换逻辑
  → 检查 PIXEL_GOAL_DETAILED_GUIDE.md 的数据验证

审查训练脚本
  → 验证 TRAINING_STRATEGY_REPORT.md 的参数设置
```

### 对于新成员入门:
```
第1周: 快速了解
  → 方案A阅读 (30分钟)
  → 运行示例数据处理脚本

第2周: 深入学习
  → 方案B阅读 (2小时)
  → 尝试修改数据处理参数

第3周: 实践应用
  → 根据方案C按需查阅
  → 独立调试数据问题
```

---

## 📊 文档统计信息

```
总文档数: 3份
总文件大小: ~62 KB
总行数: ~1,700行
总表格数: ~40+ 个
总代码示例: ~30+ 个
总流程图: ~10+ 个
```

---

## 🔗 相关文件位置

```
/data/houdekai/InternNav_/
├── TRAINING_STRATEGY_REPORT.md              # 训练策略报告
├── DATA_TRANSFORMATION_GUIDE.md             # 数据转换指南
├── PIXEL_GOAL_DETAILED_GUIDE.md            # 像素目标详细指南
│
├── internnav/trainer/internvla_n1_trainer.py       # 参考实现
├── internnav/dataset/internvla_n1_lerobot_dataset.py  # 参考实现
│
└── scripts/train/qwenvl_train/
    ├── train_compressor_baseline_2b.sh      # 训练脚本
    ├── zero2.json                            # 配置文件
    └── ...
```

---

## ✅ 使用检查清单

- [ ] 已阅读 TRAINING_STRATEGY_REPORT.md
- [ ] 已理解模型参数分布
- [ ] 已阅读 DATA_TRANSFORMATION_GUIDE.md
- [ ] 已理解数据处理流程
- [ ] 已阅读 PIXEL_GOAL_DETAILED_GUIDE.md
- [ ] 已理解像素目标点的处理方式
- [ ] 已能独立解释数据转换的每一步
- [ ] 已能回答常见问题
- [ ] 已阅读相关源代码
- [ ] 已准备好在中期报告中使用

---

## 📝 文档版本信息

| 文档 | 版本 | 创建日期 | 行数 | 大小 |
|------|------|---------|------|------|
| TRAINING_STRATEGY_REPORT.md | v1.0 | 2026-03-15 | 189 | 6.0 KB |
| DATA_TRANSFORMATION_GUIDE.md | v1.0 | 2026-03-15 | 678 | 27 KB |
| PIXEL_GOAL_DETAILED_GUIDE.md | v1.0 | 2026-03-15 | 833 | 29 KB |

---

## 🎯 核心要点总结

### TRAINING_STRATEGY_REPORT.md
> 使用Pure LoRA策略，在Qwen3-VL-2B基础上微调39.7M参数（1.83%），通过LoRA对11个关键模块进行低秩适配，冻结所有规范化层，实现高效的参数优化。

### DATA_TRANSFORMATION_GUIDE.md
> 将结构化的LeRobot多模态导航数据（MP4视频+Parquet低维+JSON文本）通过7个转换步骤（加载→验证→分类→采样→对话→Token化→组装），转换为标准的VLM训练样本格式。

### PIXEL_GOAL_DETAILED_GUIDE.md
> 像素目标点坐标是连接图像和导航动作的关键桥梁，通过从Parquet中提取、验证、分类、文本化的处理管道，使模型学会从图像预测下一步目标位置，支持端到端的监督学习。

---

## 📞 文档导航

需要快速找到某个主题？

| 主题 | 位置 |
|------|------|
| LoRA配置 | TRAINING_STRATEGY_REPORT.md §3.0 |
| 数据加载 | DATA_TRANSFORMATION_GUIDE.md §2.1 |
| 视频处理 | DATA_TRANSFORMATION_GUIDE.md §2.2 |
| Token化 | DATA_TRANSFORMATION_GUIDE.md §2.3 |
| 像素目标格式 | PIXEL_GOAL_DETAILED_GUIDE.md §2.0 |
| 目标点处理 | PIXEL_GOAL_DETAILED_GUIDE.md §3.0 |
| 转换示例 | PIXEL_GOAL_DETAILED_GUIDE.md §4.0 |
| 常见问题 | 各文档 §6.0 |

---

**本文档集为InternNav SFT训练项目的完整技术文档，包含从数据处理到模型训练的所有关键信息。**

建议将本文档与其他三份详细指南一起使用，以获得完整的项目理解。
