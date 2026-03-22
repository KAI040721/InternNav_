# R2R数据集到训练样本的完整转换流程

## 概述

本文档详细介绍了如何将原始R2R数据集转换为用于InternVLA-N1监督学习训练的样本数据。整个过程包括三个主要阶段：

1. **原始数据格式**: LeRobot格式的多模态导航数据
2. **中间转换**: 提取、处理、编码视觉和文本信息
3. **最终训练样本**: 标准化的多轮对话+视觉内容组合

---

## 第一部分: 原始R2R数据结构

### 1.1 数据目录布局

```
traj_data/r2r/                        # R2R数据集根目录
├── <scene_id>/                       # 61个不同的3D室内场景
│   ├── meta/                         # 元数据文件夹
│   │   ├── info.json                 # 场景基本信息 (fps, splits, chunks等)
│   │   ├── episodes.jsonl            # 逐行JSON格式的导航任务列表
│   │   ├── episodes_stats.jsonl      # 每个任务的统计信息
│   │   └── tasks.jsonl               # 原始任务描述
│   ├── data/                         # 结构化数据文件夹
│   │   └── chunk-000/                # 数据块 (每块1000条episode)
│   │       ├── episode_000000.parquet  # Parquet格式的低维数据
│   │       ├── episode_000001.parquet  #   包含: 动作、位姿、目标点等
│   │       └── ...
│   └── videos/                       # 视频文件夹
│       ├── chunk-000/                # 对应的视频块
│       │   ├── 125cm_0deg/           # 不同高度和俯仰角的视频
│       │   │   ├── episode_000000.mp4
│       │   │   └── ...
│       │   ├── 125cm_30deg/
│       │   └── ...
│       └── ...
```

### 1.2 单个Episode的原始内容

#### 元数据 (episodes.jsonl)
```json
{
  "episode_index": 0,
  "tasks": [
    "Exit the bedroom, enter the bathroom, wait at the toilet."
  ],
  "length": 46
}
```

**字段说明:**
- `episode_index`: 全局唯一的任务ID
- `tasks`: 自然语言指令列表 (可能有多个指令变体)
- `length`: 视频帧数 / 动作序列长度

#### 低维数据 (episode_XXXXXX.parquet)

使用Apache Parquet列式存储格式，包含以下数据列:

| 字段 | 数据类型 | 形状 | 说明 |
|------|---------|------|------|
| `action` | int32 | [46, 1] | 动作索引序列 (0=STOP, 1=↑前进, 2=←左转, 3=→右转, 5=↓后退) |
| `pose.125cm_0deg` | float32 | [46, 4, 4] | 机器人位姿 (4×4变换矩阵) @ 125cm高度0°俯仰 |
| `pose.125cm_30deg` | float32 | [46, 4, 4] | 机器人位姿 @ 125cm高度30°俯仰 |
| `goal.125cm_0deg` | int32 | [46, 2] | 像素空间目标点坐标 @ 125cm_0deg |
| `goal.125cm_30deg` | int32 | [46, 2] | 像素空间目标点坐标 @ 125cm_30deg |
| `relative_goal_frame_id.125cm_0deg` | int32 | [46, 1] | 相对目标帧ID |
| `relative_goal_frame_id.125cm_30deg` | int32 | [46, 1] | 相对目标帧ID |

#### 视频数据 (episode_XXXXXX.mp4)

- **格式**: MP4视频
- **分辨率**: 原始尺寸 (通常512×512或更高)
- **帧率**: 30 FPS
- **帧数**: 与`length`字段对应
- **变体**: 多个视点
  - `125cm_0deg`: 高度125cm，无俯仰角
  - `125cm_30deg`: 高度125cm，30°俯仰角
  - `60cm_15_15`: 高度60cm，两个15°俯仰角

### 1.3 原始数据的关键特征

| 特征 | 数值 |
|------|------|
| **总场景数** | 61个不同的3D室内环境 |
| **总Episode数** | R2R: 104,870个导航任务 |
| **总帧数** | ~1,500,000+ 视频帧 |
| **平均Episode长度** | ~14-50帧 |
| **动作空间** | 5个离散动作 |
| **指令类型** | 自然语言导航指令 |
| **多模态** | 视频 + 低维状态信息 + 文本 |
| **存储格式** | LeRobot (Parquet + MP4) |

---

## 第二部分: 转换过程详解

### 2.1 数据加载与初始化

#### 步骤1: 配置加载 (internvla_n1_lerobot_dataset.py: 150-200行)

```python
# 数据集配置示例
R2R_125CM_0_30 = {
    "data_path": "traj_data/r2r",      # LeRobot格式数据路径
    "height": 125,                      # 相机高度 (cm)
    "pitch_1": 0,                       # 主要俯仰角 (度)
    "pitch_2": 30,                      # 备用俯仰角 (度)
    "sampling_rate": 1.0                # 采样率 (0.0-1.0)
}

# 创建数据集实例时，支持多个数据集组合
# 例: "r2r_125cm_0_30,rxr_125cm_0_30%50"
# 含义: R2R全量 + RxR的50%采样
```

#### 步骤2: 原始数据读取 (行313-330)

```python
for data in dataset_list:
    # 1. 读取annotations (JSON或JSONL格式)
    if file_format == "jsonl":
        annotations = read_jsonl(data["annotation_path"])
    else:
        annotations = json.load(open(data["annotation_path"]))
    
    # 2. 应用采样率 (如果设置 < 1.0)
    sampling_rate = data.get("sampling_rate", 1.0)
    if sampling_rate < 1.0:
        random.seed(42)  # 固定种子保证复现性
        annotations = random.sample(annotations, 
                                   int(len(annotations) * sampling_rate))
    
    # 3. 为每条annotation添加data_path
    for ann in annotations:
        ann["data_path"] = data["data_path"]
    
    list_data_dict += annotations
```

**输出**: 合并后的annotations列表
```python
[
    {
        "episode_index": 0,
        "tasks": ["Exit the bedroom, ..."],
        "length": 46,
        "data_path": "traj_data/r2r"
    },
    ...
]
```

### 2.2 视频处理管道

#### 步骤3: 视频帧提取 (行372-445)

当调用`__getitem__()`获取样本时，触发视频处理:

```python
def video_decord(self, video_file):
    """使用Decord库高效提取视频帧"""
    vr = VideoReader(video_file, num_threads=4)
    total_frames = len(vr)           # 视频总帧数
    avg_fps = vr.get_avg_fps()       # 视频FPS
    video_length = total_frames / avg_fps  # 视频时长(秒)
    
    # 计算采样间隔
    interval = getattr(self.data_args, "base_interval", 4)  # 默认4帧间隔
    num_frames_to_sample = round(video_length / interval)
    
    # 限制采样帧数范围
    video_min_frames = 4
    video_max_frames = 8
    target_frames = min(max(num_frames_to_sample, video_min_frames), 
                       video_max_frames)
    
    # 均匀采样目标帧
    frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    frame_idx = np.unique(frame_idx)
    video = vr.get_batch(frame_idx).asnumpy()  # [T, H, W, C]
    
    return self.process_video_frames(video, frame_idx, video_length)
```

**示例转换:**
- 输入: 46帧 @ 30FPS = 1.53秒视频
- 间隔: 4帧 → 采样约 1.53/4 ≈ 0.4 → min(0.4, 4) = 4帧
- 输出: 4帧均匀分布的关键帧

#### 步骤4: 视觉特征处理 (行424-440)

```python
def process_video_frames(self, video, frame_idx, video_length):
    """处理提取的视频帧为模型输入"""
    fps = len(frame_idx) / video_length
    processor = copy.deepcopy(self.data_args.image_processor)
    
    # 配置处理器参数
    processor.max_pixels = self.data_args.video_max_frame_pixels  # 例: 1024*1024
    processor.min_pixels = self.data_args.video_min_frame_pixels  # 例: 256*256
    processor.size["longest_edge"] = processor.max_pixels
    processor.size["shortest_edge"] = processor.min_pixels
    
    # 预处理视频帧
    video_processed = processor.preprocess(
        images=None, 
        videos=video,           # [T, H, W, C]
        return_tensors="pt"
    )
    
    video_tensor = video_processed["pixel_values_videos"]  # [1, T, C, H, W]
    grid_thw = video_processed["video_grid_thw"][0]        # [T, H, W] of tokens
    second_per_grid_ts = [self.data_args.image_processor.temporal_patch_size / fps] * len(grid_thw)
    
    return video_tensor, grid_thw, second_per_grid_ts
```

**处理输出:**
- `video_tensor`: PyTorch张量，形状 [1, 4, 3, H, W]
- `grid_thw`: 每帧被分解为的视觉token网格 (T, H, W)
- `second_per_grid_ts`: 时间维度token的时间跨度(秒)

### 2.3 文本处理与对话构建

#### 步骤5: 对话格式化 (行145-295)

原始任务指令需要转换为多轮对话格式:

```python
# 原始输入
{
    "episode_index": 0,
    "tasks": ["Exit the bedroom, enter the bathroom, wait at the toilet."],
    "length": 46
}

# 转换为对话对象
{
    "conversations": [
        {
            "from": "human",
            "value": "<video>\n{instruction}"
        },
        {
            "from": "gpt",
            "value": "{response}"
        }
    ],
    "video": "path/to/video.mp4",
    "data_path": "traj_data/r2r"
}
```

#### 步骤6: Token化与标签生成 (行160-240)

```python
def preprocess_qwen_2_visual(sources, tokenizer, grid_thw_video):
    """将对话转换为token序列并生成监督学习标签"""
    
    input_ids = []
    targets = []
    
    for source in sources:
        input_id, target = [], []
        
        # 添加系统提示
        sys_tokens = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."}]
        )
        input_id += sys_tokens
        target += [IGNORE_INDEX] * len(sys_tokens)  # 系统提示不计算loss
        
        # 处理用户输入
        for conv in source:
            role = conv["role"]
            content = conv["content"]
            
            # 替换<video>标记为vision_start/end标记
            if "<video>" in content:
                parts = content.split("<video>")
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    replacement = (
                        "<|vision_start|>" +
                        "<|video_pad|>" * grid_thw_video[0] +  # 视频token数
                        "<|vision_end|>"
                    )
                    new_parts.append(replacement)
                new_parts.append(parts[-1])
                content = "".join(new_parts)
            
            # Token化
            conv_tokens = tokenizer.apply_chat_template([
                {"role": role, "content": content}
            ])
            input_id += conv_tokens
            
            # 生成标签: 用户输入为IGNORE_INDEX, 助手输出为token ID
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(conv_tokens)
            else:
                target_mask = conv_tokens.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3  # 忽略开始标记
                target += target_mask
        
        input_ids.append(torch.tensor(input_id, dtype=torch.long))
        targets.append(torch.tensor(target, dtype=torch.long))
    
    return {
        "input_ids": input_ids,      # [seq_len]
        "labels": targets,            # [seq_len], 带IGNORE_INDEX掩码
    }
```

**关键转换点:**
- `IGNORE_INDEX = -100`: Hugging Face的标准损失掩码值
- 用户提示和系统消息的token被标记为-100（不计算梯度）
- 只有助手响应被用于监督学习

### 2.4 完整数据项构建

#### 步骤7: 最终样本组装 (行461-650)

```python
def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    """获取单个训练样本，包含所有必要的模型输入"""
    
    sources = self.list_data_dict[i]
    
    # 步骤 A: 视频处理
    if "video" in sources:
        video_file = os.path.join(sources["data_path"], sources["video"])
        video_tensor, video_grid_thw, second_per_grid_ts = self.process_video(video_file)
    else:
        video_tensor = None
        video_grid_thw = None
    
    # 步骤 B: 构建对话
    conversations = []
    if video_tensor is not None:
        # 使用视频
        user_message = f"<video>\n{sources['instruction']}"
    else:
        user_message = sources['instruction']
    
    conversations.append({
        "from": "human",
        "value": user_message
    })
    
    # 生成模型响应 (从动作序列)
    if "actions" in sources:
        actions_text = " → ".join([
            self.idx2actions.get(a, str(a)) for a in sources["actions"]
        ])
        conversations.append({
            "from": "gpt",
            "value": actions_text
        })
    
    # 步骤 C: Token化
    tokenized = preprocess_qwen_2_visual(
        [conversations],
        self.tokenizer,
        grid_thw_video=[video_grid_thw] if video_tensor is not None else []
    )
    
    # 步骤 D: 返回完整数据项
    return {
        "input_ids": tokenized["input_ids"][0],           # [seq_len]
        "labels": tokenized["labels"][0],                  # [seq_len]
        "pixel_values_videos": video_tensor,               # [1, T, 3, H, W]
        "video_grid_thw": video_grid_thw,                  # [T, H_tokens, W_tokens]
        "second_per_grid_ts": second_per_grid_ts,         # [T]
        "image_grid_thw": None,                            # 未使用图像
        "is_history_image": False,                         # 标记: 这是当前视频，不是历史图像
    }
```

---

## 第三部分: 数据结构对比

### 3.1 原始数据 vs 训练样本

#### 原始R2R数据

```
LeRobot格式:
├── 低维状态数据 (Parquet)
│   ├── 动作序列: [46个离散动作]
│   ├── 位姿序列: [46, 4, 4] 变换矩阵
│   └── 目标点: [46, 2] 像素坐标
│
├── 视频数据 (MP4)
│   ├── 原始分辨率: 512×512
│   ├── 总帧数: 46帧 @ 30FPS
│   └── 多视点: 125cm_0deg, 125cm_30deg等
│
└── 文本数据 (JSONL)
    ├── episode_index: 0
    ├── tasks: ["Exit the bedroom, ..."]
    └── length: 46
```

**数据特点:**
- ✅ 多模态: 视频 + 低维 + 文本
- ✅ 高度结构化: Parquet列式存储
- ✅ 高保真: 原始视频、完整状态序列
- ❌ 不能直接输入Transformer: 需要转换
- ❌ 低层次: 动作而非高层推理

#### 转换后的训练样本

```
PyTorch Dataset格式:
├── input_ids: [seq_len]
│   └── Token化的文本 + 视觉占位符
│       ├── <|im_start|> system_tokens <|im_end|>
│       ├── <|im_start|> user <|vision_start|><|video_pad|>x256<|vision_end|> ... <|im_end|>
│       ├── <|im_start|> assistant response_tokens <|im_end|>
│       └── ... 总计 ~1000-2000 tokens
│
├── labels: [seq_len]
│   ├── [-100, -100, ...] (系统和用户部分)
│   └── [token_id, token_id, ...] (助手响应)
│
├── pixel_values_videos: [1, 4, 3, H, W]
│   └── 4帧采样的视频数据
│
├── video_grid_thw: [4, H_tokens, W_tokens]
│   └── 每帧被量化为的视觉token网格
│
└── second_per_grid_ts: [4]
    └── 每个时间步的时间跨度
```

**数据特点:**
- ✅ 可直接输入Transformer
- ✅ 标准化格式: HuggingFace兼容
- ✅ 损失函数兼容: IGNORE_INDEX标注
- ✅ 多模态融合: 视频+文本在同一token序列
- ✅ 时间对齐: video_grid_thw记录视觉token映射

### 3.2 详细字段对照表

| 原始数据字段 | 数据类型 | 训练样本字段 | 数据类型 | 转换说明 |
|----------|---------|-----------|---------|---------|
| `episode_index` | int | - | - | 用作样本ID (不直接传入模型) |
| `tasks[0]` | str | `input_ids` | LongTensor | 与`<video>`占位符一起Token化 |
| `videos/xxx.mp4` | MP4 | `pixel_values_videos` | FloatTensor | 解码→采样→预处理→张量化 |
| - | - | `video_grid_thw` | LongTensor | 视频处理器的输出, 记录token分布 |
| `pose.*` | float32[4,4] | - | - | 用于环境理解 (不直接输入) |
| `goal.*` | int32[2] | `labels` | LongTensor | 通过动作序列编码为文本 |
| `action` | int32[46] | `labels` | LongTensor | 转换为文本响应并编码 |
| `length` | int | - | - | 确定视频采样范围 |

### 3.3 数据量与内存估算

#### 原始R2R数据

```
单个Episode:
  - 视频 (46帧, 512×512×3): ~70 MB (未压缩MP4: ~2-5 MB)
  - Parquet数据: ~1-2 MB
  - 元数据: ~1 KB
  ─────────────────────────
  小计: 1个episode ≈ 3-10 MB

全体数据集:
  - R2R: 104,870 episodes × 5 MB ≈ 500 GB
```

#### 转换后训练样本

```
单个样本 (纯文本编码):
  - input_ids: [seq_len=1500] × 4字节 ≈ 6 KB
  - labels: [seq_len=1500] × 4字节 ≈ 6 KB
  ─────────────────────────
  小计: ~12 KB (纯参考, 实际运行时在GPU加载视频)

训练时 (Batch size=16):
  - 纯文本tokens: 16 × 12 KB ≈ 192 KB
  - 视频特征缓存: 16 × 4帧 × 256×256×3 ≈ 1.5 GB (GPU内存)
  - 注意力权重: ~2-3 GB
  ─────────────────────────
  总计GPU占用: ~4-5 GB (使用ZeRO-2)
```

---

## 第四部分: 数据流可视化

```
┌─────────────────────────────────────────────────────────────────┐
│                    原始R2R数据集                                  │
│ (LeRobot格式: MP4视频 + Parquet + JSONL)                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              数据加载 (LazySupervisedDataset)                    │
│  ├─ 读取annotations (JSONL/JSON)                               │
│  ├─ 应用采样率 (seed=42固定)                                    │
│  └─ 返回list_data_dict (列表)                                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│            DataLoader采样 (__getitem__)                         │
│  ├─ 选择样本索引i                                                │
│  └─ 调用_get_item(i)                                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌──────────┐  ┌──────────┐  ┌──────────────┐
│ 视频处理  │  │ 文本处理  │  │ 低维数据处理  │
├──────────┤  ├──────────┤  ├──────────────┤
│• 读MP4   │  │• 加载指令 │  │• 读Parquet   │
│• 提取帧  │  │• 格式化为 │  │• 提取动作    │
│• 均匀采样│  │  对话    │  │  序列        │
│• 预处理  │  │• Token化 │  │• 转文本响应  │
│  (resize)│  │• 生成标签 │  │              │
└──────────┘  └──────────┘  └──────────────┘
     │               │               │
     └───────────────┼───────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│         最终训练样本组装                                         │
└─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│         返回字典                                                  │
├─────────────────────────────────────────────────────────────────┤
│ {                                                               │
│   "input_ids": LongTensor[seq_len],                            │
│   "labels": LongTensor[seq_len],                               │
│   "pixel_values_videos": FloatTensor[1, T, 3, H, W],           │
│   "video_grid_thw": LongTensor[T, H_t, W_t],                   │
│   "second_per_grid_ts": List[T],                               │
│   "is_history_image": bool                                     │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│         Collate到Batch (batch_size=16)                          │
│  ├─ Pad sequences到最长长度                                     │
│  ├─ 堆叠视频张量                                                │
│  └─ 返回批量数据                                                │
└─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│         输入Qwen3-VL-2B模型进行前向传播                         │
│  ├─ 视觉编码器: 处理video tokens                               │
│  ├─ 语言模型: 处理input_ids                                    │
│  └─ 计算loss (仅在labels!=-100处)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第五部分: 配置示例

### 5.1 训练数据配置

```python
# 在train_compressor_baseline_2b.sh中
vln_datasets="r2r_125cm_0_30,rxr_125cm_0_30%50"

# 含义:
# - r2r_125cm_0_30: R2R全量 (104,870个episode)
# - rxr_125cm_0_30%50: RxR的50%采样 (266,908×0.5 ≈ 133,454个episode)
# - 总计: ~238,324个训练样本
```

### 5.2 视频处理参数

```python
# 参数值
base_interval: 4              # 帧采样间隔(秒)
video_min_frames: 4           # 最小采样帧数
video_max_frames: 8           # 最大采样帧数
video_max_frame_pixels: 1664  # 每帧最大像素数
video_min_frame_pixels: 256   # 每帧最小像素数

# 示例:
# 46帧@30FPS视频 (1.53秒) → interval=4秒
# num_frames = 1.53/4 ≈ 0.4 → clip(0.4, 4, 8) = 4
# 结果: 均匀采样4帧
```

### 5.3 损失函数掩码

```python
# 在Trainer中使用标准的CrossEntropyLoss
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
    ignore_index=-100  # 忽略被标记为-100的位置
)

# 只计算梯度的部分:
# - 用户消息: -100 (不计算)
# - 系统消息: -100 (不计算)
# - 助手响应: token_id (计算损失, 反向传播)
```

---

## 第六部分: 常见问题

### Q1: 为什么要采样视频帧而不使用所有帧?

**A:** 
- 推理效率: 降低计算复杂度 (4帧 vs 46帧)
- 内存约束: GPU内存有限 (batch_size需要平衡)
- 信息充分性: 4-8帧通常足以捕捉关键视觉变化
- 时间对齐: 固定帧数便于批处理

### Q2: 为什么使用IGNORE_INDEX=-100?

**A:**
- HuggingFace标准: PyTorch的CrossEntropyLoss天然支持
- 清晰语义: 显式标记哪些位置不计算损失
- 梯度控制: 防止模型学习系统/用户提示的生成
- 提升效率: 只计算有效输出的损失

### Q3: 多视点(125cm_0deg vs 125cm_30deg)如何选择?

**A:**
- 当前实现: 支持多个配置但在训练时二选一
- 高度: 125cm更接近实际机器人视角
- 俯仰角: 0度=水平, 30度=俯视
  - 0度: 适合行走时的前向视野
  - 30度: 增加垂直视野, 有助于理解房间布局
- 建议: 0度用于导航, 30度用于目标识别

### Q4: 如何处理不同长度的episode?

**A:**
```python
# 采样帧数根据视频时长自动调整
target_frames = clip(video_length / interval, min_frames, max_frames)

# 因此所有样本的帧数不一定相同(4-8帧)
# 在Batch处理时需要padding/masking
```

### Q5: 为什么要固定seed=42?

**A:**
- 分布式训练: 多个进程采样相同的数据子集
- 复现性: 相同的seed产生相同的采样
- 公平比较: 不同实验基于相同的数据分割

---

## 总结

| 阶段 | 输入 | 输出 | 关键操作 |
|------|------|------|---------|
| **原始数据** | LeRobot格式 | MP4, Parquet, JSONL | - |
| **数据加载** | 配置+采样率 | list_data_dict | 读取+采样 |
| **样本获取** | 数据索引 | 多模态样本 | 视频解码+处理 |
| **文本处理** | 指令+动作 | token序列 | Token化+标签生成 |
| **批处理** | 样本列表 | Batch张量 | Padding+堆叠 |
| **训练** | Batch张量 | 损失值 | 前向传播 |

整个转换过程的核心是: **将结构化的多模态导航数据转换为标准的language model + vision encoder输入格式，使得现成的VLM架构可以直接应用于导航任务学习。**

---

# 附录: 原始数据与转换后数据的具体样本对比

## 附录A: 原始R2R数据样本

### A.1 场景与Episode基本信息

#### 原始目录结构
```
traj_data/r2r/17DRP5sb8fy/
├── meta/
│   └── episodes.jsonl           ← 包含导航任务元数据
├── data/chunk-000/
│   ├── episode_000000.parquet   ← 第一个任务的低维数据
│   ├── episode_000001.parquet
│   └── ... (75个parquet文件)
└── videos/chunk-000/
    ├── 125cm_0deg/
    │   ├── episode_000000.mp4   ← 第一个任务的视频
    │   ├── episode_000001.mp4
    │   └── ...
    └── 125cm_30deg/
        └── ... (另一视点视频)
```

### A.2 原始数据样本 (episodes.jsonl)

#### 样本1: 简单任务

```json
{
  "episode_index": 0,
  "tasks": [
    "Exit the bedroom, enter the bathroom, wait at the toilet. "
  ],
  "length": 46
}
```

**字段说明:**
- `episode_index`: 0 - 全局唯一ID
- `tasks`: 包含1条自然语言指令
- `length`: 46 - 视频包含46帧 (@30FPS = 1.53秒)

#### 样本2: 复杂多步任务

```json
{
  "episode_index": 1,
  "tasks": [
    "Walk out of the dining area and walk straight into the bedroom that's past the living room. When in the bedroom take a left into the sitting area in the bedroom. Wait in the sitting area. "
  ],
  "length": 54
}
```

**特点:**
- 长指令: 包含3个步骤 (出餐厅→进卧室→转向坐区→等待)
- 更长的episode: 54帧 (@30FPS = 1.80秒)
- 更复杂的空间推理: 多个房间和转向

#### 样本3: 最短任务

```json
{
  "episode_index": 2,
  "tasks": [
    "Walk forward into the bathroom. Wait near the sink. "
  ],
  "length": 28
}
```

**特点:**
- 简短指令: 仅2个步骤
- 最短episode: 28帧 (@30FPS = 0.93秒)
- 简单目标: 单房间导航

### A.3 低维Parquet数据样本

#### Parquet数据列结构

文件: `episode_000000.parquet` (46行, 多列)

| 行号 | action | pose.125cm_0deg | goal.125cm_0deg | pose.125cm_30deg | goal.125cm_30deg | ... |
|------|--------|-----------------|-----------------|------------------|------------------|-----|
| 0 | 1 | [[1,0,0,x₀], [0,1,0,y₀], ...] | [256, 384] | [[1,0,0,x₀'], ...] | [512, 256] | ... |
| 1 | 1 | [[1,0,0,x₁], [0,1,0,y₁], ...] | [264, 390] | [[1,0,0,x₁'], ...] | [514, 258] | ... |
| 2 | 1 | [[1,0,0,x₂], [0,1,0,y₂], ...] | [272, 396] | [[1,0,0,x₂'], ...] | [516, 260] | ... |
| ... | ... | ... | ... | ... | ... | ... |
| 45 | 0 | [[1,0,0,x₄₅], [0,1,0,y₄₅], ...] | [-1, -1] | [[1,0,0,x₄₅'], ...] | [-1, -1] | ... |

#### 数据列详解

**动作列 (action):**
```
[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, ..., 0]
 ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑      ↑
 前进 前进 前进 ... 左转   右转     ... STOP

动作编码:
  0 = STOP (停止)
  1 = ↑ FORWARD (前进)
  2 = ← LEFT (左转)
  3 = → RIGHT (右转)
  5 = ↓ BACKWARD (后退)
```

**位姿列 (pose.125cm_0deg):**

每一行包含 [T0, T1, T2, T3] 四个向量，形成4×4变换矩阵:

```
第一帧的位姿矩阵:
[
  [1.0,  0.0,  0.0,  -4.523],   ← x轴方向 + x位移
  [0.0,  1.0,  0.0,  -3.712],   ← y轴方向 + y位移
  [0.0,  0.0,  1.0,   1.250],   ← z轴方向 + z位移(高度=125cm)
  [0.0,  0.0,  0.0,   1.0  ]    ← 齐次坐标
]

表示: 机器人在三维世界中的位置(-4.523, -3.712, 1.25)和方向(朝向)
```

**目标点列 (goal.125cm_0deg):**

```
帧索引    目标点坐标    含义
0        [256, 384]   → 在视觉帧(512×512)中的像素坐标
1        [264, 390]   → 相同场景, 目标点随机器人移动而变化
2        [272, 396]   
...
44       [512, 512]   → 接近帧的角落 (接近目标)
45       [-1, -1]     → 特殊值表示: 任务完成(STOP)
```

#### 多视点数据示例

同一episode在不同视点的目标点差异:

```
帧0:
  125cm_0deg 视点:   目标点 = [256, 384]
  125cm_30deg 视点:  目标点 = [512, 256]  ← 俯视角度下的不同位置

解释: 由于摄像头高度和俯仰角不同,
      同一个3D场景中的目标在2D图像上的投影位置不同
```

### A.4 视频数据样本

#### 视频元数据
```
文件: episode_000000.mp4
分辨率: 512 × 512 像素 (RGB)
帧率: 30 FPS
总帧数: 46 帧
时长: 46/30 = 1.533 秒
大小: ~2-5 MB (MP4压缩)
```

#### 视频内容对应关系

```
第0帧    → 机器人在卧室门口,看向浴室方向
第1帧    → 机器人向前移动一步,继续看浴室
第2帧    → 继续前进...
...
第22帧   → 机器人转向左侧(动作=2)
第23帧   → 继续转向...
...
第46帧   → 机器人停止在目标位置(动作=0)
```

---

## 附录B: 转换后的训练样本

### B.1 单个训练样本的完整结构

#### 原始导航任务
```json
{
  "episode_index": 0,
  "tasks": ["Exit the bedroom, enter the bathroom, wait at the toilet."],
  "length": 46,
  "data_path": "traj_data/r2r"
}
```

#### 转换第一步: 数据加载

```python
# 加载后的中间形式
{
  "episode_index": 0,
  "tasks": ["Exit the bedroom, enter the bathroom, wait at the toilet."],
  "length": 46,
  "data_path": "traj_data/r2r",
  "video": "17DRP5sb8fy/videos/chunk-000/125cm_0deg/episode_000000.mp4",
  "parquet_data": {
    "actions": [1, 1, 1, ..., 0],  # 46个动作
    "poses": [[...4×4矩阵...], ...],  # 46个位姿
    "goals": [[256, 384], [264, 390], ..., [-1, -1]]  # 46个目标点
  }
}
```

#### 转换第二步: 视频处理

```python
# 视频提取和采样
{
  "video_file": "traj_data/r2r/17DRP5sb8fy/videos/chunk-000/125cm_0deg/episode_000000.mp4",
  
  # 原始视频信息
  "original_frames": 46,
  "original_duration": 1.533,  # 秒
  "original_fps": 30,
  
  # 采样计算
  "base_interval": 4,  # 秒
  "num_frames_to_sample": round(1.533 / 4) = 0,  # 四舍五入后
  "clipped_frames": clip(0, min=4, max=8) = 4,
  
  # 均匀采样的帧索引
  "sampled_frame_indices": [0, 15, 31, 45],  # linspace(0, 45, 4)
  "sampled_frames": [frame_0, frame_15, frame_31, frame_45]  # 4个视觉帧
}
```

**视频帧采样示例:**

```
原始46帧时间轴:
|---0---→ 15 ---→ 31 ---→ 45---|
帧数:  0 1 2...14 15 16...30 31...45

采样后4帧:
帧0   (0/30 = 0.00秒) → 卧室门口的初始视角
帧15  (15/30 = 0.50秒) → 机器人已进入卧室
帧31  (31/30 = 1.03秒) → 机器人在转向坐区
帧45  (45/30 = 1.50秒) → 机器人接近目标位置
```

#### 转换第三步: 文本处理与对话构建

```python
# 构建多轮对话
conversations = [
    {
        "from": "human",
        "value": "<video>\nExit the bedroom, enter the bathroom, wait at the toilet."
    },
    {
        "from": "gpt",
        "value": "↑ ↑ ↑ ↑ ↑ ↑ ↑ ← ← ← → → → ↑ ↑ ↑ ... ⊗"
        # 将46个动作转换为易读的符号序列
        # ⊗ 表示 STOP
    }
]
```

**动作转文本示例:**

```
原始动作序列: [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, ..., 0]

转换步骤:
  1 → "↑" (forward)
  2 → "←" (left turn)
  3 → "→" (right turn)
  0 → "⊗" (stop)

转换结果: "↑ ↑ ↑ ↑ ↑ ↑ ↑ ← ← ← → → → ↑ ↑ ↑ ... ⊗"

Qwen3-VL的对话格式:
{
  "role": "gpt",
  "value": "↑ ↑ ↑ ↑ ↑ ↑ ↑ ← ← ← → → → ↑ ↑ ↑ ... ⊗"
}
```

#### 转换第四步: Token化与标签生成

```python
# Qwen3-VL的Tokenizer处理

# 原始对话文本
input_text = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<video>
Exit the bedroom, enter the bathroom, wait at the toilet.<|im_end|>
<|im_start|>assistant
↑ ↑ ↑ ↑ ↑ ↑ ↑ ← ← ← → → → ↑ ↑ ↑ ... ⊗<|im_end|>"""

# Token化后 (演示)
input_ids = [
    # 系统消息部分
    153152, 17953,  # <|im_start|>, system
    7258, 368, 263, ...  # "You are a helpful assistant."
    151643,  # <|im_end|>
    
    # 用户消息部分
    153152, 17894,  # <|im_start|>, user
    151655,  # VIDEO_TOKEN (151655)
    # (此处是video_pad token的占位符, 总共256个)
    10984, 4395, ...  # "Exit the bedroom, enter the bathroom, wait at the toilet."
    151643,  # <|im_end|>
    
    # 助手响应部分
    153152, 27406,  # <|im_start|>, assistant
    26855, 26855, 26855, ...  # "↑ ↑ ↑ ↑ ↑ ↑ ↑"
    6155, 6155, 6155, ...  # "← ← ←"
    ...
    151643   # <|im_end|>
]

# 标签生成 (IGNORE_INDEX = -100)
labels = [
    # 系统消息: 不计算损失
    -100, -100, -100, -100, ..., -100,  # 系统消息部分都是-100
    
    # 用户消息: 不计算损失
    -100, -100, -100, -100, -100, ..., -100,  # 用户消息部分都是-100
    
    # 助手响应: 计算损失 (真实token ID)
    26855, 26855, 26855, ...,  # "↑ ↑ ↑"
    6155, 6155, 6155, ...,  # "← ← ←"
    ...
    151643   # <|im_end|>
]

说明:
  - 系统和用户部分都是-100: 模型不学习生成这些部分
  - 助手响应是token ID: 模型学习预测这些token, 用于导航决策
  - sequence_length = len(input_ids) ≈ 1500-2000 tokens
```

### B.2 视觉特征处理

#### 视频到视觉Token的转换

```python
# 第1步: 原始视频帧
frame_0.shape = (512, 512, 3)  # RGB图像
frame_15.shape = (512, 512, 3)
frame_31.shape = (512, 512, 3)
frame_45.shape = (512, 512, 3)

# 第2步: 图像处理器的预处理
processor.preprocess(videos=[frame_0, frame_15, frame_31, frame_45])

# 输出1: 像素值张量
pixel_values_videos.shape = [1, 4, 3, 1664, 1664]
                            ↑  ↑  ↑  ↑     ↑
                          batch T C height width

# 输出2: 视觉Token网格 (video_grid_thw)
video_grid_thw = [
    (4, 26, 26),     # frame_0: 4个时间patch, 26×26空间patch
    (4, 26, 26),     # frame_15
    (4, 26, 26),     # frame_31
    (4, 26, 26)      # frame_45
]
# 总视觉token数 = 4 * 26 * 26 = 2704

# 输出3: 时间跨度
second_per_grid_ts = [
    0.375, 0.375, 0.375, 0.375  # 每个时间patch对应0.375秒
]

# 第3步: 在input_ids中的表示
# <|vision_start|> + <|video_pad|> × 2704 + <|vision_end|>
video_placeholder = [
    151655,              # <|vision_start|>
    151656, 151656, ..., # <|video_pad|> × 2704个
    151657               # <|vision_end|>
]
# 占用约2706个token位置
```

### B.3 最终训练样本的完整结构

```python
# 返回给模型的样本字典
training_sample = {
    # 文本Token相关
    "input_ids": LongTensor[
        -100处理后的长度序列, 通常 ~1500-2000 tokens
    ],
    "labels": LongTensor[
        -100处理后的标签序列, 长度与input_ids相同
    ],
    
    # 视觉特征相关
    "pixel_values_videos": FloatTensor[
        1,      # batch_size=1
        4,      # 采样的帧数
        3,      # RGB通道
        1664,   # 处理后的高度
        1664    # 处理后的宽度
    ],
    
    "video_grid_thw": LongTensor[
        [4, 26, 26],  # frame 0的token网格
        [4, 26, 26],  # frame 15
        [4, 26, 26],  # frame 31
        [4, 26, 26]   # frame 45
    ],
    
    "second_per_grid_ts": List[
        0.375, 0.375, 0.375, 0.375
    ],
    
    # 元数据
    "is_history_image": False,  # 这是当前视频, 非历史图像
}
```

---

## 附录C: 两种格式数据的详细对比

### C.1 数据维度对比

| 维度 | 原始R2R数据 | 转换后训练样本 |
|------|-----------|-------------|
| **时间维度** | 46帧完整视频 | 4帧均匀采样 |
| **空间分辨率** | 512×512像素 | 1664×1664像素(预处理后) |
| **动作表示** | int32 [1,2,3,5,0] | 文本符号 "↑←→⊗" |
| **位姿数据** | 4×4矩阵46组 | 隐含于视觉token中 |
| **目标点坐标** | 像素坐标 [x,y] | 转换为文本响应 |
| **文本指令** | 自然语言 | Token序列(已编码) |
| **存储格式** | Parquet+MP4 | PyTorch张量 |
| **文件大小** | ~5 MB | ~12 KB (纯编码), 实际加载时~1.5GB/batch |

### C.2 数据流对应关系

```
原始数据 ────────────────→ 训练样本

Episode 0 metadata (JSONL)
├─ episode_index: 0
├─ tasks[0]: "Exit the bedroom..."
└─ length: 46
        ↓
      [加载]
        ↓
list_data_dict[0] = {
    "episode_index": 0,
    "tasks": ["Exit the bedroom..."],
    "length": 46,
    "data_path": "traj_data/r2r"
}
        ↓
      [__getitem__(0)]
        ↓
┌─────────────────────────────────┐
│ 视频处理分支 (Parquet分支)      │
├─────────────────────────────────┤
│ episode_000000.mp4              │
│   (46帧 @ 30FPS)                │
│      ↓                          │
│   提取帧 (Decord)               │
│   [frame_0...frame_45]          │
│      ↓                          │
│   均匀采样: linspace(0,45,4)    │
│   [frame_0, frame_15,           │
│    frame_31, frame_45]          │
│      ↓                          │
│   预处理 (Resize)               │
│   pixel_values_videos           │
│   [1, 4, 3, 1664, 1664]        │
│      ↓                          │
│   提取token网格                 │
│   video_grid_thw = [[4,26,26]...│
│                                  │
│ 输出:                            │
│  - pixel_values_videos          │
│  - video_grid_thw               │
│  - second_per_grid_ts           │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│ 文本处理分支                     │
├─────────────────────────────────┤
│ 任务指令:                        │
│ "Exit the bedroom,              │
│  enter the bathroom,            │
│  wait at the toilet."           │
│      ↓                          │
│ 读Parquet数据:                  │
│ actions = [1,1,1,...,0]         │
│      ↓                          │
│ 转换为符号:                      │
│ "↑ ↑ ↑ ... ⊗"                  │
│      ↓                          │
│ 构建对话:                        │
│ user: "<video>\n指令"          │
│ assistant: "↑ ↑ ... ⊗"          │
│      ↓                          │
│ 替换视频占位符:                  │
│ "<video>" → "<|vision_start|>..│
│      ↓                          │
│ Token化:                        │
│ input_ids = [tokens...]         │
│ labels = [-100, ..., token_id]  │
│                                  │
│ 输出:                            │
│  - input_ids                    │
│  - labels                       │
└─────────────────────────────────┘
        ↓
    [合并输出]
        ↓
training_sample = {
    "input_ids": LongTensor[~1500],
    "labels": LongTensor[~1500],
    "pixel_values_videos": FloatTensor[1,4,3,1664,1664],
    "video_grid_thw": LongTensor[4,2,2],
    "second_per_grid_ts": List[4],
    "is_history_image": False
}
```

### C.3 具体数值示例对比

#### Episode 0 的完整转换过程

**原始数据:**
```
Video: episode_000000.mp4
- 帧数: 46
- 时长: 1.533秒
- 分辨率: 512×512

Parquet: episode_000000.parquet
- action: [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, ...]
- goal: [[256,384], [264,390], [272,396], ..., [-1,-1]]
- pose: [4×4矩阵, ...] (46个)

Metadata:
- instruction: "Exit the bedroom, enter the bathroom, wait at the toilet."
- length: 46
```

**转换过程:**

```
步骤1: 视频采样
  计算: num_frames = round(1.533 / 4) = 0 → clip(0, 4, 8) = 4
  采样帧索引: [0, 15, 31, 45]
  采样结果: 4帧

步骤2: 文本处理
  原始动作: [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, ..., 0]
  转换后:   "↑ ↑ ↑ ↑ ↑ ↑ ↑ ← ← ← → → → ↑ ↑ ↑ ... ⊗"
  
  指令: "Exit the bedroom, enter the bathroom, wait at the toilet."

步骤3: 构建对话
  消息1(系统): "You are a helpful assistant."
  消息2(用户): "<video>\nExit the bedroom, enter the bathroom, wait at the toilet."
  消息3(助手): "↑ ↑ ↑ ↑ ↑ ↑ ↑ ← ← ← → → → ↑ ↑ ↑ ... ⊗"

步骤4: Token化
  系统消息tokens: [153152, 17953, 7258, 368, ..., 151643]  (长度~20)
  用户消息tokens: [153152, 17894, 151655, ..., 151643]    (长度~250)
  助手响应tokens: [153152, 27406, 26855, ..., 151643]    (长度~80)
  总长度: ~350 tokens (加padding到统一长度, 通常1024或2048)

步骤5: 生成标签
  系统消息标签:   [-100, -100, -100, ...]      (不计算损失)
  用户消息标签:   [-100, -100, -100, ...]      (不计算损失)
  助手响应标签:   [token_id, token_id, ...]    (计算损失)
```

**最终样本:**
```python
{
    "input_ids": LongTensor[1024]           # 右填充到1024
    "labels": LongTensor[1024]              # 同样大小
    "pixel_values_videos": FloatTensor[1, 4, 3, 1664, 1664]
    "video_grid_thw": LongTensor[4, 26, 26]
    "second_per_grid_ts": [0.375, 0.375, 0.375, 0.375]
    "is_history_image": False
}
```

---

## 附录D: 多个样本的批处理

### D.1 Batch内多样性

假设Batch包含4个样本:

```python
batch = {
    "input_ids": LongTensor[4, 1024],              # 4个样本
    "labels": LongTensor[4, 1024],
    "pixel_values_videos": FloatTensor[4, 4, 3, 1664, 1664],
    "video_grid_thw": LongTensor[4, 4, 2, 2],
    "second_per_grid_ts": List[List[4]],
    "is_history_image": BoolTensor[4]
}

# 多样性来源:
# - 样本0: 简单任务 (长度28帧) → 采样4帧
# - 样本1: 中等任务 (长度46帧) → 采样4帧
# - 样本2: 复杂任务 (长度54帧) → 采样4帧
# - 样本3: 长任务 (长度80帧) → 采样4帧
#
# 不同的:
# - 不同场景 (场景ID不同)
# - 不同时间长度 (但都采样4帧)
# - 不同指令复杂度
# - 不同动作序列
```

### D.2 GPU内存占用

```
Batch size = 16

文本tokens部分:
  input_ids: 16 × 1024 × 4字节 = 64 KB
  labels: 16 × 1024 × 4字节 = 64 KB
  小计: ~128 KB

视觉特征部分:
  pixel_values_videos: 16 × 4 × 3 × 1664 × 1664 × 4字节
                      = 16 × 4 × 3 × 2.77M × 4
                      = ~1.7 GB
  video_grid_thw: 16 × 4 × 26 × 26 × 8字节 = ~54 MB
  小计: ~1.75 GB

模型激活值 (ZeRO-2):
  ~2-3 GB (取决于中间特征图)

总GPU占用: ~4-5 GB (H100的22%)
```

---

## 附录E: 关键转换公式与参数

### E.1 视频采样公式

```
input_frames = 46              # 原始帧数
fps = 30                       # 帧率
video_length = input_frames / fps = 1.533秒

base_interval = 4秒            # 采样间隔
num_frames_candidate = round(video_length / base_interval)
                     = round(1.533 / 4)
                     = round(0.383)
                     = 0

target_frames = clip(num_frames_candidate, min=4, max=8)
              = clip(0, 4, 8)
              = 4

frame_indices = linspace(0, input_frames-1, target_frames, dtype=int)
              = linspace(0, 45, 4)
              = [0, 15, 31, 45]
```

### E.2 Token网格计算

```
输入分辨率: 1664 × 1664像素
patch_size: 14 × 14像素

num_patches_h = 1664 / 14 = 118.86 → 119 (四舍五入)
num_patches_w = 1664 / 14 = 118.86 → 119

实际使用: 假设 Qwen3VL 内部处理为 128 × 128 patch grid (符合标准)
或更可能是处理为多尺度: [(64, 64), (128, 128), ...]

输出grid_thw: [T=4, H=26, W=26]  # 实际值可能不同
总token数: 4 × 26 × 26 = 2704
```

### E.3 文本长度计算

```
系统消息: "You are a helpful assistant."
  tokens ≈ 10-15个

用户消息: "<video>\nExit the bedroom, enter the bathroom, wait at the toilet."
  <video>占位符: 1 token
  <|vision_start|>: 1 token
  video_pad × 2704: 2704 tokens
  <|vision_end|>: 1 token
  指令文本: "Exit..." ≈ 20-25个tokens
  小计: ~2730 tokens

助手响应: "↑ ↑ ↑ ↑ ↑ ↑ ↑ ← ← ← → → → ↑ ↑ ↑ ... ⊗"
  46个动作 × 1.5 tokens/动作 ≈ 70 tokens
  (包括空格token)

总长度: ~2810 tokens
填充到: 2048或4096 (标准值)
```

---

## 附录F: 数据转换检查清单

### 转换质量检查

在使用转换后的数据进行训练前，应验证：

| 检查项 | 原始数据 | 转换后数据 | 验证方法 |
|-------|---------|----------|---------|
| **样本数量** | R2R: 104,870 | R2R: 104,870 (无损失) | `len(dataset)` |
| **视频完整性** | 46帧可读 | 4帧可加载 | 能否成功调用`__getitem__` |
| **动作有效性** | action ∈ {0,1,2,3,5} | token ∈ vocab | 检查labels中的token值 |
| **目标点有效性** | goal ∈ [0,512]×[0,512] | 隐含于响应 | 检查原始parquet未损坏 |
| **指令完整性** | tasks[0]非空 | input_ids非空 | 检查input_ids长度 > 100 |
| **Token化正确** | 原始文本 | input_ids + labels | 验证解码后的文本与原始一致 |
| **掩码正确** | - | labels中-100比例 ~60-70% | 统计labels为-100的比例 |
| **GPU可加载** | N/A | 张量可上传GPU | 不报CUDA OOM错误 |

