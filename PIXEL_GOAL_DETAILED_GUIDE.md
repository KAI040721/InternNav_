# 像素目标点坐标数据详细指南

## 概述

本文档深入介绍R2R数据集中**像素目标点坐标 (Pixel Goal Coordinates)** 的生成方式、数据结构、处理管道和实际应用，并通过具体例子说明原始LeRobot格式与最终训练样本格式的区别。

---

## 第一部分: 像素目标点的物理意义

### 1.1 什么是像素目标点？

在视觉导航任务中，机器人需要理解：
1. **当前视角**: 摄像头看到的第一人称图像
2. **目标位置**: 下一步应该去往何处

**像素目标点**就是在当前视角图像上**标注出目标位置的像素坐标 (x, y)**。

```
摄像头视图 (512×512图像):
┌─────────────────────────────────────┐
│                                     │
│     走廊               目标房间      │
│     ░░░░░░░░░░░░░░░░░░░░░░░░░░     │
│     ░░          ◎ ← 目标点坐标     │
│     ░░          (320, 280)         │
│     ░░░░░░░░░░░░░░░░░░░░░░░░░░     │
│                                     │
│  🤖 机器人位置                       │
│  (图像下方中心)                     │
│                                     │
└─────────────────────────────────────┘

坐标含义:
- x = 320: 从左到右, 距离左边界320像素
- y = 280: 从上到下, 距离上边界280像素
```

### 1.2 为什么需要像素目标点？

**传统方法的问题:**
- ❌ 仅有动作序列 (前进/左转/右转): 无法学习**看到哪里就去哪里**的能力
- ❌ 仅有3D世界坐标: 需要额外的视觉理解模块进行转换

**像素目标点的优势:**
- ✅ **端到端可学习**: 直接从图像→目标点坐标
- ✅ **视觉一致性**: 坐标在同一个摄像头视角下定义
- ✅ **易于监督学习**: 图像+坐标对，自然的视觉-语言对齐
- ✅ **泛化能力强**: 跨越不同场景和照明条件

### 1.3 数据生成流程概览

```
原始3D导航轨迹
    ↓
视频渲染(多视点)
    ↓
视觉特征提取 + 目标点投影
    ↓
像素坐标计算 + 验证
    ↓
存储为Parquet列
    ↓
训练数据加载 + 处理
```

---

## 第二部分: 原始LeRobot格式的目标点数据

### 2.1 数据存储位置和格式

```
文件路径:
traj_data/r2r/<scene_id>/data/chunk-000/episode_000000.parquet

Parquet列结构:
┌─────────────────────────────────────────────┐
│ Column Name          | Type    | Shape       │
├─────────────────────────────────────────────┤
│ action               | int32   | [T, 1]      │
│ pose.125cm_0deg      | float32 | [T, 4, 4]   │
│ goal.125cm_0deg      | int32   | [T, 2]      │
│ relative_goal_frame  | int32   | [T, 1]      │
│   _id.125cm_0deg     |         |             │
│ pose.125cm_30deg     | float32 | [T, 4, 4]   │
│ goal.125cm_30deg     | int32   | [T, 2]      │
│ relative_goal_frame  | int32   | [T, 1]      │
│   _id.125cm_30deg    |         |             │
└─────────────────────────────────────────────┘

其中 T = 46 (episode帧数)
```

### 2.2 目标点数据的具体含义

#### 字段 1: `goal.{setting}` → 像素坐标

```
类型: int32
形状: [T, 2]  表示 T帧 × 2个坐标值 (x, y)

例子 (episode_000000的前3帧):
frame_0: [320, 240]    # 第0帧的目标点在 x=320, y=240
frame_1: [325, 245]    # 第1帧的目标点在 x=325, y=245 (目标移动了)
frame_2: [330, 250]    # 第2帧的目标点在 x=330, y=250

含义解释:
- [320, 240]: 在512×512的图像中, 目标在(320, 240)像素位置
- 值域: x ∈ [0, 512), y ∈ [0, 512)
- 特殊值: [-1, -1] 表示当前帧无有效目标 (见2.3)
```

#### 字段 2: `relative_goal_frame_id.{setting}` → 路径长度

```
类型: int32
形状: [T, 1]

例子 (同上episode的前3帧):
frame_0: [10]    # 从当前帧到目标需要10帧 (约0.33秒@30FPS)
frame_1: [9]     # 从当前帧到目标需要9帧
frame_2: [8]     # 从当前帧到目标需要8帧

含义解释:
- 表示当前帧距离目标的"时间距离" (以帧数为单位)
- 可用于学习**距离感知**
- 路径长度计算时使用

特殊值:
- -1: 表示当前帧无有效目标
- 0: 表示已到达目标
```

### 2.3 特殊标记: -1 代表无效目标

在某些情况下，当前帧**没有有效的像素目标点**：

```
场景 1: 接近目标
┌─────────────────────────────────┐
│ frame_40: goal=[-1, -1]         │
│           relative_id=[-1]      │
│                                 │
│ 原因: 已非常接近目标,           │
│      像素空间中无法准确定位     │
└─────────────────────────────────┘

场景 2: 目标超出视野
┌─────────────────────────────────┐
│ frame_20: goal=[-1, -1]         │
│           relative_id=[-1]      │
│                                 │
│ 原因: 目标在身后或侧方,         │
│      当前视角看不到            │
└─────────────────────────────────┘

场景 3: 需要转向
┌─────────────────────────────────┐
│ frame_15: goal=[-1, -1]         │
│           relative_id=[-1]      │
│                                 │
│ 原因: 机器人需要先转向,         │
│      然后才能看到目标          │
└─────────────────────────────────┘
```

### 2.4 多视点目标点的区别

LeRobot数据包含**多个视点**下的目标点：

```
视点1: 125cm_0deg (水平视角)
┌──────────────────────────────────────┐
│ goal.125cm_0deg = [320, 240]         │
│ 特点: 前向视野清晰, 适合导航        │
└──────────────────────────────────────┘
        ↓
    512×512 图像
        ↓
 ◎ 在(320, 240)的目标点


视点2: 125cm_30deg (俯视角, 30°向下)
┌──────────────────────────────────────┐
│ goal.125cm_30deg = [350, 280]        │
│ 特点: 看到更多地面, 便于识别障碍    │
└──────────────────────────────────────┘
        ↓
    512×512 图像
        ↓
 ◎ 在(350, 280)的目标点
```

**为什么坐标不同?**
- 不同的俯仰角导致图像内容不同
- 同一个3D世界点在两个视点中的投影位置不同
- 两套坐标都有效, 取决于训练时选择哪个视点

---

## 第三部分: 目标点数据的处理与转换

### 3.1 原始数据加载 (代码行780-810)

```python
# 1. 从Parquet文件读取原始数据
import pyarrow.parquet as pq

parquet_path = "traj_data/r2r/S9hNv5qa7GM/data/chunk-000/episode_000000.parquet"
table = pq.read_table(parquet_path)
df = table.to_pandas()

# DataFrame 结构:
# 行数: 46 (episode长度)
# 列: action, pose.125cm_0deg, goal.125cm_0deg, relative_goal_frame_id.125cm_0deg, ...

# 2. 提取目标点列
goal_key = "goal.125cm_0deg"          # 选择125cm水平视角
relative_goal_id_key = "relative_goal_frame_id.125cm_0deg"

goal_column = df[goal_key]            # 类型: numpy array, 形状 [46, 2]
relative_id_column = df[relative_goal_id_key]  # 形状 [46, 1]

# 3. 转换为列表格式用于后续处理
ep_pixel_goals = [
    [df[relative_goal_id_key][idx].tolist(),    # 路径长度 [-1] 或 [10]
     df[goal_key][idx].tolist()]                 # 像素坐标 [-1, -1] 或 [320, 240]
    for idx in range(len(df))
]

# ep_pixel_goals 结构:
# [
#   [[10], [320, 240]],      # frame_0
#   [[9], [325, 245]],       # frame_1
#   [[8], [330, 250]],       # frame_2
#   ...
#   [[-1], [-1, -1]],        # frame_45 (无有效目标)
# ]
```

### 3.2 数据验证与质量检查

```python
def validate_pixel_goal(pixel_goal, frame_id, max_resolution=512):
    """验证像素目标点的有效性"""
    
    relative_goal_id, coord = pixel_goal
    relative_goal_id = relative_goal_id[0]  # 取标量值
    x, y = coord[0], coord[1]
    
    # 检查 1: -1 标记 (无效目标)
    if relative_goal_id == -1 and x == -1 and y == -1:
        return True, "Invalid goal marker"
    
    # 检查 2: 坐标范围
    if not (0 <= x < max_resolution and 0 <= y < max_resolution):
        return False, f"Out of bounds: ({x}, {y})"
    
    # 检查 3: 路径长度有效性
    if relative_goal_id < 0:
        return False, f"Negative relative_id: {relative_goal_id}"
    
    if relative_goal_id > 100:  # 超过episode长度
        return False, f"Relative_id too large: {relative_goal_id}"
    
    return True, "Valid goal"

# 验证示例
pixel_goal = [[10], [320, 240]]
is_valid, msg = validate_pixel_goal(pixel_goal, 0)
print(f"Frame 0: {msg}")  # "Frame 0: Valid goal"
```

### 3.3 目标点的使用场景分类

在训练数据构建中，根据像素目标点的有效性，将样本分为三类：

```python
# 代码行860-940

pixel_goal_list = []  # 有效目标 (像素坐标)
turn_list = []        # 转向任务 (需要原地转)
stop_list = []        # 停止任务 (到达终点)

for frame_id in range(episode_length):
    pixel_goal = pixel_goals[frame_id]
    
    if pixel_goal[0] == -1:  # -1 标记 = 无有效目标
        # 情况 A: 当前是前进动作
        if actions[frame_id] == 1:  # action=1 表示前进
            continue  # 跳过 (矛盾数据)
        else:
            # 情况 B: 需要原地转向
            turn_list.append((ep_id, ..., turn_actions, None))
    else:
        # 情况 C: 有有效的像素目标
        goal_len = pixel_goal[0]  # 相对目标帧ID
        
        if goal_len < 3:  # 路径太短
            continue  # 跳过 (数据不足)
        
        action = pixel_goal[1]  # 目标像素坐标 [x, y]
        pixel_goal_list.append((ep_id, ..., action, pose))
```

---

## 第四部分: 数据转换示例

### 4.1 具体的数据转换案例

让我们用一个真实的Episode来演示从原始数据到训练样本的完整转换：

#### **原始LeRobot数据** (Parquet 格式)

```
Episode ID: 42
Scene: S9hNv5qa7GM (Bedroom)
Duration: 46 帧 @ 30FPS = 1.53秒
Setting: 125cm_0deg (水平视角, 高度125cm)

┌──────┬────────┬─────────────────────┬──────────────────┐
│Frame │Action  │ goal.125cm_0deg     │ relative_goal_id │
├──────┼────────┼─────────────────────┼──────────────────┤
│  0   │   1    │  [300, 250]         │      12          │
│  1   │   1    │  [305, 252]         │      11          │
│  2   │   1    │  [310, 254]         │      10          │
│  3   │   1    │  [315, 256]         │       9          │
│  4   │   2    │  [320, 258]         │       8          │  ← 开始转向
│  5   │   3    │  [330, 255]         │       7          │
│  6   │   1    │  [340, 250]         │       6          │  ← 继续前进
│  7   │   1    │  [350, 245]         │       5          │
│  8   │   1    │  [-1, -1]           │      -1          │  ← 接近目标, 无效目标
│  9   │   1    │  [-1, -1]           │      -1          │
│ 10   │   0    │  [-1, -1]           │      -1          │  ← 停止 (到达目标)
│ ...  │  ...   │  ...                │     ...          │
└──────┴────────┴─────────────────────┴──────────────────┘

数据特点:
- Frame 0-3: 直线前进, 目标点逐渐右移 (视点中目标靠近)
- Frame 4-7: 边转向边前进, 坐标跳跃变化大
- Frame 8-9: 接近目标, 无法定位 ([-1, -1])
- Frame 10: 停止, 任务完成
```

#### **转换过程**

```python
# Step 1: 加载Parquet数据
import pyarrow.parquet as pq
parquet_path = "episode_000042.parquet"
df = pq.read_table(parquet_path).to_pandas()

# Step 2: 提取目标点数据
ep_pixel_goals = [
    [[df["relative_goal_frame_id.125cm_0deg"][i]],
     [df["goal.125cm_0deg"][i][0], df["goal.125cm_0deg"][i][1]]]
    for i in range(len(df))
]

# 结果:
# [
#   [[12], [300, 250]],
#   [[11], [305, 252]],
#   ...
#   [[-1], [-1, -1]],
# ]

# Step 3: 分类处理
pixel_goal_list = []
for start_frame_id in range(len(ep_actions)):
    pixel_goal = ep_pixel_goals[start_frame_id]
    
    if pixel_goal[0][0] == -1:  # 无效目标
        # → 分配到 turn_list (需要转向)
        pass
    else:
        goal_len = pixel_goal[0][0]  # 相对帧数 = 12
        if goal_len >= 3:  # 有效路径
            # → 创建样本
            sample = {
                'ep_id': 42,
                'instruction': "Exit the bedroom and enter the living room",
                'start_frame': start_frame_id,
                'end_frame': start_frame_id + goal_len + 1,
                'pixel_goal_coord': pixel_goal[1],      # [300, 250]
                'path_length': goal_len,                # 12
                'poses': [...],  # 12+1 个4×4矩阵
            }
            pixel_goal_list.append(sample)
```

#### **处理后的数据** (训练样本格式)

```python
训练样本示例:

{
    "instruction": "Exit the bedroom and enter the living room",
    "image": [经过预处理的多帧图像],
    "pixel_goal": [300, 250],           # 目标坐标
    "target_frames": [0, 1, 2, ..., 12],  # 经历的帧序列
    "input_ids": [150, 151, 156, ...],  # Token化的指令
    "labels": [-100, -100, ..., 540, 541],  # 监督标签 (仅计算目标部分)
}
```

### 4.2 两种格式数据的详细对比

| 维度 | 原始LeRobot格式 | 训练样本格式 |
|------|----------------|------------|
| **存储位置** | Parquet列式存储 | PyTorch Dataset内存 |
| **访问方式** | 逐行读取表格 | Random access by index |
| **格式** | Apache Parquet | Python Dict |
| **数据完整性** | 所有帧都包含 | 仅包含有效样本 |
| **坐标格式** | `[x, y]` 或 `[-1, -1]` | `[x, y]` (已验证) |
| **标注** | Raw值 | 已标准化/掩码化 |

#### **具体数据示例对比**

**原始LeRobot格式:**
```
Parquet 表格 (46行 × 多列)

row 0: action=1, goal=[300, 250], relative_id=12
row 1: action=1, goal=[305, 252], relative_id=11
row 2: action=1, goal=[310, 254], relative_id=10
row 3: action=1, goal=[315, 256], relative_id=9
row 4: action=2, goal=[320, 258], relative_id=8
row 5: action=3, goal=[330, 255], relative_id=7
row 6: action=1, goal=[340, 250], relative_id=6
row 7: action=1, goal=[350, 245], relative_id=5
row 8: action=1, goal=[-1, -1], relative_id=-1  ← 特殊标记
row 9: action=1, goal=[-1, -1], relative_id=-1
row 10: action=0, goal=[-1, -1], relative_id=-1
...
row 45: action=0, goal=[-1, -1], relative_id=-1
```

**转换后的训练样本:**
```python
样本 1:
{
    'ep_id': 42,
    'start_frame': 0,
    'end_frame': 13,              # 0 + 12 + 1
    'instruction': "Exit the bedroom and enter the living room",
    'pixel_goal_coord': [300, 250],
    'path_length': 12,
    'type': 'pixel_goal',  # 分类: 有像素目标
    'input_ids': torch.LongTensor([...]),
    'labels': torch.LongTensor([...]),
    'pixel_values': torch.FloatTensor([...]),  # 13帧图像
}

样本 2 (如果row 8-10被分类为stop):
{
    'ep_id': 42,
    'start_frame': 8,
    'end_frame': 11,  # 10 - 8 = 2帧
    'instruction': "Exit the bedroom and enter the living room",
    'action': 0,  # STOP
    'type': 'stop',  # 分类: 停止任务
    'input_ids': torch.LongTensor([...]),
    'labels': torch.LongTensor([...]),
    'pixel_values': torch.FloatTensor([...]),
}
```

### 4.3 坐标数值的含义解读

```
原始坐标 [320, 240]:
- x=320: 
  • 图像宽度 512 像素
  • 目标在宽度方向的 320/512 ≈ 62.5% 位置
  • 即右偏约3/8处

- y=240:
  • 图像高度 512 像素  
  • 目标在高度方向的 240/512 ≈ 47% 位置
  • 即稍微上方一点 (接近中心)

视觉解读:
┌────────────────────────────────────┐
│                                    │
│    左上[0,0]                  右上  │
│    ┌──────────────────────────┐    │
│    │                          │    │
│    │                          │    │
│    │                      ◎   │ ← 目标在右偏, 稍上
│    │                 (320,240)│    │
│    │                          │    │
│    │     🤖 机器人视点中心     │    │
│    │        (256, 256)        │    │
│    │                          │    │
│    └──────────────────────────┘    │
│    左下                        右下  │
└────────────────────────────────────┘
```

---

## 第五部分: 像素目标在模型中的应用

### 5.1 从原始坐标到模型输入

当使用像素目标进行训练时，坐标被编码为**自然语言响应**：

```python
# 代码行1070-1090

if pose is not None:  # 有有效的像素目标
    # 目标坐标被转换为文本
    action = pixel_goal[1]  # 例: [300, 250]
    
    chat_sources[0].extend([
        {
            'from': 'gpt',  # 模型输出
            'value': f'{action[0]} {action[1]}'  # "300 250"
        }
    ])

# 最终的对话:
# User:  "You are an autonomous navigation assistant. Your task is to exit the bedroom. 
#         Where should you go next? <image>"
# 
# Assistant: "300 250"  # 模型需要预测这个坐标对
```

### 5.2 Token化与监督学习

```python
# 原始对话
conversation = {
    'from': 'human',
    'value': "...<image> Where should you go next?"
}
{
    'from': 'gpt',
    'value': "300 250"  # 两个数字
}

# Token化后
input_ids = [
    ...,                    # 系统提示 + 用户问题的tokens
    151655,                 # <image> token
    ...,
    300,   # 数字"300"的token ID
    250,   # 数字"250"的token ID
    ...
]

labels = [
    -100, -100, ..., -100,  # 用户部分: 不计算损失
    300,                    # "300" 的token: 计算损失
    250,                    # "250" 的token: 计算损失
]

# 模型学习: 给定图像 → 预测坐标值
```

### 5.3 多任务学习框架

```python
# 训练数据包含三种类型的样本:

样本类型1: 像素目标 (Pixel Goal)
  input:  图像 + "下一步应该去哪?"
  output: "300 250"  (坐标)

样本类型2: 转向任务 (Turn)
  input:  图像 + "应该做什么?"
  output: "←→→"  (转向动作)

样本类型3: 停止任务 (Stop)
  input:  图像 + "应该做什么?"
  output: "STOP"  (停止)

# 代码行940-942:
list_data_dict = pixel_goal_list
if not self.pixel_goal_only:
    list_data_dict += turn_list
    list_data_dict += stop_list * 5  # 停止任务过采样(因为数量少)
```

---

## 第六部分: 常见问题与深入理解

### Q1: 为什么需要多视点目标点？

**A:** 不同的俯仰角捕捉不同的信息：

```
任务: "到达走廊另一端的房间"

视点1: 125cm_0deg (水平)
  - 看清目标房间的门
  - 适合学习"面向目标"的行为
  - 坐标例: [400, 256]

视点2: 125cm_30deg (俯视)
  - 看清地面布局和障碍
  - 适合学习"避免碰撞"的行为
  - 坐标例: [420, 300] (同一个目标, 不同投影)

多视点数据让模型学到:
✅ 多角度目标定位
✅ 视角变化的不变性
✅ 更鲁棒的导航能力
```

### Q2: [-1, -1] 标记如何处理？

**A:** 根据具体场景分类处理：

```
Case 1: 接近目标
  action=1 (前进) + goal=[-1, -1]
  → 表示"已足够接近, 无法定位"
  → 将该帧及后续帧标记为STOP样本
  → 模型学习"何时停止"

Case 2: 目标超出视野
  action=2/3 (转向) + goal=[-1, -1]
  → 表示"需要转身看目标"
  → 标记为TURN样本
  → 模型学习"如何转向定位"

Case 3: 视角遮挡
  action=1 (前进) + goal=[-1, -1]
  → 可能是暂时遮挡
  → 检查相邻帧判断是否转向
  → 可能跳过此样本
```

### Q3: 坐标值的有效范围是多少？

**A:** 
```python
# 标准配置
image_resolution = 512  # 512×512 图像

valid_range_x = [0, 512)
valid_range_y = [0, 512)

# 但通常会有一些edge cases:
边界效应 (边缘采样):
  - 某些目标会在 [490, 512) 范围
  - 表示"目标在右边界附近"
  - 有时会超出512 (算法bug或投影误差)
  
数据清洗:
  - 开发时通常clip到 [0, 512)
  - 或者filter掉异常值
```

### Q4: 如何从像素坐标恢复3D世界坐标？

**A:**
```python
# 需要相机内参和外参
K = 相机内参矩阵 (3×3)
T = 相机位姿 (4×4变换矩阵, 已在数据中)

# 逆投影:
# 从pixel_coord [x, y] → 3D点
# 需要额外信息: 深度值 depth_z

P_pixel = [x, y, 1]  # 齐次坐标
P_camera = K^(-1) * P_pixel * depth_z  # 相机坐标系
P_world = T^(-1) * P_camera  # 世界坐标系

# 实际应用:
# LeRobot数据已包含深度信息(depth图)
# 可从depth图在[x,y]处采样获得深度值
```

---

## 第七部分: 完整处理流程总结

```
╔════════════════════════════════════════════════════════════════╗
║           像素目标点数据的完整处理管道                         ║
╚════════════════════════════════════════════════════════════════╝

┌──────────────────────────┐
│   原始LeRobot数据        │
│  (Parquet + MP4 + JSON)  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│   步骤1: 数据加载                    │
│  ├─ 读Parquet文件                   │
│  ├─ 提取goal.{setting}列           │
│  └─ 提取relative_goal_frame_id列   │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│   步骤2: 数据验证                    │
│  ├─ 检查坐标范围 [0, 512)×[0, 512) │
│  ├─ 识别特殊标记 [-1, -1]          │
│  └─ 验证路径长度有效性              │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│   步骤3: 分类处理                    │
│  ├─ 像素目标类 (goal_len ≥ 3)      │
│  ├─ 转向类 (goal=[-1,-1], a≠1)    │
│  └─ 停止类 (action=STOP)          │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│   步骤4: 帧序列提取                  │
│  ├─ start_frame_id                  │
│  ├─ end_frame_id                    │
│  └─ 视频帧采样                      │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│   步骤5: 文本对话构建                │
│  ├─ 用户: 指令 + 图像占位符        │
│  └─ 助手: 坐标 或 动作 或 STOP    │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│   步骤6: Token化                     │
│  ├─ 对话→token_ids                  │
│  ├─ 生成labels (IGNORE_INDEX掩码)  │
│  └─ 坐标→数字tokens                │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│   步骤7: 最终样本组装                │
│  ├─ input_ids: [seq_len]           │
│  ├─ labels: [seq_len]              │
│  ├─ pixel_values: [T, 3, H, W]     │
│  └─ metadata: ep_id, type, ...     │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│   训练样本 (PyTorch Dataset)         │
│  可直接输入Qwen3-VL-2B模型         │
└──────────────────────────────────────┘
```

---

## 附录: 代码参考

### 完整的数据加载流程

```python
# 文件: internnav/dataset/internvla_n1_lerobot_dataset.py

def get_annotations_from_lerobot_data(data_path, setting):
    """加载LeRobot数据并生成带像素目标的annotations"""
    import pyarrow.parquet as pq
    
    annotations = {"episodes": []}
    scene_ids = [d for d in os.listdir(data_path) 
                 if os.path.isdir(os.path.join(data_path, d))]
    
    for scene_id in scene_ids:
        scene_path = os.path.join(data_path, scene_id)
        episodes = read_jsonl(os.path.join(scene_path, "meta", "episodes.jsonl"))
        
        for ep in episodes:
            ep_id = ep["episode_index"]
            ep_instructions = ep["tasks"][0].split("<INSTRUCTION_SEP>")
            ep_len = ep["length"]
            
            # 读Parquet文件
            parquet_path = os.path.join(
                scene_path, "data", f"chunk-{ep_id // 1000:03d}",
                f"episode_{ep_id:06d}.parquet"
            )
            
            df = pq.read_table(parquet_path).to_pandas()
            
            # 提取动作和目标点
            ep_actions = df["action"].tolist()
            
            goal_key = f"goal.{setting}"
            relative_goal_frame_id_key = f"relative_goal_frame_id.{setting}"
            
            ep_pixel_goals = [
                [df[relative_goal_frame_id_key][idx].tolist(),
                 df[goal_key][idx].tolist()]
                for idx in range(len(df))
            ]
            
            # 为每个指令创建episode记录
            for ep_instruction in ep_instructions:
                episode = {
                    "id": ep_id,
                    "instructions": ep_instruction,
                    "actions": ep_actions,
                    "pixel_goals": ep_pixel_goals,  # ← 核心: 像素目标
                    "length": ep_len,
                }
                annotations["episodes"].append(episode)
    
    return annotations
```

---

## 总结

**像素目标点坐标 (Pixel Goal Coordinates)** 是连接**视觉图像**和**导航动作**的关键桥梁：

| 方面 | 说明 |
|------|------|
| **物理含义** | 在当前摄像头视图中标注目标位置的像素坐标(x, y) |
| **数据格式** | LeRobot格式: `goal.{setting}` (Parquet列) |
| **存储结构** | [T, 2] 数组, 特殊值[-1, -1]表示无效 |
| **处理方式** | 验证→分类(3类样本)→提取帧→文本化→token化 |
| **训练用途** | 监督学习模型从图像预测下一步目标坐标 |
| **优势** | 端到端可学习, 视觉自然, 泛化能力强 |

通过这个详细的处理管道, 原始的结构化导航数据被转换为可用于VLM监督学习的多模态样本。
