# Count 脚本使用说明

本文档说明仓库根目录下 `count*.py` 脚本的用途、差异与典型用法。

## 1. 环境准备

建议使用 Python 3.10+，并确保安装以下依赖：

- `ultralytics`
- `opencv-python`
- `numpy`
- `shapely`
- `torch`
- （可选）`pyrealsense2`（读取 `.bag` 时需要）

示例：

```bash
pip install ultralytics opencv-python numpy shapely torch
# 如果需要处理 RealSense .bag
pip install pyrealsense2
```

---

## 2. 各脚本定位

### `count1.py`
- 最基础版本：YOLO 跟踪 + 区域计数。
- 支持可视化窗口与视频保存。
- 命令行参数里含有默认模型路径/视频路径，通常需要改成你自己的文件路径。

### `count2.py`
- 在 `count1.py` 基础上增加了 GUI（Tkinter）：
  - 可点击选择模型文件
  - 可点击选择视频文件
  - 点击按钮运行
- 使用 `seen_ids` 做了按跟踪 ID 去重计数，避免同一目标重复累计。

### `count3.py`
- 稳健命令行版本，支持普通视频与 RealSense `.bag`。
- 具备：
  - 目标跟踪与区域计数
  - `.bag` 场景下基于地面平面的株高估计
  - 高度时序平滑
- 当前版本会**去掉画面中的区域多边形边框**，并把计数统一显示在左上角。


### `count3_target.py`
- `count3.py` 的目标版本（文件头注释仍写作 `count3.py`，但实际文件名为 `count3_target.py`）。
- 与 `count3.py` 主体流程基本一致（同样有动态 `custom_tracker.yaml`、`.bag`/普通视频兼容、株高估计和平滑）。
- 关键差异：左上角统计额外显示 `Total: <总计>`，即把所有区域计数求和后单独展示。

### `count3_261514bak.py`
- 较早的稳定备份版本（无动态追踪器阈值配置）。
- 保留普通视频/`.bag` 双通道与株高估计逻辑，但追踪器参数更多依赖 Ultralytics 默认值。
- 与当前 `count3.py` 的主要差异：
  - **没有**在启动时写入 `custom_tracker.yaml`（因此不会主动把追踪阈值降到 0.25）。
  - `_draw_counts_top_left` 不统计 `Total` 行。
  - 不包含 `_resolve_classes`（类别名称到类别 ID 的解析辅助函数）。

### `count3_backup.py`
- `count3.py` 的备份副本（创建时与原 `count3.py` 完全一致）。
- 便于对比修改前后逻辑，或在实验失败时快速回退。

### `count4.py`
- GUI 版本，支持选择视频或 `.bag`。
- 提供计数与基础的高度叠加显示。

### `count5.py`
- GUI + `.bag` + 平面拟合/平滑思路的综合版本。
- 适合在交互式场景下调试更复杂的测高流程。

### `count33.py`
- 融合版本（兼容老接口 + 引入平面测高与平滑策略）。
- 适合在保留原使用习惯的同时测试增强逻辑。

---

## 3. 常用运行方式

## 3.1 运行 `count3.py`（推荐命令行）

```bash
python count3.py \
  --weights models/best.pt \
  --source /path/to/video_or_bag \
  --device auto \
  --save-img
```

参数说明：

- `--weights`：模型文件路径（`.pt`）
- `--source`：输入文件路径，支持视频或 `.bag`
- `--device`：`auto` / `cpu` / `0`（CUDA）
- `--classes`：可选，按类别过滤
- `--save-img`：保存可视化结果视频

> 若输入为普通视频：执行检测+计数。  
> 若输入为 `.bag`：执行检测+计数+株高估计。

## 3.2 运行 GUI 脚本（`count2.py` / `count4.py` / `count5.py`）

```bash
python count2.py
# 或
python count4.py
# 或
python count5.py
```

运行后在窗口中选择模型和输入文件，点击“运行”。

---

## 4. 输出说明

- 大部分脚本会输出带标注的视频（默认在 `outputs/` 或 `ultralytics_rc_output/exp*` 下）。
- 终端会打印各区域计数结果。
- `count3.py` 在 `.bag` 场景下还会输出株高统计信息（满足样本数量时）。

---

## 5. 使用建议

1. **先用 `count3_backup.py` 保存稳定基线**，再在 `count3.py` 上做实验。  
2. 普通视频任务优先使用 `count3.py`（命令行更易批处理）。  
3. 需要现场交互选择文件时使用 GUI 版（`count2/4/5.py`）。  
4. 处理 `.bag` 前先确认 `pyrealsense2` 可正常导入。

