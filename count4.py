#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os, time
from typing import Optional

# 允许多线程使用
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============== 新增：可选导入 pyrealsense2（读 .bag 用） ==============
try:
    import pyrealsense2 as rs
    HAVE_RS = True
except Exception:
    HAVE_RS = False

# 全局变量初始化
track_history = defaultdict(list)
current_region = None  # 用于记录正在拖拽的区域

# 定义计数区域（保持不变）
counting_regions = [
    {
        'name': 'YOLOv8 Polygon Region',
        'polygon': Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),
        'counts': 0,
        'seen_ids': set(),
        'dragging': False,
        'region_color': (255, 42, 4),
        'text_color': (255, 255, 255)
    },
    {
        'name': 'YOLOv8 Rectangle Region',
        'polygon': Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),
        'counts': 0,
        'seen_ids': set(),
        'dragging': False,
        'region_color': (37, 255, 225),
        'text_color': (0, 0, 0)
    }
]

def mouse_callback(event, x, y, flags, param):
    global current_region
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region['polygon'].contains(Point((x, y))):
                current_region = region
                current_region['dragging'] = True
                current_region['offset_x'] = x
                current_region['offset_y'] = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region['dragging']:
            dx = x - current_region['offset_x']
            dy = y - current_region['offset_y']
            current_region['polygon'] = Polygon([
                (p[0] + dx, p[1] + dy) for p in current_region['polygon'].exterior.coords])
            current_region['offset_x'] = x
            current_region['offset_y'] = y
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region['dragging']:
            current_region['dragging'] = False

# ============== 新增：工具函数（最少改动） ==============
def _is_bag(path: str) -> bool:
    return str(path).lower().endswith('.bag')

def _iterate_bag_frames(bag_path: str):
    """返回对齐到彩色的 (color_bgr, depth_m)"""
    if not HAVE_RS:
        raise RuntimeError("未安装 pyrealsense2，无法读取 .bag（pip3 install pyrealsense2）")
    cfg = rs.config()
    cfg.enable_device_from_file(bag_path, repeat_playback=False)
    cfg.enable_all_streams()
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
    playback = prof.get_device().as_playback()
    playback.set_real_time(False)
    try:
        while True:
            fs = pipe.wait_for_frames()
            if not fs: break
            af = align.process(fs)
            d, c = af.get_depth_frame(), af.get_color_frame()
            if not d or not c: continue
            color = np.asanyarray(c.get_data())
            depth_m = np.asanyarray(d.get_data()).astype(np.float32) * depth_scale
            yield color, depth_m
    except Exception:
        pass
    finally:
        pipe.stop()

def _estimate_z0_from_frame(depth_m: np.ndarray) -> Optional[float]:
    """用 95% 分位估计‘远’的平面，作为相机->苗床基准距离 z0（米）"""
    if depth_m is None: return None
    d = depth_m[np.isfinite(depth_m)]
    d = d[d > 0.05]
    if d.size < 1000: return None
    return float(np.percentile(d, 95))

# ================== 你的 run()：仅在内部加了对 .bag 的处理 + 高度叠加 ==================
def run(
    weights,
    source,
    device='auto',            # ← 默认改成 'auto'，自动判定
    view_img=False,
    save_img=False,
    exist_ok=False,
    classes=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2
):
    """
    weights : 模型 .pt 文件路径
    source  : 视频路径 或 RealSense .bag
    device  : 'auto' | 'cpu' | '0' / 'cuda' / 'cuda:0'
    """
    import torch  # 函数内部引用，避免全局依赖

    # ① 解析 device
    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'

    # ② 校验输入
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # ③ 加载模型到指定设备
    model = YOLO(weights)
    if device in ('0', 'cuda', 'cuda:0'):
        model.to('cuda:0'); print('[INFO] 推理设备: GPU (cuda:0)')
    else:
        model.to('cpu');    print('[INFO] 推理设备: CPU')

    names = model.model.names

    # ④ 输出路径（保持不变）
    save_dir = increment_path(Path('ultralytics_rc_output') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f'{Path(source).stem}.mp4'

    # ========== 分两种源：普通视频 或 .bag ==========
    use_bag = _is_bag(source)

    if not use_bag:
        # 普通视频：行为保持不变
        videocapture = cv2.VideoCapture(source)
        frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
        fps        = int(videocapture.get(5))
        fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_width, frame_height))
    else:
        # .bag：延迟到第一帧再创建 writer（默认 25fps）
        videocapture = None
        video_writer = None
        default_bag_fps = 25

    # ⑤ 主循环
    vid_frame_count = 0
    z0_m = None               # 在线估计相机->苗床距离
    z0_buf = []               # 收集前若干帧用于稳定估计
    Z0_FRAMES = 20            # 收集 20 帧后固定

    try:
        if not use_bag:
            # ===== 普通视频（无真实深度，不计算株高）=====
            while videocapture.isOpened():
                success, frame = videocapture.read()
                if not success:
                    break
                vid_frame_count += 1

                results = model.track(frame, persist=True, classes=classes)

                # 原有绘制/计数逻辑（不改）
                if results and results[0].boxes.id is not None:
                    boxes     = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    clss      = results[0].boxes.cls.cpu().tolist()
                    annotator = Annotator(frame, line_width=line_thickness, example=str(names))

                    for box, track_id, cls in zip(boxes, track_ids, clss):
                        annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                        bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

                        track = track_history[track_id]
                        track.append((float(bbox_center[0]), float(bbox_center[1])))
                        if len(track) > 30:
                            track.pop(0)
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False,
                                      color=colors(cls, True), thickness=track_thickness)

                        for region in counting_regions:
                            if region['polygon'].contains(Point((bbox_center[0], bbox_center[1]))):
                                if track_id not in region['seen_ids']:
                                    region['counts'] += 1
                                    region['seen_ids'].add(track_id)

                # 绘区域 & 文本（不改）
                for region in counting_regions:
                    pts = np.array(region['polygon'].exterior.coords[:-1], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, region['region_color'], thickness=region_thickness)
                    text = f"{region['name']}: {region['counts']}"
                    x, y = int(pts[0][0][0]), int(pts[0][0][1]) - 10
                    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region['text_color'], 2)

                # 显示 / 保存（不改）
                if view_img:
                    if vid_frame_count == 1:
                        cv2.namedWindow('Ultralytics YOLOv8 Region Counter Movable')
                        cv2.setMouseCallback('Ultralytics YOLOv8 Region Counter Movable', mouse_callback)
                    cv2.imshow('Ultralytics YOLOv8 Region Counter Movable', frame)
                if save_img:
                    video_writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        else:
            # ===== RealSense .bag（对齐到彩色，计算株高）=====
            for color_bgr, depth_m in _iterate_bag_frames(source):
                frame = color_bgr
                h, w = frame.shape[:2]
                vid_frame_count += 1

                # 第一次拿到尺寸后创建 writer（mp4）
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(str(out_path), fourcc, default_bag_fps, (w, h))

                # 在线估计 z0（前 Z0_FRAMES 帧的 95% 分位中值）
                if z0_m is None:
                    z_est = _estimate_z0_from_frame(depth_m)
                    if z_est is not None:
                        z0_buf.append(z_est)
                        if len(z0_buf) >= Z0_FRAMES:
                            z0_m = float(np.median(z0_buf))
                            print(f"[INFO] 估计 z0 ≈ {z0_m:.3f} m（相机->苗床）")

                # YOLO
                results = model.track(frame, persist=True, classes=classes)

                if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    clss  = results[0].boxes.cls.cpu().tolist() if results[0].boxes.cls is not None else [0]*len(boxes)
                    tids  = results[0].boxes.id
                    tids  = tids.int().cpu().tolist() if tids is not None else [None]*len(boxes)

                    annotator = Annotator(frame, line_width=line_thickness, example=str(names))
                    for box, track_id, cls in zip(boxes, tids, clss):
                        annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w-1, x2), min(h-1, y2)

                        # 轨迹（不改）
                        if track_id is not None:
                            cx = int((x1+x2)/2); cy = int((y1+y2)/2)
                            tr = track_history[track_id]
                            tr.append((float(cx), float(cy)))
                            if len(tr)>30: tr.pop(0)
                            pts = np.hstack(tr).astype(np.int32).reshape((-1,1,2))
                            cv2.polylines(frame, [pts], False, colors(cls, True), track_thickness)

                        # ========== 新增：株高（有深度且 z0_m 已估计）==========
                        if z0_m is not None:
                            roi = depth_m[y1:y2, x1:x2]
                            valid = np.isfinite(roi) & (roi > 0.05)
                            if valid.sum() > 50:
                                z_canopy = float(np.percentile(roi[valid], 10))  # 冠层“更近”用 10% 分位
                                height_m = max(0.0, z0_m - z_canopy)
                                cv2.putText(frame, f"H={height_m*100:.1f}cm",
                                            (x1, max(0, y1-8)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                        # 区域计数（不改）
                        bbox_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        for region in counting_regions:
                            if region['polygon'].contains(Point(bbox_center)):
                                if track_id is not None and track_id not in region['seen_ids']:
                                    region['counts'] += 1
                                    region['seen_ids'].add(track_id)

                # 画区域（不改）
                for region in counting_regions:
                    pts = np.array(region['polygon'].exterior.coords[:-1], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, region['region_color'], thickness=region_thickness)
                    text = f"{region['name']}: {region['counts']}"
                    x, y = int(pts[0][0][0]), int(pts[0][0][1]) - 10
                    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region['text_color'], 2)

                # 显示 / 保存（保持原逻辑）
                if view_img:
                    if vid_frame_count == 1:
                        cv2.namedWindow('Ultralytics YOLOv8 Region Counter Movable')
                        cv2.setMouseCallback('Ultralytics YOLOv8 Region Counter Movable', mouse_callback)
                    cv2.imshow('Ultralytics YOLOv8 Region Counter Movable', frame)
                if save_img and video_writer is not None:
                    video_writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # ⑥ 资源释放（保持不变）
        if 'video_writer' in locals() and video_writer is not None:
            video_writer.release()
        if 'videocapture' in locals() and videocapture is not None:
            videocapture.release()
        cv2.destroyAllWindows()

    # ⑦ 打印计数（保持不变）
    for region in counting_regions:
        print(f"Region: {region['name']}, Total Counts: {region['counts']}")

# 下面 GUI 相关保持不变
def select_file():
    file_path = filedialog.askopenfilename(
        title='选择视频文件（支持 .mp4/.avi/.mov 或 .bag）',
        filetypes=[('视频/包文件', ('*.mp4', '*.avi', '*.mov', '*.bag')), ('所有文件', '*.*')]
    )
    if file_path:
        video_path_var.set(file_path)

def select_model():
    file_path = filedialog.askopenfilename(
        title='选择模型文件',
        filetypes=[('PyTorch 模型', '*.pt'), ('所有文件', '*.*')]
    )
    if file_path:
        model_path_var.set(file_path)

def on_run_button_click():
    weights = model_path_var.get()
    source  = video_path_var.get()
    if not weights or not source:
        messagebox.showerror('错误', '请先选择模型和视频文件。')
        return
    try:
        run(weights=weights,
            source=source,
            device='auto',
            view_img=True,
            save_img=True)
        messagebox.showinfo('完成', '视频处理完成，输出文件已保存。')
    except Exception as e:
        messagebox.showerror('错误', f'处理视频时发生错误：{e}')

# GUI（保持不变）
root = tk.Tk()
root.title('视频处理工具')
root.geometry("700x250")
style = ttk.Style(root)
style.configure('TLabel', font=('Helvetica', 14))
style.configure('TButton', font=('Helvetica', 14))
style.configure('TEntry', font=('Helvetica', 14))
model_path_var = tk.StringVar()
video_path_var = tk.StringVar()
frame = ttk.Frame(root, padding="20")
frame.grid(row=0, column=0, sticky="nsew")
ttk.Label(frame, text="模型文件:").grid(row=0, column=0, padx=5, pady=10, sticky='e')
ttk.Entry(frame, textvariable=model_path_var, width=50).grid(row=0, column=1, padx=5, pady=10)
ttk.Button(frame, text="选择模型", command=select_model).grid(row=0, column=2, padx=5, pady=10)
ttk.Label(frame, text="视频文件:").grid(row=1, column=0, padx=5, pady=10, sticky='e')
ttk.Entry(frame, textvariable=video_path_var, width=50).grid(row=1, column=1, padx=5, pady=10)
ttk.Button(frame, text="选择视频", command=select_file).grid(row=1, column=2, padx=5, pady=10)
ttk.Button(frame, text="运行", command=on_run_button_click).grid(row=2, column=1, padx=5, pady=20)
root.mainloop()

