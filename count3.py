#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count3.py（稳健回放版）
- 静默运行（无窗口）
- 兼容普通视频与 RealSense .bag
- 目标计数 + 株高估计（.bag 深度 + 地面平面拟合；普通视频仅计数）
- 结尾打印兼容行："区域: <名称>，计数: <数字>"
"""

import argparse
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# 允许 MKL/OpenMP 共存
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============== 可选导入 RealSense（读 .bag 用） ==============
try:
    import pyrealsense2 as rs
    HAVE_RS = True
except Exception:
    HAVE_RS = False

# ---------------- 计数区域（与旧版一致） ----------------
counting_regions = [
    {
        'name': 'YOLOv8 Polygon Region',
        'polygon': Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),
        'counts': 0,
        'seen_ids': set(),
        'region_color': (255, 42, 4),
        'text_color': (255, 255, 255),
    },
    {
        'name': 'YOLOv8 Rectangle Region',
        'polygon': Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),
        'counts': 0,
        'seen_ids': set(),
        'region_color': (37, 255, 225),
        'text_color': (0, 0, 0),
    },
]

# ---------------- 轨迹与株高平滑 ----------------
track_history = defaultdict(list)
heights_win = defaultdict(lambda: deque(maxlen=5))  # 中值窗口
heights_ema = {}                                    # EMA 平滑
EMA_ALPHA = 0.25

def _resolve_classes(model_names, classes=None, class_names=None):
    """将类别过滤参数统一解析为 YOLO 所需的类别 id 列表。"""
    if classes and class_names:
        raise ValueError("--classes 与 --class-names 只能二选一。")
    if classes:
        return classes
    if not class_names:
        return None

    name_to_id = {str(v).lower(): int(k) for k, v in model_names.items()}
    resolved = []
    unknown = []
    for name in class_names:
        key = str(name).strip().lower()
        if key in name_to_id:
            resolved.append(name_to_id[key])
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(f"未知类别名: {unknown}。可选类别: {list(model_names.values())}")
    return resolved


def _draw_counts_top_left(frame, regions, origin=(12, 28), line_gap=28):
    """在左上角统一绘制各区域计数。"""
    x0, y0 = origin
    for i, region in enumerate(regions):
        display_name = region['name'].replace('YOLOv8 ', '')
        text = f"{display_name}: {region['counts']}"
        y = y0 + i * line_gap
        cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

def _smooth_height_by_id(tid: Optional[int], h_raw: Optional[float]) -> Optional[float]:
    if tid is None:
        return h_raw
    if h_raw is None:
        return heights_ema.get(tid, None)
    q = heights_win[tid]
    q.append(h_raw)
    h_med = float(np.median(q))
    prev = heights_ema.get(tid, h_med)
    h_ema = (1.0 - EMA_ALPHA) * prev + EMA_ALPHA * h_med
    MAX_DH_M = 0.02  # 2 cm/帧
    dh = h_ema - prev
    if abs(dh) > MAX_DH_M:
        h_ema = prev + np.sign(dh) * MAX_DH_M
    heights_ema[tid] = h_ema
    return h_ema

# ---------------- 基础工具 ----------------
def _is_bag(path: str) -> bool:
    return str(path).lower().endswith('.bag')

def _rs_postprocess_depth_frame(depth_frame):
    """RealSense 深度后处理：降采样 + 空间/时间滤波 + 补洞"""
    dec = rs.decimation_filter()
    spat = rs.spatial_filter()
    temp = rs.temporal_filter()
    hole = rs.hole_filling_filter()
    df = dec.process(depth_frame)
    df = spat.process(df)
    df = temp.process(df)
    df = hole.process(df)
    return df

def _iterate_bag_frames(bag_path: str, max_consec_timeout: int = 20, first_frame_retries: int = 8, first_retry_sleep_ms: int = 100):
    """
    迭代返回 (color_bgr, depth_m, intr)。增强：
    - 首帧失败重试（避免“写入未完全”或“开头空包”直接失败）
    - 中途“帧未到达 5000ms”连续出现 max_consec_timeout 次后才结束
    - 优雅退出，不抛异常给上层
    """
    if not HAVE_RS:
        raise RuntimeError("未安装 pyrealsense2，无法读取 .bag。请在 yolocode 环境安装：pip install pyrealsense2")

    pipe = None
    try:
        cfg = rs.config()
        # 如果文件很短/未完整，RealSense 可能在开头给不到帧；先允许多次尝试
        cfg.enable_device_from_file(bag_path, repeat_playback=False)
        cfg.enable_all_streams()
        pipe = rs.pipeline()
        prof = pipe.start(cfg)

        # 离线回放（非实时），便于尽快取帧
        playback = prof.get_device().as_playback()
        playback.set_real_time(False)

        align = rs.align(rs.stream.color)
        depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()

        consec_timeout = 0
        got_any_frame = False

        while True:
            try:
                fs = pipe.wait_for_frames()
            except RuntimeError as e:
                # 典型：Frame didn't arrive within 5000
                consec_timeout += 1
                if not got_any_frame and consec_timeout <= first_frame_retries:
                    # 首帧到不了：短暂等待重试
                    cv2.waitKey(first_retry_sleep_ms)
                    continue
                if consec_timeout >= max_consec_timeout:
                    # 认为流已结束或文件不完整，优雅退出
                    break
                # 继续尝试拿帧
                continue

            if not fs:
                consec_timeout += 1
                if consec_timeout >= max_consec_timeout:
                    break
                continue

            consec_timeout = 0
            af = align.process(fs)
            d, c = af.get_depth_frame(), af.get_color_frame()
            if not d or not c:
                continue

            d = _rs_postprocess_depth_frame(d)
            intr = d.get_profile().as_video_stream_profile().get_intrinsics()

            color = np.asanyarray(c.get_data())
            depth_m = np.asanyarray(d.get_data()).astype(np.float32) * depth_scale

            # 去毛刺
            if depth_m.shape[0] >= 5 and depth_m.shape[1] >= 5:
                depth_m = cv2.medianBlur(depth_m, 5)

            got_any_frame = True
            yield color, depth_m, intr

    except Exception:
        # 不把异常抛给外面，交由上层用已有帧完成收尾
        return
    finally:
        try:
            if pipe is not None:
                pipe.stop()
        except Exception:
            pass

def _roi_points_from_depth(depth_m: np.ndarray, intr, x1, y1, x2, y2, step=4):
    H, W = depth_m.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    xs = np.arange(x1, x2, step, dtype=np.int32)
    ys = np.arange(y1, y2, step, dtype=np.int32)
    if xs.size == 0 or ys.size == 0:
        return None
    gx, gy = np.meshgrid(xs, ys)
    z = depth_m[gy, gx].astype(np.float32)
    mask = np.isfinite(z) & (z > 0.05) & (z < 10.0)
    if not np.any(mask):
        return None
    u = gx[mask].astype(np.float32)
    v = gy[mask].astype(np.float32)
    z = z[mask]
    X = (u - intr.ppx) / intr.fx * z
    Y = (v - intr.ppy) / intr.ppy * 0 + (v - intr.ppy) / intr.fy * z  # 展开写法避免误读
    Y = (v - intr.ppy) / intr.fy * z
    P = np.stack([X, Y, z], axis=1)
    return P

def _fit_plane_ransac(points: np.ndarray, iters=300, dist_thresh=0.004) -> Optional[Tuple[np.ndarray, float]]:
    if points is None or len(points) < 100:
        return None
    pts = points
    N = pts.shape[0]
    best_inl = -1
    best_n = None; best_d = None; best_mask = None
    rng = np.random.default_rng(123)

    for _ in range(iters):
        idx = rng.choice(N, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        v1 = p1 - p0; v2 = p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = n / norm
        d = -np.dot(n, p0)

        dist = np.abs(pts @ n + d)
        mask = dist < dist_thresh
        ninl = int(mask.sum())
        if ninl > best_inl:
            best_inl = ninl
            best_n = n; best_d = d
            best_mask = mask

    if best_inl < 100:
        return None

    P = pts[best_mask]
    cen = P.mean(axis=0)
    Q = P - cen
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    n = vh[-1, :]
    n = n / np.linalg.norm(n)
    d = -np.dot(n, cen)
    if n[2] > 0:
        n = -n; d = -d
    return (n.astype(np.float32), float(d))

def _estimate_ground_plane(depth_m: np.ndarray, intr):
    H, W = depth_m.shape[:2]
    y0 = int(H * 0.85)  # 底部 15%
    pts = _roi_points_from_depth(depth_m, intr, 0, y0, W, H, step=4)
    if pts is None:
        return None
    return _fit_plane_ransac(pts, iters=300, dist_thresh=0.004)

def _height_from_box_plane(depth_m, box, intr, plane, top_ratio=0.02, step=2) -> Optional[float]:
    n, d = plane
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    w = x2 - x1; h = y2 - y1
    if w < 6 or h < 6:
        return None
    shrink = 0.12
    x1 = x1 + int(w * shrink)
    x2 = x2 - int(w * shrink)
    y2 = y1 + int(h * 0.70)  # 只用上部 70%
    pts = _roi_points_from_depth(depth_m, intr, x1, y1, x2, y2, step=step)
    if pts is None or pts.shape[0] < 40:
        return None
    h = pts @ n + d
    h = h[(h > 0.0) & (h < 1.5)]
    if h.size < 20:
        return None
    k = max(10, int(top_ratio * h.size))
    idx = np.argpartition(h, -k)[-k:]
    return float(np.median(h[idx]))

# ================== 核心运行逻辑 ==================
def run(
    weights='models/best.pt',
    source=None,
    device='auto',          # 'auto' | 'cpu' | '0'(cuda)
    view_img=False,         # 兼容旧参，忽略（静默）
    save_img=True,
    exist_ok=False,
    classes=None,
    class_names=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2
):
    import torch

    # 设备
    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'

    # 路径检查
    if not source or not Path(source).exists():
        raise FileNotFoundError(f"视频/包文件 '{source}' 不存在。")
    if not Path(weights).exists():
        raise FileNotFoundError(f"模型文件 '{weights}' 不存在。")

    # 模型
    model = YOLO(weights)
    if device in ('0', 'cuda', 'cuda:0'):
        model.to('cuda:0')
    else:
        model.to('cpu')
    names = model.model.names
    classes = _resolve_classes(names, classes=classes, class_names=class_names)

    # 输出路径
    save_dir = Path('outputs')
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f'{Path(source).stem}.mp4'
    default_bag_fps = 25

    # 平面估计缓存
    PLANE_FRAMES = 10
    plane_buf = []
    plane_nd = None  # (n, d)

    heights_all = []

    if not _is_bag(source):
        # 普通视频：仅计数
        cap = cv2.VideoCapture(source)
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        fps = int(cap.get(5)) or default_bag_fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            results = model.track(frame, persist=True, classes=classes)
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                tids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                ann = Annotator(frame, line_width=line_thickness, example=str(names))
                for box, tid, cls in zip(boxes, tids, clss):
                    ann.box_label(box, str(names[cls]), color=colors(cls, True))
                    cx = int((box[0] + box[2]) / 2); cy = int((box[1] + box[3]) / 2)
                    tr = track_history[tid]
                    tr.append((float(cx), float(cy)))
                    if len(tr) > 30:
                        tr.pop(0)
                    pts = np.hstack(tr).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, colors(cls, True), track_thickness)
                    for region in counting_regions:
                        if region['polygon'].contains(Point((cx, cy))):
                            if tid not in region['seen_ids']:
                                region['counts'] += 1
                                region['seen_ids'].add(tid)

                frame = ann.result()

            _draw_counts_top_left(frame, counting_regions)

            if save_img:
                writer.write(frame)

        writer.release()
        cap.release()

    else:
        # .bag：计数 + 株高
        writer = None
        got_any_frame = False
        for color_bgr, depth_m, intr in _iterate_bag_frames(source):
            got_any_frame = True
            frame = color_bgr
            h, w = frame.shape[:2]

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(out_path), fourcc, default_bag_fps, (w, h))

            if plane_nd is None:
                est = _estimate_ground_plane(depth_m, intr)
                if est is not None:
                    plane_buf.append(est)
                    if len(plane_buf) >= PLANE_FRAMES:
                        ns = np.stack([p[0] for p in plane_buf], axis=0)
                        ns = ns / np.linalg.norm(ns, axis=1, keepdims=True)
                        n_mean = ns.mean(axis=0)
                        n_mean = n_mean / np.linalg.norm(n_mean)
                        ds = np.array([p[1] for p in plane_buf], dtype=np.float32)
                        d_med = float(np.median(ds))
                        if d_med < 0:
                            n_mean = -n_mean; d_med = -d_med
                        plane_nd = (n_mean.astype(np.float32), d_med)

            results = model.track(frame, persist=True, classes=classes)
            if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:
                boxes = results[0].boxes.xyxy.cpu()
                clss = results[0].boxes.cls.cpu().tolist() if results[0].boxes.cls is not None else [0]*len(boxes)
                tids_t = results[0].boxes.id
                tids = tids_t.int().cpu().tolist() if tids_t is not None else [None]*len(boxes)

                ann = Annotator(frame, line_width=line_thickness, example=str(names))
                for box, tid, cls in zip(boxes, tids, clss):
                    ann.box_label(box, str(names[cls]), color=colors(cls, True))
                    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    if tid is not None:
                        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
                        tr = track_history[tid]
                        tr.append((float(cx), float(cy)))
                        if len(tr) > 30:
                            tr.pop(0)
                        pts = np.hstack(tr).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], False, colors(cls, True), track_thickness)
                    else:
                        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)

                    for region in counting_regions:
                        if region['polygon'].contains(Point((cx, cy))):
                            if tid not in region['seen_ids']:
                                region['counts'] += 1
                                region['seen_ids'].add(tid)

                    if plane_nd is not None:
                        h_raw = _height_from_box_plane(depth_m, (x1, y1, x2, y2), intr, plane_nd,
                                                       top_ratio=0.02, step=2)
                        if h_raw is None:
                            pts_fb = _roi_points_from_depth(depth_m, intr, x1, y1, x2, y2, step=3)
                            if pts_fb is not None and pts_fb.shape[0] >= 20:
                                hh = pts_fb @ plane_nd[0] + plane_nd[1]
                                hh = hh[(hh > 0.0) & (hh < 1.5)]
                                if hh.size >= 10:
                                    h_raw = float(np.percentile(hh, 95.0))
                        h_show = _smooth_height_by_id(tid, h_raw) if tid is not None else h_raw
                        if h_show is not None:
                            heights_all.append(h_show)
                            cv2.putText(frame, f"H={h_show*100:.1f}cm",
                                        (x1, max(0, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                frame = ann.result()

            _draw_counts_top_left(frame, counting_regions)

            if save_img and writer is not None:
                writer.write(frame)

        # 若没拿到任何帧，仍需创建一个空的输出占位，避免上游“找不到文件”
        if writer is None:
            # 尝试用 640x480 占位（防止前端上传时报错）
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_path), fourcc, default_bag_fps, (640, 480))
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No frames from .bag", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            writer.write(blank)

        writer.release()

    # ---------------- 结束：打印计数（兼容旧解析） ----------------
    for region in counting_regions:
        print(f"区域: {region['name']}，计数: {region['counts']}")

    if len(heights_all) >= 5:
        h_arr = np.array(heights_all, dtype=np.float32)
        h_med = float(np.median(h_arr)) * 100.0  # cm
        h_p95 = float(np.percentile(h_arr, 95)) * 100.0
        print(f"株高统计: 中位数≈{h_med:.1f} cm，95分位≈{h_p95:.1f} cm")

# ---------------- CLI ----------------
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/best.pt', help='模型文件路径')
    parser.add_argument('--device', default='auto', help="推理设备: 'auto'|'cpu'|'0'")
    parser.add_argument('--source', type=str, required=True, help='视频/包文件路径（支持 .mp4/.avi/.mov/.bag）')
    parser.add_argument('--view-img', action='store_true', help='兼容旧参数，忽略（静默）')
    parser.add_argument('--save-img', action='store_true', default=True, help='保存标注后的视频到 outputs/')
    parser.add_argument('--exist-ok', action='store_true', help='保留参数占位，不影响输出位置')
    parser.add_argument('--classes', nargs='+', type=int, help='按类别 id 过滤目标（如: --classes 0 1）')
    parser.add_argument('--class-names', nargs='+', type=str, help='按类别名称过滤目标（如: --class-names chill weed）')
    parser.add_argument('--line-thickness', type=int, default=2, help='边框粗细')
    parser.add_argument('--track-thickness', type=int, default=2, help='追踪线粗细')
    parser.add_argument('--region-thickness', type=int, default=4, help='区域线粗细')
    return parser.parse_args()

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
