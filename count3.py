#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count3.py（稳健回放版 - 追踪器阈值修正版）
- 修复了新模型因置信度低于官方默认追踪阈值 (0.6)，导致 tid 为 None 无法计数的问题。
- 动态生成 custom_tracker.yaml，放宽追踪及格线至 0.25，并补全必要的 track_buffer 参数。
- 保持原有多边形区域与株高逻辑完全不变。
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

# ================= 动态生成适合新模型的追踪器配置 =================
TRACKER_PATH = "custom_tracker.yaml"
# 💡 补全了 track_buffer 参数，防止 ultralytics 底层报错
with open(TRACKER_PATH, "w", encoding="utf-8") as f:
    f.write("""
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.05
new_track_thresh: 0.25
track_buffer: 30
match_thresh: 0.8
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
""")

# ============== 可选导入 RealSense ==============
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
heights_win = defaultdict(lambda: deque(maxlen=5))
heights_ema = {}
EMA_ALPHA = 0.25


def _resolve_classes(model_names, classes=None, class_names=None):
    if classes and class_names:
        raise ValueError("--classes 与 --class-names 只能二选一。")
    if classes:
        return classes
    if not class_names:
        return None

    name_to_id = {str(v).lower(): int(k) for k, v in model_names.items()}
    resolved = []
    for name in class_names:
        key = str(name).strip().lower()
        if key in name_to_id:
            resolved.append(name_to_id[key])
    return resolved


def _draw_counts_top_left(frame, regions, origin=(12, 28), line_gap=28):
    x0, y0 = origin
    total = 0
    for i, region in enumerate(regions):
        display_name = region['name'].replace('YOLOv8 ', '')
        text = f"{display_name}: {region['counts']}"
        y = y0 + i * line_gap
        cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        total += int(region['counts'])
    cv2.putText(frame, f"Total: {total}", (x0, y0 + len(regions) * line_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)


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
    MAX_DH_M = 0.02
    dh = h_ema - prev
    if abs(dh) > MAX_DH_M:
        h_ema = prev + np.sign(dh) * MAX_DH_M
    heights_ema[tid] = h_ema
    return h_ema


def _is_bag(path: str) -> bool:
    return str(path).lower().endswith('.bag')


def _rs_postprocess_depth_frame(depth_frame):
    dec = rs.decimation_filter()
    spat = rs.spatial_filter()
    temp = rs.temporal_filter()
    hole = rs.hole_filling_filter()
    df = dec.process(depth_frame)
    df = spat.process(df)
    df = temp.process(df)
    df = hole.process(df)
    return df


def _iterate_bag_frames(bag_path: str, max_consec_timeout: int = 20, first_frame_retries: int = 8,
                        first_retry_sleep_ms: int = 100):
    if not HAVE_RS:
        return
    pipe = None
    try:
        cfg = rs.config()
        cfg.enable_device_from_file(bag_path, repeat_playback=False)
        cfg.enable_all_streams()
        pipe = rs.pipeline()
        prof = pipe.start(cfg)
        playback = prof.get_device().as_playback()
        playback.set_real_time(False)
        align = rs.align(rs.stream.color)
        depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
        consec_timeout = 0
        got_any_frame = False

        while True:
            try:
                fs = pipe.wait_for_frames()
            except RuntimeError:
                consec_timeout += 1
                if not got_any_frame and consec_timeout <= first_frame_retries:
                    cv2.waitKey(first_retry_sleep_ms)
                    continue
                if consec_timeout >= max_consec_timeout:
                    break
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

            if depth_m.shape[0] >= 5 and depth_m.shape[1] >= 5:
                depth_m = cv2.medianBlur(depth_m, 5)

            got_any_frame = True
            yield color, depth_m, intr

    except Exception:
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
    Y = (v - intr.ppy) / intr.fy * z
    P = np.stack([X, Y, z], axis=1)
    return P


def _fit_plane_ransac(points: np.ndarray, iters=300, dist_thresh=0.004):
    if points is None or len(points) < 100:
        return None
    pts = points
    N = pts.shape[0]
    best_inl = -1
    best_n = None;
    best_d = None;
    best_mask = None
    rng = np.random.default_rng(123)

    for _ in range(iters):
        idx = rng.choice(N, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        v1 = p1 - p0;
        v2 = p2 - p0
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
            best_n = n;
            best_d = d
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
        n = -n;
        d = -d
    return (n.astype(np.float32), float(d))


def _estimate_ground_plane(depth_m: np.ndarray, intr):
    H, W = depth_m.shape[:2]
    y0 = int(H * 0.85)
    pts = _roi_points_from_depth(depth_m, intr, 0, y0, W, H, step=4)
    if pts is None:
        return None
    return _fit_plane_ransac(pts, iters=300, dist_thresh=0.004)


def _height_from_box_plane(depth_m, box, intr, plane, top_ratio=0.02, step=2) -> Optional[float]:
    n, d = plane
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    w = x2 - x1;
    h = y2 - y1
    if w < 6 or h < 6:
        return None
    shrink = 0.12
    x1 = x1 + int(w * shrink)
    x2 = x2 - int(w * shrink)
    y2 = y1 + int(h * 0.70)
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
        device='auto',
        view_img=False,
        save_img=True,
        exist_ok=False,
        classes=None,
        class_names=None,
        line_thickness=2,
        track_thickness=2,
        region_thickness=2
):
    import torch

    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'

    if not source or not Path(source).exists():
        raise FileNotFoundError(f"视频/包文件 '{source}' 不存在。")
    if not Path(weights).exists():
        raise FileNotFoundError(f"模型文件 '{weights}' 不存在。")

    model = YOLO(weights)
    if device in ('0', 'cuda', 'cuda:0'):
        model.to('cuda:0')
    else:
        model.to('cpu')
    names = model.model.names
    classes = _resolve_classes(names, classes=classes, class_names=class_names)

    save_dir = Path('outputs')
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f'{Path(source).stem}.mp4'
    default_bag_fps = 25

    PLANE_FRAMES = 10
    plane_buf = []
    plane_nd = None

    heights_all = []

    if not _is_bag(source):
        cap = cv2.VideoCapture(source)
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        fps = int(cap.get(5)) or default_bag_fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            results = model.track(frame, persist=True, classes=classes, conf=0.25, tracker=TRACKER_PATH)

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                tids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                ann = Annotator(frame, line_width=line_thickness, example=str(names))
                for box, tid, cls in zip(boxes, tids, clss):
                    ann.box_label(box, 'target', color=colors(0, True))
                    cx = int((box[0] + box[2]) / 2);
                    cy = int((box[1] + box[3]) / 2)
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
                            n_mean = -n_mean;
                            d_med = -d_med
                        plane_nd = (n_mean.astype(np.float32), d_med)

            results = model.track(frame, persist=True, classes=classes, conf=0.25, tracker=TRACKER_PATH)

            if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:
                boxes = results[0].boxes.xyxy.cpu()
                clss = results[0].boxes.cls.cpu().tolist() if results[0].boxes.cls is not None else [0] * len(boxes)
                tids_t = results[0].boxes.id
                tids = tids_t.int().cpu().tolist() if tids_t is not None else [None] * len(boxes)

                ann = Annotator(frame, line_width=line_thickness, example=str(names))
                for box, tid, cls in zip(boxes, tids, clss):
                    ann.box_label(box, 'target', color=colors(0, True))
                    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    if tid is not None:
                        cx = int((x1 + x2) / 2);
                        cy = int((y1 + y2) / 2)
                        tr = track_history[tid]
                        tr.append((float(cx), float(cy)))
                        if len(tr) > 30:
                            tr.pop(0)
                        pts = np.hstack(tr).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], False, colors(cls, True), track_thickness)
                    else:
                        cx = int((x1 + x2) / 2);
                        cy = int((y1 + y2) / 2)

                    for region in counting_regions:
                        if region['polygon'].contains(Point((cx, cy))):
                            if tid is not None and tid not in region['seen_ids']:
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
                            cv2.putText(frame, f"H={h_show * 100:.1f}cm",
                                        (x1, max(0, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                frame = ann.result()

            _draw_counts_top_left(frame, counting_regions)

            if save_img and writer is not None:
                writer.write(frame)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_path), fourcc, default_bag_fps, (640, 480))
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No frames from .bag", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            writer.write(blank)

        writer.release()

    for region in counting_regions:
        print(f"区域: {region['name']}，计数: {region['counts']}")

    if len(heights_all) >= 5:
        h_arr = np.array(heights_all, dtype=np.float32)
        h_med = float(np.median(h_arr)) * 100.0
        h_p95 = float(np.percentile(h_arr, 95)) * 100.0
        print(f"株高统计: 中位数≈{h_med:.1f} cm，95分位≈{h_p95:.1f} cm")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/best.pt', help='模型文件路径')
    parser.add_argument('--device', default='auto', help="推理设备")
    parser.add_argument('--source', type=str, required=True, help='视频文件路径')
    parser.add_argument('--view-img', action='store_true', help='静默')
    parser.add_argument('--save-img', action='store_true', default=True, help='保存输出')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--class-names', nargs='+', type=str)
    parser.add_argument('--line-thickness', type=int, default=2)
    parser.add_argument('--track-thickness', type=int, default=2)
    parser.add_argument('--region-thickness', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))
