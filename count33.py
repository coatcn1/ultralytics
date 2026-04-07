#!/usr/bin/env python3
import argparse
from collections import defaultdict, deque
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os
from typing import Optional

# 允许多线程使用
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============== 可选导入 RealSense（用于 .bag） ==============
try:
    import pyrealsense2 as rs
    HAVE_RS = True
except Exception:
    HAVE_RS = False

# ============== 高度时序平滑参数（移植自代码二） ==============
heights_win = defaultdict(lambda: deque(maxlen=5))  # 每目标窗口中值
heights_ema = {}                                     # 每目标 EMA
EMA_ALPHA = 0.25                                     # 0.15~0.35 可调

# 地面平面 EMA（允许相机轻微移动）
PLANE_ALPHA = 0.15
PRINT_PLANE_EVERY = 30  # 每隔多少帧打印一次平面，防刷屏

# 全局（和代码一一致）
track_history = defaultdict(list)

# ============== 计数区域（保留代码一逻辑，可拖拽） ==============
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
current_region = None

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
        if current_region is not None and current_region.get('dragging', False):
            dx = x - current_region['offset_x']
            dy = y - current_region['offset_y']
            current_region['polygon'] = Polygon([
                (p[0] + dx, p[1] + dy) for p in current_region['polygon'].exterior.coords
            ])
            current_region['offset_x'] = x
            current_region['offset_y'] = y
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region.get('dragging', False):
            current_region['dragging'] = False

# ============== 深度/平面/高度 —— 移植自代码二 ==============
def _is_bag(path: str) -> bool:
    return str(path).lower().endswith('.bag')

def _rs_postprocess_depth_frame(depth_frame):
    """RealSense 稳深度：抽点+空间+时域+补洞"""
    dec = rs.decimation_filter()
    spat = rs.spatial_filter()
    temp = rs.temporal_filter()
    hole = rs.hole_filling_filter()
    df = dec.process(depth_frame)
    df = spat.process(df)
    df = temp.process(df)
    df = hole.process(df)
    return df

def _iterate_bag_frames(bag_path: str):
    """返回 (color_bgr, depth_m, intr)。深度已与彩色对齐并稳深度。"""
    if not HAVE_RS:
        raise RuntimeError("未安装 pyrealsense2，无法读取 .bag（pip install pyrealsense2）")
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
            if not fs:
                break
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
            yield color, depth_m, intr
    except Exception:
        pass
    finally:
        pipe.stop()

def _roi_points_from_depth(depth_m, intr, x1, y1, x2, y2, step=4):
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
    P = np.stack([X, Y, z], axis=1)  # N x 3
    return P

def _fit_plane_ransac(points, iters=300, dist_thresh=0.004):
    """RANSAC 拟合平面 ax+by+cz+d=0，返回 (n, d)，|n|=1。法向朝向相机（n_z<0）。"""
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

    # 统一法向朝向相机：n_z 应 < 0
    if n[2] > 0:
        n = -n; d = -d

    return (n.astype(np.float32), float(d))

def _estimate_ground_plane(depth_m, intr):
    """仅用画面底部 15% 点云拟合地面，更抗遮挡。"""
    H, W = depth_m.shape[:2]
    y0 = int(H * 0.85)
    pts = _roi_points_from_depth(depth_m, intr, 0, y0, W, H, step=4)
    if pts is None:
        return None
    return _fit_plane_ransac(pts, iters=300, dist_thresh=0.004)

def _ema_plane(prev, cur, alpha=0.15):
    """平面 (n,d) 的 EMA；允许相机轻微移动仍保持稳定。"""
    if cur is None:
        return prev
    n1, d1 = cur
    if prev is None:
        return (n1.astype(np.float32), float(d1))
    n0, d0 = prev
    n = (1.0 - alpha) * n0 + alpha * n1
    n_norm = np.linalg.norm(n)
    n = n1 if n_norm < 1e-8 else n / n_norm
    d = (1.0 - alpha) * d0 + alpha * d1
    return (n.astype(np.float32), float(d))

def _height_from_box_plane(depth_m, box, intr, plane, top_ratio=0.02, step=2):
    """给定 bbox 与地面平面，返回 bbox 内冠层的正交高度（米）。"""
    n, d = plane
    x1, y1, x2, y2 = [int(round(v)) for v in box]

    # 收缩左右、只用上部（靠近相机的一端）以避开地面/噪点
    w = x2 - x1; h = y2 - y1
    if w < 6 or h < 6:
        return None
    shrink = 0.12
    x1 = x1 + int(w * shrink)
    x2 = x2 - int(w * shrink)
    y2 = y1 + int(h * 0.70)

    pts = _roi_points_from_depth(depth_m, intr, x1, y1, x2, y2, step=step)
    if pts is None or pts.shape[0] < 40:
        return None

    # 点到平面距离（正交）
    hvals = pts @ n + d
    hvals = hvals[(hvals > 0.0) & (hvals < 1.5)]  # 合理阈
    if hvals.size < 20:
        return None

    # 取最高的 top_ratio 部分（或至少10个）做中位数
    k = max(10, int(top_ratio * hvals.size))
    idx = np.argpartition(hvals, -k)[-k:]
    return float(np.median(hvals[idx]))

def _smooth_height_by_id(tid, h_raw):
    """同一 track 的高度做“窗口中值 + EMA + 每帧限幅”；无新值时沿用上一帧。"""
    if tid is None:
        return h_raw
    if h_raw is None:
        return heights_ema.get(tid, None)
    q = heights_win[tid]
    q.append(h_raw)
    h_med = float(np.median(q))
    prev = heights_ema.get(tid, h_med)
    h_ema = (1.0 - EMA_ALPHA) * prev + EMA_ALPHA * h_med
    MAX_DH_M = 0.02  # 每帧最大变化（约 2cm）
    dh = h_ema - prev
    if abs(dh) > MAX_DH_M:
        h_ema = prev + np.sign(dh) * MAX_DH_M
    heights_ema[tid] = h_ema
    return h_ema

# ============== 主流程（保留代码一接口，融合深度测高） ==============
def run(
    weights='models/best.pt',
    source=None,
    device='auto',
    view_img=False,
    save_img=False,
    exist_ok=False,   # 占位保持接口一致
    classes=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=4
):
    import torch

    # 自动判定设备
    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'

    if not source or not Path(source).exists():
        raise FileNotFoundError(f"视频/包文件 '{source}' 不存在，请检查路径。")
    if not Path(weights).exists():
        raise FileNotFoundError(f"模型文件 '{weights}' 不存在，请检查路径。")

    model = YOLO(weights)
    if device in ('0', 'cuda', 'cuda:0'):
        model.to('cuda:0'); print('[INFO] 推理设备: GPU (cuda:0)')
    else:
        model.to('cpu');    print('[INFO] 推理设备: CPU')

    names = model.model.names
    use_bag = _is_bag(source)

    save_dir = Path('outputs')
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f'{Path(source).stem}.mp4'

    if not use_bag:
        # 普通视频：行为同代码一
        videocapture = cv2.VideoCapture(source)
        frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
        fps = int(videocapture.get(5))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_width, frame_height))
    else:
        videocapture = None
        video_writer = None
        default_bag_fps = 25

    vid_frame_count = 0
    plane_nd = None  # 地面平面 (n,d)

    try:
        if not use_bag:
            # -------- 普通视频分支（只检测/计数） --------
            while videocapture.isOpened():
                success, frame = videocapture.read()
                if not success:
                    break
                vid_frame_count += 1

                results = model.track(frame, persist=True, classes=classes)

                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes     = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    clss      = results[0].boxes.cls.cpu().tolist()
                    annotator = Annotator(frame, line_width=line_thickness, example=str(names))

                    for box, track_id, cls in zip(boxes, track_ids, clss):
                        annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                        bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

                        # 轨迹（代码一里是注释掉的，这里保持不画）
                        track = track_history[track_id]
                        track.append((float(bbox_center[0]), float(bbox_center[1])))
                        if len(track) > 30:
                            track.pop(0)

                        # 区域计数
                        for region in counting_regions:
                            if region['polygon'].contains(Point((bbox_center[0], bbox_center[1]))):
                                if track_id not in region['seen_ids']:
                                    region['counts'] += 1
                                    region['seen_ids'].add(track_id)

                # 区域可视化
                for region in counting_regions:
                    pts = np.array(region['polygon'].exterior.coords[:-1], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, region['region_color'], thickness=region_thickness)
                    text = f"{region['name']}: {region['counts']}"
                    x, y = int(pts[0][0][0]), int(pts[0][0][1]) - 10
                    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region['text_color'], 2)

                # 显示/保存
                if view_img:
                    if vid_frame_count == 1:
                        cv2.namedWindow('标注窗口')
                        cv2.setMouseCallback('标注窗口', mouse_callback)
                    cv2.imshow('标注窗口', frame)
                if save_img:
                    video_writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        else:
            # -------- RealSense .bag 分支（检测 + 计数 + 高度） --------
            for color_bgr, depth_m, intr in _iterate_bag_frames(source):
                frame = color_bgr
                h, w = frame.shape[:2]
                vid_frame_count += 1

                # 第一次拿到尺寸后创建 writer
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(str(out_path), fourcc, default_bag_fps, (w, h))

                # 持续重估地面平面 + EMA 平滑（允许相机有轻微移动）
                est = _estimate_ground_plane(depth_m, intr)
                if est is not None:
                    plane_nd = _ema_plane(plane_nd, est, alpha=PLANE_ALPHA)
                    if (vid_frame_count % PRINT_PLANE_EVERY) == 1:
                        print(f"[INFO] 平面更新: n={plane_nd[0]}, d={plane_nd[1]:.3f} m")
                # 若本帧无法估出，则沿用上一帧的 plane_nd

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

                        # 轨迹（不画线，保持和代码一一致）
                        if track_id is not None:
                            cx = int((x1+x2)/2); cy = int((y1+y2)/2)
                            tr = track_history[track_id]
                            tr.append((float(cx), float(cy)))
                            if len(tr) > 30: tr.pop(0)

                        # 高度（有平面时才算）
                        if plane_nd is not None:
                            h_raw = _height_from_box_plane(depth_m, (x1, y1, x2, y2), intr, plane_nd,
                                                           top_ratio=0.02, step=2)
                            # 兜底：整框稀疏采样 + 分位数
                            if h_raw is None:
                                pts_fb = _roi_points_from_depth(depth_m, intr, x1, y1, x2, y2, step=3)
                                if pts_fb is not None and pts_fb.shape[0] >= 20:
                                    hh = pts_fb @ plane_nd[0] + plane_nd[1]
                                    hh = hh[(hh > 0.0) & (hh < 1.5)]
                                    if hh.size >= 10:
                                        h_raw = float(np.percentile(hh, 95.0))

                            h_show = _smooth_height_by_id(track_id, h_raw) if track_id is not None else h_raw
                            if h_show is None and (track_id in heights_ema):
                                h_show = heights_ema[track_id]
                            if h_show is not None:
                                cv2.putText(frame, f"H={h_show * 100:.1f}cm",
                                            (x1, max(0, y1 - 8)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # 区域计数
                        bbox_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        for region in counting_regions:
                            if region['polygon'].contains(Point(bbox_center)):
                                if track_id is not None and track_id not in region['seen_ids']:
                                    region['counts'] += 1
                                    region['seen_ids'].add(track_id)

                # 区域可视化
                for region in counting_regions:
                    pts = np.array(region['polygon'].exterior.coords[:-1], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, region['region_color'], thickness=region_thickness)
                    text = f"{region['name']}: {region['counts']}"
                    x, y = int(pts[0][0][0]), int(pts[0][0][1]) - 10
                    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region['text_color'], 2)

                # 显示/保存（窗口名与代码一一致）
                if view_img:
                    if vid_frame_count == 1:
                        cv2.namedWindow('标注窗口')
                        cv2.setMouseCallback('标注窗口', mouse_callback)
                    cv2.imshow('标注窗口', frame)
                if save_img and video_writer is not None:
                    video_writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        if video_writer is not None:
            video_writer.release()
        if videocapture is not None:
            videocapture.release()
        cv2.destroyAllWindows()

    # 打印计数（保持风格）
    for region in counting_regions:
        print(f"区域: {region['name']}，计数: {region['counts']}")

# ============== 与代码一一致的 CLI ==============
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/best.pt', help='模型文件路径')
    parser.add_argument('--device', default='auto', help="推理设备: 'auto'|'cpu'|'0'")
    parser.add_argument('--source', type=str, default=None, help='视频或 .bag 文件路径（留空将弹出选择框）')
    parser.add_argument('--view-img', action='store_true', help='显示视频')
    parser.add_argument('--save-img', action='store_true', default=True, help='保存标注后的视频')
    parser.add_argument('--exist-ok', action='store_true', help='占位（保持接口一致）')
    parser.add_argument('--classes', nargs='+', type=int, help='过滤目标类别')
    parser.add_argument('--line-thickness', type=int, default=2, help='边框粗细')
    parser.add_argument('--track-thickness', type=int, default=2, help='追踪线粗细（此版本不绘制）')
    parser.add_argument('--region-thickness', type=int, default=4, help='区域线粗细')
    return parser.parse_args()

def resolve_source(src):
    """src 为空时弹窗选择；失败再用命令行输入兜底。"""
    if src is not None and Path(src).exists():
        return src

    # 1) 优先弹出文件选择框
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title='选择视频文件（支持 .mp4/.avi/.mov 或 .bag）',
            filetypes=[('视频/包文件', ('*.mp4', '*.avi', '*.mov', '*.bag')),
                       ('所有文件', '*.*')]
        )
        root.destroy()
        if path and Path(path).exists():
            return path
    except Exception:
        # 部分无图形界面环境会抛异常，留给命令行兜底
        pass

    # 2) 命令行兜底
    try:
        path = input('请输入视频或 .bag 路径：').strip().strip('"').strip("'")
    except EOFError:
        path = ''
    if not path or not Path(path).exists():
        raise FileNotFoundError('未指定有效的 --source，且未在弹窗/输入中选择到有效文件。')
    return path

def main(opt):
    opt.source = resolve_source(opt.source)
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
