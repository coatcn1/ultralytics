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
import os

# 允许多线程使用
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 全局变量初始化
track_history = defaultdict(list)
current_region = None  # 用于记录正在拖拽的区域

# 定义计数区域
counting_regions = [
    {
        'name': 'YOLOv8 Polygon Region',
        'polygon': Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),
        'counts': 0,
        'seen_ids': set(),  # 初始化 ID 记录
        'dragging': False,
        'region_color': (255, 42, 4),
        'text_color': (255, 255, 255)
    },
    {
        'name': 'YOLOv8 Rectangle Region',
        'polygon': Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),
        'counts': 0,
        'seen_ids': set(),  # 初始化 ID 记录
        'dragging': False,
        'region_color': (37, 255, 225),
        'text_color': (0, 0, 0)
    }
]

def mouse_callback(event, x, y, flags, param):
    global current_region  # 声明为全局变量

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
            # 更新区域的坐标
            current_region['polygon'] = Polygon([
                (p[0] + dx, p[1] + dy) for p in current_region['polygon'].exterior.coords])
            current_region['offset_x'] = x
            current_region['offset_y'] = y

    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region['dragging']:
            current_region['dragging'] = False

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
    source  : 视频路径
    device  : 'auto' | 'cpu' | '0' / 'cuda' / 'cuda:0'
              'auto' 会优先使用 GPU（若可用），否则退回 CPU
    其余参数保持不变
    """
    import torch  # 函数内部引用，避免全局依赖

    # ---------- ① 解析 device ----------
    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'

    # ---------- ② 校验输入 ----------
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # ---------- ③ 加载模型并放到指定设备 ----------
    model = YOLO(weights)
    if device in ('0', 'cuda', 'cuda:0'):
        model.to('cuda:0')
        print('[INFO] 推理设备: GPU (cuda:0)')
    else:
        model.to('cpu')
        print('[INFO] 推理设备: CPU')

    names = model.model.names

    # ---------- ④ 打开视频 ----------
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps        = int(videocapture.get(5))
    fourcc     = cv2.VideoWriter_fourcc(*'mp4v')

    save_dir = increment_path(Path('ultralytics_rc_output') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(
        str(save_dir / f'{Path(source).stem}.mp4'),
        fourcc, fps,
        (frame_width, frame_height)
    )

    # ---------- ⑤ 主循环 ----------
    vid_frame_count = 0
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        results = model.track(frame, persist=True, classes=classes)

        # ====== 原有绘制/计数逻辑保持不变 ======
        if results[0].boxes.id is not None:
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

        # 绘制计数区域及计数文本
        for region in counting_regions:
            pts = np.array(region['polygon'].exterior.coords[:-1], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, region['region_color'], thickness=region_thickness)
            text = f"{region['name']}: {region['counts']}"
            x, y = int(pts[0][0][0]), int(pts[0][0][1]) - 10
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region['text_color'], 2)

        # ====== 显示 / 保存 ======
        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow('Ultralytics YOLOv8 Region Counter Movable')
                cv2.setMouseCallback('Ultralytics YOLOv8 Region Counter Movable', mouse_callback)
            cv2.imshow('Ultralytics YOLOv8 Region Counter Movable', frame)

        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---------- ⑥ 资源释放 ----------
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()

    # ---------- ⑦ 打印计数 ----------
    for region in counting_regions:
        print(f"Region: {region['name']}, Total Counts: {region['counts']}")

def select_file():
    # 使用多个扩展名的元组作为文件过滤条件
    file_path = filedialog.askopenfilename(
        title='选择视频文件',
        filetypes=[('视频文件', ('*.mp4', '*.avi', '*.mov')), ('所有文件', '*.*')]
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
        # device 不再写死，由 run() 内部自动判定
        run(weights=weights,
            source=source,
            device='auto',        # ← 关键：自动优先 GPU
            view_img=True,
            save_img=True)
        messagebox.showinfo('完成', '视频处理完成，输出文件已保存。')
    except Exception as e:
        messagebox.showerror('错误', f'处理视频时发生错误：{e}')

# 创建主窗口并使用 ttk 改善样式
root = tk.Tk()
root.title('视频处理工具')
root.geometry("700x250")

# 使用 ttk.Style 设置统一字体和样式
style = ttk.Style(root)
style.configure('TLabel', font=('Helvetica', 14))
style.configure('TButton', font=('Helvetica', 14))
style.configure('TEntry', font=('Helvetica', 14))

# 定义全局变量存储文件路径
model_path_var = tk.StringVar()
video_path_var = tk.StringVar()

# 使用 ttk 控件布局
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
