#!/usr/bin/env python3
import os, cv2, sys, time, argparse, datetime, csv
import numpy as np

def main():
    parser = argparse.ArgumentParser("Record Intel RealSense streams to files")
    parser.add_argument("--out", type=str, default="rs_out", help="输出目录")
    parser.add_argument("--seconds", type=int, default=30, help="录制时长（秒）。<=0 表示手动模式（直到按 q 停止）")
    parser.add_argument("--width", type=int, default=640, help="宽度")
    parser.add_argument("--height", type=int, default=480, help="高度")
    parser.add_argument("--fps", type=int, default=15, help="帧率")
    parser.add_argument("--no-bag", action="store_true", help="不保存 .bag（默认会保存）")
    parser.add_argument("--no-color-mp4", action="store_true", help="不保存彩色视频（默认会保存）")
    parser.add_argument("--no-depth-vis", action="store_true", help="不保存深度可视化视频（默认会保存）")
    parser.add_argument("--save-depth-png", type=int, default=10, help="保存多少张16位深度PNG（0 关闭）")
    parser.add_argument("--max-depth-m", type=float, default=2.0, help="深度可视化最大范围（米）")
    args = parser.parse_args()

    try:
        import pyrealsense2 as rs
    except Exception as e:
        print("请先安装 pyrealsense2： pip3 install pyrealsense2")
        sys.exit(1)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, ts)
    os.makedirs(out_dir, exist_ok=True)

    print("输出目录：", out_dir)

    # ---------- 设备与管线 ----------
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("未检测到 RealSense 设备！")
        sys.exit(1)
    dev = ctx.devices[0]
    print("设备：", dev.get_info(rs.camera_info.name),
          "| 序列号：", dev.get_info(rs.camera_info.serial_number))

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    bag_path = os.path.join(out_dir, f"record_{ts}.bag")
    if not args.no_bag:
        cfg.enable_record_to_file(bag_path)
        print("将保存 .bag：", bag_path)

    profile = pipe.start(cfg)

    # 对齐：深度 -> 彩色
    align = rs.align(rs.stream.color)

    # 深度比例（把 z16 转米要乘这个）
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度比例（深度单位）：{depth_scale} m/单位")

    # 允许打开投射器，提升纹理
    for s in profile.get_device().sensors:
        if s.get_info(rs.camera_info.name) == "Stereo Module":
            if s.supports(rs.option.emitter_enabled):
                s.set_option(rs.option.emitter_enabled, 1)
            break

    # ---------- 视频写出（首帧到手后再初始化，确保分辨率匹配） ----------
    color_writer = None
    depth_writer = None

    # 记录元数据
    csv_path = os.path.join(out_dir, "frames_meta.csv")
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["index", "timestamp_ms", "depth_mean_m_in_roi"])

    # 深度 PNG 保存计数
    want_png = max(0, int(args.save_depth_png))
    saved_png = 0

    manual_mode = (args.seconds <= 0)
    if manual_mode:
        print("开始录制（手动模式）：按 q 停止。")
    else:
        print("开始录制（固定时长）：按 q 可提前停止 / 或自动到时长：", args.seconds, "秒")

    t0 = time.time()
    idx = 0
    try:
        while True:
            # 固定时长模式：到时停
            if (not manual_mode) and ((time.time() - t0) >= args.seconds):
                break

            frames = pipe.wait_for_frames()
            frames = align.process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue

            # numpy
            color_img = np.asanyarray(color.get_data())             # uint8, HxWx3, BGR
            depth_z16 = np.asanyarray(depth.get_data())             # uint16, z16

            H, W = color_img.shape[:2]
            if color_writer is None and not args.no_color_mp4:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                color_writer = cv2.VideoWriter(os.path.join(out_dir, "color.mp4"),
                                               fourcc, args.fps, (W, H))
                if not color_writer.isOpened():
                    print("[WARN] 彩色视频写入器打开失败（OpenCV编译可能不带mp4），将不保存 color.mp4")
                    color_writer = None

            if depth_writer is None and not args.no_depth_vis:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                depth_writer = cv2.VideoWriter(os.path.join(out_dir, "depth_vis.mp4"),
                                               fourcc, args.fps, (W, H))
                if not depth_writer.isOpened():
                    print("[WARN] 深度可视化视频写入器打开失败，将不保存 depth_vis.mp4")
                    depth_writer = None

            # 写彩色视频
            if color_writer is not None:
                color_writer.write(color_img)

            # 生成深度可视化帧（伪彩）
            if depth_writer is not None:
                depth_m = depth_z16.astype(np.float32) * depth_scale
                # 0~max_depth_m 映射到 0~255
                max_m = max(0.2, float(args.max_depth_m))
                depth_norm = np.clip((depth_m / max_m) * 255.0, 0, 255).astype(np.uint8)
                depth_cmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                depth_writer.write(depth_cmap)

            # 保存若干张 16位 原始深度PNG
            if saved_png < want_png:
                png_path = os.path.join(out_dir, f"depth_raw_{idx:06d}.png")
                # OpenCV 会按 16-bit PNG 写出，数值保持不变（单位仍为 z16）
                cv2.imwrite(png_path, depth_z16)
                saved_png += 1

            # 简单统计一个 ROI 的平均深度（中心 20x20）
            cx, cy, k = W // 2, H // 2, 10
            roi = depth_z16[max(0, cy-k):min(H, cy+k), max(0, cx-k):min(W, cx+k)]
            valid = roi[roi > 0]
            mean_m = float(valid.mean() * depth_scale) if valid.size else 0.0

            ts_ms = int(color.get_timestamp())
            csv_w.writerow([idx, ts_ms, round(mean_m, 4)])

            # 可视化窗口（叠加累计时长/帧数，手动模式更直观）
            vis = color_img.copy()
            elapsed = time.time() - t0
            cv2.rectangle(vis, (cx-k, cy-k), (cx+k, cy+k), (0,255,255), 2)
            status_txt = f"{'MANUAL' if manual_mode else 'FIXED'}  t={elapsed:6.1f}s  n={idx}"
            depth_txt  = f"mean_depth_center={mean_m:.3f} m"
            cv2.putText(vis, status_txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(vis, depth_txt,  (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            tip_txt = "Press 'q' to stop"
            cv2.putText(vis, tip_txt,    (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

            cv2.imshow("RealSense Recording (q to stop)", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        if color_writer is not None:
            color_writer.release()
        if depth_writer is not None:
            depth_writer.release()
        csv_f.close()
        pipe.stop()
        cv2.destroyAllWindows()

    print("\n录制完成！文件一览：")
    if not args.no_bag:
        print("  -", bag_path)
    if color_writer is not None:
        print("  -", os.path.join(out_dir, "color.mp4"))
    if depth_writer is not None:
        print("  -", os.path.join(out_dir, "depth_vis.mp4"))
    if want_png > 0:
        print(f"  - {saved_png} 张 16-bit 深度PNG（如 depth_raw_000000.png …）")
    print("  -", os.path.join(out_dir, "frames_meta.csv"))

if __name__ == "__main__":
    main()
