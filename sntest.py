import pyrealsense2 as rs
ctx = rs.context()
for dev in ctx.query_devices():
    print("Name:", dev.get_info(rs.camera_info.name),
          "SN:", dev.get_info(rs.camera_info.serial_number))
