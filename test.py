# # import sys
# # sys.path.append("autodl-tmp/yolov8/")
#
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # 直接使用预训练模型创建模型
    # model = YOLO('yolov8n.pt')
    # model.train(**{'cfg':'ultralytics/cfg/default.yaml', 'data':'ultralytics/models/yolo/detect/mydata/traffic.yaml'}, epochs=10, imgsz=640, batch=32)

    #使用yaml配置文件来创建模型，并导入预训练权重
    # model = YOLO('ultralytics/cfg/models/myyaml/yolov8-MobileNetV3_2.yaml')  # build a new model from YAML
    # model.load('yolov8n.pt')
    # model.train(**{'cfg': 'ultralytics/cfg/default.yaml', 'data': 'ultralytics/cfg/datasets/Mydata.yaml'},
    #             epochs=500, imgsz=640, batch=32, name='train')  # name：是此次训练结果保存的文件夹   数据集是我自己的数据集

# #     # 模型验证：用验证集
#     model = YOLO('runs/detect/train/weights/best.pt')
#     model.val(**{'data':'ultralytics/models/yolo/detect/mydata/traffic.yaml', 'name':'val', 'batch':32}) #模型验证用验证集
#     model.val(**{'data':'ultralytics/models/yolo/detect/mydata/traffic.yaml', 'split':'test', 'iou':0.9}) #模型验证用测试集

    # 模型推理：
    model = YOLO('runs/detect/train63/weights/best.pt')
    model.predict(source='D:/y/ultralytics-main/test/lettuce2', name='predict', **{'save':True})


