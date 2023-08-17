if __name__ == '__main__':
    import torch
    import ultralytics
    from ultralytics import YOLO
    # print(torch.cuda.is_available())
    # print('='*20)
    # print(ultralytics.checks())


    model = YOLO('yolov8n.pt')
    model.train(data='C:/Users/user/PycharmProjects/yoloproject/AQUA_PROJ-2/data.yaml', epochs=30, imgsz=640, batch = 8, workers=3, device=0)


