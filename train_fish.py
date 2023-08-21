if __name__ == '__main__':
    import torch
    import ultralytics
    from ultralytics import YOLO
    # print(torch.cuda.is_available())
    # print('='*20)
    # print(ultralytics.checks())


    model = YOLO('yolov8n.pt')
    model.train(data=r'C:\Users\user\PycharmProjects\yoloproject\Fish-44\data.yaml', epochs=30, imgsz=640, batch = 8, workers=3, device=0)
