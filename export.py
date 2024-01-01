import torch
from models.yolo import Model


if __name__ == "__main__":
    model = Model(cfg="./models/yolov4-p5.yaml")
    x = torch.randn(1, 3, 640, 640)
    try:
        torch.export.export(model, (x, ))
        print ("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print ("[JIT] torch.export failed.")
        raise e
