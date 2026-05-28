import torch
import torch.nn.functional as F

from .models.common import DetectMultiBackend
from .utils.augmentations import classify_transforms
from .utils.general import check_img_size
import cv2
import os


class FaceClassify:
    def __init__(self) -> None:
        self.weights = ""
        self.source = "",
        self.imgsz = (224, 224),
        self.device = '',
        self.half = False

    def load_model(self):
        # Load model
        self.device = "cpu" if self.device == "cpu" else f"cuda:{self.device}"
        self.device = torch.device(self.device)
        self.model = DetectMultiBackend(
            self.weights, device=self.device, fp16=self.half, dnn=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check image size
        self.transforms = classify_transforms(self.imgsz[0], half=self.half, device=self.device)

    @torch.no_grad()
    def predict(self, img):
        im = self.transforms(img)
        # im = torch.Tensor(im).to(self.model.device)
        # im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        results = self.model(im)
        pred = F.softmax(results, dim=1)  # probabilities
        for i, prob in enumerate(pred):  # per image
            # Print results
            top1i = prob.tolist()  # top 5 indices
            accuracy = max(top1i)
            cls = top1i.index(accuracy)
            return cls, accuracy

