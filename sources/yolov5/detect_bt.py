import numpy as np
import torch
import torch.nn.functional as f
from ..yolov5.models.common import DetectMultiBackend
from ..yolov5.trackers.multi_tracker_zoo import create_tracker
from ..yolov5.utils.augmentations import letterbox, classify_transforms
from ..yolov5.utils.general import (check_img_size, non_max_suppression, scale_boxes)
from ..yolov5.utils.torch_utils import select_device
# from config import HALF_MODEL
import time
import cv2
from ..yolov5.trackers.ocsort.ocsort import OCSort
import torchvision.transforms as T

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


def relu(x):
    return max(0, int(x))


class Detection:
    def __init__(self):
        self._tracker = None
        self.model_file = ''
        self.imgsz = 640
        self.conf_thres = 0.5
        self.iou_thres = 0.25
        self.max_det = 1000
        self.device = 'cpu'
        self.classes = [0]
        self.agnostic_nms = True
        self.half = True

    def setup_model(self, a_model, classes, conf_thres, img_size, device, data_, half):
        self.conf_thres = conf_thres
        self.imgsz = img_size
        self.device = device
        self.model_file = a_model
        self.classes = classes
        self.half = half
        # self._tracker = create_tracker('ocsort', self.model_file, self.device, self.half)
        self._tracker = OCSort(
            det_thresh=0.45,
            iou_threshold=0.2,
            use_byte=False
        )
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.model_file, device=self.device, data=data_, fp16=self.half)
        self.model.eval()
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def preprocess_img(self, origin_img):
        img = letterbox(origin_img, new_shape=self.imgsz, stride=self.stride, auto=self.pt)[0]
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    @torch.no_grad()
    def detect(self, processed_img, origin_img):
        detect_list = []

        pred = self.model(processed_img)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                   self.agnostic_nms,
                                   max_det=self.max_det)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                det = det.cpu().numpy()
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(processed_img.shape[2:], det[:, :4], origin_img.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = list(map(relu, xyxy))
                    detect_list.append([x1, y1, x2, y2, int(cls), float(conf)])
        return detect_list


class Tracking(Detection):
    def __init__(self):
        super().__init__()

    def setup_model(self, a_model, classes, conf_thres, img_size, device, data, half):
        self.conf_thres = conf_thres
        self.imgsz = img_size
        self.device = device
        self.model_file = a_model
        self.classes = classes
        self.half = half
        self._tracker = create_tracker('ocsort', self.model_file, self.device, self.half)
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.model_file, device=self.device, data=data, fp16=self.half)
        self.model.eval()
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        if self.half:
            self.model.half()  # to FP16

    @torch.no_grad()
    def track(self, processed_img, origin_img):
        id_dict = {}
        pred = self.model(processed_img, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                   self.agnostic_nms, max_det=self.max_det)

        for det in pred:
            if det is not None and len(det):
                det = det.detach().cpu()
                det[:, :4] = scale_boxes(processed_img.shape[2:], det[:, :4], origin_img.shape).round()
                outputs = self._tracker.update(det, origin_img)
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, det[:, 4])):
                        x1, y1, x2, y2 = list(map(relu, output[0:4]))
                        ids = output[4]
                        cls = output[5]
                        x_min, y_min, x_max, y_max = int(x1), int(y1), int(x2), int(y2)
                        y_max = int(y_max * 1)
                        id_dict[ids] = (x_min, y_min, x_max, y_max, cls)
        return id_dict


class VehicleTracking(Tracking):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def track(self, processed_img, origin_img):
        id_dict = {}
        pred = self.model(processed_img)
        pred = non_max_suppression(pred,
                                   self.conf_thres,
                                   self.iou_thres,
                                   self.classes,
                                   self.agnostic_nms,
                                   max_det=self.max_det)
        for det in pred:
            if det is None:
                continue

            if not len(det):
                continue

            det = det.detach().cpu()
            det[:, :4] = scale_boxes(processed_img.shape[2:], det[:, :4], origin_img.shape).round()
            outputs = self._tracker.update(det, origin_img)
            if len(outputs):
                for j, (output, conf) in enumerate(zip(outputs, det[:, 4])):
                    x1, y1, x2, y2 = list(map(relu, output[0:4]))
                    ids = output[4]
                    cls = output[5]
                    if cls == 0:
                        y2 = int(y2 * 1.05)  # Expand height for bottom car
                    id_dict[ids] = (x1, y1, x2, y2, cls)
        return id_dict


class Classify:
    def __init__(self):
        super().__init__()
        self.pt = None
        self.stride = None
        self.imgsz = (112, 112)
        self.device = '0'
        self.half = False
        self.dnn = False
        self.model = None
        self.file_model = None
        self.names = None
        self.auto = False

    def setup_model(self, a_model, classes, conf_thres, img_size, device):
        self.conf_thres = conf_thres
        self.imgsz = img_size
        self.device = device
        self.model_file = a_model
        self.classes = classes

        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.model_file, device=self.device, dnn=self.dnn, fp16=self.half)
        self.model.eval()
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.transforms = classify_transforms(self.imgsz[0], half=self.half, device=self.device)

    @torch.no_grad()
    def classify(self, image):
        result = []
        im = self.transforms(image)
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        results = self.model(im)
        pred = f.softmax(results, dim=1)  # probabilities
        conf = []
        for i, prob in enumerate(pred):  # per image
            top5i = prob.argsort(0, descending=True)[:5].tolist()  # top 5 indices
            result = top5i
            conf = prob.tolist()
        return self.names[result[0]], conf[result[0]]
