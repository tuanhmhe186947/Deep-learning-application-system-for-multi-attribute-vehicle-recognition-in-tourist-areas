import os
import cv2
import numpy as np
import time
# import onnx
# from onnx2pytorch import ConvertModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

sys.path.insert(0, './')
import torch
import torch.nn.functional as F

from ..yolov5.models.common import DetectMultiBackend
from ..yolov5.utils.augmentations import letterbox, classify_transforms
from ..yolov5.models.experimental import attempt_load
from ..yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes, scale_segments
from ..yolov5.utils.torch_utils import select_device
# from yolov5.sort import *
from ..yolov5.utils.segment.general import process_mask, masks2segments


def relu(x):
    return max(0, int(x))


class Tracking():
    def __init__(self):
        self.model_file = ''
        self.imgsz = 640  # inference size (height, width)
        self.conf_thres = 0.5  # confidence threshold
        self.iou_thres = 0.3  # NMS IOU threshold
        self.max_det = 25  # maximum detections per image
        self.device = 0  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.classes = [0, 1]  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = True  # class-agnostic NMS
        self.line_thickness = 1  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference

        # self.sort_tracker = Sort(max_age=150, min_hits=10, iou_threshold=0.3)

    def setup_model(self, a_model, classes, conf_thres, img_size, device):
        self.conf_thres = conf_thres
        self.imgsz = img_size
        self.device = device
        self.model_file = a_model
        self.classes = classes
        self.device = select_device(self.device)
        self.model = attempt_load(
            self.model_file, device=self.device)
        stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=stride)
        if self.half:
            self.model.half()

    def detect(self, img):
        with torch.no_grad():
            id_dict = {}
            id_list = []
            img_copy = img.copy()
            img = letterbox(img, new_shape=self.imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                       self.agnostic_nms,
                                       max_det=self.max_det)
            # Process detections
            dets_to_sort = np.empty((0, 6))
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(
                        img.shape[2:], det[:, :4], img_copy.shape).round()

                    for x1, y1, x2, y2, conf, cls in det.cpu().numpy():
                        dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, cls])))

            tracked_det = self.sort_tracker.update(dets_to_sort)
            if len(tracked_det) > 0:
                bbox_xyxy = tracked_det[:, :4]
                indentities = tracked_det[:, 8]
                categories = tracked_det[:, 4]
                for i in range(len(bbox_xyxy)):
                    x1, y1, x2, y2 = list(map(relu, bbox_xyxy[i]))
                    y2 = int(y2 * 1)
                    id_ = int(indentities[i])
                    id_dict[id_] = (x1, y1, x2, y2, categories[i])
                    id_list.append((x1, y1, x2, y2, categories[i], id_))

            return id_list


class Detection():
    def __init__(self, dnn):
        self.model_file = ''
        self.imgsz = 640  # inference size (height, width)
        self.conf_thres = 0.5  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = 0  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = True  # class-agnostic NMS
        self.half = False  # use FP16 half-precision inference
        self.dnn = dnn  # use OpenCV DNN for ONNX inference
        self.name = None
        self.stride = 32

    def setup_model(self, a_model, classes, conf_thres, img_size, device):
        self.conf_thres = conf_thres
        self.imgsz = img_size
        self.device = device
        self.model_file = a_model
        self.classes = classes
        self.device = select_device(self.device)
        # self.data_yml = data_yml
        # self.model = attempt_load(
        #     self.model_file, device=self.device)  # load FP32 model

        self.model = DetectMultiBackend(a_model, device=self.device, dnn=self.dnn, fp16=self.half)

        # stride = int(self.model.stride.max())  # model stride
        self.stride = int(self.model.stride)
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names

    def detect(self, img):
        with torch.no_grad():
            detect_list = []
            img_copy = img.copy()
            # img = cv2.resize(img, (256, 256))
            # print(img.shape)
            img = letterbox(img, new_shape=self.imgsz, stride=self.stride, auto=False)[0]
            # print(img.shape)
            # Convert
            # BGR to RGB, to 3x416x416
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img[None]
            # Inference
            pre_time = time.time()
            pred = self.model(img, augment=False)
            # print("pred shape: ", pred.shape)
            # print("inference_time: ", int((time.time() - pre_time) * 1000), "ms")
            # Apply NMS
            pred = non_max_suppression(pred,
                                       self.conf_thres,
                                       self.iou_thres,
                                       self.classes,
                                       self.agnostic_nms,
                                       max_det=self.max_det)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_copy.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = list(map(relu, xyxy))
                        detect_list.append([x1, y1, x2, y2, self.names[int(cls)], float(conf)])
            return detect_list


class Classify():
    def __init__(self):
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

    def setup_model(self, model_file):
        self.file_model = model_file
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.file_model, device=self.device, dnn=self.dnn, data=None, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.transforms = classify_transforms(self.imgsz[0], half=self.half, device=self.device)

    def classify(self, image):
        result = []
        image_copy = image.copy()
        # self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
        im = self.transforms(image_copy)
        im = torch.Tensor(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        results = self.model(im)
        pred = F.softmax(results, dim=1)  # probabilities
        for i, prob in enumerate(pred):  # per image
            top5i = prob.argsort(0, descending=True)[:5].tolist()  # top 5 indices
            result = top5i

        return self.names[result[0]]


class Segment:
    def __init__(self):
        self.model_file = r'E:\ProjectAtin\Plate\weight\yolov5m-seg.pt'
        self.imgsz = 640  # inference size (height, width)
        self.conf_thres = 0.5  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = 'cuda:0'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = True  # class-agnostic NMS
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.name = None
        self.retina_masks = False

        self.model = None

    def setup(self, model_file, classes, conf_thres, img_size, device):
        self.model_file = model_file

        self.conf_thres = conf_thres
        self.imgsz = img_size
        self.device = device
        self.classes = classes

        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.model_file, device=self.device, dnn=self.dnn, data=None, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        print('names: ', self.names)
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # self.transforms = classify_transforms(self.imgsz[0], half=self.half, device=self.device)

    def segment(self, img):
        detect_list = []
        img_copy = img
        img = letterbox(img, new_shape=self.imgsz)[0]
        # Convert

        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference

        pred, proto = self.model(img, augment=False)[:2]

        # Apply NMS
        pred = non_max_suppression(pred,
                                   self.conf_thres,
                                   self.iou_thres,
                                   self.classes,
                                   self.agnostic_nms,
                                   max_det=self.max_det)

        for i, det in enumerate(pred):
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = list(map(relu, xyxy))
                    detect_list.append([x1, y1, x2, y2, float(cls), float(conf)])
        return detect_list


if __name__ == "__main__":
    detect = Detection()
    detect.setup_model(r"E:\ProjectAtin\CaoToc\weight\yolov5s.pt",
                       None, 0.25, 640, 'cuda:0')
    path = r"E:\ProjectAtin\Plate\video\camera 3006 14h35 21-03-2023.mp4"
    cap = cv2.VideoCapture(path)
    # yolov8.model.warmup(imgsz=(1, 3, *yolov8.imgsz))
    while True:
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(path)
            continue
        id_list = detect.detect(frame)
        for result in id_list:
            x1, y1, x2, y2, conf, cls = result
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, str(int(cls)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        frame = cv2.resize(frame, (600, 400))
        cv2.imshow("yolov5", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
