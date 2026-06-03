import cv2
import numpy as np

from sources.models.digit_config import digit_config


def normalize_bbox(bbox) -> list:
    if bbox is None or len(bbox) < 4:
        return []

    try:
        return [int(round(float(value))) for value in bbox[:4]]
    except (TypeError, ValueError):
        return []


def bbox_area(bbox) -> int:
    box = normalize_bbox(bbox)
    if not box:
        return 0

    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def clip_bbox(bbox, image_shape, padding=0) -> list:
    box = normalize_bbox(bbox)
    if not box or image_shape is None or len(image_shape) < 2:
        return []

    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return []

    return [x1, y1, x2, y2]


def crop_bbox(image, bbox, padding=0):
    box = clip_bbox(bbox, image.shape if image is not None else None, padding=padding)
    if not box:
        return None, []

    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None, []

    return crop, box


def max_size_boundingbox(list_bbox) -> list:
    if not list_bbox:
        return []

    return max(list_bbox, key=bbox_area)


def check_plate(style, plate) -> bool:
    if len(plate) != len(style):
        return False

    for i in range(len(style)):
        if style[i] == "N":
            if not plate[i].isdigit():
                return False
        elif style[i] == "C":
            if not plate[i].isalpha():
                return False

    return True


def check_alpha_digit(plate, list_style_digit, list_style_alpha) -> bool:
    ss = plate[:2]
    if ss.isdigit():
        if ss in list_style_digit:
            return False
    if ss.isalpha():
        if ss in list_style_alpha:
            return True
        return False
    return True


def filter_text_digit(plate, list_style) -> bool:
    for style in list_style:
        if check_plate(style, plate):
            return True
    return False


def get_angle(image, bbox) -> float:
    cy = image.shape[0] // 2
    l1 = []
    l2 = []

    for item in bbox:
        x1, y1, x2, y2 = item[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        _, cby = (x1 + x2) // 2, (y1 + y2) // 2
        if cby < cy:
            l1.append(item)
        else:
            l2.append(item)

    if len(l1) == 0 or len(l2) == 0:
        chosen_line = l1 if len(l1) > 0 else l2
    else:
        chosen_line = l2

    two_point = sorted(chosen_line, key=lambda x: x[0], reverse=True)[:2]
    center_two_point = [
        ((item[0] + item[2]) // 2, (item[1] + item[3]) // 2)
        for item in two_point
    ]
    center_two_point = sorted(center_two_point, key=lambda x: x[0])

    if len(center_two_point) < 2:
        return 0.0

    vector_1 = np.array(
        [
            center_two_point[1][0] - center_two_point[0][0],
            center_two_point[1][1] - center_two_point[0][1],
        ],
        dtype=np.float32,
    )
    vector_2 = np.array([0, image.shape[1]], dtype=np.float32)
    norm_1 = np.linalg.norm(vector_1)
    norm_2 = np.linalg.norm(vector_2)
    if norm_1 == 0 or norm_2 == 0:
        return 0.0

    unit_vector_1 = vector_1 / norm_1
    unit_vector_2 = vector_2 / norm_2
    dot_product = np.clip(np.dot(unit_vector_1, unit_vector_2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    angle_in_degree = np.degrees(angle)
    if angle_in_degree > 90:
        angle_in_degree = -np.abs(angle_in_degree - 90)
    else:
        angle_in_degree = np.abs(angle_in_degree - 90)

    return angle_in_degree


def rotate(image, angle) -> np.ndarray:
    h, w = image.shape[:2]
    c_x, c_y = w // 2, h // 2
    matrix = cv2.getRotationMatrix2D((c_x, c_y), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def check_type_plate(line_1) -> int:
    line = ""
    for item in line_1:
        _, _, _, _, label, _ = item
        line = line + str(label)
    line = line.upper()

    if not line:
        return -1
    if len(line) < 3:
        if not line[0].isalpha():
            return 0
        return 1
    if "LB" in line or "LD" in line:
        return 1
    if filter_text_digit(line, digit_config.STYLE_ALPHA_CAR):
        return 1
    return 0


def process_square_lp(bboxes):
    line_1 = []
    line_2 = []
    all_y = [box[1] for box in bboxes]
    if len(all_y) == 0:
        return [], -1

    average_y = sum(all_y) / len(all_y)
    for bbox in bboxes:
        if bbox[1] < average_y:
            line_1.append(bbox)
        else:
            line_2.append(bbox)

    line_1 = sorted(line_1, key=lambda x: x[0])
    line_2 = sorted(line_2, key=lambda x: x[0])
    return line_1 + line_2, check_type_plate(line_1)


def is_square_lp(id_list) -> bool:
    if not len(id_list):
        return False

    list_x = []
    length_digit = 0
    for result in id_list:
        box = normalize_bbox(result)
        if not box:
            continue
        x1, _, x2, _ = box
        list_x.append(x1)
        list_x.append(x2)
        length_digit += x2 - x1

    if length_digit <= 0 or len(list_x) < 2:
        return False

    list_x.sort()
    length_plate = list_x[-1] - list_x[0]
    return (length_plate / length_digit) < digit_config.CONF_PLATE_SQUARE
