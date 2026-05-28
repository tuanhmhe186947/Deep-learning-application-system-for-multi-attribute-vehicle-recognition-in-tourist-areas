import argparse
import base64
import time
from pathlib import Path

import cv2
import requests

DEFAULT_HOST = "http://127.0.0.1:8484"


def image_to_base64(img):
    ok, buffer = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("Could not encode image as JPEG")
    return base64.b64encode(buffer).decode("ascii")


def draw_boxes(img, boxes, color=(0, 255, 0), label=""):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box["bbox"])
        conf = box.get("confidence", 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return img


def call_api(host, endpoint, image_b64):
    url = f"{host.rstrip('/')}{endpoint}"
    try:
        response = requests.post(url, json={"image": image_b64}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        print(f"[ERROR] {endpoint}: {exc}")
        return None


def test_image(path, host, output_dir):
    img = cv2.imread(str(path))
    if img is None:
        print(f"Could not read image: {path}")
        return

    image_b64 = image_to_base64(img)

    car_res = call_api(host, "/detect/car", image_b64)
    if car_res:
        print("\n[CAR]")
        print(car_res)
        img = draw_boxes(img, car_res.get("boxes", []), (0, 255, 0), "car")

    moto_res = call_api(host, "/detect/moto", image_b64)
    if moto_res:
        print("\n[MOTO]")
        print(moto_res)
        img = draw_boxes(img, moto_res.get("boxes", []), (255, 0, 0), "moto")

    plate_res = call_api(host, "/detect/plate", image_b64)
    plate_boxes = plate_res.get("plates", []) if plate_res else []
    if plate_res:
        print("\n[PLATE DETECT]")
        print(plate_res)
        img = draw_boxes(img, plate_boxes, (0, 0, 255), "plate")

    for idx, plate in enumerate(plate_boxes):
        x1, y1, x2, y2 = map(int, plate["bbox"])
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        ocr = call_api(host, "/ocr/plate", image_to_base64(crop))
        if not ocr:
            continue

        print(f"\n[OCR PLATE {idx}]")
        print(ocr)

        text = ocr.get("text", "")
        conf = ocr.get("confidence", 0)
        type_plate = ocr.get("typePlate", -1)
        cv2.putText(
            img,
            f"{text} | {conf:.2f} | T:{type_plate}",
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"result_{int(time.time())}.jpg"
    cv2.imwrite(str(output_path), img)
    print(f"\nSaved annotated image: {output_path}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test the vehicle recognition API")
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--host", default=DEFAULT_HOST, help="API host URL")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    test_image(args.image, args.host, args.output_dir)


if __name__ == "__main__":
    main()
