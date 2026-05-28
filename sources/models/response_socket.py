import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
from loguru import logger


@dataclass
class SocketResponse:
    camera_id: int
    plate_text: str
    plate_relative_path: Optional[str] = None
    image_relative_path: Optional[str] = None
    overview_relative_path: Optional[str] = None
    video_relative_path: Optional[str] = None

    def __post_init__(self):
        self.date_created = datetime.now().strftime("%Y-%m-%d")
        self.time_created = datetime.now().strftime("%H:%M:%S")
        self.date_time = f"{self.date_created}_{self.time_created}".replace(":", "-")

    @staticmethod
    def _safe_name(value: str) -> str:
        safe_name = "".join(char for char in value if char.isalnum() or char in "-_")
        return safe_name or "unk"

    def get_and_save_relative_path(
        self,
        root,
        plate_text,
        image_plate,
        image_full,
        image_overview,
        param,
    ):
        param = param or "default"
        plate_name = self._safe_name(str(self.plate_text or plate_text or "unk"))
        save_root = os.path.join(
            root,
            str(self.camera_id),
            self.date_created,
            "app_images",
            param,
        )

        try:
            os.makedirs(save_root, exist_ok=True)
        except Exception as exc:
            logger.exception(f"Could not create storage directory: {exc}")

        if image_full is not None:
            filename = f"{plate_name}_{self.date_time}_full.jpg"
            self.image_relative_path = os.path.join(
                str(self.camera_id), self.date_created, "app_images", param, filename
            )
            save_image_path = os.path.join(save_root, filename)
            cv2.imwrite(save_image_path, image_full)
            logger.info(f"Saved image to: {save_image_path}")
        else:
            self.image_relative_path = ""

        if image_plate is not None:
            filename = f"{plate_name}_{self.date_time}_plate.jpg"
            self.plate_relative_path = os.path.join(
                str(self.camera_id), self.date_created, "app_images", param, filename
            )
            save_plate_path = os.path.join(save_root, filename)
            cv2.imwrite(save_plate_path, image_plate)
            logger.info(f"Saved plate image to: {save_plate_path}")
        else:
            self.plate_relative_path = ""

        if image_overview is not None:
            filename = f"{plate_name}_{self.date_time}_overview.jpg"
            self.overview_relative_path = os.path.join(
                str(self.camera_id), self.date_created, "app_images", param, filename
            )
            save_overview_path = os.path.join(save_root, filename)
            cv2.imwrite(save_overview_path, image_overview)
            logger.info(f"Saved overview image to: {save_overview_path}")
        else:
            self.overview_relative_path = ""

    def _response(self):
        return {
            "client_id": 1,
            "data": {
                "type": 1,
                "result": [
                    {
                        "camId": str(self.camera_id),
                        "Type": int(0),
                        "personIdCode": "",
                        "imageFace": "",
                        "plate": str(self.plate_text),
                        "imagePlateUrl": str(self.plate_relative_path),
                        "fullImageUrl": str(self.image_relative_path),
                        "overviewImageUrl": str(self.overview_relative_path),
                        "videoURL": "",
                        "accessTime": str(self.date_time),
                    }
                ],
            },
        }
