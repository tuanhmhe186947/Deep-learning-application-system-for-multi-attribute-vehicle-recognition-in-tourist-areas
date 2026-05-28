from typing import List

from pydantic import BaseModel, ConfigDict


class PlateResponse(BaseModel):
    cameraId: int
    plateBox: List[int]
    plateText: str
    typePlate: int
    status: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cameraId": 1,
                "plateBox": [1, 2, 3, 4],
                "plateText": "88A88888",
                "typePlate": 1,
                "status": 200,
            }
        }
    )
