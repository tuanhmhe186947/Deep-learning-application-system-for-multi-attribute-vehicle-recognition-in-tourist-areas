from typing import List

from pydantic import BaseModel, ConfigDict, Field


class PlateBs64Response(BaseModel):
    cameraId: int
    plateBox: List[int]
    plateText: str
    typePlate: int
    confidence: float = Field(default=0.0, ge=0.0)
    imgBs64: str
    status: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cameraId": 1,
                "plateBox": [1, 2, 3, 4],
                "plateText": "88A88888",
                "typePlate": 1,
                "confidence": 0.92,
                "imgBs64": "/9j//******",
                "status": 200,
            }
        }
    )
