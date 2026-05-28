from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AppDataRequest(BaseModel):
    image: Optional[str] = Field(default=None, description="Image encoded as base64 JPEG/PNG")
    imageOverview: Optional[str] = Field(
        default=None,
        description="Overview image encoded as base64 JPEG/PNG",
    )
    cameraId: Optional[int] = Field(default=None, description="Camera identifier")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image": "/9j/*****",
                "imageOverview": "/9j/*****",
                "cameraId": 1,
            }
        }
    )
