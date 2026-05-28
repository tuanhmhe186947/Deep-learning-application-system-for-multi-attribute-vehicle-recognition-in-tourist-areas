from pydantic import BaseModel, ConfigDict, Field


class DataRequest(BaseModel):
    image: str = Field(..., min_length=1, description="Image encoded as base64 JPEG/PNG")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image": "/9j/*****",
            }
        }
    )
