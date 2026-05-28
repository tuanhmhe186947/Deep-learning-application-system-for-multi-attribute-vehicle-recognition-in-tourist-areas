from pydantic import BaseModel, ConfigDict


class DigitResponse(BaseModel):
    cameraId: int
    plateText: str
    typePlate: int
    status: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cameraId": 1,
                "plateText": "88A88888",
                "typePlate": 1,
                "status": 200,
            }
        }
    )
