from typing import List

from pydantic import BaseModel, ConfigDict


class VehicleResponse(BaseModel):
    vehicleBox: List[List[int]]
    status: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vehicleBox": [[1, 2, 3, 4], [5, 6, 7, 8]],
                "status": 200,
            }
        }
    )
