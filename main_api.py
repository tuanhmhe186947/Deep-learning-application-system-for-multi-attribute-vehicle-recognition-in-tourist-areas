import time

from fastapi import FastAPI

from sources.controller.main_controller import api_controller
from sources.models import AppDataRequest, DataRequest, MainConfig
from sources.util.common import set_logger


app = FastAPI(
    title="Multi-Task Vehicle Recognition API",
    description="Vehicle detection, license plate detection, and license plate OCR service.",
    version="1.0.0",
)
main_config = MainConfig()

time_logger = set_logger(key="timer_log")


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok"}


@app.get("/ready", tags=["System"])
async def readiness_check():
    return api_controller.readiness()


@app.post("/detect/vehicle", description="Detect all configured vehicles", tags=["Detect"])
async def detect_vehicle(payload: DataRequest):
    start = time.time()
    result = await api_controller.detect_vehicle_api(payload)
    time_logger.info(f"vehicle_detect {round(time.time() - start, 5)}s")
    return result


@app.post("/detect/car", description="Detect cars", tags=["Detect"])
async def detect_car(payload: DataRequest):
    start = time.time()
    result = await api_controller.detect_car(payload)
    time_logger.info(f"car_detect {round(time.time() - start, 5)}s")
    return result


@app.post("/detect/moto", description="Detect motorcycles", tags=["Detect"])
async def detect_moto(payload: DataRequest):
    start = time.time()
    result = await api_controller.detect_moto(payload)
    time_logger.info(f"moto_detect {round(time.time() - start, 5)}s")
    return result


@app.post("/detect/plate", description="Detect license plates", tags=["Detect"])
async def detect_plate(payload: DataRequest):
    start = time.time()
    result = await api_controller.detect_plate_api(payload)
    time_logger.info(f"plate_detect {round(time.time() - start, 5)}s")
    return result


@app.post("/ocr/plate", description="Recognize license plate text", tags=["OCR"])
async def ocr_plate(payload: DataRequest):
    start = time.time()
    result = await api_controller.ocr_plate(payload)
    time_logger.info(f"plate_ocr {round(time.time() - start, 5)}s")
    return result


@app.post("/recognize/plate", description="Detect and recognize the best license plate", tags=["Recognize"])
async def recognize_plate(payload: AppDataRequest):
    start = time.time()
    result = await api_controller.recognize_plate(payload)
    time_logger.info(f"plate_recognize {round(time.time() - start, 5)}s")
    return result


@app.post(
    "/recognize/plate/store",
    description="Detect, recognize, and store license plate images",
    tags=["Recognize"],
)
async def recognize_plate_and_store(payload: AppDataRequest):
    start = time.time()
    result = await api_controller.recognize_plate_and_store(payload)
    time_logger.info(f"plate_recognize_store {round(time.time() - start, 5)}s")
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=main_config.HOST,
        port=main_config.PORT,
        reload=main_config.RELOAD,
        timeout_keep_alive=main_config.TIMEOUT_KEEP_ALIVE,
    )
