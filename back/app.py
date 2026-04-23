from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from back.config.database import init_database
from back.routes.detection_routes import router as detection_router
from back.routes.saved_detection_routes import router as saved_detection_router
from back.services.detection_service import init_detectors

app = FastAPI(title="DefineLogic", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)
app.include_router(saved_detection_router)


@app.on_event("startup")
def startup_event() -> None:
    init_detectors()
    init_database()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
