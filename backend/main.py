from fastapi import FastAPI

from routers import registration_router, liveness_router
from log_decorator import RequestContextMiddleware


app = FastAPI(title="Face Gaze Auth API", version="1.0.0")

app.add_middleware(RequestContextMiddleware)

app.include_router(registration_router)
app.include_router(liveness_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok"}

