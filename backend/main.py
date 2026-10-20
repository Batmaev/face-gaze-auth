from fastapi import FastAPI

from routers import registration_router, liveness_router
from log_decorator import RequestContextMiddleware


# Caddy и Vite срезают /api перед проксированием. Без префикса Swagger
# запрашивает /openapi.json с корня сайта и попадает во фронтенд.
app = FastAPI(
    title="Face Gaze Auth API",
    version="1.0.0",
    root_path="/api",
)

app.add_middleware(RequestContextMiddleware)

app.include_router(registration_router)
app.include_router(liveness_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok"}

