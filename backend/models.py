from typing import Literal
from pydantic import BaseModel, Field


# --- Registration Models ---

class RegisterResponse(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the registered user")

    model_config = {
        "json_schema_extra": {
            "examples": [{"user_id": "abc123"}]
        }
    }


# --- Liveness Models ---

class StimulusResponse(BaseModel):
    type: Literal["moving-dot"] = Field("moving-dot", description="Type of stimulus")
    fps: int = Field(60, description="Frames per second")
    duration_sec: int = Field(5, description="Duration in seconds")
    token: str = Field(..., description="Stimulus will be saved on the server for 3 minutes with this token")
    x: list[float]
    y: list[float]
    ctrl: list[list[float]] = Field(
        ...,
        description="Spline control points, shape (n, 2). Shorter than knots plus coefficients; another worker can rebuild x, y from them",
    )


class LivenessResponse(BaseModel):
    is_live: bool = Field(..., example=True)
    score: float = Field(
        ...,
        description="1 minus the conformal p-value of the shown stimulus among random candidates. is_live is score >= 0.96",
        example=0.98,
    )
    gaze_x: list[float | None] = Field(
        ...,
        description="Min-max normalized gaze on the compared window, null where the face was missing",
    )
    gaze_y: list[float | None]
    inlier: list[bool] = Field(
        ...,
        description="Frame is within 0.15 of the stimulus on both axes after min-max. Blinks are false",
    )
    fps: float = Field(..., description="Frame rate of gaze_x, gaze_y and inlier", example=15)
