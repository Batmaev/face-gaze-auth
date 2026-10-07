from typing import Literal
from pydantic import BaseModel, Field

from liveness.example_values import ExampleValues


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


class LivenessResponse(BaseModel):
    is_live: bool = Field(..., example=True)
    score: float = Field(..., example=0.85)
    gaze_fit_x: list[float]
    gaze_fit_y: list[float]
    inlier: list[bool]
    lag: int = Field(
        ..., 
        description="Estimated lag between stimulus and gaze in frames",
        example=10
    )


LIVENESS_EXAMPLE = {
    "stim_x": ExampleValues.stim_x,
    "stim_y": ExampleValues.stim_y,
    "gaze_x": ExampleValues.gaze_x,
    "gaze_y": ExampleValues.gaze_y,
    "left_eye_blink": ExampleValues.left_eye_blink,
    "right_eye_blink": ExampleValues.right_eye_blink,
}

