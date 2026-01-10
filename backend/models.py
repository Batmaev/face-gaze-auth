from typing import List, Literal
from pydantic import BaseModel, Field

from liveness.example_values import ExampleValues


# --- Face ID Models ---

class RegisterResponse(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the registered user")

    model_config = {
        "json_schema_extra": {
            "examples": [{"user_id": "abc123"}]
        }
    }


class FacialArea(BaseModel):
    x: int
    y: int
    w: int
    h: int
    left_eye: list[int] | None = None
    right_eye: list[int] | None = None


class FacialAreas(BaseModel):
    img1: FacialArea
    img2: FacialArea


class VerifyResponse(BaseModel):
    verified: bool = Field(..., description="Whether the face matches the registered user")
    distance: float = Field(..., description="Distance between face embeddings")
    threshold: float = Field(..., description="Threshold for verification")
    confidence: float = Field(..., description="Confidence score (inverse of distance)")
    model: str = Field(..., description="Face recognition model used")
    detector_backend: str = Field(..., description="Face detector backend used")
    similarity_metric: str = Field(..., description="Similarity metric used")
    facial_areas: FacialAreas = Field(..., description="Detected facial areas in both images")
    time: float = Field(..., description="Processing time in seconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "verified": False,
                    "distance": 0.97095,
                    "threshold": 0.68,
                    "confidence": 3.87,
                    "model": "ArcFace",
                    "detector_backend": "opencv",
                    "similarity_metric": "cosine",
                    "facial_areas": {
                        "img1": {
                            "x": 216,
                            "y": 318,
                            "w": 923,
                            "h": 923,
                            "left_eye": [836, 710],
                            "right_eye": [507, 682]
                        },
                        "img2": {
                            "x": 0,
                            "y": 0,
                            "w": 767,
                            "h": 919,
                            "left_eye": None,
                            "right_eye": None
                        }
                    },
                    "time": 1.45
                }
            ]
        }
    }


class AsyncVerifyResponse(BaseModel):
    result_id: str = Field(..., description="ID to retrieve the verification result")

    model_config = {
        "json_schema_extra": {
            "examples": [{"result_id": "abc123"}]
        }
    }


class VerifyResultResponse(BaseModel):
    status: Literal["pending", "completed", "error"] = Field(..., description="Status of the verification")
    result: VerifyResponse | None = Field(None, description="Verification result if completed")
    error: str | None = Field(None, description="Error message if failed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"status": "pending", "result": None, "error": None},
                {"status": "completed", "result": {"verified": True, "distance": 0.3, "threshold": 0.68, "confidence": 2.3, "model": "ArcFace", "detector_backend": "opencv", "similarity_metric": "cosine", "facial_areas": {"img1": {"x": 0, "y": 0, "w": 100, "h": 100}, "img2": {"x": 0, "y": 0, "w": 100, "h": 100}}, "time": 1.2}, "error": None},
                {"status": "error", "result": None, "error": "Registration photo not found"},
            ]
        }
    }


# --- Liveness Models ---

class StimulusResponse(BaseModel):
    type: Literal["moving-dot"] = Field("moving-dot", description="Type of stimulus")
    fps: int = Field(60, description="Frames per second")
    duration_sec: int = Field(5, description="Duration in seconds")
    token: str = Field(..., description="Stimulus will be saved on the server for 3 minutes with this token")
    x: List[float]
    y: List[float]


class LivenessResponse(BaseModel):
    is_live: bool = Field(..., example=True)
    score: float = Field(..., example=0.85)
    gaze_fit_x: List[float]
    gaze_fit_y: List[float]
    inlier: List[bool]
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

