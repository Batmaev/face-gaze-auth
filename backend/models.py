from typing import Annotated, Literal
from pydantic import BaseModel, Field


# --- Registration Models ---

class RegisterResponse(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the registered user")

    model_config = {
        "json_schema_extra": {
            "examples": [{"user_id": "abc123"}]
        }
    }


# --- Stimulus ---

Coord = Annotated[float, Field(allow_inf_nan=False)]


class SplineStimulus(BaseModel):
    """Кубический сплайн через контрольные точки, параметр равномерно по времени.

    Кривая нормируется min-max в [0, 1]² по сетке из fps × duration_sec кадров,
    см. stimulus.generate.eval_spline.
    """

    type: Literal["spline"] = "spline"
    ctrl: list[tuple[Coord, Coord]] = Field(
        ...,
        min_length=4,
        description="Control points (x, y), at least 4. The curve is min-max normalized to [0, 1]² on the frame grid",
    )
    fps: float = Field(60, gt=0, description="Frame rate the stimulus was shown at")
    duration_sec: float = Field(5, gt=0)


# Компактное описание стимула. Сервер стимулы не хранит: клиент получает `stim`
# из GET /stim и присылает его обратно вместе с записью, траекторию восстановит
# любой инстанс. Новый способ генерации — новый класс со своим `type`, а здесь
# Annotated[Union[...], Field(discriminator="type")].
Stimulus = SplineStimulus


class StimulusResponse(BaseModel):
    stim: Stimulus = Field(..., description="Send this object back with the recording")
    x: list[float] = Field(..., description="Frames to show, in [0, 1], at stim.fps")
    y: list[float]


# --- Liveness Models ---

class ArkitLivenessRequest(BaseModel):
    stim: Stimulus
    gaze_x: list[float]
    gaze_y: list[float]
    left_eye_blink: list[float]
    right_eye_blink: list[float]
    fps: float = Field(60, gt=0, description="Frame rate of the gaze series")



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
