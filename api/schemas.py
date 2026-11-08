import math

from pydantic import BaseModel, Field, field_validator


class BreastCancerInput(BaseModel):
    features: dict[str, float] = Field(min_length=1)

    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, features):
        if not all(math.isfinite(value) for value in features.values()):
            raise ValueError("All feature values must be finite numbers.")
        return features


class PredictionResponse(BaseModel):
    prediction: int
    diagnosis: str
    malignant_probability: float


class BatchPredictionRequest(BaseModel):
    samples: list[BreastCancerInput] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    status: str
    message: str
    details: list[ErrorDetail]
