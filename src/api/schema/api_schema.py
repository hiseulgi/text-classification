from typing import List

import rootutils
from pydantic import BaseModel, Field

from src.api.schema.predictions_schema import PredictionsResultSchema

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


class BaseApiResponseSchema(BaseModel):
    """Base API Response Schema"""

    status: str = Field(..., description="API Response Status", example="success")
    message: str = Field(..., description="API Response Message", example="OK")


class PredictionsRequestSchema(BaseModel):
    """Predictions Request Schema"""

    text: str = Field(..., description="Input text to be classified")


class PredictionResponseSchema(BaseApiResponseSchema):
    """Prediction Response Schema"""

    results: List[PredictionsResultSchema] = Field(
        ..., description="List of Predictions Result"
    )