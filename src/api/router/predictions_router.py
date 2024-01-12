from typing import List

import rootutils
from fastapi import APIRouter, Depends

from src.api.core.bilstm_core import BilstmCore
from src.api.schema.api_schema import PredictionResponseSchema, PredictionsRequestSchema
from src.api.utils.logger import get_logger
from src.api.utils.utils import clean_text

log = get_logger()

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


# initialize bilstm core
bilstm_core = BilstmCore()
bilstm_core.setup()

# initialize router
router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/bilstm",
    tags=["predictions"],
    summary="Classify text(s) into prediction result.",
    response_model=PredictionResponseSchema,
)
async def bilstm_predictions(
    request: PredictionsRequestSchema = Depends(),
) -> PredictionResponseSchema:
    """BiLSTM predictions endpoint."""

    cleaned_text = await clean_text(request.text)

    predictions = bilstm_core.predict(cleaned_text)

    response = PredictionResponseSchema(
        status="success", message="Text processed successfully.", results=predictions
    )

    log.info(f"Text processed successfully.")

    return response
