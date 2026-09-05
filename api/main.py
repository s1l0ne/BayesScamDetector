from fastapi import FastAPI, HTTPException, Request, Depends
from contextlib import asynccontextmanager
from api.model_service import ModelService
from api.schemas import PredictRequest, PredictResponse
from typing import cast, Annotated

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = ModelService()

    yield


app = FastAPI(title='BayesScamDetector', lifespan=lifespan)


def get_model_service(request: Request) -> ModelService:
    return cast(ModelService, request.app.state.model_service)


ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(data: PredictRequest, service: ModelServiceDep) -> PredictResponse:
    return PredictResponse(
        is_scam=service.model.predict([data.text])[0],
        probability=service.model.predict_proba([data.text])[0][1]
    )
