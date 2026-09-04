from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from api.model_service import ModelService
from api.schemas import PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = ModelService()

    yield


app = FastAPI(title='BayesScamDetector', lifespan=lifespan)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(request: Request, data: PredictRequest) -> PredictResponse:
    return PredictResponse(
        is_scam=request.app.state.model_service.model.predict([data.text])[0],
        probability=request.app.state.model_service.model.predict_proba([data.text])[0][1]
    )
