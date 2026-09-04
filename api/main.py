from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    pass


app = FastAPI(title='BayesScamDetector')


@app.get('/health')
def health():
    return {'status': 'ok'}
