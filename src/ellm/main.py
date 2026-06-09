from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from ellm.model import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load resources at startup
    app.state.model, app.state.tokenizer = load_model()
    yield
    # Clean up the models and release the resources
    app.state.model = None
    app.state.tokenizer = None


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
def generate_response(request: Request):
    model = app.state.model
    tokenizer = app.state.tokenizer
    return model, tokenizer


@app.get("/health")
def get_health(request: Request):
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics(request: Request):
    return "Getting metrics..."
