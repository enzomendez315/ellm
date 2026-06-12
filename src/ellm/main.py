from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel

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


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50


@app.post("/generate")
def generate_response(request: GenerateRequest):
    prompt = request.prompt
    model = app.state.model
    tokenizer = app.state.tokenizer

    encoded_prompt = tokenizer.encode(text=prompt, return_tensors="pt")
    encoded_response = model.generate(encoded_prompt)
    response = tokenizer.decode(encoded_response)

    return response


@app.get("/health")
def get_health(request: Request):
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics(request: Request):
    return "Getting metrics..."
