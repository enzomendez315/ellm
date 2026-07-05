from contextlib import asynccontextmanager
from threading import Thread

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer

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


@app.post("/generate", response_class=StreamingResponse)
async def generate_response(request: GenerateRequest):
    model = app.state.model
    tokenizer = app.state.tokenizer

    inputs = tokenizer(text=request.prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer)

    # Run the generation in a separate thread so that it's non-blocking
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=request.max_tokens)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for new_text in streamer:
        yield new_text


@app.get("/health")
async def get_health(request: Request):
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics(request: Request):
    return "Getting metrics..."
