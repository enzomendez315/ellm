import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ellm.model import load_model


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load resources at startup
    app.state.queue = asyncio.Queue()
    app.state.model, app.state.tokenizer = load_model()
    app.state.task = asyncio.create_task(
        start_engine_loop(app.state.queue, app.state.model, app.state.tokenizer)
    )
    yield

    # Clean up and release resources
    app.state.task.cancel()
    try:
        await app.state.task
    except asyncio.CancelledError:
        pass
    app.state.model = None
    app.state.tokenizer = None
    app.state.queue = None


def run_generation(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, request: GenerateRequest
) -> str:
    encoded_prompt = tokenizer.encode(text=request.prompt, return_tensors="pt")
    encoded_completion = model.generate(encoded_prompt, max_new_tokens=request.max_tokens)
    completion = tokenizer.decode(encoded_completion[0], skip_special_tokens=True)
    return completion


async def start_engine_loop(
    queue: asyncio.Queue, model: AutoModelForCausalLM, tokenizer: AutoTokenizer
):
    while True:
        # Pull an item from the queue and run generation in a separate thread
        request, future = await queue.get()
        try:
            completion = await asyncio.to_thread(run_generation, model, tokenizer, request)
            future.set_result(completion)
        except Exception as e:
            future.set_exception(e)


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate_response(request: GenerateRequest):
    queue: asyncio.Queue = app.state.queue
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await queue.put((request, future))
    result = await future
    return {"completion": result}


@app.get("/health")
async def get_health(request: Request):
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics(request: Request):
    return "Getting metrics..."
