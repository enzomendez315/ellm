import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ellm.config import Settings
from ellm.model import load_model


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load resources at startup
    app.state.settings = Settings()
    app.state.queue = asyncio.Queue(app.state.settings.max_queue_size)
    app.state.model, app.state.tokenizer = load_model(app.state.settings.model_name)
    app.state.task = asyncio.create_task(
        start_engine_loop(
            app.state.queue, app.state.model, app.state.tokenizer, app.state.settings.max_batch_size
        )
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
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, requests: list[GenerateRequest]
) -> list[str]:
    prompts = [r.prompt for r in requests]
    max_tokens = max(r.max_tokens for r in requests)

    # Tokenizer returns a dict with input_ids and attention_mask
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


async def start_engine_loop(
    queue: asyncio.Queue, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_batch_size: int
):
    while True:
        # Pull at least one item from the queue
        request, future = await queue.get()
        batch = [(request, future)]

        # Increase the batch size if possible
        while len(batch) < max_batch_size:
            try:
                batch.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        requests = [r for r, _ in batch]
        futures = [f for _, f in batch]
        # Run generations in a separate thread
        try:
            completions = await asyncio.to_thread(run_generation, model, tokenizer, requests)
            for future, completion in zip(futures, completions, strict=True):
                if not future.cancelled():
                    future.set_result(completion)
        except Exception as e:
            for future in futures:
                if not future.cancelled():
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
async def get_health():
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    return "Getting metrics..."
