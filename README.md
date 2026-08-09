# eLLM

An LLM inference server built from scratch to understand how model serving actually works.

This is a small version of what production inference engines like vLLM do: accept requests over HTTP, decouple request handling from inference, batch concurrent requests into single forward passes, and expose the metrics that tell you whether any of it is working.

The point wasn't to build something faster than vLLM. It was to build a small enough version that reading vLLM's source afterward makes sense.

## What it does

- Serves an open-weight transformer model (Qwen2.5-0.5B-Instruct by default) over a REST API
- Decouples HTTP handling from inference through an asyncio queue feeding a single background generation loop
- Groups concurrently-arriving requests into one forward pass via dynamic batching
- Exposes Prometheus metrics for throughput, queue depth, and batch size
- Runs in a multi-stage Docker container as a non-root user, with persistent caching for model weights

## Architecture

```
POST /generate
      │
      ▼
  handler ──── creates a Future, enqueues (request, future), awaits the Future
      │
      ▼
 asyncio.Queue ──── requests collect here so they can be batched
      │
      ▼
 engine loop ──── blocks for one request, drains the queue up to max_batch_size,
      │           runs generation on a thread, sets each Future's result
      ▼
model.generate() ──── one forward pass for the whole batch
```

The handler never touches the model. It creates an `asyncio.Future`, puts it on a shared queue with the request, and awaits it — suspending without blocking the event loop. A single background engine loop, launched at startup, pulls items off the queue and runs generation. When a batch finishes, each completion is routed back to its originating request's Future, which wakes the handler that's been waiting on it.

The decoupling exists so batching is possible. If each request ran generation inline in its own handler, there'd be no point where multiple requests exist together and could be grouped.

Generation itself runs via `asyncio.to_thread`. `model.generate()` is blocking and CPU-bound — calling it directly on the event loop would freeze the server for the duration of every generation, including health checks. Async can't help here because `generate()` never yields; a thread can be preempted regardless.

## Dynamic batching

The engine loop blocks for the first request, then drains whatever else is already queued (up to `max_batch_size`) without waiting. Under no load this batches one request and proceeds immediately. Under concurrent load, requests that arrive while a batch is generating pile up and get scooped into the next one.

Batching raises throughput because a forward pass is dominated by moving model weights, not by the data pushed through them. Running eight sequences through one pass costs barely more than running one, so the fixed cost is amortized across requests. The tradeoff is latency: the first request in a batch waits for companions it wouldn't have waited for alone.

Batched sequences have to form a rectangular tensor, so prompts are padded to equal length. Padding goes on the **left** — generation continues from the last position of each sequence, and right-padding would mean continuing from pad tokens. The attention mask marks padded positions so the model ignores them.

## Metrics

`GET /metrics` exposes Prometheus-format metrics:

| Metric | Type | What it tells you |
|---|---|---|
| `ellm_tokens_generated_total` | Counter | Token throughput (rate is computed by Prometheus, not stored) |
| `ellm_requests_total` | Counter | Completed requests |
| `ellm_requests_failed_total` | Counter | Failed requests |
| `ellm_queue_depth` | Gauge | Requests currently waiting — the overload signal |
| `ellm_batch_size` | Histogram | Batch size per generation step; `sum / count` is batch utilization |

The token counter measures generated positions including post-EOS padding, since that reflects compute actually performed.

## Running it

With Docker:

```bash
docker compose up
```

The first start downloads model weights into a named volume; subsequent starts reuse them.

Locally:

```bash
uv sync
uv run fastapi dev src/ellm/main.py
```

Then:

```bash
curl -X POST localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of Italy is", "max_tokens": 20}'
```

Interactive docs at `localhost:8000/docs`.

## Configuration

Settings are read from environment variables via `pydantic-settings`, with defaults in `src/ellm/config.py`.

| Setting | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model ID to load at startup |
| `MAX_BATCH_SIZE` | `10` | Upper bound on requests grouped into one forward pass |
| `MAX_QUEUE_SIZE` | `0` | Queue capacity; `0` means unbounded |

`MAX_QUEUE_SIZE=0` means the queue accepts requests indefinitely — under sustained overload it grows without limit rather than rejecting work. A bounded queue would apply backpressure instead, which is the behavior you'd want in production.

`MAX_BATCH_SIZE` is a ceiling, not a target. Actual batch sizes depend on how many requests happen to be queued when the engine loop picks up work; `ellm_batch_size` reports what's really happening.

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/generate` | POST | Prompt in, completion out |
| `/health` | GET | Liveness check |
| `/metrics` | GET | Prometheus metrics |

## Tests

```bash
uv run pytest
```

Tests assert on the response contract — status code, shape, non-empty completion — rather than on generated text, which isn't deterministic.

## Design decisions

**Futures rather than a per-request queue.** A Future holds one value, which fits the current non-streaming response. Streaming through the decoupled engine would require replacing it with a per-request queue that the engine pushes tokens onto.

**Batch shares one token budget.** Because a batch runs a single `generate()` loop, all requests in it share one `max_new_tokens`. Taking the max across the batch means nobody gets cut short, at the cost of over-generating for requests that asked for less. Per-sequence stopping is what continuous batching solves.

**Weights in a volume, not baked into the image.** Keeps the image smaller and lets the model change without a rebuild, at the cost of a first-run download. Baking weights in becomes more attractive under orchestration, where pods start frequently.

**Exceptions propagate to the failing request, not the loop.** If generation throws, the exception is set on each Future in that batch rather than allowed to escape. An unhandled exception would kill the engine loop and hang every subsequent request silently.

## Not built yet

- **Streaming.** Was working before the queue refactor; reintroducing it means replacing the per-request Future with a per-request queue.
- **TTFT and inter-token latency.** Both require knowing when each token was produced, which requires streaming.
- **GPU metrics.** Everything so far runs on CPU.
- **Continuous batching.** Adding and removing sequences from a batch at the token level, rather than waiting for the whole batch to finish. This is the actual vLLM magic.
- **Kubernetes, CI/CD, Terraform, benchmarks.**

## Stack

Python 3.12 · PyTorch · Hugging Face Transformers · FastAPI · asyncio · Prometheus · Docker · uv · ruff · pytest