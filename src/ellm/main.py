from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model
    yield
    # Clean up the models and release the resources


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
def generate_response():
    print("Generating response!")


@app.get("/health")
def get_health():
    print("Getting health...")


@app.get("/metrics")
def get_metrics():
    print("Getting metrics...")
