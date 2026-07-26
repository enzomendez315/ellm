from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_batch_size: int = 10
    max_queue_size: int = 0  # 0 = unbounded
    default_max_tokens: int = 50
