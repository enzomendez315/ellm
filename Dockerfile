FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependencies and README
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Source code
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Create unprivileged user
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "src/ellm/main.py", "--host", "0.0.0.0", "--port", "8000"]