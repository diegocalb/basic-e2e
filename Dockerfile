# ---- Builder Stage ----
# This stage installs dependencies into a virtual environment.
FROM python:3.10-slim as builder

WORKDIR /app

RUN pip install uv

COPY pyproject.toml poetry.lock uv.lock README.md ./
COPY src ./src

# Install dependencies into a virtual environment within the image
# This leverages Docker's layer caching. Dependencies are only re-installed
# if pyproject.toml or poetry.lock change.
RUN uv venv && uv sync --extra dev

# ---- Final Stage ----
# This stage creates the final, lean production image.
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY src ./src
COPY data ./data

# Variable de entorno para MLflow
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

ENTRYPOINT ["train-model"]
