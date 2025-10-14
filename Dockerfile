# ---- Builder Stage ----
# This stage installs dependencies into a virtual environment.
FROM python:3.10-slim as builder

# Set up the working directory
WORKDIR /app

# Install uv, a fast Python package installer
RUN pip install uv

# Copy dependency definition files
COPY pyproject.toml poetry.lock ./

# Install dependencies into a virtual environment within the image
# This leverages Docker's layer caching. Dependencies are only re-installed
# if pyproject.toml or poetry.lock change.
RUN uv venv && uv pip sync poetry.lock

# ---- Final Stage ----
# This stage creates the final, lean production image.
FROM python:3.10-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv ./.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application source code and data
COPY src ./src
COPY data ./data

# Set the entrypoint to run the training script
ENTRYPOINT ["train-model"]