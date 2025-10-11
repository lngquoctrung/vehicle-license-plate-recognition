# First stage: Setup dependencies and libraries
FROM python:3.13-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV ENV_PATH="/otp/.venv"
RUN python -m venv "${ENV_PATH}"
# Set virtual environment into PATH
ENV PATH="${ENV_PATH}/bin:$PATH"

# Copy requirements file and install libraries
COPY requirements.gradio.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.gradio.txt

# Cleanup
RUN find "${ENV_PATH}" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true              \
    && find "${ENV_PATH}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true     \
    && find "${ENV_PATH}" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true            \
    && find "${ENV_PATH}" -type d -name "docs" -exec rm -rf {} + 2>/dev/null || true            \
    && find "${ENV_PATH}" -type d -name "benchmarks" -exec rm -rf {} + 2>/dev/null || true      \
    && find "${ENV_PATH}" -type f -name "*.pyc" -delete                                         \
    && find "${ENV_PATH}" -type f -name "*.pyo" -delete                                         \
    && find "${ENV_PATH}" -type f -name "*.a" -delete                                          

# Second stage: Copy environment from first stage
FROM python:3.13-slim-bookworm AS runtime

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    libpng16-16 \
    libopenblas0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copy virtual environment from builder
ENV ENV_PATH="/otp/.venv"
COPY --from=builder "${ENV_PATH}" "${ENV_PATH}"
ENV PATH="${ENV_PATH}/bin:$PATH"

WORKDIR /app
COPY gradio_app/ ./gradio_app/
COPY src/ ./src/

EXPOSE 7001
CMD ["python", "./gradio_app/app.py"]
