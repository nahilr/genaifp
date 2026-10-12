FROM python:3.10-slim

# Prevent interactive prompts and Python output buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    GENAIFP_ALLOW_REMOTE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Install system libraries needed by OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-optimized PyTorch build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir timm

# Copy application source, models, and CLI entrypoint
COPY src/ src/
COPY models/ models/
COPY main.py .

EXPOSE 8000

# Container healthcheck matching the FastAPI /api/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "main.py", "web", "--host", "0.0.0.0", "--port", "8000"]
