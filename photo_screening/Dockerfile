# ToothAI Photo Screening API (CPU). Build from the repository root:
#   docker build -f photo_screening/Dockerfile -t toothai-photo-api .
# Weights are mounted at runtime (see docker-compose.yml), never baked into the image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PHOTO_WEIGHTS_DIR=/app/photo_screening/weights PHOTO_DEVICE=cpu YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY photo_screening/requirements.txt photo_screening/requirements.txt
# CPU-only torch first, so pip doesn't pull the multi-GB CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" "torchvision>=0.17" \
    && pip install -r photo_screening/requirements.txt

# The PDF reuses the main app's report styling, brand assets and fonts.
COPY tooth_ai/ tooth_ai/
COPY assets/ assets/
COPY static/fonts/ static/fonts/
COPY photo_screening/ photo_screening/

RUN useradd --create-home app && mkdir -p /tmp/ultralytics && chown -R app /app /tmp/ultralytics
USER app
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://localhost:8001/health || exit 1
CMD ["uvicorn", "photo_screening.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
