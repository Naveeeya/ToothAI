# ToothAI app (Streamlit: OPG analysis with Detectron2 + the Photo screening page), CPU only.
# Build from the repository root on an arm64 Mac (same architecture as the Oracle Ampere VM):
#   docker build -f docker/ui.Dockerfile -t toothai-ui:<version> .
# OPG weights are mounted read-only at /app/output at runtime, never baked into the image.

# ---- build: compile Detectron2 once, into a wheel
FROM python:3.11-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*
# Same torch and Detectron2 commit as the paper_eval runs (paper_eval/kaggle/run_eval.ipynb), CPU wheels.
ARG TORCH=2.5.1 TORCHVISION=0.20.1 D2_COMMIT=fd27788985af0f4ca800bca563acdb700bb890e2
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH}" "torchvision==${TORCHVISION}"
RUN pip wheel --no-build-isolation --no-deps -w /wheels \
    "git+https://github.com/facebookresearch/detectron2.git@${D2_COMMIT}"

# ---- runtime
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PHOTO_API_URL=http://photo-api:8001 MPLCONFIGDIR=/tmp/mpl
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ARG TORCH=2.5.1 TORCHVISION=0.20.1
COPY requirements-app.txt /tmp/requirements-app.txt
COPY --from=build /wheels /wheels
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH}" "torchvision==${TORCHVISION}" \
    && pip install /wheels/*.whl \
    && grep -v -E "^(cairosvg|pytest)" /tmp/requirements-app.txt > /tmp/req.txt && pip install -r /tmp/req.txt \
    && rm -rf /wheels /tmp/req*.txt

# Only what the app reads. paper_eval: the four result files evidence.py parses, nothing else.
COPY app.py ./
COPY .streamlit/ .streamlit/
COPY tooth_ai/ tooth_ai/
COPY assets/ assets/
COPY static/ static/
COPY paper_eval/results/coco_eval_716.csv paper_eval/results/decomposition.csv \
     paper_eval/results/paired_deltas.txt paper_eval/results/engine_v2.csv paper_eval/results/
COPY photo_screening/__init__.py photo_screening/palette.py photo_screening/
COPY photo_screening/ui/ photo_screening/ui/
COPY photo_screening/samples/ photo_screening/samples/
# Brand Streamlit's HTML shell (tab title, favicon, link previews); SITE_URL makes the preview image absolute.
COPY docker/brand_shell.py docker/brand_shell.py
ARG SITE_URL=http://localhost:8501
RUN python docker/brand_shell.py "$SITE_URL"

RUN useradd --create-home app && mkdir -p /tmp/mpl && chown -R app /app /tmp/mpl
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD curl -fs http://localhost:8501/_stcore/health || exit 1
# tooth_ai.serve = streamlit run app.py, with the OPG models loaded at start-up (see the module docstring)
CMD ["python", "-m", "tooth_ai.serve", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
