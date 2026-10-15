"""FastAPI app: ``uvicorn photo_screening.app.main:app --port 8001`` (run from the repo root)."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import ValidationError
from starlette.datastructures import UploadFile

import photo_screening
from photo_screening.app import logs
from photo_screening.app.config import Settings, get_settings, load_thresholds
from photo_screening.app.imaging import ImageDecodeError
from photo_screening.app.models.registry import Registry
from photo_screening.app.report import Upload, build_report
from photo_screening.app.schemas import VIEWS, ModelStatus, ScreeningReport

log = logging.getLogger("photo_screening.api")
PREFIX = "/v1/photo-screening"
_inference_lock = threading.Lock()          # one analysis at a time: models are shared, demo-scale service


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logs.configure()
    app.state.settings = settings
    app.state.registry = Registry.load(settings)
    app.state.registry.warm_up()
    from photo_screening.app.vlm import build_vlm

    app.state.vlm = build_vlm(settings)
    log.info("service ready", extra={"loaded": app.state.registry.loaded, "vlm": settings.vlm_enabled})
    yield


app = FastAPI(title="ToothAI Photo Screening API", version=photo_screening.__version__, lifespan=lifespan,
              description="Smartphone intraoral photos to a screening report. "
                          "A screening aid, not a diagnosis. " + photo_screening.DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def get_registry(request: Request) -> Registry:
    return request.app.state.registry


def get_app_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


def get_vlm(request: Request):
    return getattr(request.app.state, "vlm", None)


async def _uploads_from_form(request: Request, settings: Settings) -> tuple[list[Upload], bool]:
    form = await request.form()
    mirrored_views = {v.strip() for v in str(form.get("mirrored", "")).split(",") if v.strip()}
    if mirrored_views - set(VIEWS):
        raise HTTPException(422, f"mirrored must list views from: {', '.join(VIEWS)}")
    uploads = []
    limit = settings.max_upload_mb * 1024 * 1024
    for view in VIEWS:
        f = form.get(view)
        if not isinstance(f, UploadFile):
            continue
        data = await f.read()
        if not data:
            continue
        if len(data) > limit:
            raise HTTPException(413, f"The {view} photo is larger than {settings.max_upload_mb} MB.")
        uploads.append(Upload(view, data, (view in mirrored_views) if "mirrored" in form else None))
    if not any(u.view == "front" for u in uploads):
        raise HTTPException(422, "A front photo is required.")
    use_vlm = str(form.get("use_vlm", "false")).lower() in ("1", "true", "yes", "on")
    return uploads, use_vlm


def _run(uploads: list[Upload], registry: Registry, settings: Settings, use_vlm: bool, vlm) -> ScreeningReport:
    if "alphadent" not in registry.loaded:
        raise HTTPException(503, "The primary photo model is not loaded. Run scripts/fetch_weights.py and restart.")
    # One analysis at a time (2 CPU cores on the public server): a second request waits briefly, then is
    # told the service is busy instead of queueing behind it.
    if not _inference_lock.acquire(timeout=settings.busy_wait_s):
        raise HTTPException(503, "The service is analysing someone else's photos. Try again in a minute.",
                            headers={"Retry-After": "30"})
    try:
        return build_report(uploads, registry, settings, use_vlm=use_vlm, vlm=vlm)
    except ImageDecodeError as e:
        raise HTTPException(400, "One of the photos isn't a readable JPG, PNG or HEIC image.") from e
    finally:
        _inference_lock.release()


@app.post(f"{PREFIX}/analyze", response_model=ScreeningReport,
          summary="Analyse 1-5 intraoral photos (multipart: front, upper, lower, left, right; use_vlm; mirrored)")
async def analyze(request: Request, registry: Registry = Depends(get_registry),
                  settings: Settings = Depends(get_app_settings), vlm=Depends(get_vlm)) -> ScreeningReport:
    uploads, use_vlm = await _uploads_from_form(request, settings)
    return await run_in_threadpool(_run, uploads, registry, settings, use_vlm, vlm)


@app.post(f"{PREFIX}/report.pdf", summary="PDF report from a ScreeningReport JSON body, or from the analyze form",
          response_class=Response, responses={200: {"content": {"application/pdf": {}}}})
async def report_pdf(request: Request, registry: Registry = Depends(get_registry),
                     settings: Settings = Depends(get_app_settings), vlm=Depends(get_vlm)) -> Response:
    if request.headers.get("content-type", "").startswith("application/json"):
        body = await request.body()
        if len(body) > settings.max_report_mb * 1024 * 1024:
            raise HTTPException(413, f"The report is larger than {settings.max_report_mb} MB.")
        try:
            report = ScreeningReport.model_validate_json(body)
        except ValidationError as e:
            raise HTTPException(422, "Body is not a valid ScreeningReport.") from e
    else:
        uploads, use_vlm = await _uploads_from_form(request, settings)
        report = await run_in_threadpool(_run, uploads, registry, settings, use_vlm, vlm)
    from photo_screening.app.pdf import build_pdf

    pdf = await run_in_threadpool(build_pdf, report, settings)
    filename = f"toothai-screening-{report.report_id}.pdf"
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get(f"{PREFIX}/models", summary="Loaded models, versions (SHA-256 prefix), licences and thresholds")
def models(registry: Registry = Depends(get_registry), settings: Settings = Depends(get_app_settings),
           vlm=Depends(get_vlm)) -> dict:
    return {
        "models": [m.model_dump() for m in registry.status_list()],
        "thresholds": load_thresholds(settings, registry.loaded),
        "device": settings.resolved_device(),
        "tta": settings.tta,
        "vlm": vlm.status().model_dump() if vlm else {"enabled": False},
        "app_version": photo_screening.__version__,
    }


@app.get("/health")
def health(registry: Registry = Depends(get_registry)) -> dict:
    missing: list[ModelStatus] = [s for s in registry.status_list() if s.required and s.status != "loaded"]
    return {"status": "ok" if not missing else "degraded", "models_loaded": registry.loaded,
            "required_missing": [s.model_id for s in missing]}
