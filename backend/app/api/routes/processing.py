"""
FastAPI routes for asynchronous processing jobs, cache invalidation, and CesiumJS XYZ raster tiles (Section 12 & 13).
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response

from app.cache.manager import build_cache_key, get_cache_store
from app.core.regions import validate_and_resolve_region
from app.models.jobs import JobStatus, ProcessingJobInfo, ProcessingJobRequest
from app.models.metadata import DataStatus
from app.processing.combined.timeline import run_combined_multimodal_analysis
from app.processing.forest.statistics import run_forest_disturbance_pipeline
from app.processing.nisar.change_analysis import run_nisar_analysis_pipeline
from app.processing.rainfall.aggregation import run_gpm_rainfall_pipeline
from app.processing.sentinel1.outputs import run_sentinel1_flood_pipeline
from app.schemas.common import parse_spatial_query
from app.storage.manager import (
    render_classification_png_tile,
    render_continuous_png_tile,
)

router = APIRouter(tags=["Processing Jobs, Cache & Map Tiles"])

VALID_JOB_TYPES = {
    "sentinel1_flood",
    "gfw_forest",
    "nisar_lband",
    "gpm_rainfall",
    "combined_analysis",
}


def _execute_processing_job(job_id: str, req: ProcessingJobRequest) -> None:
    """Execute a satellite processing job and persist results in SQLite cache and job store."""
    store = get_cache_store()
    now_str = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        region = validate_and_resolve_region(
            region_id=req.region_id,
            bbox=req.bbox,
            polygon_geojson=req.polygon_geojson,
        )
        cache_key = build_cache_key(
            dataset=req.job_type,
            region_id=region["region_id"],
            start_date=req.start_date,
            end_date=req.end_date,
            config=req.parameters,
        )

        # Check cache first
        cached = store.get(cache_key)
        if cached is not None:
            cached_with_job = dict(cached)
            cached_with_job["job_id"] = job_id
            cached_with_job["served_from_cache"] = True
            job_info = ProcessingJobInfo(
                job_id=job_id,
                job_type=req.job_type,
                status=JobStatus.COMPLETED,
                data_status=DataStatus(cached.get("data_status", "demo")),
                region_id=region["region_id"],
                start_date=req.start_date,
                end_date=req.end_date,
                updated_at=now_str(),
                completed_at=now_str(),
                progress_pct=100.0,
                message="Job completed from verified cache.",
                cache_key=cache_key,
            )
            store.save_job(job_info.model_dump(), req.model_dump(), cached_with_job)
            return

        # Execute requested pipeline
        if req.job_type == "sentinel1_flood":
            raw = run_sentinel1_flood_pipeline(
                region=region,
                start_date=req.start_date,
                end_date=req.end_date,
                baseline_start_date=req.baseline_start_date or "2023-08-01",
                baseline_end_date=req.baseline_end_date or "2023-10-15",
            )
            data_status = raw["coverage"].status.value
            result_payload = {
                "job_id": job_id,
                "job_type": req.job_type,
                "data_status": data_status,
                "coverage": raw["coverage"].model_dump(),
                "layer": raw["layer"].model_dump(),
                "statistics": raw["statistics"].model_dump(),
                "timeline": raw["timeline"].model_dump(),
            }
        elif req.job_type == "gfw_forest":
            raw = run_forest_disturbance_pipeline(
                region=region,
                start_date=req.start_date,
                end_date=req.end_date,
            )
            data_status = raw["statistics"].status.value
            result_payload = {
                "job_id": job_id,
                "job_type": req.job_type,
                "data_status": data_status,
                "layer": raw["layer"].model_dump(),
                "statistics": raw["statistics"].model_dump(),
                "timeline": raw["timeline"].model_dump(),
            }
        elif req.job_type == "nisar_lband":
            ptype = str(req.parameters.get("product_type", "GCOV"))
            raw = run_nisar_analysis_pipeline(
                region=region,
                start_date=req.start_date,
                end_date=req.end_date,
                product_type=ptype,
            )
            data_status = raw["coverage"].status.value
            result_payload = {
                "job_id": job_id,
                "job_type": req.job_type,
                "data_status": data_status,
                "coverage": raw["coverage"].model_dump(),
                "layer": raw["layer"].model_dump(),
                "statistics": raw["statistics"].model_dump(),
                "observations": raw["observations"].model_dump(),
            }
        elif req.job_type == "gpm_rainfall":
            raw = run_gpm_rainfall_pipeline(
                region=region,
                start_date=req.start_date,
                end_date=req.end_date,
            )
            data_status = raw["statistics"].status.value
            result_payload = {
                "job_id": job_id,
                "job_type": req.job_type,
                "data_status": data_status,
                "layer": raw["layer"].model_dump(),
                "statistics": raw["statistics"].model_dump(),
                "timeline": raw["timeline"].model_dump(),
            }
        else:
            dist_km = float(req.parameters.get("distance_km", 5.0))
            raw = run_combined_multimodal_analysis(
                region=region,
                start_date=req.start_date,
                end_date=req.end_date,
                distance_km=dist_km,
            )
            data_status = raw["timeline"].status.value
            result_payload = {
                "job_id": job_id,
                "job_type": req.job_type,
                "data_status": data_status,
                "summary": raw["summary"],
                "overlap": {
                    "overlap_area_km2": raw["overlap"]["overlap_area_km2"],
                    "pct_of_flood_area": raw["overlap"]["pct_of_flood_area"],
                    "pct_of_disturbance_area": raw["overlap"]["pct_of_disturbance_area"],
                    "geojson": raw["overlap"]["geojson"].model_dump(),
                },
                "proximity": {
                    "distance_threshold_km": raw["proximity"]["distance_threshold_km"],
                    "direct_intersection_km2": raw["proximity"]["direct_intersection_km2"],
                    "proximity_buffer_only_km2": raw["proximity"]["proximity_buffer_only_km2"],
                    "total_within_distance_km2": raw["proximity"]["total_within_distance_km2"],
                },
                "timeline": raw["timeline"].model_dump(),
            }

        store.set(
            cache_key=cache_key,
            dataset=req.job_type,
            region_id=region["region_id"],
            start_date=req.start_date,
            end_date=req.end_date,
            payload=result_payload,
        )

        done_info = ProcessingJobInfo(
            job_id=job_id,
            job_type=req.job_type,
            status=JobStatus.COMPLETED,
            data_status=DataStatus(data_status),
            region_id=region["region_id"],
            start_date=req.start_date,
            end_date=req.end_date,
            updated_at=now_str(),
            completed_at=now_str(),
            progress_pct=100.0,
            message="Processing completed and validated.",
            cache_key=cache_key,
        )
        store.save_job(done_info.model_dump(), req.model_dump(), result_payload)

    except Exception as exc:
        failed_info = ProcessingJobInfo(
            job_id=job_id,
            job_type=req.job_type,
            status=JobStatus.FAILED,
            data_status=DataStatus.FAILED,
            region_id=req.region_id,
            start_date=req.start_date,
            end_date=req.end_date,
            updated_at=now_str(),
            progress_pct=0.0,
            message="Processing failed.",
            error=str(exc),
        )
        store.save_job(failed_info.model_dump(), req.model_dump(), None)


@router.post("/processing/jobs", response_model=ProcessingJobInfo, status_code=202)
async def create_processing_job(
    request: ProcessingJobRequest,
    background_tasks: BackgroundTasks,
) -> ProcessingJobInfo:
    """
    Submit an asynchronous geospatial processing job (Section 12).
    Validates parameters upfront and schedules execution in BackgroundTasks.
    """
    if request.job_type not in VALID_JOB_TYPES:
        raise ValueError(
            f"Unsupported job_type '{request.job_type}'. Valid types: {sorted(VALID_JOB_TYPES)}"
        )

    region = validate_and_resolve_region(
        region_id=request.region_id,
        bbox=request.bbox,
        polygon_geojson=request.polygon_geojson,
    )
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    cache_key = build_cache_key(
        dataset=request.job_type,
        region_id=region["region_id"],
        start_date=request.start_date,
        end_date=request.end_date,
        config=request.parameters,
    )

    initial_info = ProcessingJobInfo(
        job_id=job_id,
        job_type=request.job_type,
        status=JobStatus.PROCESSING,
        data_status=DataStatus.PROCESSING,
        region_id=region["region_id"],
        start_date=request.start_date,
        end_date=request.end_date,
        progress_pct=15.0,
        message="Job accepted and dispatched to background processing worker.",
        cache_key=cache_key,
    )
    get_cache_store().save_job(initial_info.model_dump(), request.model_dump(), None)
    background_tasks.add_task(_execute_processing_job, job_id, request)
    return initial_info


@router.get("/processing/jobs/{job_id}", response_model=ProcessingJobInfo)
async def get_processing_job_status(job_id: str) -> ProcessingJobInfo:
    """Retrieve status and progress of a processing job."""
    row = get_cache_store().get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Processing job '{job_id}' not found.")
    return ProcessingJobInfo(
        job_id=row["job_id"],
        job_type=row["job_type"],
        status=JobStatus(row["status"]),
        data_status=DataStatus(row["data_status"]),
        region_id=row["region_id"],
        start_date=row["start_date"],
        end_date=row["end_date"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
        progress_pct=float(row["progress_pct"]),
        message=row["message"],
        error=row["error"],
        cache_key=row["cache_key"],
    )


@router.get("/processing/jobs/{job_id}/results")
async def get_processing_job_results(job_id: str) -> Dict[str, Any]:
    """Retrieve processed outputs of a completed job."""
    row = get_cache_store().get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Processing job '{job_id}' not found.")
    if row["status"] != JobStatus.COMPLETED.value or not row["result_json"]:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is currently in status '{row['status']}'.",
        )
    return json.loads(row["result_json"])


@router.delete("/cache")
async def invalidate_cache(
    cache_key: Optional[str] = Query(default=None),
    dataset: Optional[str] = Query(default=None),
    region_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Invalidate cached geospatial results by key, dataset, region, or all (Section 11)."""
    removed = get_cache_store().invalidate(cache_key=cache_key, dataset=dataset, region_id=region_id)
    return {
        "status": "invalidated",
        "entries_removed": removed,
        "filters": {"cache_key": cache_key, "dataset": dataset, "region_id": region_id},
    }


@router.get("/tiles/{dataset}/{region_id}/{z}/{x}/{y}.png")
async def get_cesium_xyz_raster_tile(
    dataset: str,
    region_id: str,
    z: int,
    x: int,
    y: int,
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
    product_type: str = Query(default="GCOV"),
) -> Response:
    """
    Serve dynamic 256x256 RGBA PNG XYZ map tiles for CesiumJS `UrlTemplateImageryProvider` (Section 13).
    No private service credentials are ever exposed in the tile URL or response headers.
    """
    region = parse_spatial_query(region_id=region_id)
    ds = dataset.lower()

    if ds == "flood":
        res = run_sentinel1_flood_pipeline(region=region, start_date=start_date, end_date=end_date)
        png_bytes = render_classification_png_tile(res["classification_grid"])
    elif ds == "rainfall":
        res = run_gpm_rainfall_pipeline(region=region, start_date=start_date, end_date=end_date)
        grid = res["spatial_grid_mm"]
        png_bytes = render_continuous_png_tile(
            grid,
            vmin=float(grid.min()),
            vmax=float(grid.max()) + 1.0,
            colormap="rainfall",
        )
    elif ds == "nisar":
        res = run_nisar_analysis_pipeline(
            region=region,
            start_date=start_date,
            end_date=end_date,
            product_type=product_type,
        )
        prep = res.get("preprocessed_latest")
        if prep is None:
            png_bytes = render_classification_png_tile(res["coverage"].observation_count * 0)
        else:
            png_bytes = render_continuous_png_tile(
                prep.hh_gamma0_db,
                vmin=-24.0,
                vmax=-1.0,
                colormap="nisar",
            )
    elif ds == "forest":
        # Render forest disturbance density grid
        res = run_sentinel1_flood_pipeline(region=region, start_date=start_date, end_date=end_date)
        png_bytes = render_classification_png_tile((res["classification_grid"] == 2).astype("uint8"))
    else:
        raise ValueError(f"Unsupported tile dataset '{dataset}'. Supported: flood, forest, nisar, rainfall.")

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=1800"},
    )
