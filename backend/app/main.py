"""FastAPI application entrypoint with security, CORS, request size limits, and error handling."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import analysis, flood, forest, health, nisar, processing, rainfall
from app.core.config import get_settings


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds max_request_size_bytes (Section 16)."""

    def __init__(self, app: FastAPI, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": f"Request payload exceeds maximum allowed size of {self.max_bytes} bytes."
                },
            )
        return await call_next(request)


def create_app() -> FastAPI:
    """Create and configure the Amazon Sentinel FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Environmental monitoring data acquisition, scientific SAR/optical/rainfall processing, "
            "and CesiumJS delivery backend for the Amazon Basin."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Accept"],
    )
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_size_bytes)

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc), "error_type": "validation_error"},
        )

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(flood.router, prefix=settings.api_prefix)
    app.include_router(forest.router, prefix=settings.api_prefix)
    app.include_router(nisar.router, prefix=settings.api_prefix)
    app.include_router(rainfall.router, prefix=settings.api_prefix)
    app.include_router(analysis.router, prefix=settings.api_prefix)
    app.include_router(processing.router, prefix=settings.api_prefix)

    return app


app = create_app()
