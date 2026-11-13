"""Google Earth Engine authentication, initialization, and cloud-side helpers (Section 4.1, 5, 8)."""

import logging
from typing import Any, Dict, Optional

from app.core.config import get_settings

logger = logging.getLogger("amazon_sentinel.earth_engine")


class EarthEngineService:
    """
    Manages Google Earth Engine initialization and cloud-side execution.
    Never downloads full scenes; all operations execute on Google Earth Engine servers
    or fall back transparently to calibrated local reference observations (with status='demo')
    when GEE credentials are not configured in the environment.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._ee_module: Optional[Any] = None
        self._last_error: Optional[str] = None

    def initialize(self) -> bool:
        """Attempt to initialize Google Earth Engine using service account or application default credentials."""
        if self._initialized:
            return True

        settings = get_settings()
        if not settings.has_gee_configured():
            self._last_error = "GEE_PROJECT_ID is not configured in environment (.env)."
            return False

        try:
            import ee  # type: ignore

            self._ee_module = ee
            if settings.gee_service_account_email and settings.gee_private_key_path:
                credentials = ee.ServiceAccountCredentials(
                    settings.gee_service_account_email,
                    settings.gee_private_key_path,
                )
                ee.Initialize(credentials=credentials, project=settings.gee_project_id)
            else:
                ee.Initialize(project=settings.gee_project_id)

            self._initialized = True
            self._last_error = None
            logger.info("Initialized Google Earth Engine for project %s", settings.gee_project_id)
            return True
        except Exception as exc:
            self._last_error = f"Earth Engine initialization unavailable: {exc}"
            logger.warning("Earth Engine live connection unavailable (%s); using calibrated reference fallback.", exc)
            return False

    @property
    def is_live(self) -> bool:
        """Return True if live Earth Engine connection is initialized."""
        return self._initialized or self.initialize()

    @property
    def ee(self) -> Any:
        """Return the imported ee module if available."""
        if not self.is_live:
            raise RuntimeError(self._last_error or "Google Earth Engine is not initialized.")
        return self._ee_module

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def region_to_ee_geometry(self, region: Dict[str, Any]) -> Any:
        """Convert validated study region dictionary to an ee.Geometry."""
        ee = self.ee
        return ee.Geometry(region["geometry"])


_gee_service: Optional[EarthEngineService] = None


def get_earth_engine_service() -> EarthEngineService:
    """Return singleton EarthEngineService instance."""
    global _gee_service
    if _gee_service is None:
        _gee_service = EarthEngineService()
    return _gee_service
