"""NASA–ISRO NISAR L-band SAR cloud access service via Alaska Satellite Facility DAAC (Section 7.1 & 7.2)."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, SecretStr

from app.core.config import get_settings

logger = logging.getLogger("amazon_sentinel.nisar_access")

# Official NISAR L-band product specifications (ASF DAAC documentation, Section 7.1)
NISAR_PRODUCT_SPECS: Dict[str, Dict[str, Any]] = {
    "GCOV": {
        "product_type": "GCOV",
        "level": "L2",
        "name": "Geocoded Polarimetric Covariance Matrix",
        "description": "Orthorectified, terrain-flattened L-band radar backscatter gamma0 covariance (HHHH, HVHV, VVVV) on UTM/projected grid.",
        "hdf5_group_path": "/science/LSAR/GCOV/grids/frequencyA",
        "calibration_quantity": "gamma0_terrain_flattened",
        "nominal_resolution_m": 20.0,
        "supported_for_backscatter_analysis": True,
    },
    "GUNW": {
        "product_type": "GUNW",
        "level": "L2",
        "name": "Geocoded Unwrapped Interferogram",
        "description": "Geocoded L-band wrapped/unwrapped interferometric phase and coherence between repeat-pass pairs.",
        "hdf5_group_path": "/science/LSAR/GUNW/grids/frequencyA",
        "calibration_quantity": "interferometric_phase_and_coherence",
        "nominal_resolution_m": 80.0,
        "supported_for_backscatter_analysis": True,
    },
    "RSLC": {
        "product_type": "RSLC",
        "level": "L1",
        "name": "Range-Doppler Single Look Complex",
        "description": "Focused complex radar imagery in slant-range / zero-Doppler radar coordinates.",
        "hdf5_group_path": "/science/LSAR/RSLC/swaths/frequencyA",
        "calibration_quantity": "beta0_complex_slant_range",
        "nominal_resolution_m": 7.0,
        "supported_for_backscatter_analysis": False,
    },
}


class TemporaryS3Credentials(BaseModel):
    """Temporary 1-hour AWS STS credentials for ASF Earthdata Cloud (`us-west-2`). Never serialized to API responses."""

    access_key_id: SecretStr
    secret_access_key: SecretStr
    session_token: SecretStr
    expiration: datetime
    aws_region: str = "us-west-2"

    def is_expired(self, safety_margin_seconds: int = 300) -> bool:
        now = datetime.now(timezone.utc)
        exp = self.expiration if self.expiration.tzinfo else self.expiration.replace(tzinfo=timezone.utc)
        return now >= (exp - timedelta(seconds=safety_margin_seconds))


class ASFNisarAccessService:
    """
    Manages official NASA Alaska Satellite Facility (ASF DAAC) Earthdata Cloud authentication,
    temporary S3 STS credential refresh (`https://nisar.asf.earthdatacloud.nasa.gov/s3credentials`),
    and AWS `us-west-2` configuration without ever exposing credentials in logs or frontend responses.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._cached_creds: Optional[TemporaryS3Credentials] = None

    @property
    def has_earthdata_credentials(self) -> bool:
        return self.settings.has_earthdata_configured()

    def get_temporary_s3_credentials(self, force_refresh: bool = False) -> Optional[TemporaryS3Credentials]:
        """
        Obtain or refresh temporary 1-hour AWS S3 credentials from the official ASF endpoint.
        Returns None if Earthdata credentials are not configured or endpoint is unreachable.
        """
        if not self.has_earthdata_credentials:
            return None

        if not force_refresh and self._cached_creds is not None and not self._cached_creds.is_expired():
            return self._cached_creds

        headers: Dict[str, str] = {"Accept": "application/json"}
        auth = None
        if self.settings.earthdata_bearer_token and self.settings.earthdata_bearer_token.get_secret_value():
            headers["Authorization"] = f"Bearer {self.settings.earthdata_bearer_token.get_secret_value()}"
        elif self.settings.earthdata_username and self.settings.earthdata_password:
            auth = (
                self.settings.earthdata_username,
                self.settings.earthdata_password.get_secret_value(),
            )

        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                resp = client.get(
                    self.settings.asf_s3_credential_endpoint,
                    headers=headers,
                    auth=auth,
                )
            if resp.status_code == 200:
                payload = resp.json()
                exp_str = str(payload.get("expiration", ""))
                exp_dt = (
                    datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
                    if exp_str
                    else datetime.now(timezone.utc) + timedelta(hours=1)
                )
                self._cached_creds = TemporaryS3Credentials(
                    access_key_id=SecretStr(payload["accessKeyId"]),
                    secret_access_key=SecretStr(payload["secretAccessKey"]),
                    session_token=SecretStr(payload["sessionToken"]),
                    expiration=exp_dt,
                    aws_region=self.settings.asf_aws_region,
                )
                logger.info(
                    "Obtained temporary ASF S3 credentials in region %s valid until %s",
                    self.settings.asf_aws_region,
                    exp_dt.isoformat(),
                )
                return self._cached_creds
            logger.warning("ASF S3 credential endpoint returned HTTP %d", resp.status_code)
            return None
        except Exception as exc:
            logger.warning("ASF S3 credential exchange failed: %s", exc)
            return None

    def get_public_connection_summary(self) -> Dict[str, Any]:
        """Return safe, non-sensitive summary of ASF cloud access state for API metadata."""
        has_creds = self.has_earthdata_credentials
        s3_ready = self._cached_creds is not None and not self._cached_creds.is_expired()
        return {
            "daac": "NASA Alaska Satellite Facility (ASF DAAC)",
            "search_endpoint": self.settings.asf_search_api_url,
            "s3_credential_endpoint": self.settings.asf_s3_credential_endpoint,
            "aws_region": self.settings.asf_aws_region,
            "earthdata_credentials_configured": has_creds,
            "active_s3_session": s3_ready,
            "access_mode": "windowed_s3_hdf5_subset" if s3_ready else "asf_cmr_discovery_and_calibrated_lband_reference",
        }


_nisar_service: Optional[ASFNisarAccessService] = None


def get_nisar_access_service() -> ASFNisarAccessService:
    """Return singleton ASFNisarAccessService instance."""
    global _nisar_service
    if _nisar_service is None:
        _nisar_service = ASFNisarAccessService()
    return _nisar_service
