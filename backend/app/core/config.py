"""Application configuration and secret management."""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecretRedactingFilter(logging.Filter):
    """Logging filter that scrubs credentials, tokens, and AWS keys from logs."""

    PATTERNS = [
        re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
        re.compile(r"(AKIA|ASIA)[0-9A-Z]{16}"),
        re.compile(r"(x-api-key['\"]?\s*[:=]\s*['\"]?)[^'\"\s]+", re.IGNORECASE),
        re.compile(r"(aws_secret_access_key['\"]?\s*[:=]\s*['\"]?)[^'\"\s]+", re.IGNORECASE),
        re.compile(r"(aws_session_token['\"]?\s*[:=]\s*['\"]?)[^'\"\s]+", re.IGNORECASE),
        re.compile(r"(password['\"]?\s*[:=]\s*['\"]?)[^'\"\s]+", re.IGNORECASE),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            msg = record.msg
            for pattern in self.PATTERNS:
                msg = pattern.sub(r"\1[REDACTED]", msg)
            record.msg = msg
        return True


class Settings(BaseSettings):
    """Centralized backend settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Amazon Sentinel Data Engineering & Processing System", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    api_prefix: str = Field(default="/api", alias="API_PREFIX")
    allowed_origins_raw: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="ALLOWED_ORIGINS",
    )
    max_request_size_bytes: int = Field(default=5 * 1024 * 1024, alias="MAX_REQUEST_SIZE_BYTES")

    # Google Earth Engine
    gee_project_id: Optional[str] = Field(default=None, alias="GEE_PROJECT_ID")
    gee_service_account_email: Optional[str] = Field(default=None, alias="GEE_SERVICE_ACCOUNT_EMAIL")
    gee_private_key_path: Optional[str] = Field(default=None, alias="GEE_PRIVATE_KEY_PATH")

    # Global Forest Watch
    gfw_api_base_url: str = Field(default="https://data-api.globalforestwatch.org", alias="GFW_API_BASE_URL")
    gfw_api_key: Optional[SecretStr] = Field(default=None, alias="GFW_API_KEY")

    # NASA Earthdata / ASF DAAC (NISAR)
    earthdata_username: Optional[str] = Field(default=None, alias="EARTHDATA_USERNAME")
    earthdata_password: Optional[SecretStr] = Field(default=None, alias="EARTHDATA_PASSWORD")
    earthdata_bearer_token: Optional[SecretStr] = Field(default=None, alias="EARTHDATA_BEARER_TOKEN")
    asf_search_api_url: str = Field(
        default="https://api.daac.asf.alaska.edu/services/search/param",
        alias="ASF_SEARCH_API_URL",
    )
    asf_s3_credential_endpoint: str = Field(
        default="https://nisar.asf.earthdatacloud.nasa.gov/s3credentials",
        alias="ASF_S3_CREDENTIAL_ENDPOINT",
    )
    asf_aws_region: str = Field(default="us-west-2", alias="ASF_AWS_REGION")

    # Storage & Caching
    storage_dir: Path = Field(default=Path("./data_storage"), alias="STORAGE_DIR")
    cache_backend: str = Field(default="sqlite", alias="CACHE_BACKEND")
    cache_db_path: Path = Field(default=Path("./data_storage/cache.db"), alias="CACHE_DB_PATH")
    cache_default_ttl_seconds: int = Field(default=3600, alias="CACHE_DEFAULT_TTL_SECONDS")
    redis_url: Optional[SecretStr] = Field(default=None, alias="REDIS_URL")
    database_url: Optional[SecretStr] = Field(default=None, alias="DATABASE_URL")

    @property
    def allowed_origins(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins_raw.split(",") if origin.strip()]

    def has_gee_configured(self) -> bool:
        """Check if non-placeholder Google Earth Engine configuration is provided."""
        if not self.gee_project_id or self.gee_project_id == "your-gcp-earth-engine-project-id":
            return False
        return True

    def has_gfw_configured(self) -> bool:
        """Check if non-placeholder Global Forest Watch API key is provided."""
        if not self.gfw_api_key:
            return False
        val = self.gfw_api_key.get_secret_value()
        return bool(val and val != "your-gfw-api-key")

    def has_earthdata_configured(self) -> bool:
        """Check if NASA Earthdata credentials or bearer token are configured."""
        if self.earthdata_bearer_token and self.earthdata_bearer_token.get_secret_value():
            return True
        if (
            self.earthdata_username
            and self.earthdata_username != "your-earthdata-username"
            and self.earthdata_password
            and self.earthdata_password.get_secret_value() != "your-earthdata-password"
        ):
            return True
        return False


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure root and application logger with secret redaction."""
    logger = logging.getLogger("amazon_sentinel")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        handler.setFormatter(formatter)
        handler.addFilter(SecretRedactingFilter())
        logger.addHandler(handler)
    return logger


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    settings = Settings()
    configure_logging(settings.log_level)
    return settings
