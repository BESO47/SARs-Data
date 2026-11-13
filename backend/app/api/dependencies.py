"""Shared FastAPI dependencies for query parameters, settings, and cache access."""

from fastapi import Depends

from app.cache.manager import CacheAndJobStore, get_cache_store
from app.core.config import Settings, get_settings


def get_app_settings() -> Settings:
    return get_settings()


def get_app_cache_store(
    store: CacheAndJobStore = Depends(get_cache_store),
) -> CacheAndJobStore:
    return store
