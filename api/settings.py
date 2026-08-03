"""
api/settings.py — Module 21: Production REST API Security & Configuration
Environment-based settings manager for API security, rate limiting, and CORS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class APISettings:
    """
    Centralized configuration settings loaded from environment variables.
    Supports development, testing, and production environments.
    """

    # Environment
    ENVIRONMENT: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development").lower()
    )
    DEBUG: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    )

    # API Keys & Auth
    # AUTH_ENABLED defaults to True in production/testing, False only if explicitly disabled in dev
    AUTH_REQUIRED: bool = field(
        default_factory=lambda: os.getenv("AUTH_REQUIRED", "true").lower() in ("true", "1", "yes")
    )
    # Comma-separated list of valid API keys
    API_KEYS_RAW: str = field(
        default_factory=lambda: os.getenv("API_KEYS", os.getenv("API_KEY", ""))
    )

    # Documentation Control
    ENABLE_DOCS: bool = field(
        default_factory=lambda: os.getenv("ENABLE_DOCS", "true").lower() in ("true", "1", "yes")
    )

    # CORS Allowed Origins
    CORS_ORIGINS_RAW: str = field(
        default_factory=lambda: os.getenv(
            "ALLOWED_CORS_ORIGINS",
            "http://localhost:3000,http://localhost:7860,http://127.0.0.1:8000,http://127.0.0.1:7860"
        )
    )

    # File Upload Security Limits
    MAX_UPLOAD_SIZE_BYTES: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(20 * 1024 * 1024)))  # 20MB
    )
    ALLOWED_MIME_TYPES: Set[str] = field(
        default_factory=lambda: {
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/bmp",
            "image/tiff",
            "application/dicom",
            "application/octet-stream",
        }
    )
    ALLOWED_EXTENSIONS: Set[str] = field(
        default_factory=lambda: {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}
    )

    # Rate Limiting (Requests per minute)
    RATE_LIMIT_PREDICT_PER_MIN: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PREDICT_PER_MIN", "20"))
    )
    RATE_LIMIT_HISTORY_PER_MIN: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_HISTORY_PER_MIN", "60"))
    )
    RATE_LIMIT_HEALTH_PER_MIN: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_HEALTH_PER_MIN", "120"))
    )
    RATE_LIMIT_DEFAULT_PER_MIN: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_DEFAULT_PER_MIN", "100"))
    )

    @property
    def valid_api_keys(self) -> List[str]:
        """Return parsed list of valid API keys."""
        if not self.API_KEYS_RAW:
            return []
        return [k.strip() for k in self.API_KEYS_RAW.split(",") if k.strip()]

    @property
    def cors_origins(self) -> List[str]:
        """Return parsed list of allowed CORS origins."""
        if self.ENVIRONMENT == "production":
            # In production, require explicit origins list; no wildcard
            origins = [o.strip() for o in self.CORS_ORIGINS_RAW.split(",") if o.strip() and o.strip() != "*"]
            return origins or ["https://medical-ai-platform.internal"]
        # Development / Testing
        return [o.strip() for o in self.CORS_ORIGINS_RAW.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT == "testing"


# Singleton instance
settings = APISettings()
