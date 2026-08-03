"""
Medical AI Engine — EngineConfiguration

Manages configuration loading, validation, default settings, and YAML parsing.
Supports timeouts, thread limits, logging levels, health monitoring interval,
statistics interval, and future async/distributed settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from medical_ai_engine.core.exceptions import ConfigurationException


@dataclass
class EngineConfiguration:
    """
    Configuration dataclass for MedicalAIEngine.
    Thread-safe and loaded from YAML or dict.
    """

    # ── Engine Core Settings ───────────────────────────────────────────────
    engine_name: str = "MedicalAIEngineCore"
    version: str = "0.14.2.1"
    environment: str = "production"

    # ── Execution Controls ─────────────────────────────────────────────────
    request_timeout_seconds: float = 30.0
    max_concurrent_threads: int = 16
    thread_pool_prefix: str = "MedAI-EngineWorker"

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file_path: str = "logs/medical_ai_engine.log"
    enable_console_logging: bool = True
    enable_file_logging: bool = True

    # ── Health & Statistics ────────────────────────────────────────────────
    health_check_interval_seconds: float = 60.0
    statistics_report_interval_seconds: float = 300.0
    enable_health_monitor: bool = True

    # ── Future Async & Distributed Flags ───────────────────────────────────
    enable_async_execution: bool = False
    enable_distributed_mode: bool = False
    node_id: str = "node_local_01"

    # ── Extensible Custom Settings ─────────────────────────────────────────
    extra_settings: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> EngineConfiguration:
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: File path to YAML configuration file.

        Returns:
            EngineConfiguration instance.

        Raises:
            ConfigurationException: If file not found or YAML invalid.
        """
        path = Path(yaml_path)
        if not path.exists():
            raise ConfigurationException(
                f"Configuration file not found at '{path}'.",
                config_path=str(path),
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationException(
                f"Failed to parse YAML configuration: {e}",
                config_path=str(path),
                cause=e,
            )

        return cls.from_dict(content, config_path=str(path))

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config_path: Optional[str] = None) -> EngineConfiguration:
        """
        Build EngineConfiguration from a dictionary.

        Args:
            data: Dictionary containing configuration key-values.
            config_path: Optional file path string for exception context.

        Returns:
            EngineConfiguration instance.
        """
        engine_cfg = data.get("engine", {})
        exec_cfg = data.get("execution", {})
        logging_cfg = data.get("logging", {})
        health_cfg = data.get("health", {})
        async_cfg = data.get("async_distributed", {})

        kwargs = {
            "engine_name": engine_cfg.get("name", "MedicalAIEngineCore"),
            "version": engine_cfg.get("version", "0.14.2.1"),
            "environment": engine_cfg.get("environment", "production"),
            "request_timeout_seconds": float(exec_cfg.get("request_timeout_seconds", 30.0)),
            "max_concurrent_threads": int(exec_cfg.get("max_concurrent_threads", 16)),
            "thread_pool_prefix": exec_cfg.get("thread_pool_prefix", "MedAI-EngineWorker"),
            "log_level": logging_cfg.get("level", "INFO"),
            "log_file_path": logging_cfg.get("file_path", "logs/medical_ai_engine.log"),
            "enable_console_logging": bool(logging_cfg.get("enable_console", True)),
            "enable_file_logging": bool(logging_cfg.get("enable_file", True)),
            "health_check_interval_seconds": float(health_cfg.get("check_interval_seconds", 60.0)),
            "statistics_report_interval_seconds": float(health_cfg.get("statistics_interval_seconds", 300.0)),
            "enable_health_monitor": bool(health_cfg.get("enable_monitor", True)),
            "enable_async_execution": bool(async_cfg.get("enable_async", False)),
            "enable_distributed_mode": bool(async_cfg.get("enable_distributed", False)),
            "node_id": async_cfg.get("node_id", "node_local_01"),
            "extra_settings": data.get("extra", {}),
        }

        cfg = cls(**kwargs)
        cfg.validate(config_path=config_path)
        return cfg

    def validate(self, config_path: Optional[str] = None) -> None:
        """
        Validate configuration values.

        Raises:
            ConfigurationException: If validation fails.
        """
        if self.request_timeout_seconds <= 0:
            raise ConfigurationException(
                f"request_timeout_seconds must be > 0, got {self.request_timeout_seconds}",
                config_path=config_path,
            )

        if self.max_concurrent_threads <= 0:
            raise ConfigurationException(
                f"max_concurrent_threads must be > 0, got {self.max_concurrent_threads}",
                config_path=config_path,
            )

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ConfigurationException(
                f"log_level must be one of {valid_levels}, got '{self.log_level}'",
                config_path=config_path,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a structured dictionary."""
        return {
            "engine": {
                "name": self.engine_name,
                "version": self.version,
                "environment": self.environment,
            },
            "execution": {
                "request_timeout_seconds": self.request_timeout_seconds,
                "max_concurrent_threads": self.max_concurrent_threads,
                "thread_pool_prefix": self.thread_pool_prefix,
            },
            "logging": {
                "level": self.log_level,
                "file_path": self.log_file_path,
                "enable_console": self.enable_console_logging,
                "enable_file": self.enable_file_logging,
            },
            "health": {
                "check_interval_seconds": self.health_check_interval_seconds,
                "statistics_interval_seconds": self.statistics_report_interval_seconds,
                "enable_monitor": self.enable_health_monitor,
            },
            "async_distributed": {
                "enable_async": self.enable_async_execution,
                "enable_distributed": self.enable_distributed_mode,
                "node_id": self.node_id,
            },
            "extra": self.extra_settings,
        }
