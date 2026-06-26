"""Dashboard configuration — reads from environment with sane defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DashboardConfig:
    api_base_url: str = field(
        default_factory=lambda: os.getenv("NEUROFORGE_API_URL", "http://localhost:8000")
    )
    api_timeout: int = int(os.getenv("NEUROFORGE_API_TIMEOUT", "30"))
    health_timeout: int = 5
    poll_interval_seconds: int = 2
    page_title: str = "NeuroForge"
    page_icon: str = "🧠"


CONFIG = DashboardConfig()