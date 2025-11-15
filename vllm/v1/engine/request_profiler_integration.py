# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration helpers for v1 request profiler with AsyncLLM engine."""

import os
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.engine.request_profiler import (
    V1RequestProfilerLogger,
    create_v1_request_profiler_factory,
    is_v1_request_profiling_enabled,
    get_v1_log_file_path
)
from vllm.v1.metrics.loggers import StatLoggerFactory

logger = init_logger(__name__)


def maybe_add_v1_request_profiler(
    stat_loggers: Optional[list[StatLoggerFactory]] = None,
    log_file_path: Optional[str] = None
) -> Optional[list[StatLoggerFactory]]:
    """Automatically add v1 request profiler if environment variable is set.

    This function checks the VLLM_ENABLE_REQUEST_PROFILING environment variable
    and adds the request profiler to the stat loggers list if enabled.

    Args:
        stat_loggers: Existing list of StatLoggerFactory functions (can be None)
        log_file_path: Optional custom log file path

    Returns:
        Updated stat_loggers list if profiling enabled, otherwise original list
    """
    if not is_v1_request_profiling_enabled():
        return stat_loggers

    if stat_loggers is None:
        stat_loggers = []

    # Create profiler factory
    profiler_factory = create_v1_request_profiler_factory(
        log_file_path or get_v1_log_file_path()
    )

    # Add to loggers list
    stat_loggers.append(profiler_factory)

    logger.info(f"V1 Request profiler enabled, logging to: {log_file_path or get_v1_log_file_path()}")

    return stat_loggers


def create_v1_profiler_only(log_file_path: Optional[str] = None) -> list[StatLoggerFactory]:
    """Create a stat_loggers list with only the request profiler.

    Useful when you want to disable default loggers and only use request profiling.

    Args:
        log_file_path: Optional custom log file path

    Returns:
        List containing only the request profiler factory
    """
    return [create_v1_request_profiler_factory(log_file_path or get_v1_log_file_path())]


class V1ProfilerConfig:
    """Configuration helper for v1 request profiling."""

    def __init__(self,
                 log_file_path: Optional[str] = None,
                 auto_enable: bool = True):
        """Initialize profiler configuration.

        Args:
            log_file_path: Custom log file path (if None, uses env var or default)
            auto_enable: Whether to check environment variable for auto-enabling
        """
        self.log_file_path = log_file_path or get_v1_log_file_path()
        self.auto_enable = auto_enable
        self.enabled = auto_enable and is_v1_request_profiling_enabled()

    def get_stat_loggers(self,
                        existing_loggers: Optional[list[StatLoggerFactory]] = None,
                        replace_existing: bool = False) -> Optional[list[StatLoggerFactory]]:
        """Get stat loggers list with request profiler if enabled.

        Args:
            existing_loggers: Existing stat loggers to extend
            replace_existing: If True, replace existing loggers instead of extending

        Returns:
            Updated stat loggers list or None if profiling disabled
        """
        if not self.enabled:
            return existing_loggers

        profiler_factory = create_v1_request_profiler_factory(self.log_file_path)

        if replace_existing or existing_loggers is None:
            return [profiler_factory]
        else:
            return existing_loggers + [profiler_factory]


# Environment-based helpers for common use cases
def get_v1_profiling_stat_loggers() -> Optional[list[StatLoggerFactory]]:
    """Get stat loggers for v1 profiling if enabled via environment.

    Returns:
        List with request profiler if VLLM_ENABLE_REQUEST_PROFILING=1, else None
    """
    return maybe_add_v1_request_profiler()


def get_v1_profiling_stat_loggers_only() -> Optional[list[StatLoggerFactory]]:
    """Get stat loggers with ONLY request profiler if enabled via environment.

    This disables all default loggers and only enables request profiling.

    Returns:
        List with only request profiler if enabled, else None
    """
    if is_v1_request_profiling_enabled():
        return create_v1_profiler_only()
    return None