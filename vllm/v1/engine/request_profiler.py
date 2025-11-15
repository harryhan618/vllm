# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time
from pathlib import Path
from typing import Optional, TextIO

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

logger = init_logger(__name__)


class V1RequestProfilerLogger(StatLoggerBase):
    """V1 engine-compatible request profiler that logs detailed per-request statistics.

    This logger captures comprehensive metrics for each request processed by the v1 engine:
    - Precise timing: arrival, queued, scheduled, first token, completion timestamps
    - Latency breakdown: prefill time, decode time, queue time, e2e latency
    - Token counts: prompt tokens, generated tokens
    - Performance metrics: tokens/second, time-to-first-token
    - Request metadata: finish reason, sampling parameters

    All metrics are logged as JSON Lines format to the specified file.
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 engine_index: int = 0,
                 log_file_path: Optional[str] = None):
        """Initialize the v1 request profiler logger.

        Args:
            vllm_config: vLLM configuration object
            engine_index: Engine index for multi-engine setups
            log_file_path: Path to log file. If None, uses default /tmp/vllm_v1_request_profiles.jsonl
        """
        self.vllm_config = vllm_config
        self.engine_index = engine_index

        # Determine log file path
        if log_file_path is None:
            import os
            log_file_path = os.environ.get(
                'VLLM_REQUEST_PROFILE_LOG_PATH',
                f'/tmp/vllm_v1_request_profiles_{engine_index}.jsonl'
            )

        self.log_file_path = Path(log_file_path)
        self.log_file: Optional[TextIO] = None
        self._open_log_file()

    def _open_log_file(self) -> None:
        """Open the log file for writing."""
        try:
            # Create parent directories if needed
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Open in append mode for multi-engine support
            self.log_file = open(self.log_file_path, 'w', buffering=1)  # Line buffered

            # Write header for new files
            if self.log_file_path.stat().st_size == 0:
                self.log_file.write("# vLLM v1 Request Profile Log\n")
                self.log_file.write("# Format: One JSON object per completed request\n")
                self.log_file.write("# Engine index: {}\n".format(self.engine_index))
                self.log_file.write("# Fields: request_id, timestamps, latencies, tokens, performance\n")

            logger.info(f"V1 Request profiler logging to: {self.log_file_path}")

        except Exception as e:
            logger.error(f"Failed to open v1 request profile log {self.log_file_path}: {e}")
            self.log_file = None

    def record(self,
               scheduler_stats: Optional[SchedulerStats],
               iteration_stats: Optional[IterationStats],
               engine_idx: int = 0):
        """Process iteration stats and log completed requests."""
        if not self.log_file or not iteration_stats:
            return

        # Log all finished requests in this iteration
        for finished_req in iteration_stats.finished_requests:
            self._log_finished_request(finished_req, iteration_stats.iteration_timestamp)

    def _log_finished_request(self, finished_req, iteration_timestamp: float) -> None:
        """Log a completed request with detailed metrics.

        Args:
            finished_req: FinishedRequestStats object
            iteration_timestamp: Timestamp when iteration completed
        """
        if not self.log_file:
            return

        try:
            # Extract timing and metrics from FinishedRequestStats
            # The v1 engine provides much more detailed timing info
            arrival_timestamp = iteration_timestamp - finished_req.e2e_latency
            scheduled_timestamp = arrival_timestamp + finished_req.queued_time
            first_token_timestamp = scheduled_timestamp + finished_req.prefill_time
            completion_timestamp = iteration_timestamp

            # Calculate derived metrics
            total_tokens = finished_req.num_prompt_tokens + finished_req.num_generation_tokens
            tokens_per_second = (finished_req.num_generation_tokens / finished_req.decode_time
                               if finished_req.decode_time > 0 else 0.0)
            time_to_first_token = finished_req.queued_time + finished_req.prefill_time

            # Create comprehensive log entry
            log_entry = {
                # Request identification
                "request_id": f"v1_req_{int(completion_timestamp * 1000000)}_{hash(str(arrival_timestamp)) % 10000:04d}",
                "engine_index": self.engine_index,

                # Timestamps (all in seconds since epoch)
                "iteration_timestamp": iteration_timestamp,
                "arrival_timestamp": arrival_timestamp,
                "scheduled_timestamp": scheduled_timestamp,
                "first_token_timestamp": first_token_timestamp,
                "completion_timestamp": completion_timestamp,

                # Timing breakdown (all in seconds)
                "queued_time": finished_req.queued_time,
                "prefill_time": finished_req.prefill_time,
                "decode_time": finished_req.decode_time,
                "inference_time": finished_req.inference_time,
                "e2e_latency": finished_req.e2e_latency,
                "time_to_first_token": time_to_first_token,

                # Token counts
                "num_prompt_tokens": finished_req.num_prompt_tokens,
                "num_generation_tokens": finished_req.num_generation_tokens,
                "total_tokens": total_tokens,
                "max_tokens_param": finished_req.max_tokens_param,

                # Performance metrics
                "tokens_per_second": tokens_per_second,

                # Request outcome
                "finish_reason": str(finished_req.finish_reason),

                # Model info
                "model_name": self.vllm_config.model_config.model,
                "served_model_name": self.vllm_config.model_config.served_model_name,
            }

            # Write JSON line
            self.log_file.write(json.dumps(log_entry) + '\n')
            self.log_file.flush()

        except Exception as e:
            logger.error(f"Failed to write v1 request profile entry: {e}")

    def log_engine_initialized(self):
        """Called when the engine is initialized."""
        if self.log_file:
            init_entry = {
                "event": "engine_initialized",
                "timestamp": time.time(),
                "engine_index": self.engine_index,
                "model": self.vllm_config.model_config.model,
                "max_model_len": self.vllm_config.model_config.max_model_len,
            }
            self.log_file.write(json.dumps(init_entry) + '\n')
            self.log_file.flush()

    def close(self) -> None:
        """Close the log file."""
        if self.log_file:
            # Write closing event
            try:
                close_entry = {
                    "event": "engine_shutdown",
                    "timestamp": time.time(),
                    "engine_index": self.engine_index
                }
                self.log_file.write(json.dumps(close_entry) + '\n')
                self.log_file.flush()
            except Exception:
                pass  # Don't fail on shutdown

            self.log_file.close()
            self.log_file = None

    def __del__(self):
        """Ensure log file is closed on destruction."""
        self.close()


def create_v1_request_profiler_factory(log_file_path: Optional[str] = None):
    """Create a StatLoggerFactory function for the v1 request profiler.

    Args:
        log_file_path: Path to log file. If None, uses environment variable or default.

    Returns:
        StatLoggerFactory function that can be passed to v1 engine
    """
    def factory(vllm_config: VllmConfig, engine_index: int = 0) -> V1RequestProfilerLogger:
        return V1RequestProfilerLogger(vllm_config, engine_index, log_file_path)

    return factory


def is_v1_request_profiling_enabled() -> bool:
    """Check if v1 request profiling should be enabled via environment."""
    import os
    return os.environ.get('VLLM_ENABLE_REQUEST_PROFILING', '').lower() in ('1', 'true', 'on')


def get_v1_log_file_path() -> str:
    """Get the log file path from environment or default."""
    import os
    return os.environ.get('VLLM_REQUEST_PROFILE_LOG_PATH', '/tmp/vllm_v1_request_profiles.jsonl')