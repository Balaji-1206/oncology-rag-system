import time
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Tuple
from flask import request


class RateLimiter:
    """Thread-safe sliding-window rate limiter."""

    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 20) -> None:
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.window_seconds = 60.0
        self.client_records: Dict[str, List[float]] = defaultdict(list)
        self.lock = Lock()

    def _get_client_identifier(self) -> str:
        """Determines client key based on API key or IP address."""
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key.strip()}"

        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        return f"ip:{request.remote_addr or 'unknown'}"

    def is_allowed(self) -> Tuple[bool, int, float]:
        """
        Checks if the current request is allowed under rate limits.
        Returns:
            (is_allowed, remaining_requests, retry_after_seconds)
        """
        client_id = self._get_client_identifier()
        now = time.time()
        window_start = now - self.window_seconds

        with self.lock:
            timestamps = self.client_records[client_id]

            # Prune timestamps outside current sliding window
            valid_timestamps = [t for t in timestamps if t > window_start]
            self.client_records[client_id] = valid_timestamps

            if len(valid_timestamps) >= self.requests_per_minute:
                oldest_timestamp = valid_timestamps[0]
                retry_after = max(1.0, round((oldest_timestamp + self.window_seconds) - now, 1))
                return False, 0, retry_after

            # Check burst limit in last 5 seconds
            recent_burst = [t for t in valid_timestamps if t > (now - 5.0)]
            if len(recent_burst) >= self.burst_limit:
                return False, 0, 5.0

            valid_timestamps.append(now)
            remaining = self.requests_per_minute - len(valid_timestamps)
            return True, max(0, remaining), 0.0

    def reset(self) -> None:
        """Resets all tracked rate limits."""
        with self.lock:
            self.client_records.clear()


_GLOBAL_RATE_LIMITER = RateLimiter(requests_per_minute=120, burst_limit=30)


def get_rate_limiter() -> RateLimiter:
    """Returns global RateLimiter singleton."""
    return _GLOBAL_RATE_LIMITER
