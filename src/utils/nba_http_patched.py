# src/utils/nba_http_patched.py

from __future__ import annotations

import os
import time
from typing import Any, Optional, Tuple

import requests
from nba_api.stats.library.http import NBAStatsHTTP
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class PatchedNBAStatsHTTP(NBAStatsHTTP):
    """
    Hardened NBAStatsHTTP for CI use.

    Changes vs upstream:
      - fresh Session per request (avoids stale keep-alive hangs in CI)
      - retries/backoff for transient failures
      - default timeout as (connect, read)
      - avoids mutating shared headers dict
      - optional Connection: close in GitHub Actions
    """

    DEFAULT_TIMEOUT: Tuple[float, float] = (10.0, 30.0)

    RETRY_TOTAL = 1
    BACKOFF_FACTOR = 1.0
    STATUS_FORCELIST = (429, 500, 502, 503, 504)
    ALLOWED_METHODS = frozenset(["GET", "HEAD", "OPTIONS"])

    @classmethod
    def get_session(cls) -> requests.Session:
        # Fresh session each time (per your request)
        s = requests.Session()

        retry = Retry(
            total=cls.RETRY_TOTAL,
            connect=cls.RETRY_TOTAL,
            read=cls.RETRY_TOTAL,
            status=cls.RETRY_TOTAL,
            backoff_factor=cls.BACKOFF_FACTOR,
            status_forcelist=cls.STATUS_FORCELIST,
            allowed_methods=cls.ALLOWED_METHODS,
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    def send_api_request(
        self,
        endpoint: str,
        parameters: dict[str, Any],
        referer: Optional[str] = None,
        proxy: Optional[Any] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[Any] = None,
        raise_exception_on_error: bool = False,
    ):
        # Normalize timeout. requests supports either number or (connect, read).
        if timeout is None:
            timeout = self.DEFAULT_TIMEOUT
        elif isinstance(timeout, (int, float)):
            # If nba_api passes timeout=30, convert to (connect, read) for more control
            timeout = (10.0, float(timeout))

        # Copy headers so we never mutate shared dicts from the library
        if headers is None:
            request_headers = (
                dict(self.headers) if getattr(self, "headers", None) else {}
            )
        else:
            request_headers = dict(headers)

        if referer:
            request_headers["Referer"] = referer
            request_headers.setdefault("Origin", "https://stats.nba.com")

        # CI reliability: mitigate half-open keep-alive sockets
        if os.getenv("GITHUB_ACTIONS") == "true":
            request_headers.setdefault("Connection", "close")

        # Call parent implementation but with our session and safe headers.
        # Parent uses cls.get_session(), so overriding get_session() is enough,
        # but we also pass our safe headers and normalized timeout.
        return super().send_api_request(
            endpoint=endpoint,
            parameters=parameters,
            referer=referer,
            proxy=proxy,
            headers=request_headers,
            timeout=timeout,
            raise_exception_on_error=raise_exception_on_error,
        )
