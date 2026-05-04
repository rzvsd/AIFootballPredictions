from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import requests

DEFAULT_API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
DEFAULT_API_FOOTBALL_RAPIDAPI_BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _read_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return data
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return data
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        data[key] = value
    return data


def _candidate_env_paths() -> list[Path]:
    paths: list[Path] = []
    explicit = os.getenv("API_FOOTBALL_ENV_FILE")
    if explicit:
        paths.append(Path(explicit).expanduser())
    paths.append(Path.cwd() / ".env")
    try:
        repo_root = Path(__file__).resolve().parents[1]
        paths.append(repo_root / ".env")
    except Exception:
        pass
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _resolve_api_key(explicit_key: str | None) -> str | None:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    env_key_names = (
        "API_FOOTBALL_KEY",
        "API_FOOTBALL_ODDS_KEY",
        "APISPORTS_KEY",
    )
    for name in env_key_names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()

    for env_path in _candidate_env_paths():
        env_data = _read_env_file(env_path)
        for name in env_key_names:
            value = env_data.get(name)
            if value and value.strip():
                return value.strip()
    return None


def _resolve_provider(base_url: str) -> str:
    provider = (os.getenv("API_FOOTBALL_PROVIDER") or "").strip().lower()
    if provider in {"api-sports", "apisports", "rapidapi"}:
        return provider
    if "rapidapi.com" in base_url:
        return "rapidapi"
    return "api-sports"


def _resolve_base_url(base_url: str, provider: str) -> str:
    explicit = (os.getenv("API_FOOTBALL_BASE_URL") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    if provider == "rapidapi" and base_url.rstrip("/") == DEFAULT_API_FOOTBALL_BASE_URL:
        return DEFAULT_API_FOOTBALL_RAPIDAPI_BASE_URL
    return base_url.rstrip("/")


class APIFootballClient:
    """
    Thin API-Football v3 client with free-tier guardrails:
      - auth from constructor or API_FOOTBALL_KEY env var
      - request budget guardrail (max requests/day)
      - per-minute rate limiting (default: 10/min)
      - retry/backoff for 429 and 5xx responses
      - JSON cache helpers on disk
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_API_FOOTBALL_BASE_URL,
        cache_dir: str | os.PathLike[str] = "data/cache/api_football",
        timeout_seconds: int = 25,
        rate_per_minute: int = 10,
        max_requests_per_day: int | None = None,
        max_retries: int = 4,
        backoff_seconds: float = 1.25,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = _resolve_api_key(api_key)
        if not self.api_key:
            raise ValueError(
                "Missing API key. Set API_FOOTBALL_KEY (env or .env), "
                "or pass api_key explicitly."
            )

        self.provider = _resolve_provider(base_url)
        self.base_url = _resolve_base_url(base_url, self.provider)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.timeout_seconds = int(timeout_seconds)
        self.rate_per_minute = max(0, int(rate_per_minute))
        self.max_requests_per_day = (
            None if max_requests_per_day is None else int(max_requests_per_day)
        )
        if self.max_requests_per_day is not None and self.max_requests_per_day <= 0:
            raise ValueError("max_requests_per_day must be > 0 when provided.")

        self.max_retries = max(0, int(max_retries))
        self.backoff_seconds = max(0.1, float(backoff_seconds))

        self.session = session or requests.Session()
        auth_headers: dict[str, str] = {"accept": "application/json"}
        if self.provider == "rapidapi":
            rapid_host = (
                os.getenv("API_FOOTBALL_RAPIDAPI_HOST")
                or "api-football-v1.p.rapidapi.com"
            ).strip()
            auth_headers["x-rapidapi-host"] = rapid_host
            auth_headers["x-rapidapi-key"] = self.api_key
        else:
            auth_headers["x-apisports-key"] = self.api_key
        self.session.headers.update(auth_headers)

        self._request_times: deque[float] = deque()
        self._budget_state_path = self.cache_dir / "_request_budget_state.json"
        self._process_request_attempts = 0

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "APIFootballClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    @staticmethod
    def _today_utc_iso() -> str:
        return dt.datetime.now(dt.timezone.utc).date().isoformat()

    @staticmethod
    def _extract_payload_error(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        errors = payload.get("errors")
        if errors in (None, "", [], {}):
            return None
        if isinstance(errors, dict):
            parts: list[str] = []
            for key, value in errors.items():
                if value in (None, "", [], {}):
                    continue
                key_s = str(key).strip()
                value_s = str(value).strip()
                if not value_s:
                    continue
                if key_s:
                    parts.append(f"{key_s}: {value_s}")
                else:
                    parts.append(value_s)
            if parts:
                return "; ".join(parts)
            return None
        return str(errors).strip() or None

    def _raise_if_payload_error(self, payload: Any) -> None:
        api_error = self._extract_payload_error(payload)
        if api_error:
            if "Missing application key" in api_error and self.provider == "api-sports":
                api_error = (
                    f"{api_error} | hint: API_FOOTBALL_KEY is not accepted by API-Sports. "
                    "If you use RapidAPI, set API_FOOTBALL_PROVIDER=rapidapi."
                )
            raise RuntimeError(f"API-Football returned error payload: {api_error}")

    def _load_budget_state(self) -> dict[str, Any]:
        today = self._today_utc_iso()
        fallback = {"date": today, "count": 0}
        if not self._budget_state_path.exists():
            return fallback
        try:
            raw = json.loads(self._budget_state_path.read_text(encoding="utf-8"))
            date_val = str(raw.get("date", today))
            count_val = int(raw.get("count", 0))
            if count_val < 0:
                count_val = 0
            return {"date": date_val, "count": count_val}
        except Exception:
            return fallback

    def _save_budget_state(self, state: Mapping[str, Any]) -> None:
        payload = {
            "date": str(state.get("date", self._today_utc_iso())),
            "count": int(state.get("count", 0)),
        }
        tmp_path = self._budget_state_path.with_suffix(".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self._budget_state_path)

    def _consume_budget(self) -> None:
        self._process_request_attempts += 1
        if self.max_requests_per_day is None:
            return

        state = self._load_budget_state()
        today = self._today_utc_iso()
        if state.get("date") != today:
            state = {"date": today, "count": 0}

        used = int(state.get("count", 0))
        if used >= self.max_requests_per_day:
            raise RuntimeError(
                "Daily API request budget exhausted "
                f"({used}/{self.max_requests_per_day})."
            )

        state["count"] = used + 1
        self._save_budget_state(state)

    def _wait_for_rate_limit(self) -> None:
        if self.rate_per_minute <= 0:
            return

        window_seconds = 60.0
        now = time.monotonic()
        while self._request_times and now - self._request_times[0] >= window_seconds:
            self._request_times.popleft()

        if len(self._request_times) >= self.rate_per_minute:
            oldest = self._request_times[0]
            sleep_seconds = window_seconds - (now - oldest)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            now = time.monotonic()
            while self._request_times and now - self._request_times[0] >= window_seconds:
                self._request_times.popleft()

        self._request_times.append(time.monotonic())

    @staticmethod
    def _sorted_param_pairs(params: Mapping[str, Any] | None) -> list[tuple[str, str]]:
        if not params:
            return []
        pairs: list[tuple[str, str]] = []
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    pairs.append((str(key), str(item)))
            elif isinstance(value, bool):
                pairs.append((str(key), "true" if value else "false"))
            else:
                pairs.append((str(key), str(value)))
        pairs.sort(key=lambda item: (item[0], item[1]))
        return pairs

    def build_cache_key(
        self,
        namespace: str,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
    ) -> str:
        endpoint_clean = endpoint.strip("/").replace("/", "_") or "root"
        query = urlencode(self._sorted_param_pairs(params), doseq=True)
        fingerprint = hashlib.sha1(
            f"{endpoint_clean}?{query}".encode("utf-8")
        ).hexdigest()[:16]
        namespace_clean = namespace.strip("/").replace("\\", "/")
        filename = f"{endpoint_clean}_{fingerprint}.json"
        if not namespace_clean:
            return filename
        return f"{namespace_clean}/{filename}"

    def cache_path(self, cache_key: str | os.PathLike[str]) -> Path:
        key_path = Path(str(cache_key))
        if key_path.is_absolute():
            raise ValueError("cache_key must be a relative path.")
        path = self.cache_dir / key_path
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")

        root = self.cache_dir.resolve()
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError("cache_key escapes cache_dir.")
        return path

    def read_json_cache(
        self,
        cache_key: str | os.PathLike[str],
        *,
        max_age_seconds: int | None = None,
    ) -> Any | None:
        path = self.cache_path(cache_key)
        if not path.exists():
            return None
        if max_age_seconds is not None:
            age = time.time() - path.stat().st_mtime
            if age > max_age_seconds:
                return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def write_json_cache(
        self,
        cache_key: str | os.PathLike[str],
        payload: Any,
    ) -> Path:
        path = self.cache_path(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return path

    def _next_backoff(self, attempt_idx: int, response: requests.Response | None) -> float:
        retry_after = None
        if response is not None:
            header = response.headers.get("Retry-After")
            if header:
                try:
                    retry_after = max(0.0, float(header))
                except ValueError:
                    retry_after = None
        if retry_after is not None:
            return min(retry_after, 120.0)
        exp_backoff = self.backoff_seconds * (2 ** attempt_idx)
        jitter = random.uniform(0.0, 0.35)
        return min(exp_backoff + jitter, 120.0)

    def get_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        pairs = self._sorted_param_pairs(params)
        clean_params: list[tuple[str, str]] = pairs
        last_error: Exception | None = None

        for attempt_idx in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            self._consume_budget()
            try:
                response = self.session.get(
                    url,
                    params=clean_params,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt_idx >= self.max_retries:
                    break
                time.sleep(self._next_backoff(attempt_idx, None))
                continue

            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt_idx >= self.max_retries:
                    tail = response.text[:240].replace("\n", " ").strip()
                    raise RuntimeError(
                        f"API-Football request failed after retries: "
                        f"{response.status_code} {tail}"
                    )
                time.sleep(self._next_backoff(attempt_idx, response))
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                tail = response.text[:240].replace("\n", " ").strip()
                raise RuntimeError(
                    f"API-Football request failed: {response.status_code} {tail}"
                ) from exc

            try:
                payload = response.json()
            except ValueError as exc:
                raise RuntimeError("API-Football returned non-JSON response.") from exc
            self._raise_if_payload_error(payload)
            return payload

        if last_error is not None:
            raise RuntimeError(f"API-Football request failed: {last_error}") from last_error
        raise RuntimeError("API-Football request failed for unknown reason.")

    def cached_get_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
        cache_key: str | os.PathLike[str] | None = None,
        max_age_seconds: int | None = None,
        force_refresh: bool = False,
    ) -> Any:
        if cache_key and not force_refresh:
            cached = self.read_json_cache(cache_key, max_age_seconds=max_age_seconds)
            if cached is not None:
                if self._extract_payload_error(cached) is None:
                    return cached

        payload = self.get_json(endpoint, params=params)
        self._raise_if_payload_error(payload)
        if cache_key:
            self.write_json_cache(cache_key, payload)
        return payload

    def fixtures(
        self,
        *,
        league_id: int,
        season: int,
        status: str | list[str] | tuple[str, ...] | set[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        cache_key: str | os.PathLike[str] | None = None,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"league": int(league_id), "season": int(season)}
        if status:
            if isinstance(status, (list, tuple, set)):
                params["status"] = "-".join(str(s).strip() for s in status if str(s).strip())
            else:
                params["status"] = str(status).strip()
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to

        payload = self.cached_get_json(
            "/fixtures",
            params=params,
            cache_key=cache_key,
            force_refresh=force_refresh,
        )
        if isinstance(payload, dict):
            response = payload.get("response", [])
            if isinstance(response, list):
                return response
        return []

    def fixture_statistics(
        self,
        fixture_id: int,
        *,
        cache_key: str | os.PathLike[str] | None = None,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        params = {"fixture": int(fixture_id)}
        payload = self.cached_get_json(
            "/fixtures/statistics",
            params=params,
            cache_key=cache_key,
            force_refresh=force_refresh,
        )
        if isinstance(payload, dict):
            response = payload.get("response", [])
            if isinstance(response, list):
                return response
        return []

    def fixture_odds(
        self,
        fixture_id: int,
        *,
        cache_key: str | os.PathLike[str] | None = None,
        max_age_seconds: int | None = None,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        params = {"fixture": int(fixture_id)}
        payload = self.cached_get_json(
            "/odds",
            params=params,
            cache_key=cache_key,
            max_age_seconds=max_age_seconds,
            force_refresh=force_refresh,
        )
        if isinstance(payload, dict):
            response = payload.get("response", [])
            if isinstance(response, list):
                return response
        return []

    def budget_status(self) -> dict[str, Any]:
        state = self._load_budget_state()
        today = self._today_utc_iso()
        if state.get("date") != today:
            state = {"date": today, "count": 0}
        return {
            "date_utc": state.get("date"),
            "used_requests_today": int(state.get("count", 0)),
            "max_requests_per_day": self.max_requests_per_day,
            "process_request_attempts": self._process_request_attempts,
        }
