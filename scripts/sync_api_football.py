"""
Sync fixtures and match stats from API-Football into local CSVs.
Odds fetching is optional (disabled by default for low-call operation).

Outputs under --data-dir:
  - history_fixtures.csv      (finished statuses: FT/AET/PEN)
  - upcoming_fixtures.csv     (upcoming status: NS)
  - fixture_quality_report.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from providers.api_football import APIFootballClient
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from providers.api_football import APIFootballClient

try:
    import config
except Exception:  # pragma: no cover - optional import when run standalone
    config = None  # type: ignore[assignment]


FINISHED_STATUSES = {"FT", "AET", "PEN"}
UPCOMING_STATUS = "NS"
DEFAULT_LEAGUE_IDS_CSV = "39"
DEFAULT_DATA_DIR = "data/api_football"
DEFAULT_HISTORY_DAYS = 365
DEFAULT_HORIZON_DAYS = 14
DEFAULT_MAX_REQUESTS = 100
DEFAULT_RATE_PER_MINUTE = 10
DEFAULT_MIN_ODDS_COVERAGE = 0.60
DEFAULT_MIN_STATS_COVERAGE = 0.60
DEFAULT_FETCH_ODDS = False

OUTPUT_COLUMNS = [
    "fixture_id",
    "status",
    "fixture_datetime_utc",
    "date",
    "time",
    "country",
    "league",
    "league_id",
    "season",
    "home_id",
    "away_id",
    "home",
    "away",
    "score_home",
    "score_away",
    "shots_home",
    "shots_away",
    "shots_on_target_home",
    "shots_on_target_away",
    "corners_home",
    "corners_away",
    "possession_home",
    "possession_away",
    # Extended match stats (optional, when API provides them)
    "shots_off_home",
    "shots_off_away",
    "blocked_shots_home",
    "blocked_shots_away",
    "goal_attempts_home",
    "goal_attempts_away",
    "attacks_home",
    "attacks_away",
    "dangerous_attacks_home",
    "dangerous_attacks_away",
    "counter_attacks_home",
    "counter_attacks_away",
    "cross_attacks_home",
    "cross_attacks_away",
    "goalkeeper_saves_home",
    "goalkeeper_saves_away",
    "fouls_home",
    "fouls_away",
    "offsides_home",
    "offsides_away",
    "free_kicks_home",
    "free_kicks_away",
    "throwins_home",
    "throwins_away",
    "yellow_cards_home",
    "yellow_cards_away",
    "red_cards_home",
    "red_cards_away",
    "substitutions_home",
    "substitutions_away",
    "odds_over_2_5",
    "odds_under_2_5",
    "odds_btts_yes",
    "odds_btts_no",
    "datameci",
    "orameci",
    "txtechipa1",
    "txtechipa2",
    "codechipa1",
    "codechipa2",
    "scor1",
    "scor2",
    "sut",
    "sutt",
    "cor",
    "ballp",
    "ballph",
    "ballpa",
    "cotao",
    "cotau",
    "gg",
    "ng",
]


def _default_season(today: dt.date | None = None) -> int:
    now = today or dt.date.today()
    return now.year if now.month >= 7 else now.year - 1


def _cfg_value(name: str, fallback: Any) -> Any:
    if config is None:
        return fallback
    return getattr(config, name, fallback)


def _default_data_dir() -> str:
    env = os.getenv("API_FOOTBALL_DATA_DIR")
    if env:
        return env.strip()
    cfg = _cfg_value("API_FOOTBALL_DEFAULT_DATA_DIR", DEFAULT_DATA_DIR)
    cfg_str = str(cfg or DEFAULT_DATA_DIR).strip()
    return cfg_str or DEFAULT_DATA_DIR


def _default_league_ids_csv() -> str:
    env = os.getenv("API_FOOTBALL_LEAGUE_IDS")
    if env:
        return env.strip()
    cfg = _cfg_value("API_FOOTBALL_DEFAULT_LEAGUE_IDS", None)
    if isinstance(cfg, (list, tuple, set)):
        tokens = [str(v).strip() for v in cfg if str(v).strip()]
        if tokens:
            return ",".join(tokens)
    if isinstance(cfg, str) and cfg.strip():
        return cfg.strip()
    return DEFAULT_LEAGUE_IDS_CSV


def _int_from_env_or_cfg(env_key: str, cfg_key: str, fallback: int) -> int:
    raw_env = os.getenv(env_key)
    if raw_env is not None and raw_env.strip():
        try:
            return int(raw_env.strip())
        except ValueError:
            pass
    cfg = _cfg_value(cfg_key, fallback)
    try:
        return int(cfg)
    except Exception:
        return int(fallback)


def _bool_from_env_or_cfg(env_key: str, cfg_key: str, fallback: bool) -> bool:
    raw_env = os.getenv(env_key)
    if raw_env is not None and raw_env.strip():
        token = raw_env.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    cfg = _cfg_value(cfg_key, fallback)
    if isinstance(cfg, bool):
        return cfg
    token = str(cfg).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return bool(fallback)


def _parse_league_ids(raw: str) -> list[int]:
    league_ids: list[int] = []
    for token in raw.split(","):
        value = token.strip()
        if not value:
            continue
        league_ids.append(int(value))
    if not league_ids:
        raise ValueError("At least one league id is required.")
    return sorted(set(league_ids))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    text = text.replace("%", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _parse_api_datetime_utc(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        fixed = text.replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(fixed)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def _format_pair(home: Any, away: Any) -> str:
    if _is_missing(home) or _is_missing(away):
        return ""
    return f"{int(round(float(home)))}-{int(round(float(away)))}"


def _team_code(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except Exception:
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        return text


def _normalize_stat_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _extract_team_stats(entry: dict[str, Any]) -> dict[str, float | None]:
    stats_by_name: dict[str, Any] = {}
    for stat in entry.get("statistics") or []:
        name = _normalize_stat_name(stat.get("type"))
        stats_by_name[name] = stat.get("value")

    def pick(*aliases: str) -> float | None:
        for alias in aliases:
            if alias in stats_by_name:
                return _to_float(stats_by_name[alias])
        return None

    return {
        "shots": pick("total shots", "shots total"),
        "shots_on_target": pick("shots on goal", "shots on target"),
        "shots_off": pick("shots off goal", "shots off target"),
        "blocked_shots": pick("blocked shots"),
        "goal_attempts": pick("goal attempts", "goal attempt"),
        "attacks": pick("attacks", "total attacks"),
        "dangerous_attacks": pick("dangerous attacks", "dangerous attack", "danger attacks"),
        "counter_attacks": pick("counter attacks", "counter attack"),
        "cross_attacks": pick("cross attacks", "cross attack"),
        "goalkeeper_saves": pick("goalkeeper saves", "saves"),
        "fouls": pick("fouls"),
        "offsides": pick("offsides", "offsides"),
        "free_kicks": pick("free kicks", "freekicks"),
        "throwins": pick("throwins", "throw-ins", "throw in"),
        "yellow_cards": pick("yellow cards", "yellow card"),
        "red_cards": pick("red cards", "red card"),
        "substitutions": pick("substitutions", "substitution"),
        "corners": pick("corner kicks", "corners"),
        "possession": pick("ball possession", "possession"),
    }


def _fixture_stats_from_payload(
    payload: list[dict[str, Any]],
    home_id: int | None,
    away_id: int | None,
) -> dict[str, float | None]:
    parsed_by_team_id: dict[int, dict[str, float | None]] = {}
    parsed_in_order: list[dict[str, float | None]] = []

    for entry in payload:
        team = entry.get("team") or {}
        team_id = team.get("id")
        parsed = _extract_team_stats(entry)
        parsed_in_order.append(parsed)
        if team_id is not None:
            try:
                parsed_by_team_id[int(team_id)] = parsed
            except Exception:
                pass

    home_stats = None
    away_stats = None
    if home_id is not None:
        home_stats = parsed_by_team_id.get(int(home_id))
    if away_id is not None:
        away_stats = parsed_by_team_id.get(int(away_id))

    if (home_stats is None or away_stats is None) and len(parsed_in_order) >= 2:
        if home_stats is None:
            home_stats = parsed_in_order[0]
        if away_stats is None:
            away_stats = parsed_in_order[1]

    home_stats = home_stats or {}
    away_stats = away_stats or {}

    return {
        "shots_home": home_stats.get("shots"),
        "shots_away": away_stats.get("shots"),
        "shots_on_target_home": home_stats.get("shots_on_target"),
        "shots_on_target_away": away_stats.get("shots_on_target"),
        "shots_off_home": home_stats.get("shots_off"),
        "shots_off_away": away_stats.get("shots_off"),
        "blocked_shots_home": home_stats.get("blocked_shots"),
        "blocked_shots_away": away_stats.get("blocked_shots"),
        "goal_attempts_home": home_stats.get("goal_attempts"),
        "goal_attempts_away": away_stats.get("goal_attempts"),
        "attacks_home": home_stats.get("attacks"),
        "attacks_away": away_stats.get("attacks"),
        "dangerous_attacks_home": home_stats.get("dangerous_attacks"),
        "dangerous_attacks_away": away_stats.get("dangerous_attacks"),
        "counter_attacks_home": home_stats.get("counter_attacks"),
        "counter_attacks_away": away_stats.get("counter_attacks"),
        "cross_attacks_home": home_stats.get("cross_attacks"),
        "cross_attacks_away": away_stats.get("cross_attacks"),
        "goalkeeper_saves_home": home_stats.get("goalkeeper_saves"),
        "goalkeeper_saves_away": away_stats.get("goalkeeper_saves"),
        "fouls_home": home_stats.get("fouls"),
        "fouls_away": away_stats.get("fouls"),
        "offsides_home": home_stats.get("offsides"),
        "offsides_away": away_stats.get("offsides"),
        "free_kicks_home": home_stats.get("free_kicks"),
        "free_kicks_away": away_stats.get("free_kicks"),
        "throwins_home": home_stats.get("throwins"),
        "throwins_away": away_stats.get("throwins"),
        "yellow_cards_home": home_stats.get("yellow_cards"),
        "yellow_cards_away": away_stats.get("yellow_cards"),
        "red_cards_home": home_stats.get("red_cards"),
        "red_cards_away": away_stats.get("red_cards"),
        "substitutions_home": home_stats.get("substitutions"),
        "substitutions_away": away_stats.get("substitutions"),
        "corners_home": home_stats.get("corners"),
        "corners_away": away_stats.get("corners"),
        "possession_home": home_stats.get("possession"),
        "possession_away": away_stats.get("possession"),
    }


def _is_full_time_market(name: str) -> bool:
    exclude_markers = (
        "1st",
        "2nd",
        "first half",
        "second half",
        "half-time",
        "halftime",
        "interval",
    )
    return not any(marker in name for marker in exclude_markers)


def _is_over_under_market(name: str) -> bool:
    if "over/under" not in name and "goals over/under" not in name:
        return False
    # Exclude non-goals O/U style markets.
    exclude_markers = (
        "corner",
        "card",
        "booking",
        "result",
        "both teams",
        "clean sheet",
        "odd/even",
        "team",
    )
    return not any(marker in name for marker in exclude_markers)


def _is_btts_market(name: str) -> bool:
    # Keep only direct BTTS markets (exclude combo/derived variants).
    if name in {"both teams to score", "both teams score", "gg/ng"}:
        return True
    if ("both teams to score" in name or "both teams score" in name) and not any(
        marker in name for marker in ("result", "total goals", "/", "1st", "2nd", "half")
    ):
        return True
    return False


def _odds_from_payload(payload: list[dict[str, Any]]) -> dict[str, float | None]:
    over_25 = None
    under_25 = None
    btts_yes = None
    btts_no = None

    for fixture_odds in payload:
        for bookmaker in fixture_odds.get("bookmakers") or []:
            for bet in bookmaker.get("bets") or []:
                market_name = _normalize_stat_name(bet.get("name"))
                if not market_name or not _is_full_time_market(market_name):
                    continue

                values = bet.get("values") or []
                if _is_over_under_market(market_name):
                    over_in_bet = None
                    under_in_bet = None
                    for value in values:
                        odd = _to_float(value.get("odd"))
                        if odd is None:
                            continue
                        value_text = _normalize_stat_name(
                            " ".join(
                                [
                                    str(value.get("value", "")),
                                    str(value.get("label", "")),
                                    str(value.get("handicap", "")),
                                    str(value.get("line", "")),
                                ]
                            )
                        )
                        if "2.5" not in value_text:
                            continue
                        tokens = set(re.findall(r"[a-z0-9\.]+", value_text))
                        is_over = "over" in tokens
                        is_under = "under" in tokens
                        if is_over and not is_under and over_in_bet is None:
                            over_in_bet = odd
                        if is_under and not is_over and under_in_bet is None:
                            under_in_bet = odd
                    # Prefer a paired O/U line from the same market bet.
                    if over_25 is None and over_in_bet is not None:
                        over_25 = over_in_bet
                    if under_25 is None and under_in_bet is not None:
                        under_25 = under_in_bet

                if _is_btts_market(market_name):
                    yes_in_bet = None
                    no_in_bet = None
                    for value in values:
                        odd = _to_float(value.get("odd"))
                        if odd is None:
                            continue
                        value_text = _normalize_stat_name(
                            " ".join([str(value.get("value", "")), str(value.get("label", ""))])
                        )
                        tokens = set(re.findall(r"[a-z0-9\.]+", value_text))
                        if ("yes" in tokens or "gg" in tokens) and yes_in_bet is None:
                            yes_in_bet = odd
                        if ("no" in tokens or "ng" in tokens) and no_in_bet is None:
                            no_in_bet = odd
                    if btts_yes is None and yes_in_bet is not None:
                        btts_yes = yes_in_bet
                    if btts_no is None and no_in_bet is not None:
                        btts_no = no_in_bet

            if all(v is not None for v in (over_25, under_25, btts_yes, btts_no)):
                break
        if all(v is not None for v in (over_25, under_25, btts_yes, btts_no)):
            break

    return {
        "odds_over_2_5": over_25,
        "odds_under_2_5": under_25,
        "odds_btts_yes": btts_yes,
        "odds_btts_no": btts_no,
    }


def _normalize_fixture_row(
    fixture_row: dict[str, Any],
    stats: dict[str, float | None],
    odds: dict[str, float | None],
) -> dict[str, Any]:
    fixture = fixture_row.get("fixture") or {}
    league = fixture_row.get("league") or {}
    teams = fixture_row.get("teams") or {}
    goals = fixture_row.get("goals") or {}

    home_team = teams.get("home") or {}
    away_team = teams.get("away") or {}

    fixture_id = fixture.get("id")
    status = str((fixture.get("status") or {}).get("short") or "").strip()
    dt_utc = _parse_api_datetime_utc(fixture.get("date"))
    date_str = dt_utc.date().isoformat() if dt_utc else str(fixture.get("date") or "")[:10]
    time_str = dt_utc.strftime("%H:%M") if dt_utc else ""
    time_hhmm = dt_utc.strftime("%H%M") if dt_utc else ""

    home_id = home_team.get("id")
    away_id = away_team.get("id")
    home_name = str(home_team.get("name") or "").strip()
    away_name = str(away_team.get("name") or "").strip()

    score_home = _to_int(goals.get("home"))
    score_away = _to_int(goals.get("away"))

    shots_home = _to_int(stats.get("shots_home"))
    shots_away = _to_int(stats.get("shots_away"))
    sot_home = _to_int(stats.get("shots_on_target_home"))
    sot_away = _to_int(stats.get("shots_on_target_away"))
    corners_home = _to_int(stats.get("corners_home"))
    corners_away = _to_int(stats.get("corners_away"))
    poss_home = _to_float(stats.get("possession_home"))
    poss_away = _to_float(stats.get("possession_away"))
    shots_off_home = _to_int(stats.get("shots_off_home"))
    shots_off_away = _to_int(stats.get("shots_off_away"))
    blocked_shots_home = _to_int(stats.get("blocked_shots_home"))
    blocked_shots_away = _to_int(stats.get("blocked_shots_away"))
    goal_attempts_home = _to_int(stats.get("goal_attempts_home"))
    goal_attempts_away = _to_int(stats.get("goal_attempts_away"))
    attacks_home = _to_int(stats.get("attacks_home"))
    attacks_away = _to_int(stats.get("attacks_away"))
    dangerous_attacks_home = _to_int(stats.get("dangerous_attacks_home"))
    dangerous_attacks_away = _to_int(stats.get("dangerous_attacks_away"))
    counter_attacks_home = _to_int(stats.get("counter_attacks_home"))
    counter_attacks_away = _to_int(stats.get("counter_attacks_away"))
    cross_attacks_home = _to_int(stats.get("cross_attacks_home"))
    cross_attacks_away = _to_int(stats.get("cross_attacks_away"))
    goalkeeper_saves_home = _to_int(stats.get("goalkeeper_saves_home"))
    goalkeeper_saves_away = _to_int(stats.get("goalkeeper_saves_away"))
    fouls_home = _to_int(stats.get("fouls_home"))
    fouls_away = _to_int(stats.get("fouls_away"))
    offsides_home = _to_int(stats.get("offsides_home"))
    offsides_away = _to_int(stats.get("offsides_away"))
    free_kicks_home = _to_int(stats.get("free_kicks_home"))
    free_kicks_away = _to_int(stats.get("free_kicks_away"))
    throwins_home = _to_int(stats.get("throwins_home"))
    throwins_away = _to_int(stats.get("throwins_away"))
    yellow_cards_home = _to_int(stats.get("yellow_cards_home"))
    yellow_cards_away = _to_int(stats.get("yellow_cards_away"))
    red_cards_home = _to_int(stats.get("red_cards_home"))
    red_cards_away = _to_int(stats.get("red_cards_away"))
    substitutions_home = _to_int(stats.get("substitutions_home"))
    substitutions_away = _to_int(stats.get("substitutions_away"))

    over_25 = _to_float(odds.get("odds_over_2_5"))
    under_25 = _to_float(odds.get("odds_under_2_5"))
    btts_yes = _to_float(odds.get("odds_btts_yes"))
    btts_no = _to_float(odds.get("odds_btts_no"))

    return {
        "fixture_id": fixture_id,
        "status": status,
        "fixture_datetime_utc": dt_utc.isoformat() if dt_utc else "",
        "date": date_str,
        "time": time_str,
        "country": str(league.get("country") or "").strip(),
        "league": str(league.get("name") or "").strip(),
        "league_id": league.get("id"),
        "season": league.get("season"),
        "home_id": home_id,
        "away_id": away_id,
        "home": home_name,
        "away": away_name,
        "score_home": score_home,
        "score_away": score_away,
        "shots_home": shots_home,
        "shots_away": shots_away,
        "shots_on_target_home": sot_home,
        "shots_on_target_away": sot_away,
        "corners_home": corners_home,
        "corners_away": corners_away,
        "possession_home": poss_home,
        "possession_away": poss_away,
        "shots_off_home": shots_off_home,
        "shots_off_away": shots_off_away,
        "blocked_shots_home": blocked_shots_home,
        "blocked_shots_away": blocked_shots_away,
        "goal_attempts_home": goal_attempts_home,
        "goal_attempts_away": goal_attempts_away,
        "attacks_home": attacks_home,
        "attacks_away": attacks_away,
        "dangerous_attacks_home": dangerous_attacks_home,
        "dangerous_attacks_away": dangerous_attacks_away,
        "counter_attacks_home": counter_attacks_home,
        "counter_attacks_away": counter_attacks_away,
        "cross_attacks_home": cross_attacks_home,
        "cross_attacks_away": cross_attacks_away,
        "goalkeeper_saves_home": goalkeeper_saves_home,
        "goalkeeper_saves_away": goalkeeper_saves_away,
        "fouls_home": fouls_home,
        "fouls_away": fouls_away,
        "offsides_home": offsides_home,
        "offsides_away": offsides_away,
        "free_kicks_home": free_kicks_home,
        "free_kicks_away": free_kicks_away,
        "throwins_home": throwins_home,
        "throwins_away": throwins_away,
        "yellow_cards_home": yellow_cards_home,
        "yellow_cards_away": yellow_cards_away,
        "red_cards_home": red_cards_home,
        "red_cards_away": red_cards_away,
        "substitutions_home": substitutions_home,
        "substitutions_away": substitutions_away,
        "odds_over_2_5": over_25,
        "odds_under_2_5": under_25,
        "odds_btts_yes": btts_yes,
        "odds_btts_no": btts_no,
        "datameci": date_str,
        "orameci": time_hhmm,
        "txtechipa1": home_name,
        "txtechipa2": away_name,
        "codechipa1": _team_code(home_id),
        "codechipa2": _team_code(away_id),
        "scor1": score_home,
        "scor2": score_away,
        "sut": _format_pair(shots_home, shots_away),
        "sutt": _format_pair(sot_home, sot_away),
        "cor": _format_pair(corners_home, corners_away),
        "ballp": _format_pair(poss_home, poss_away),
        "ballph": poss_home,
        "ballpa": poss_away,
        "cotao": over_25,
        "cotau": under_25,
        "gg": btts_yes,
        "ng": btts_no,
    }


def _ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS]
    if "fixture_datetime_utc" in out.columns and len(out) > 0:
        out = out.sort_values(
            by=["fixture_datetime_utc", "fixture_id"],
            kind="stable",
        ).reset_index(drop=True)
    return out


def _coverage(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {
            "rows": 0,
            "with_stats": 0,
            "with_ou_odds": 0,
            "with_btts_odds": 0,
        }

    stats_mask = (
        df["shots_home"].notna()
        & df["shots_away"].notna()
        & df["shots_on_target_home"].notna()
        & df["shots_on_target_away"].notna()
        & df["corners_home"].notna()
        & df["corners_away"].notna()
        & df["possession_home"].notna()
        & df["possession_away"].notna()
    )
    ou_mask = df["odds_over_2_5"].notna() & df["odds_under_2_5"].notna()
    btts_mask = df["odds_btts_yes"].notna() & df["odds_btts_no"].notna()

    return {
        "rows": int(len(df)),
        "with_stats": int(stats_mask.sum()),
        "with_ou_odds": int(ou_mask.sum()),
        "with_btts_odds": int(btts_mask.sum()),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _missing_fixture_ids(
    df: pd.DataFrame,
    required_cols: list[str],
    limit: int = 100,
) -> list[int]:
    if df.empty or "fixture_id" not in df.columns:
        return []
    mask = pd.Series(False, index=df.index)
    for col in required_cols:
        if col in df.columns:
            mask = mask | df[col].isna()
    missing_ids = (
        df.loc[mask, "fixture_id"]
        .dropna()
        .astype("Int64")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    return missing_ids[:limit]


def _dedupe_fixtures_by_id(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    no_id: list[dict[str, Any]] = []
    for item in fixtures:
        fixture = item.get("fixture") or {}
        fixture_id = fixture.get("id")
        if fixture_id is None:
            no_id.append(item)
            continue
        try:
            by_id[int(fixture_id)] = item
        except Exception:
            no_id.append(item)
    return list(by_id.values()) + no_id


def _fetch_fixtures(
    client: APIFootballClient,
    league_ids: list[int],
    season: int,
    history_days: int,
    horizon_days: int,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], list[str]]:
    today = dt.date.today()
    history_from = (today - dt.timedelta(days=history_days)).isoformat()
    history_to = today.isoformat()
    upcoming_from = today.isoformat()
    upcoming_to = (today + dt.timedelta(days=horizon_days)).isoformat()

    finished_rows: list[dict[str, Any]] = []
    upcoming_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for league_id in league_ids:
        try:
            finished_cache_key = client.build_cache_key(
                "fixtures",
                "/fixtures",
                {
                    "league": league_id,
                    "season": season,
                    "status": "FT-AET-PEN",
                    "from": history_from,
                    "to": history_to,
                },
            )
            chunk_finished = client.fixtures(
                league_id=league_id,
                season=season,
                status=["FT", "AET", "PEN"],
                date_from=history_from,
                date_to=history_to,
                cache_key=finished_cache_key,
            )
            chunk_finished = [
                row
                for row in chunk_finished
                if str((((row.get("fixture") or {}).get("status") or {}).get("short") or "").strip())
                in FINISHED_STATUSES
            ]
            finished_rows.extend(chunk_finished)
            logger.info(
                "league=%s finished fixtures=%s window=%s..%s",
                league_id,
                len(chunk_finished),
                history_from,
                history_to,
            )
        except Exception as exc:
            errors.append(f"fixtures finished league={league_id}: {exc}")
            logger.warning("failed finished fixtures league=%s (%s)", league_id, exc)

        try:
            upcoming_cache_key = client.build_cache_key(
                "fixtures",
                "/fixtures",
                {
                    "league": league_id,
                    "season": season,
                    "status": "NS",
                    "from": upcoming_from,
                    "to": upcoming_to,
                },
            )
            chunk_upcoming = client.fixtures(
                league_id=league_id,
                season=season,
                status=UPCOMING_STATUS,
                date_from=upcoming_from,
                date_to=upcoming_to,
                cache_key=upcoming_cache_key,
            )
            chunk_upcoming = [
                row
                for row in chunk_upcoming
                if str((((row.get("fixture") or {}).get("status") or {}).get("short") or "").strip())
                == UPCOMING_STATUS
            ]
            upcoming_rows.extend(chunk_upcoming)
            logger.info(
                "league=%s upcoming fixtures=%s window=%s..%s",
                league_id,
                len(chunk_upcoming),
                upcoming_from,
                upcoming_to,
            )
        except Exception as exc:
            errors.append(f"fixtures upcoming league={league_id}: {exc}")
            logger.warning("failed upcoming fixtures league=%s (%s)", league_id, exc)

    finished_rows = _dedupe_fixtures_by_id(finished_rows)
    upcoming_rows = _dedupe_fixtures_by_id(upcoming_rows)

    windows = {
        "history_from": history_from,
        "history_to": history_to,
        "upcoming_from": upcoming_from,
        "upcoming_to": upcoming_to,
    }
    return finished_rows, upcoming_rows, windows, errors


def _hydrate_fixtures(
    client: APIFootballClient,
    finished_fixtures: list[dict[str, Any]],
    upcoming_fixtures: list[dict[str, Any]],
    logger: logging.Logger,
    *,
    fetch_odds: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int | bool], list[str]]:
    history_rows: list[dict[str, Any]] = []
    upcoming_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    upcoming_odds_cache_ttl_seconds = 6 * 60 * 60

    counters: dict[str, int | bool] = {
        "odds_cache_hits": 0,
        "odds_api_fetches": 0,
        "odds_errors": 0,
        "stats_cache_hits": 0,
        "stats_api_fetches": 0,
        "stats_errors": 0,
        "budget_exhausted": False,
    }

    all_rows = finished_fixtures + upcoming_fixtures
    for idx, fixture_row in enumerate(all_rows, start=1):
        fixture = fixture_row.get("fixture") or {}
        teams = fixture_row.get("teams") or {}
        home_id = _to_int((teams.get("home") or {}).get("id"))
        away_id = _to_int((teams.get("away") or {}).get("id"))
        fixture_id = _to_int(fixture.get("id"))
        status = str((fixture.get("status") or {}).get("short") or "").strip()

        stats: dict[str, float | None] = {}
        odds: dict[str, float | None] = {}

        if fetch_odds and fixture_id is not None:
            odds_cache_key = f"odds/{fixture_id}"
            odds_cache_exists = client.cache_path(odds_cache_key).exists()
            try_odds = odds_cache_exists or not bool(counters["budget_exhausted"])
            if try_odds:
                try:
                    odds_force_refresh = False
                    odds_max_age_seconds = None
                    cached_odds_payload = None
                    if status == UPCOMING_STATUS:
                        odds_max_age_seconds = upcoming_odds_cache_ttl_seconds
                        cached_odds_payload = client.read_json_cache(
                            odds_cache_key,
                            max_age_seconds=odds_max_age_seconds,
                        )
                        if cached_odds_payload is not None:
                            cached_odds = _odds_from_payload(
                                (cached_odds_payload.get("response", []) if isinstance(cached_odds_payload, dict) else [])
                            )
                            # Upcoming odds appear late. If a fresh cache is empty, refetch once instead
                            # of letting an early "no odds yet" snapshot stick forever.
                            if not any(v is not None for v in cached_odds.values()):
                                odds_force_refresh = True
                    odds_payload = client.fixture_odds(
                        fixture_id=fixture_id,
                        cache_key=odds_cache_key,
                        max_age_seconds=odds_max_age_seconds,
                        force_refresh=odds_force_refresh,
                    )
                    if odds_cache_exists and not odds_force_refresh and cached_odds_payload is not None:
                        counters["odds_cache_hits"] = int(counters["odds_cache_hits"]) + 1
                    else:
                        counters["odds_api_fetches"] = int(counters["odds_api_fetches"]) + 1
                    odds = _odds_from_payload(odds_payload)
                except Exception as exc:
                    counters["odds_errors"] = int(counters["odds_errors"]) + 1
                    errors.append(f"odds fixture={fixture_id}: {exc}")
                    if "budget exhausted" in str(exc).lower():
                        counters["budget_exhausted"] = True

        if fixture_id is not None and status in FINISHED_STATUSES:
            stats_cache_key = f"stats/{fixture_id}"
            stats_cache_exists = client.cache_path(stats_cache_key).exists()
            try_stats = stats_cache_exists or not bool(counters["budget_exhausted"])
            if try_stats:
                try:
                    stats_payload = client.fixture_statistics(
                        fixture_id=fixture_id,
                        cache_key=stats_cache_key,
                    )
                    if stats_cache_exists:
                        counters["stats_cache_hits"] = int(counters["stats_cache_hits"]) + 1
                    else:
                        counters["stats_api_fetches"] = int(counters["stats_api_fetches"]) + 1
                    stats = _fixture_stats_from_payload(
                        payload=stats_payload,
                        home_id=home_id,
                        away_id=away_id,
                    )
                except Exception as exc:
                    counters["stats_errors"] = int(counters["stats_errors"]) + 1
                    errors.append(f"stats fixture={fixture_id}: {exc}")
                    if "budget exhausted" in str(exc).lower():
                        counters["budget_exhausted"] = True

        row = _normalize_fixture_row(fixture_row, stats, odds)
        if status in FINISHED_STATUSES:
            history_rows.append(row)
        elif status == UPCOMING_STATUS:
            upcoming_rows.append(row)

        if idx % 50 == 0:
            logger.info("hydrated fixtures %s/%s", idx, len(all_rows))

    return history_rows, upcoming_rows, counters, errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync API-Football fixtures/stats/odds")
    parser.add_argument(
        "--data-dir",
        default=_default_data_dir(),
        help="Output directory for CSV + report files.",
    )
    parser.add_argument(
        "--league-ids",
        default=_default_league_ids_csv(),
        help="Comma-separated API-Football league IDs (e.g. 39,78,61).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=_default_season(),
        help="Season start year (e.g. 2025 for 2025/26).",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=_int_from_env_or_cfg(
            "API_FOOTBALL_HISTORY_DAYS",
            "API_FOOTBALL_DEFAULT_HISTORY_DAYS",
            DEFAULT_HISTORY_DAYS,
        ),
        help="How many days back to query finished fixtures.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=_int_from_env_or_cfg(
            "API_FOOTBALL_HORIZON_DAYS",
            "API_FOOTBALL_DEFAULT_HORIZON_DAYS",
            DEFAULT_HORIZON_DAYS,
        ),
        help="How many days forward to query upcoming fixtures.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=_int_from_env_or_cfg(
            "API_FOOTBALL_MAX_REQUESTS",
            "API_FOOTBALL_MAX_REQUESTS_FREE",
            DEFAULT_MAX_REQUESTS,
        ),
        help="Daily request guardrail. Use 0 to disable.",
    )
    parser.add_argument(
        "--rate-per-minute",
        type=int,
        default=_int_from_env_or_cfg(
            "API_FOOTBALL_RATE_PER_MINUTE",
            "API_FOOTBALL_RATE_PER_MIN_FREE",
            DEFAULT_RATE_PER_MINUTE,
        ),
        help="Per-minute request rate limit.",
    )
    parser.add_argument(
        "--fetch-odds",
        dest="fetch_odds",
        action="store_true",
        help="Fetch odds endpoints for fixtures (uses more API calls).",
    )
    parser.add_argument(
        "--no-fetch-odds",
        dest="fetch_odds",
        action="store_false",
        help="Skip odds endpoints; fixtures+stats only.",
    )
    parser.set_defaults(
        fetch_odds=_bool_from_env_or_cfg(
            "API_FOOTBALL_FETCH_ODDS",
            "API_FOOTBALL_FETCH_ODDS_DEFAULT",
            DEFAULT_FETCH_ODDS,
        )
    )
    parser.add_argument(
        "--min-odds-coverage",
        type=float,
        default=DEFAULT_MIN_ODDS_COVERAGE,
        help="Minimum required odds coverage ratio for quality gate (0..1).",
    )
    parser.add_argument(
        "--min-stats-coverage",
        type=float,
        default=DEFAULT_MIN_STATS_COVERAGE,
        help="Minimum required history stats coverage ratio for quality gate (0..1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch/normalize only; do not write output CSV/report files.",
    )
    parser.add_argument(
        "--strict-quality-gate",
        action="store_true",
        help=(
            "Fail with non-zero exit when quality gate does not pass. "
            "Default behavior is soft (write outputs + warn)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("sync_api_football")

    league_ids = _parse_league_ids(args.league_ids)
    max_requests = args.max_requests if args.max_requests and args.max_requests > 0 else None
    data_dir = Path(args.data_dir)
    cache_dir = data_dir / "cache" / "api_football"

    logger.info(
        "starting sync leagues=%s season=%s history_days=%s horizon_days=%s fetch_odds=%s dry_run=%s",
        league_ids,
        args.season,
        args.history_days,
        args.horizon_days,
        bool(args.fetch_odds),
        args.dry_run,
    )

    with APIFootballClient(
        cache_dir=cache_dir,
        max_requests_per_day=max_requests,
        rate_per_minute=args.rate_per_minute,
    ) as client:
        finished_fixtures, upcoming_fixtures, windows, fetch_errors = _fetch_fixtures(
            client=client,
            league_ids=league_ids,
            season=args.season,
            history_days=args.history_days,
            horizon_days=args.horizon_days,
            logger=logger,
        )

        history_rows, upcoming_rows, hydrate_stats, hydrate_errors = _hydrate_fixtures(
            client=client,
            finished_fixtures=finished_fixtures,
            upcoming_fixtures=upcoming_fixtures,
            logger=logger,
            fetch_odds=bool(args.fetch_odds),
        )

        history_df = _ensure_output_columns(pd.DataFrame(history_rows))
        upcoming_df = _ensure_output_columns(pd.DataFrame(upcoming_rows))
        history_cov = _coverage(history_df)
        upcoming_cov = _coverage(upcoming_df)
        stats_coverage = _safe_ratio(history_cov["with_stats"], history_cov["rows"])
        if (not bool(args.fetch_odds)) or upcoming_cov["rows"] <= 0:
            # Odds disabled (or no upcoming fixtures): skip odds gate in sync stage.
            ou_cov = 1.0
            btts_cov = 1.0
            odds_coverage = 1.0
            odds_gate_skipped = True
            odds_gate_skipped_fetch_disabled = not bool(args.fetch_odds)
        else:
            ou_cov = _safe_ratio(upcoming_cov["with_ou_odds"], upcoming_cov["rows"])
            btts_cov = _safe_ratio(upcoming_cov["with_btts_odds"], upcoming_cov["rows"])
            odds_coverage = min(ou_cov, btts_cov)
            odds_gate_skipped = False
            odds_gate_skipped_fetch_disabled = False

        quality_reasons: list[str] = []
        total_rows = int(len(history_df) + len(upcoming_df))
        fetch_error_count = int(len(fetch_errors))
        if stats_coverage < args.min_stats_coverage:
            quality_reasons.append(
                f"stats_coverage {stats_coverage:.1%} < min {args.min_stats_coverage:.1%}"
            )
        if (not odds_gate_skipped) and odds_coverage < args.min_odds_coverage:
            quality_reasons.append(
                f"odds_coverage {odds_coverage:.1%} < min {args.min_odds_coverage:.1%}"
            )
        if fetch_error_count > 0 and total_rows == 0:
            quality_reasons.append(
                "failed to load fixtures from API (check API key/access and request params)"
            )
        quality_gate_passed = len(quality_reasons) == 0

        report = {
            "generated_at_utc": (
                dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            ),
            "inputs": {
                "data_dir": str(data_dir),
                "league_ids": league_ids,
                "season": int(args.season),
                "history_days": int(args.history_days),
                "horizon_days": int(args.horizon_days),
                "fetch_odds": bool(args.fetch_odds),
                "max_requests": max_requests,
                "rate_per_minute": int(args.rate_per_minute),
                "dry_run": bool(args.dry_run),
            },
            "windows": windows,
            "counts": {
                "fixtures_finished": int(len(history_df)),
                "fixtures_upcoming": int(len(upcoming_df)),
                "fixtures_total": int(len(history_df) + len(upcoming_df)),
            },
            "coverage": {
                "history": history_cov,
                "upcoming": upcoming_cov,
            },
            "quality_metrics": {
                "stats_coverage": stats_coverage,
                "odds_coverage": odds_coverage,
                "upcoming_ou_odds_coverage": ou_cov,
                "upcoming_btts_odds_coverage": btts_cov,
                "min_stats_coverage": float(args.min_stats_coverage),
                "min_odds_coverage": float(args.min_odds_coverage),
                "odds_gate_skipped_no_upcoming": odds_gate_skipped,
                "odds_gate_skipped_fetch_disabled": odds_gate_skipped_fetch_disabled,
                "strict_quality_gate": bool(args.strict_quality_gate),
            },
            "quality_gate_passed": quality_gate_passed,
            "quality_gate_reasons": quality_reasons,
            "missing_fixture_ids": {
                "history_missing_stats": _missing_fixture_ids(
                    history_df,
                    [
                        "shots_home",
                        "shots_away",
                        "shots_on_target_home",
                        "shots_on_target_away",
                        "corners_home",
                        "corners_away",
                        "possession_home",
                        "possession_away",
                    ],
                ),
                "history_missing_ou_odds": _missing_fixture_ids(
                    history_df,
                    ["odds_over_2_5", "odds_under_2_5"],
                ),
                "upcoming_missing_ou_odds": _missing_fixture_ids(
                    upcoming_df,
                    ["odds_over_2_5", "odds_under_2_5"],
                ),
                "upcoming_missing_btts_odds": _missing_fixture_ids(
                    upcoming_df,
                    ["odds_btts_yes", "odds_btts_no"],
                ),
            },
            "hydration": hydrate_stats,
            "request_budget": client.budget_status(),
            "errors": (fetch_errors + hydrate_errors)[:200],
        }

        if args.dry_run:
            logger.info("dry-run: no output files written.")
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            history_path = data_dir / "history_fixtures.csv"
            upcoming_path = data_dir / "upcoming_fixtures.csv"
            report_path = data_dir / "fixture_quality_report.json"

            history_df.to_csv(history_path, index=False)
            upcoming_df.to_csv(upcoming_path, index=False)
            report_path.write_text(
                json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            logger.info("wrote %s rows -> %s", len(history_df), history_path)
            logger.info("wrote %s rows -> %s", len(upcoming_df), upcoming_path)
            logger.info("wrote quality report -> %s", report_path)

        summary = {
            "finished_rows": int(len(history_df)),
            "upcoming_rows": int(len(upcoming_df)),
            "odds_api_fetches": int(hydrate_stats["odds_api_fetches"]),
            "stats_api_fetches": int(hydrate_stats["stats_api_fetches"]),
            "budget_exhausted": bool(hydrate_stats["budget_exhausted"]),
            "error_count": int(len(fetch_errors) + len(hydrate_errors)),
            "quality_gate_passed": bool(quality_gate_passed),
            "dry_run": bool(args.dry_run),
        }
        logger.info("summary: %s", json.dumps(summary, ensure_ascii=True))
        if not quality_gate_passed:
            joined_reasons = "; ".join(quality_reasons) if quality_reasons else "unknown reason"
            if total_rows <= 0:
                logger.error("quality gate FAILED with zero fixtures loaded: %s", joined_reasons)
                raise SystemExit(2)
            if args.strict_quality_gate:
                logger.error("quality gate FAILED (strict mode): %s", joined_reasons)
                raise SystemExit(2)
            logger.warning("quality gate FAILED (soft mode): %s", joined_reasons)
            logger.warning("continuing with generated outputs (history/upcoming/report were written)")


if __name__ == "__main__":
    main()
