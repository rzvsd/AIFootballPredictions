import math
from typing import Optional


# Standard pitch dimensions (meters) for conversion from Understat's [0,1] coords
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_Y_CENTER = 0.5  # Understat coords center on Y=0.5
HALF_GOAL_WIDTH_M = 7.32 / 2.0  # 3.66m


def xy_understat_to_meters(x: float, y: float) -> tuple[float, float]:
    """Convert Understat normalized coords (X in [0,1] to the right goal, Y in [0,1])
    to meters relative to the shooting position with origin at the goal line.

    We assume the attacking direction is always toward X=1 (Understat convention).
    dx is the horizontal distance to the goal line; dy is vertical offset from goal center.
    """
    dx = (1.0 - float(x)) * PITCH_LENGTH_M
    dy = (float(y) - GOAL_Y_CENTER) * PITCH_WIDTH_M
    return dx, dy


def shot_distance_m(x: float, y: float) -> float:
    """Euclidean distance from shot location to goal center in meters."""
    dx, dy = xy_understat_to_meters(x, y)
    return math.hypot(dx, dy)


def shot_open_angle_rad(x: float, y: float) -> float:
    """Opening angle (radians) to the two goalposts from the shot location.

    Uses geometry with the two posts at +/- HALF_GOAL_WIDTH_M around the goal center.
    """
    dx, dy = xy_understat_to_meters(x, y)
    # Angle to top and bottom posts
    top = math.atan2(HALF_GOAL_WIDTH_M - dy, dx)
    bot = math.atan2(-HALF_GOAL_WIDTH_M - dy, dx)
    ang = abs(top - bot)
    # Numerical safety
    if ang < 0:
        ang = 0.0
    if ang > math.pi:
        ang = math.pi
    return ang


def shot_open_angle_deg(x: float, y: float) -> float:
    return math.degrees(shot_open_angle_rad(x, y))


def is_header(shot_type: Optional[str], body_part: Optional[str]) -> int:
    """Heuristic: mark header based on shot_type or body_part labels (case-insensitive)."""
    s = (shot_type or "").lower()
    b = (body_part or "").lower()
    return 1 if ("head" in s or "header" in s or b == "head") else 0

