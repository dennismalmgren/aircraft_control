from dataclasses import dataclass

@dataclass(frozen=True)
class TaskRange:
    delta_altitude_max: float
    delta_altitude_min: float
    delta_speed_max: float
    delta_speed_min: float
    delta_heading_max: float
    delta_heading_min: float
