from clib_interface.aircraft_simulator import AircraftCLibInitialConditions
import random

class CurriculumManagerCLib:
    def __init__(self, 
                 min_long_gc_deg: float,
                 max_long_gc_deg: float,
                 min_lat_geod_deg: float,
                 max_lat_geod_deg: float,
                 min_altitude: float, 
                 max_altitude: float, 
                 min_speed: float,
                 max_speed: float,
                 min_heading: float,
                 max_heading: float):
        self.min_long_gc_deg = min_long_gc_deg
        self.max_long_gc_deg = max_long_gc_deg
        self.min_lat_geod_deg = min_lat_geod_deg
        self.max_lat_geod_deg = max_lat_geod_deg
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_heading = min_heading
        self.max_heading = max_heading        
        self.m_to_ft = 1.0 / 0.3048

    def get_initial_conditions(self) -> AircraftCLibInitialConditions:
        ic_long_gc_deg = random.uniform(self.min_long_gc_deg, self.max_long_gc_deg)
        ic_lat_geod_deg = random.uniform(self.min_lat_geod_deg, self.max_lat_geod_deg)
        ic_altitude = random.uniform(self.min_altitude, self.max_altitude) * self.m_to_ft
        ic_speed = random.uniform(self.min_speed, self.max_speed) * self.m_to_ft
        ic_heading = random.uniform(self.min_heading, self.max_heading)

        initial_conditions = AircraftCLibInitialConditions(
            long_gc_deg = ic_long_gc_deg,
            lat_geod_deg = ic_lat_geod_deg,
            h_sl_ft = ic_altitude,
            u_fps = ic_speed,
            psi_true_deg = ic_heading
        )

        return initial_conditions
    #     class AircraftJSBSimInitialConditions:
    # long_gc_deg: float = 120.0 # geocentric longitude [deg]
    # lat_geod_deg: float = 60.0  # geodetic latitude  [deg]
    # h_sl_ft: float = 6000      # altitude above mean sea level [ft]
    # psi_true_deg: float = 0.0   # initial (true) heading [deg] (0, 360)
    # u_fps: float = 800.0        # body frame x-axis velocity [ft/s]  (-2200, 2200)
    # v_fps: float = 0.0          # body frame y-axis velocity [ft/s]  (-2200, 2200)
    # w_fps: float = 0.0          # body frame z-axis velocity [ft/s]  (-2200, 2200)
    # p_rad_sec: float = 0.0      # roll rate  [rad/s]  (-2 * pi, 2 * pi)
    # q_rad_sec: float = 0.0      # pitch rate [rad/s]  (-2 * pi, 2 * pi)
    # r_rad_sec: float = 0.0      # yaw rate   [rad/s]  (-2 * pi, 2 * pi)
    # roc_fpm: float = 0.0        # initial rate of climb [ft/min]
    # terrain_elevation_ft: float = 0.0     # terrain elevation [ft] 
    