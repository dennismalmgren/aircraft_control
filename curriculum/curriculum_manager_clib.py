from clib_interface.aircraft_simulator import AircraftCLibInitialConditions
import random

class CurriculumManagerCLib:
    def __init__(self, 
                 min_X: float,
                 max_X: float,
                 min_Y: float,
                 max_Y: float,
                 min_Z: float, 
                 max_Z: float, 
                 min_speed: float,
                 max_speed: float,
                 min_heading: float,
                 max_heading: float):
        self.min_X = min_X
        self.max_X = max_X
        self.min_Y = min_Y
        self.max_Y = max_Y
        self.min_Z = min_Z
        self.max_Z = max_Z
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_heading = min_heading
        self.max_heading = max_heading        

    def get_initial_conditions(self) -> AircraftCLibInitialConditions:
        ic_X = random.uniform(self.min_X, self.min_X)
        ic_Y = random.uniform(self.min_Y, self.min_Y)
        ic_Z = random.uniform(self.min_Z, self.max_Z) 
        ic_speed = random.uniform(self.min_speed, self.max_speed)
        ic_heading = random.uniform(self.min_heading, self.max_heading)

        initial_conditions = AircraftCLibInitialConditions(
            x_rt90 = ic_X,
            y_rt90 = ic_Y,
            z_rt90 = ic_Z,
            u_mps = ic_speed,
            psi_deg = ic_heading
        )

        return initial_conditions
