from dataclasses import dataclass
from enum import Enum
import math

@dataclass
class AircraftProperty:
    name: str
    description: str
    min: float
    max: float
    access: str
    #todo: spaces, clipped, update...

@dataclass
class CLibCatalog:
    position_lat_geod_deg = AircraftProperty("position/lat-geod-deg", "geodetic latitude [deg]", -90, 90, "R")
    position_lat_geod_rad = AircraftProperty("position/lat-geod-rad", "geodetic latitude [rad]", float("-inf"), float("+inf"), "R")
    position_lat_gc_deg = AircraftProperty("position/lat-gc-deg", "geocentric latitude [deg]", float("-inf"), float("+inf"), "RW")
    position_lat_gc_rad = AircraftProperty("position/lat-gc-rad", "geocentric latitude [rad]", float("-inf"), float("+inf"), "RW")
    position_long_gc_deg = AircraftProperty("position/long-gc-deg", "geocentric longitude [deg]", -180, 180, "RW")
    position_long_gc_rad = AircraftProperty("position/long-gc-rad", "geocentric longitude [rad]", float("-inf"), float("+inf"), "RW")
    position_distance_from_start_mag_mt = AircraftProperty(
        "position/distance-from-start-mag-mt", "distance travelled from starting position [m]", float("-inf"), float("+inf"), "R"
    )
    position_distance_from_start_lat_mt = AircraftProperty("position/distance-from-start-lat-mt", "mt", float("-inf"), float("+inf"), "R")
    position_distance_from_start_lon_mt = AircraftProperty("position/distance-from-start-lon-mt", "mt", float("-inf"), float("+inf"), "R")
    position_epa_rad = AircraftProperty("position/epa-rad", "rad", float("-inf"), float("+inf"), "R")
    position_geod_alt_ft = AircraftProperty("position/geod-alt-ft", "ft", float("-inf"), float("+inf"), "R")
    position_radius_to_vehicle_ft = AircraftProperty("position/radius-to-vehicle-ft", "ft", float("-inf"), float("+inf"), "R")
    position_terrain_elevation_asl_ft = AircraftProperty("position/terrain-elevation-asl-ft", "ft", float("-inf"), float("+inf"), "R")
    position_h_agl_ft = AircraftProperty(
        "position/h-agl-ft", "altitude above ground level [ft]", -1400, 85000, "RW"
    )
    position_h_sl_ft = AircraftProperty("position/h-sl-ft", "altitude above mean sea level [ft]", -1400, 85000, "RW")
    
    
    attitude_psi_deg = AircraftProperty("attitude/psi-deg", "heading [deg]", 0, 360, "R")
    attitude_psi_rad = AircraftProperty("attitude/psi-rad", "rad", float("-inf"), float("+inf"), "R")
    attitude_pitch_rad = AircraftProperty("attitude/pitch-rad", "pitch [rad]", -0.5 * math.pi, 0.5 * math.pi, "R")
    attitude_theta_rad = AircraftProperty("attitude/theta-rad", "rad", float("-inf"), float("+inf"), "R")
    attitude_theta_deg = AircraftProperty("attitude/theta-deg", "deg", float("-inf"), float("+inf"), "R")
    attitude_roll_rad = AircraftProperty("attitude/roll-rad", "roll [rad]", -math.pi, math.pi, "R")
    attitude_phi_rad = AircraftProperty("attitude/phi-rad", "rad", float("-inf"), float("+inf"), "R")
    attitude_phi_deg = AircraftProperty("attitude/phi-deg", "deg", float("-inf"), float("+inf"), "R")
    #Psi and true heading are the same
    attitude_heading_true_rad = AircraftProperty("attitude/heading-true-rad", "rad", float("-inf"), float("+inf"), "R")

    # velocities
    velocities_u_fps = AircraftProperty("velocities/u-fps", "body frame x-axis velocity [ft/s]", -2200, 2200,"R")
    velocities_v_fps = AircraftProperty("velocities/v-fps", "body frame y-axis velocity [ft/s]", -2200, 2200, "R")
    velocities_w_fps = AircraftProperty("velocities/w-fps", "body frame z-axis velocity [ft/s]", -2200, 2200, "R")
    velocities_v_north_fps = AircraftProperty("velocities/v-north-fps", "velocity true north [ft/s]", -2200, 2200, "R")
    velocities_v_east_fps = AircraftProperty("velocities/v-east-fps", "velocity east [ft/s]", -2200, 2200, "R")
    velocities_v_down_fps = AircraftProperty("velocities/v-down-fps", "velocity downwards [ft/s]", -2200, 2200, "R")
    velocities_vc_fps = AircraftProperty("velocities/vc-fps", "airspeed [ft/s]]", 0, 4400, "R")
    velocities_vg_fps = AircraftProperty("velocities/vg-fps", "groundspeed [ft/s]", 0, 4400, "R")

    velocities_h_dot_fps = AircraftProperty("velocities/h-dot-fps", "rate of altitude change [ft/s]", float("-inf"), float("+inf"), "R")
    velocities_u_aero_fps = AircraftProperty("velocities/u-aero-fps", "fps", float("-inf"), float("+inf"), "R")
    velocities_v_aero_fps = AircraftProperty("velocities/v-aero-fps", "fps", float("-inf"), float("+inf"), "R")
    velocities_w_aero_fps = AircraftProperty("velocities/w-aero-fps", "fps", float("-inf"), float("+inf"), "R")
    velocities_mach = AircraftProperty("velocities/mach", "", float("-inf"), float("+inf"), "R")
    velocities_machU = AircraftProperty("velocities/machU", "", float("-inf"), float("+inf"), "R")
    velocities_eci_velocity_mag_fps = AircraftProperty("velocities/eci-velocity-mag-fps", "fps", float("-inf"), float("+inf"), "R")
    velocities_vc_kts = AircraftProperty("velocities/vc-kts", "kts", float("-inf"), float("+inf"), "R")
    velocities_ve_fps = AircraftProperty("velocities/ve-fps", "fps", float("-inf"), float("+inf"), "R")
    velocities_ve_kts = AircraftProperty("velocities/ve-kts", "kts", float("-inf"), float("+inf"), "R")
    velocities_vg_fps = AircraftProperty("velocities/vg-fps", "fps", float("-inf"), float("+inf"), "R")
    velocities_vt_fps = AircraftProperty("velocities/vt-fps", "fps", float("-inf"), float("+inf"), "R")
    velocities_p_rad_sec = AircraftProperty("velocities/p-rad_sec", "roll rate [rad/s]", -2 * math.pi, 2 * math.pi, "R")
    velocities_q_rad_sec = AircraftProperty("velocities/q-rad_sec", "pitch rate [rad/s]", -2 * math.pi, 2 * math.pi, "R")
    velocities_r_rad_sec = AircraftProperty("velocities/r-rad_sec", "yaw rate [rad/s]", -2 * math.pi, 2 * math.pi, "R")
    velocities_p_aero_rad_sec = AircraftProperty("velocities/p-aero-rad_sec", "rad/sec", float("-inf"), float("+inf"), "R")
    velocities_q_aero_rad_sec = AircraftProperty("velocities/q-aero-rad_sec", "rad/sec", float("-inf"), float("+inf"), "R")
    velocities_r_aero_rad_sec = AircraftProperty("velocities/r-aero-rad_sec", "rad/sec", float("-inf"), float("+inf"), "R")
    velocities_phidot_rad_sec = AircraftProperty("velocities/phidot-rad_sec", "rad/s", -2 * math.pi, 2 * math.pi, "R")
    velocities_thetadot_rad_sec = AircraftProperty("velocities/thetadot-rad_sec", "rad/s", -2 * math.pi, 2 * math.pi, "R")
    velocities_psidot_rad_sec = AircraftProperty("velocities/psidot-rad_sec", "rad/sec", -2 * math.pi, 2 * math.pi, "R")

    #atmosphere
    atmosphere_crosswind_fps = AircraftProperty("atmosphere/crosswind-fps", "fps", -100, 100, "R")
    atmosphere_headwind_fps = AircraftProperty("atmosphere/headwind-fps", "fps", -100, 100, "R")
    atmosphere_rho_slugs_ft3 = AircraftProperty("atmosphere/rho-slugs_ft3", "slugs/ft^3", float("-inf"), float("+inf"), "R")
    atmosphere_a_fps = AircraftProperty("atmosphere/a-fps", "fps", float("-inf"), float("+inf"), "R")

   # Acceleration

    accelerations_pdot_rad_sec2 = AircraftProperty(
        "accelerations/pdot-rad_sec2", "rad/sÂ²", -(8 / 180) * math.pi, (8 / 180) * math.pi, "R")
    accelerations_qdot_rad_sec2 = AircraftProperty(
        "accelerations/qdot-rad_sec2", "rad/sÂ²", -(8 / 180) * math.pi, (8 / 180) * math.pi, "R")
    accelerations_rdot_rad_sec2 = AircraftProperty(
        "accelerations/rdot-rad_sec2", "rad/sÂ²", -(8 / 180) * math.pi, (8 / 180) * math.pi, "R")
    accelerations_vdot_ft_sec2 = AircraftProperty("accelerations/vdot-ft_sec2", "ft/s²", -4.0, 4.0, "R")
    accelerations_wdot_ft_sec2 = AircraftProperty("accelerations/wdot-ft_sec2", "ft/s²", -4.0, 4.0, "R")
    accelerations_udot_ft_sec2 = AircraftProperty("accelerations/udot-ft_sec2", "ft/s²", -4.0, 4.0, "R")
    accelerations_a_pilot_x_ft_sec2 = AircraftProperty("accelerations/a-pilot-x-ft_sec2", "pilot body x-axis acceleration [ft/s²]", 
                                                       float("-inf"), float("+inf"), "R")
    accelerations_a_pilot_y_ft_sec2 = AircraftProperty(
        "accelerations/a-pilot-y-ft_sec2", "pilot body y-axis acceleration [ft/sÂ²]", float("-inf"), float("+inf"), "R")
    accelerations_a_pilot_z_ft_sec2 = AircraftProperty(
        "accelerations/a-pilot-z-ft_sec2", "pilot body z-axis acceleration [ft/sÂ²]", float("-inf"), float("+inf"), "R")
    accelerations_n_pilot_x_norm = AircraftProperty(
        "accelerations/n-pilot-x-norm", "pilot body x-axis acceleration, normalised", float("-inf"), float("+inf"), "R")
    accelerations_n_pilot_y_norm = AircraftProperty(
        "accelerations/n-pilot-y-norm", "pilot body y-axis acceleration, normalised", float("-inf"), float("+inf"), "R")
    accelerations_n_pilot_z_norm = AircraftProperty(
        "accelerations/n-pilot-z-norm", "pilot body z-axis acceleration, normalised", float("-inf"), float("+inf"), "R")

    # aero

    aero_alpha_deg = AircraftProperty("aero/alpha-deg", "deg", -180, +180, "R")
    aero_beta_deg = AircraftProperty("aero/beta-deg", "sideslip [deg]", -180, +180, "R")

    #fcs
    fcs_throttle_cmd_norm = AircraftProperty("fcs/throttle-cmd-norm", "throttle command, normalised", 0, 1.0, "RW")
    fcs_aileron_cmd_norm = AircraftProperty("fcs/aileron-cmd-norm", "aileron commanded position, normalised", -1.0, 1.0, "RW")
    fcs_elevator_cmd_norm = AircraftProperty("fcs/elevator-cmd-norm", "elevator commanded position, normalised", -1.0, 1.0, "RW")
    fcs_rudder_cmd_norm = AircraftProperty("fcs/rudder-cmd-norm", "rudder commanded position, normalised", -1.0, 1.0, "RW")

    # initial conditions
    ic_h_sl_ft = AircraftProperty("ic/h-sl-ft", "initial altitude MSL [ft]", position_h_sl_ft.min, position_h_sl_ft.max, "RW")
    ic_h_agl_ft = AircraftProperty("ic/h-agl-ft", "", position_h_sl_ft.min, position_h_sl_ft.max, "RW")
    ic_geod_alt_ft = AircraftProperty("ic/geod-alt-ft", "ft", float("-inf"), float("+inf"), "RW")
    ic_sea_level_radius_ft = AircraftProperty("ic/sea-level-radius-ft", "ft", float("-inf"), float("+inf"), "RW")
    ic_terrain_elevation_ft = AircraftProperty("ic/terrain-elevation-ft", "ft", float("-inf"), float("+inf"), "RW")
    ic_long_gc_deg = AircraftProperty("ic/long-gc-deg", "initial geocentric longitude [deg]", float("-inf"), float("+inf"), "RW")
    ic_long_gc_rad = AircraftProperty("ic/long-gc-rad", "rad", float("-inf"), float("+inf"), "RW")
    ic_lat_gc_deg = AircraftProperty("ic/lat-gc-deg", "deg", float("-inf"), float("+inf"), "RW")
    ic_lat_gc_rad = AircraftProperty("ic/lat-gc-rad", "rad", float("-inf"), float("+inf"), "RW")
    ic_lat_geod_deg = AircraftProperty("ic/lat-geod-deg", "initial geodetic latitude [deg]", float("-inf"), float("+inf"), "RW")
    ic_lat_geod_rad = AircraftProperty("ic/lat-geod-rad", "rad", float("-inf"), float("+inf"), "RW")
    ic_psi_true_deg = AircraftProperty("ic/psi-true-deg", "initial (true) heading [deg]", attitude_psi_deg.min, attitude_psi_deg.max, "RW")
    ic_psi_true_rad = AircraftProperty("ic/psi-true-rad", "rad", float("-inf"), float("+inf"), "RW")
    ic_theta_deg = AircraftProperty("ic/theta-deg", "deg", float("-inf"), float("+inf"), "RW")
    ic_theta_rad = AircraftProperty("ic/theta-rad", "rad", float("-inf"), float("+inf"), "RW")
    ic_phi_deg = AircraftProperty("ic/phi-deg", "deg", float("-inf"), float("+inf"), "RW")
    ic_phi_rad = AircraftProperty("ic/phi-rad", "rad", float("-inf"), float("+inf"), "RW")
    ic_alpha_deg = AircraftProperty("ic/alpha-deg", "deg", float("-inf"), float("+inf"), "RW")
    ic_alpha_rad = AircraftProperty("ic/alpha-rad", "rad", float("-inf"), float("+inf"), "RW")
    ic_beta_deg = AircraftProperty("ic/beta-deg", "deg", float("-inf"), float("+inf"), "RW")
    ic_beta_rad = AircraftProperty("ic/beta-rad", "rad", float("-inf"), float("+inf"), "RW")
    ic_gamma_deg = AircraftProperty("ic/gamma-deg", "deg", float("-inf"), float("+inf"), "RW")
    ic_gamma_rad = AircraftProperty("ic/gamma-rad", "rad", float("-inf"), float("+inf"), "RW")
    ic_mach = AircraftProperty("ic/mach", "", float("-inf"), float("+inf"), "RW")
    ic_u_fps = AircraftProperty("ic/u-fps", "body frame x-axis velocity; positive forward [ft/s]", float("-inf"), float("+inf"), "RW")
    ic_v_fps = AircraftProperty("ic/v-fps", "body frame y-axis velocity; positive right [ft/s]", float("-inf"), float("+inf"), "RW")
    ic_w_fps = AircraftProperty("ic/w-fps", "body frame z-axis velocity; positive down [ft/s]", float("-inf"), float("+inf"), "RW")
    ic_p_rad_sec = AircraftProperty("ic/p-rad_sec", "roll rate [rad/s]", float("-inf"), float("+inf"), "RW")
    ic_q_rad_sec = AircraftProperty("ic/q-rad_sec", "pitch rate [rad/s]", float("-inf"), float("+inf"), "RW")
    ic_r_rad_sec = AircraftProperty("ic/r-rad_sec", "yaw rate [rad/s]", float("-inf"), float("+inf"), "RW")
    ic_roc_fpm = AircraftProperty("ic/roc-fpm", "initial rate of climb [ft/min]", float("-inf"), float("+inf"), "RW")
    ic_roc_fps = AircraftProperty("ic/roc-fps", "fps", float("-inf"), float("+inf"), "RW")
    ic_vc_kts = AircraftProperty("ic/vc-kts", "kts", float("-inf"), float("+inf"), "RW")
    ic_vd_fps = AircraftProperty("ic/vd-fps", "fps", float("-inf"), float("+inf"), "RW")
    ic_ve_fps = AircraftProperty("ic/ve-fps", "fps", float("-inf"), float("+inf"), "RW")
    ic_ve_kts = AircraftProperty("ic/ve-kts", "kts", float("-inf"), float("+inf"), "RW")
    ic_vg_fps = AircraftProperty("ic/vg-fps", "fps", float("-inf"), float("+inf"), "RW")
    ic_vg_kts = AircraftProperty("ic/vg-kts", "kts", float("-inf"), float("+inf"), "RW")
    ic_vn_fps = AircraftProperty("ic/vn-fps", "fps", float("-inf"), float("+inf"), "RW")
    ic_vt_fps = AircraftProperty("ic/vt-fps", "fps", float("-inf"), float("+inf"), "RW")
    ic_vt_kts = AircraftProperty("ic/vt-kts", "kts", float("-inf"), float("+inf"), "RW")
    ic_vw_bx_fps = AircraftProperty("ic/vw-bx-fps", "fps", float("-inf"), float("+inf"), "RW")
    ic_vw_by_fps = AircraftProperty("ic/vw-by-fps", "fps", float("-inf"), float("+inf"), "RW")
    ic_vw_bz_fps = AircraftProperty("ic/vw-bz-fps", "fps", float("-inf"), float("+inf"), "RW")
    ic_vw_dir_deg = AircraftProperty("ic/vw-dir-deg", "deg", float("-inf"), float("+inf"), "RW")
    ic_vw_down_fps = AircraftProperty("ic/vw-down-fps", "fps", float("-inf"), float("+inf"), "RW")
    ic_vw_east_fps = AircraftProperty("ic/vw-east-fps", "fps", float("-inf"), float("+inf"), "RW")
    ic_vw_mag_fps = AircraftProperty("ic/vw-mag-fps", "fps", float("-inf"), float("+inf"), "RW")
    ic_vw_north_fps = AircraftProperty("ic/vw-north-fps", "fps", float("-inf"), float("+inf"), "RW")
    ic_targetNlf = AircraftProperty("ic/targetNlf", "", float("-inf"), float("+inf"), "RW")
