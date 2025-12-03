import math
from dataclasses import dataclass
from typing import Tuple

Vector2 = Tuple[float, float]

@dataclass
class Wing:
    area: float              # S [m^2]
    cl_alpha: float          # dCL/d(alpha) [1/rad]
    cd0: float               # parasitic drag coefficient
    aspect_ratio: float = 6  # AR, used for induced drag
    e: float = 0.8           # Oswald efficiency factor
    rho: float = 1.225       # air density [kg/m^3]

def compute_lift_drag(
    vel: Vector2,
    wing_angle: float,   # [rad], wing chord angle in world frame
    wing: Wing
) -> Tuple[Vector2, Vector2, Vector2]:
    """
    vel: (vx, vy) in world frame [m/s]
    wing_angle: wing chord orientation angle [rad] in world frame
    Returns: (lift_vec, drag_vec, total_force_vec)
    """
    vx, vy = vel
    V = math.hypot(vx, vy)
    if V < 1e-6:
        # No meaningful airflow -> no aerodynamic forces
        return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)

    # Flow direction angle in world frame
    flow_angle = math.atan2(vy, vx)  # direction of velocity

    # Angle of attack alpha = wing angle - flow angle
    alpha = wing_angle - flow_angle  # [rad]

    # Lift and drag coefficients
    cl = wing.cl_alpha * alpha
    k = 1.0 / (math.pi * wing.aspect_ratio * wing.e)
    cd = wing.cd0 + k * cl * cl

    # Dynamic pressure
    q = 0.5 * wing.rho * V * V

    # Force magnitudes
    L = q * wing.area * cl
    D = q * wing.area * cd

    # Unit vector along velocity (drag direction is opposite this)
    v_hat_x = vx / V
    v_hat_y = vy / V

    # Drag vector (opposite velocity)
    drag_vec = (-D * v_hat_x, -D * v_hat_y)

    # Lift direction: 90° CCW from velocity
    lift_hat_x = -v_hat_y
    lift_hat_y =  v_hat_x
    lift_vec = (L * lift_hat_x, L * lift_hat_y)

    total_force = (lift_vec[0] + drag_vec[0],
                   lift_vec[1] + drag_vec[1])

    return lift_vec, drag_vec, total_force

if __name__ == "__main__":
    wing = Wing(
        area=1.0,          # m^2
        cl_alpha=2 * math.pi,  # ~2π per rad (thin airfoil theory)
        cd0=0.02
    )

    vel = (30.0, 0.0)          # 30 m/s along +x
    wing_angle = math.radians(5.0)  # wing chord at +5 degrees

    L, D, F = compute_lift_drag(vel, wing_angle, wing)
    print("Lift:", L)
    print("Drag:", D)
    print("Total force:", F)
