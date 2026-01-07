from dataclasses import dataclass
import numpy as np


@dataclass
class KFConfig1D:
    # Process noise: how much we "trust" the motion model
    accel_noise_std: float = 1.5  # m/s^2 (includes IMU noise + unmodeled accel)

    # Measurement noise: how much we "trust" sensors
    baro_alt_std: float = 2.0     # m
    gps_alt_std: float = 2.5      # m
    gps_vel_std: float = 0.8      # m/s


class KalmanFilter1D:
    """
    State x = [z, v]^T
    Input u = a (measured accel)
    Measurements:
      - baro: z
      - gps: z and v (when available)
    """
    def __init__(self, z0=0.0, v0=0.0, cfg: KFConfig1D | None = None):
        self.cfg = cfg or KFConfig1D()

        self.x = np.array([[z0], [v0]], dtype=float)  # 2x1
        self.P = np.diag([25.0, 25.0])                # initial covariance

    def predict(self, a_meas: float, dt: float):
        # State transition
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=float)

        # Control/input model (accel)
        B = np.array([[0.5 * dt * dt],
                      [dt]], dtype=float)

        # Process noise (from accel uncertainty)
        q = self.cfg.accel_noise_std ** 2
        Q = q * np.array([[0.25 * dt**4, 0.5 * dt**3],
                          [0.5 * dt**3, dt**2]], dtype=float)

        # Predict
        self.x = F @ self.x + B * float(a_meas)
        self.P = F @ self.P @ F.T + Q

    def update_baro(self, z_baro: float):
        # Measurement: z
        H = np.array([[1.0, 0.0]], dtype=float)   # 1x2
        R = np.array([[self.cfg.baro_alt_std ** 2]], dtype=float)

        y = np.array([[z_baro]], dtype=float) - (H @ self.x)  # innovation
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

    def update_gps(self, z_gps: float, v_gps: float):
        # Measurement: [z, v]
        H = np.array([[1.0, 0.0],
                      [0.0, 1.0]], dtype=float)  # 2x2
        R = np.diag([self.cfg.gps_alt_std ** 2,
                     self.cfg.gps_vel_std ** 2])

        y = np.array([[z_gps], [v_gps]], dtype=float) - (H @ self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

    @property
    def z(self) -> float:
        return float(self.x[0, 0])

    @property
    def v(self) -> float:
        return float(self.x[1, 0])
