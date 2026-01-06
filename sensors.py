from dataclasses import dataclass
import numpy as np


@dataclass
class IMUAccelModel:
    bias: float = 0.15        # m/s^2 constant bias
    noise_std: float = 0.25  # m/s^2 white noise
    seed: int = 1

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def measure(self, a_true: float) -> float:
        return a_true + self.bias + self.rng.normal(0.0, self.noise_std)


@dataclass
class BaroAltModel:
    bias: float = 1.5         # meters
    noise_std: float = 0.8    # meters
    drift_per_s: float = 0.02 # meters per second
    seed: int = 2

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.drift = 0.0

    def measure(self, z_true: float, dt: float) -> float:
        self.drift += self.drift_per_s * dt
        return z_true + self.bias + self.drift + self.rng.normal(0.0, self.noise_std)


@dataclass
class GPSModel:
    rate_hz: float = 5.0
    alt_noise_std: float = 1.8
    vel_noise_std: float = 0.6
    dropout_prob: float = 0.03
    seed: int = 3

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.t_next = 0.0

    def measure(self, t: float, z_true: float, v_true: float):
        # Only output at GPS rate
        if t + 1e-9 < self.t_next:
            return False, None, None

        self.t_next = t + 1.0 / self.rate_hz

        # Random dropout
        if self.rng.random() < self.dropout_prob:
            return False, None, None

        z = z_true + self.rng.normal(0.0, self.alt_noise_std)
        v = v_true + self.rng.normal(0.0, self.vel_noise_std)
        return True, z, v
        