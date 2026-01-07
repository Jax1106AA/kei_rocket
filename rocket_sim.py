import pandas as pd
from kalman_1d import KalmanFilter1D
from sensors import IMUAccelModel, BaroAltModel, GPSModel


def run_sim(dt=0.01, t_max=30.0):
    # 1D vertical "truth" model (Phase A simple)
    g = 9.80665
    rho = 1.225
    cd = 0.55
    area = 0.0095

    m0 = 12.0
    mdot = 0.35
    thrust = 450.0
    burn_time = 3.2

    # Sensor models
    imu = IMUAccelModel()
    baro = BaroAltModel()
    gps = GPSModel()
    kf = KalmanFilter1D(z0=0.0, v0=0.0)

    t = 0.0
    z = 0.0
    v = 0.0

    phase = "IDLE"
    rows = []

    while t < t_max:
        # Mass + thrust schedule
        m = max(m0 - mdot * min(t, burn_time), 0.1)
        T = thrust if t <= burn_time else 0.0

        # Simple flight state machine (truth-based for now)
        if phase == "IDLE" and (T > 1.0 or v > 1.0):
            phase = "POWERED_ASCENT"
        elif phase == "POWERED_ASCENT" and T <= 1.0:
            phase = "COAST"
        elif phase == "COAST" and v < -0.5:
            phase = "DESCENT"
        elif phase == "DESCENT" and z <= 0.0 and abs(v) < 2.0:
            phase = "LANDED"

        # Drag (signed)
        D = 0.5 * rho * cd * area * v * abs(v)

        # Net acceleration (up is +)
        a = (T - D - m * g) / m

        # --- Sensors (measured) ---
        imu_accel = imu.measure(a_true=a)
        baro_alt = baro.measure(z_true=z, dt=dt)
        gps_fix, gps_alt, gps_v = gps.measure(t=t, z_true=z, v_true=v)

        # --- Kalman filter ---
        kf.predict(a_meas=imu_accel, dt=dt)
        kf.update_baro(z_baro=baro_alt)
        if gps_fix:
            kf.update_gps(z_gps=gps_alt, v_gps=gps_v)

        # Integrate (semi-implicit Euler)
        v = v + a * dt
        z = z + v * dt

        # Ground clamp
        if z < 0.0:
            z = 0.0
            v = 0.0

        rows.append({
            # time
            "t": t,

            # truth
            "truth_z": z,
            "truth_v": v,
            "truth_a": a,
            "m": m,
            "thrust": T,
            "phase": phase,

            # sensors
            "imu_accel": imu_accel,
            "baro_alt": baro_alt,
            "gps_fix": gps_fix,
            "gps_alt": gps_alt,
            "gps_v": gps_v,
            "est_z": kf.z,
            "est_v": kf.v,
        })

        t += dt
        if phase == "LANDED" and t > 1.0:
            break

    return pd.DataFrame(rows)
