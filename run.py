from rocket_sim import run_sim
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("Running simulation...", flush=True)
df = run_sim()
print("Done. Last 5 rows:", flush=True)
print(df.tail(), flush=True)

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

# Save CSV log
csv_path = out_dir / "flight_log.csv"
df.to_csv(csv_path, index=False)
print(f"Saved {csv_path}", flush=True)

# Plot: altitude truth vs baro vs gps (gps is sparse)
plt.figure()
plt.plot(df["t"], df["truth_z"], label="truth_z")
plt.plot(df["t"], df["baro_alt"], label="baro_alt")

gps_df = df[df["gps_fix"] == True]
plt.plot(gps_df["t"], gps_df["gps_alt"], marker="o", linestyle="None", label="gps_alt")

plt.xlabel("t (s)")
plt.ylabel("altitude (m)")
plt.title("Altitude: Truth vs Sensors")
plt.legend()
plt.savefig(out_dir / "altitude_truth_vs_sensors.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot: velocity truth vs gps (gps is sparse)
plt.figure()
plt.plot(df["t"], df["truth_v"], label="truth_v")

plt.plot(gps_df["t"], gps_df["gps_v"], marker="o", linestyle="None", label="gps_v")

plt.xlabel("t (s)")
plt.ylabel("vertical velocity (m/s)")
plt.title("Velocity: Truth vs GPS")
plt.legend()
plt.savefig(out_dir / "velocity_truth_vs_gps.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot: accel truth vs IMU accel
plt.figure()
plt.plot(df["t"], df["truth_a"], label="truth_a")
plt.plot(df["t"], df["imu_accel"], label="imu_accel")
plt.xlabel("t (s)")
plt.ylabel("accel (m/s^2)")
plt.title("Acceleration: Truth vs IMU")
plt.legend()
plt.savefig(out_dir / "accel_truth_vs_imu.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved plots:", flush=True)
print("  outputs/altitude_truth_vs_sensors.png", flush=True)
print("  outputs/velocity_truth_vs_gps.png", flush=True)
print("  outputs/accel_truth_vs_imu.png", flush=True)
print("Program exiting.", flush=True)

plt.figure()
plt.plot(df["t"], df["truth_z"], label="truth_z")
plt.plot(df["t"], df["baro_alt"], label="baro_alt")
plt.plot(df["t"], df["est_z"], label="est_z")

gps_df = df[df["gps_fix"] == True]
plt.plot(gps_df["t"], gps_df["gps_alt"], marker="o", linestyle="None", label="gps_alt")

plt.xlabel("t (s)")
plt.ylabel("altitude (m)")
plt.title("Altitude: Truth vs Sensors vs KF Estimate")
plt.legend()
plt.savefig(out_dir / "altitude_with_kf.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(df["t"], df["truth_v"], label="truth_v")
plt.plot(df["t"], df["est_v"], label="est_v")
plt.plot(gps_df["t"], gps_df["gps_v"], marker="o", linestyle="None", label="gps_v")

plt.xlabel("t (s)")
plt.ylabel("vertical velocity (m/s)")
plt.title("Velocity: Truth vs GPS vs KF Estimate")
plt.legend()
plt.savefig(out_dir / "velocity_with_kf.png", dpi=150, bbox_inches="tight")
plt.close()


