"""
run_simulation.py
=================
Produces every deliverable:

  1. Runs the baseline two-stage ascent and prints key numbers.
  2. Confirms the ideal-rocket-equation validation.
  3. Generates four plots: altitude vs time, ground track, velocity profile,
     and dynamic pressure (max-Q).
  4. Runs a staging-time trade study: sweeps the pitch-kick angle and reports
     how final velocity and apogee respond -- turning the simulator from a
     single trajectory into a design tool.

Run:
    python scripts/run_simulation.py

Outputs land in results/.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vehicle import demo_two_stage
from src.trajectory import simulate
from src.validation import run_check

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def plot_trajectory(r):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Altitude vs time
    ax = axes[0, 0]
    ax.plot(r["t"], r["h"] / 1000, "C0")
    for e in r["events"]:
        ax.axvline(e["t"], color="grey", ls="--", alpha=0.6)
        ax.text(e["t"], ax.get_ylim()[1]*0.05, f" {e['stage']} sep",
                rotation=90, fontsize=8, va="bottom")
    ax.set_xlabel("time (s)"); ax.set_ylabel("altitude (km)")
    ax.set_title("Altitude vs. time"); ax.grid(alpha=0.3)

    # Ground track (downrange vs altitude)
    ax = axes[0, 1]
    ax.plot(r["x"] / 1000, r["h"] / 1000, "C2")
    ax.set_xlabel("downrange (km)"); ax.set_ylabel("altitude (km)")
    ax.set_title("Ascent trajectory (gravity turn)"); ax.grid(alpha=0.3)

    # Velocity profile
    ax = axes[1, 0]
    ax.plot(r["t"], r["speed"], "C3", label="speed")
    ax.plot(r["t"], r["vh"], "C0", alpha=0.6, label="vertical")
    ax.plot(r["t"], r["vx"], "C1", alpha=0.6, label="horizontal")
    ax.set_xlabel("time (s)"); ax.set_ylabel("velocity (m/s)")
    ax.set_title("Velocity profile"); ax.grid(alpha=0.3); ax.legend()

    # Dynamic pressure
    ax = axes[1, 1]
    ax.plot(r["t"], r["q"] / 1000, "C4")
    ax.axvline(r["t"][np.argmax(r["q"])], color="r", ls="--",
               label=f"max-Q {r['maxq_pa']/1000:.0f} kPa @ {r['maxq_alt_km']:.0f} km")
    ax.set_xlabel("time (s)"); ax.set_ylabel("dynamic pressure q (kPa)")
    ax.set_title("Dynamic pressure (max-Q)"); ax.grid(alpha=0.3); ax.legend()

    fig.suptitle("Two-stage ascent trajectory", fontweight="bold", fontsize=14)
    fig.tight_layout()
    path = os.path.join(RESULTS, "trajectory.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def trade_study():
    """Sweep the pitch-kick angle and record final velocity and apogee.
    A steeper kick turns the vehicle horizontal sooner (more velocity toward
    orbit, lower apogee); a shallower kick climbs higher but stays slower."""
    kicks = np.arange(2, 12.5, 1.0)
    final_v, apogee = [], []
    for k in kicks:
        r = simulate(demo_two_stage(), pitch_kick=k)
        final_v.append(r["max_speed_ms"])
        apogee.append(r["apogee_km"])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(kicks, final_v, "C0-o", label="max speed")
    ax1.set_xlabel("pitch-kick angle (deg)")
    ax1.set_ylabel("max speed (m/s)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.plot(kicks, apogee, "C3-s", label="apogee")
    ax2.set_ylabel("apogee (km)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title("Trade study: pitch-kick angle vs. performance", fontweight="bold")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS, "trade_study.png")
    fig.savefig(path, dpi=130); plt.close(fig)

    # Save the raw trade data too.
    import csv
    with open(os.path.join(RESULTS, "trade_study.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pitch_kick_deg", "max_speed_ms", "apogee_km"])
        for k, v, a in zip(kicks, final_v, apogee):
            w.writerow([f"{k:.1f}", f"{v:.1f}", f"{a:.1f}"])
    return path


def main():
    v = demo_two_stage()

    print("=== Ideal rocket equation validation ===")
    for i, s in enumerate(v.stages):
        ideal, numeric, err = run_check(s, v.mass_above_stage(i))
        print(f"  {s.name}: ideal {ideal:.1f} m/s, numeric {numeric:.1f} m/s, "
              f"rel err {err:.1e}")

    print("\n=== Baseline ascent ===")
    r = simulate(v)
    print(f"  Apogee:    {r['apogee_km']:.1f} km")
    print(f"  Max speed: {r['max_speed_ms']:.0f} m/s")
    print(f"  Max-Q:     {r['maxq_pa']/1000:.1f} kPa at {r['maxq_alt_km']:.1f} km")
    for e in r["events"]:
        print(f"  {e['stage']} separation: t={e['t']:.1f}s  "
              f"alt={e['altitude_km']:.1f}km  v={e['speed_ms']:.0f}m/s")

    p1 = plot_trajectory(r)
    p2 = trade_study()
    print(f"\nPlots written:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
