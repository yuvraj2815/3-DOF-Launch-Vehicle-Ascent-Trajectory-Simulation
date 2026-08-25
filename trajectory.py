"""
trajectory.py
=============
The core simulator: integrates a 3-DOF (planar) point-mass ascent through the
atmosphere, stage by stage, using a gravity-turn steering law.

State vector (planar, flat-Earth with altitude-dependent gravity):
    [x, h, vx, vh, m]
      x  : downrange distance (m)
      h  : altitude (m)
      vx : horizontal velocity (m/s)
      vh : vertical velocity (m/s)
      m  : current total mass (kg)

Forces on the vehicle:
    Thrust : along the velocity vector (gravity-turn), corrected for ambient
             back-pressure on the nozzle exit.
    Gravity: toward the ground, weakening with altitude (inverse-square).
    Drag   : opposite the velocity vector, from the standard atmosphere.

Steering:
    A gravity turn. The rocket rises vertically for a short kick, is given a
    small initial pitch-over, and thereafter thrust simply follows the velocity
    vector -- gravity naturally bends the trajectory toward the horizontal. This
    is the classic fuel-efficient ascent used by real launchers.

Staging:
    We integrate one stage at a time with solve_ivp, stopping at propellant
    burnout via a terminal event. At each separation we drop the spent stage's
    dry mass and restart integration with the next stage's thrust/Isp. This
    cleanly handles the mass discontinuity that a single continuous integration
    cannot.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .atmosphere import atmosphere, G0
from .vehicle import Vehicle

R_EARTH = 6.371e6  # mean Earth radius, m
MU = 3.986e14      # Earth gravitational parameter, m^3/s^2


def gravity(h: float) -> float:
    """Gravitational acceleration at altitude h (inverse-square)."""
    return MU / (R_EARTH + h) ** 2


def _derivs(t, y, thrust, mdot, exit_area, isp, vehicle, kick_time, pitch_kick):
    x, h, vx, vh, m = y

    speed = np.hypot(vx, vh)
    g = gravity(h)
    rho, P_amb, _ = atmosphere(h)

    # --- thrust direction (gravity-turn steering) ---
    if h < 1e-3 and speed < 1e-3:
        # On the pad: straight up.
        tx, th = 0.0, 1.0
    elif t < kick_time:
        # Vertical rise phase.
        tx, th = 0.0, 1.0
    elif speed < 1e-6:
        tx, th = 0.0, 1.0
    else:
        # Follow velocity, but apply a one-time pitch kick just after the
        # vertical phase to initiate the turn.
        tx, th = vx / speed, vh / speed
        if t < kick_time + 2.0:  # brief pitch-over window
            ang = np.arctan2(th, tx) - np.radians(pitch_kick)
            tx, th = np.cos(ang), np.sin(ang)

    # --- thrust magnitude with back-pressure correction ---
    # Vacuum thrust is the input; ambient pressure reduces effective thrust.
    thrust_eff = thrust - P_amb * exit_area if thrust > 0 else 0.0
    thrust_eff = max(thrust_eff, 0.0)

    Fx_thrust = thrust_eff * tx
    Fh_thrust = thrust_eff * th

    # --- drag (opposes velocity) ---
    if speed > 1e-6:
        q = 0.5 * rho * speed ** 2
        drag = q * vehicle.cd * vehicle.reference_area
        Fx_drag = -drag * (vx / speed)
        Fh_drag = -drag * (vh / speed)
    else:
        Fx_drag = Fh_drag = 0.0

    ax = (Fx_thrust + Fx_drag) / m
    ah = (Fh_thrust + Fh_drag) / m - g

    dm = -mdot if thrust > 0 else 0.0
    return [vx, vh, ax, ah, dm]


def simulate(vehicle: Vehicle, kick_time: float = 8.0, pitch_kick: float = 5.0,
             max_time: float = 600.0, dt_max: float = 0.5):
    """Run the full multi-stage ascent. Returns a dict of time-history arrays
    and a list of stage-separation events."""
    # Initial state: on the pad.
    m0 = vehicle.mass_above_stage(0)
    state = [0.0, 0.0, 0.0, 0.0, m0]

    all_t, all_x, all_h, all_vx, all_vh, all_m = [], [], [], [], [], []
    events_log = []
    t_start = 0.0

    for i, stage in enumerate(vehicle.stages):
        thrust, mdot = stage.thrust, stage.mdot
        burn = stage.burn_time

        # Terminal event: propellant exhausted (mass reaches this stage's dry
        # floor = everything above + this stage's dry mass + payload).
        dry_floor = vehicle.mass_above_stage(i) - stage.propellant_mass

        def burnout(t, y, *_args, floor=dry_floor):
            return y[4] - floor
        burnout.terminal = True
        burnout.direction = -1

        sol = solve_ivp(
            _derivs, (t_start, t_start + burn + 1.0), state,
            args=(thrust, mdot, stage.exit_area, stage.isp, vehicle,
                  kick_time, pitch_kick),
            events=burnout, max_step=dt_max, rtol=1e-7, atol=1e-6, dense_output=False,
        )

        all_t.extend(sol.t); all_x.extend(sol.y[0]); all_h.extend(sol.y[1])
        all_vx.extend(sol.y[2]); all_vh.extend(sol.y[3]); all_m.extend(sol.y[4])

        t_start = sol.t[-1]
        state = list(sol.y[:, -1])

        # Jettison the spent stage's dry mass before igniting the next stage.
        sep_alt = state[1]
        sep_speed = np.hypot(state[2], state[3])
        events_log.append({"stage": stage.name, "t": t_start,
                           "altitude_km": sep_alt / 1000, "speed_ms": sep_speed})
        state[4] -= stage.dry_mass

        # Coast a touch handled implicitly by next stage ignition (no coast here).

    # Post-burn coast to apogee (no thrust) so the trajectory shows the arc.
    def hit_ground(t, y, *_args):
        return y[1]
    hit_ground.terminal = True
    hit_ground.direction = -1

    coast = solve_ivp(
        _derivs, (t_start, max_time), state,
        args=(0.0, 0.0, 0.0, 0.0, vehicle, kick_time, pitch_kick),
        events=hit_ground, max_step=dt_max, rtol=1e-7, atol=1e-6,
    )
    all_t.extend(coast.t); all_x.extend(coast.y[0]); all_h.extend(coast.y[1])
    all_vx.extend(coast.y[2]); all_vh.extend(coast.y[3]); all_m.extend(coast.y[4])

    t = np.array(all_t); x = np.array(all_x); h = np.array(all_h)
    vx = np.array(all_vx); vh = np.array(all_vh); m = np.array(all_m)
    speed = np.hypot(vx, vh)

    # Dynamic pressure history for max-Q.
    rho = np.array([atmosphere(hi)[0] for hi in h])
    q = 0.5 * rho * speed ** 2

    return {
        "t": t, "x": x, "h": h, "vx": vx, "vh": vh, "m": m,
        "speed": speed, "q": q, "events": events_log,
        "apogee_km": h.max() / 1000,
        "max_speed_ms": speed.max(),
        "maxq_pa": q.max(),
        "maxq_alt_km": h[np.argmax(q)] / 1000,
    }


if __name__ == "__main__":
    from .vehicle import demo_two_stage
    r = simulate(demo_two_stage())
    print(f"Apogee:      {r['apogee_km']:.1f} km")
    print(f"Max speed:   {r['max_speed_ms']:.0f} m/s")
    print(f"Max-Q:       {r['maxq_pa']/1000:.1f} kPa at {r['maxq_alt_km']:.1f} km")
    for e in r["events"]:
        print(f"  sep {e['stage']}: t={e['t']:.1f}s  alt={e['altitude_km']:.1f}km  "
              f"v={e['speed_ms']:.0f}m/s")
