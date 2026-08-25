"""
validation.py
=============
Cross-checks the simulator against closed-form theory. The single most important
sanity check for any trajectory code is the ideal rocket equation (Tsiolkovsky):

    dv = Isp * g0 * ln(m_initial / m_final)

If we remove gravity and drag and thrust straight up in vacuum, the velocity
gained over a burn must equal the Tsiolkovsky dv. Matching this to a fraction of
a percent proves the propulsion/mass-flow core is correct; any later disagreement
in the full trajectory is then attributable to gravity and drag losses, which is
exactly the decomposition an interviewer will ask you to explain.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .vehicle import Stage, G0


def tsiolkovsky_dv(stage: Stage, m_initial: float) -> float:
    """Ideal delta-v for a single stage burning from m_initial to burnout."""
    m_final = m_initial - stage.propellant_mass
    return stage.isp * G0 * np.log(m_initial / m_final)


def simulate_vacuum_no_gravity(stage: Stage, m_initial: float):
    """Integrate a pure 1-D burn with NO gravity and NO drag. Velocity gained
    should equal the Tsiolkovsky dv exactly (up to integrator tolerance)."""
    mdot = stage.mdot

    def derivs(t, y):
        v, m = y
        thrust = stage.thrust
        return [thrust / m, -mdot]

    def burnout(t, y):
        return y[1] - (m_initial - stage.propellant_mass)
    burnout.terminal = True
    burnout.direction = -1

    sol = solve_ivp(derivs, (0, stage.burn_time + 1), [0.0, m_initial],
                    events=burnout, rtol=1e-10, atol=1e-12, max_step=0.1)
    return sol.y[0, -1]  # final velocity


def run_check(stage: Stage, m_initial: float):
    ideal = tsiolkovsky_dv(stage, m_initial)
    numeric = simulate_vacuum_no_gravity(stage, m_initial)
    rel_err = abs(numeric - ideal) / ideal
    return ideal, numeric, rel_err


if __name__ == "__main__":
    from .vehicle import demo_two_stage
    v = demo_two_stage()
    print("Ideal rocket equation validation (vacuum, no gravity):")
    print(f"{'stage':10s} {'ideal dv':>10s} {'numeric dv':>12s} {'rel err':>10s}")
    for i, s in enumerate(v.stages):
        ideal, numeric, err = run_check(s, v.mass_above_stage(i))
        print(f"{s.name:10s} {ideal:10.1f} {numeric:12.1f} {err:10.2e}")
