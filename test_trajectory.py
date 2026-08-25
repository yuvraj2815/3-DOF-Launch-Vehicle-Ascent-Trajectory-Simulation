"""
test_trajectory.py
==================
Physics sanity tests. These validate the simulator against closed-form results
and expected qualitative behaviour, so a regression that breaks the physics is
caught immediately.

Run:  pytest -q
"""

import numpy as np

from src.vehicle import demo_two_stage, Stage
from src.trajectory import simulate, gravity, R_EARTH
from src.validation import run_check
from src.atmosphere import atmosphere


def test_sea_level_atmosphere():
    rho, P, T = atmosphere(0)
    assert abs(rho - 1.225) < 0.01      # standard sea-level density
    assert abs(P - 101325) < 100        # standard sea-level pressure
    assert abs(T - 288.15) < 0.1


def test_gravity_decreases_with_altitude():
    assert gravity(0) > gravity(100_000) > gravity(1_000_000)
    assert abs(gravity(0) - 9.82) < 0.05  # ~9.8 at surface


def test_rocket_equation_matches_to_machine_precision():
    v = demo_two_stage()
    for i, s in enumerate(v.stages):
        _, _, err = run_check(s, v.mass_above_stage(i))
        assert err < 1e-9, f"{s.name} deviates from Tsiolkovsky: {err}"


def test_liftoff_requires_twr_above_one():
    # The demo vehicle must actually be able to lift off.
    v = demo_two_stage()
    g0 = 9.80665
    twr = v.stages[0].thrust / (v.mass_above_stage(0) * g0)
    assert twr > 1.0


def test_baseline_ascent_is_physical():
    r = simulate(demo_two_stage())
    # Reaches space, gains substantial horizontal velocity, has a sane max-Q.
    assert r["apogee_km"] > 100          # crosses the Karman line
    assert r["max_speed_ms"] > 5000      # meaningful orbital-ish velocity
    assert 20e3 < r["maxq_pa"] < 120e3   # max-Q in a realistic band
    assert 5 < r["maxq_alt_km"] < 20     # max-Q in the lower atmosphere


def test_two_separation_events_recorded():
    r = simulate(demo_two_stage())
    assert len(r["events"]) == 2
    # Second stage separates higher and faster than the first.
    assert r["events"][1]["altitude_km"] > r["events"][0]["altitude_km"]
    assert r["events"][1]["speed_ms"] > r["events"][0]["speed_ms"]


def test_mass_is_monotonically_non_increasing_during_burn():
    r = simulate(demo_two_stage())
    # Mass only drops (burn) or jumps down (staging), never rises.
    m = r["m"]
    # allow tiny numerical wiggle
    assert np.all(np.diff(m) <= 1e-6 + 0)
