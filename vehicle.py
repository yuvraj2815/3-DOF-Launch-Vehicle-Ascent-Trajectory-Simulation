"""
vehicle.py
==========
Defines the launch vehicle as a set of stages plus aerodynamic properties.
Everything the trajectory integrator needs about the rocket lives here, so
swapping in a different vehicle is a one-object change.

A Stage carries:
  - dry_mass:      structural mass that is jettisoned at burnout (kg)
  - propellant_mass: usable propellant (kg)
  - thrust:        vacuum thrust (N)  [we correct for back-pressure in the EOM]
  - isp:           specific impulse (s)
  - burn_time:     derived from propellant, thrust, isp

The Vehicle bundles stages bottom-first and adds a payload plus drag properties.
"""

from dataclasses import dataclass, field
from typing import List

G0 = 9.80665


@dataclass
class Stage:
    name: str
    dry_mass: float          # kg
    propellant_mass: float   # kg
    thrust: float            # N (vacuum)
    isp: float               # s
    exit_area: float = 0.0   # m^2, nozzle exit area for pressure thrust term

    @property
    def mdot(self) -> float:
        """Propellant mass flow rate, from thrust = mdot * isp * g0."""
        return self.thrust / (self.isp * G0)

    @property
    def burn_time(self) -> float:
        return self.propellant_mass / self.mdot


@dataclass
class Vehicle:
    stages: List[Stage]
    payload_mass: float      # kg
    diameter: float          # m, for reference area
    cd: float = 0.3          # drag coefficient (approx, transonic-averaged)

    @property
    def reference_area(self) -> float:
        import math
        return math.pi * (self.diameter / 2) ** 2

    def mass_above_stage(self, stage_index: int) -> float:
        """Total mass sitting on top of (and including) the given stage:
        this stage's full mass plus everything above it plus payload."""
        m = self.payload_mass
        for i in range(stage_index, len(self.stages)):
            s = self.stages[i]
            m += s.dry_mass + s.propellant_mass
        return m


def demo_two_stage() -> Vehicle:
    """A small illustrative two-stage vehicle (loosely small-launcher sized)."""
    stage1 = Stage(name="Stage-1", dry_mass=3000, propellant_mass=30000,
                   thrust=750_000, isp=280, exit_area=1.2)
    stage2 = Stage(name="Stage-2", dry_mass=800, propellant_mass=6000,
                   thrust=120_000, isp=330, exit_area=0.6)
    return Vehicle(stages=[stage1, stage2], payload_mass=300, diameter=1.5, cd=0.3)


if __name__ == "__main__":
    v = demo_two_stage()
    print(f"Reference area: {v.reference_area:.3f} m^2")
    liftoff = v.mass_above_stage(0)
    print(f"Liftoff mass: {liftoff:.0f} kg")
    for i, s in enumerate(v.stages):
        twr = s.thrust / (v.mass_above_stage(i) * G0)
        print(f"  {s.name}: burn {s.burn_time:5.1f} s, mdot {s.mdot:6.1f} kg/s, "
              f"initial TWR {twr:4.2f}")
