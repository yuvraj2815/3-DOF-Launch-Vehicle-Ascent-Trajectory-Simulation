"""
atmosphere.py
=============
A simplified U.S. Standard Atmosphere (1976) up to ~86 km. Gives air density and
pressure as a function of altitude, which the trajectory needs for aerodynamic
drag. Above 86 km density is negligible for our purposes and is clamped to a
tiny value so the drag term smoothly vanishes.

The model is piecewise: the atmosphere is divided into layers, each with a
constant temperature lapse rate. Within a layer, temperature is linear in
altitude and pressure follows the barometric formula; density comes from the
ideal gas law.
"""

import numpy as np

# Constants
G0 = 9.80665      # standard gravity at sea level, m/s^2
R_AIR = 287.053   # specific gas constant for air, J/(kg K)

# Layer bases: (base geopotential altitude m, base temp K, lapse rate K/m,
#               base pressure Pa)
_LAYERS = [
    (0.0,      288.15, -0.0065, 101325.0),
    (11000.0,  216.65,  0.0,    22632.06),
    (20000.0,  216.65,  0.001,  5474.889),
    (32000.0,  228.65,  0.0028, 868.0187),
    (47000.0,  270.65,  0.0,    110.9063),
    (51000.0,  270.65, -0.0028, 66.93887),
    (71000.0,  214.65, -0.002,  3.956420),
]
_TOP = 86000.0


def atmosphere(altitude: float):
    """Return (density [kg/m^3], pressure [Pa], temperature [K]) at altitude."""
    h = max(0.0, float(altitude))
    if h >= _TOP:
        return 1e-9, 0.0, 186.87  # effectively vacuum

    # Find the layer this altitude sits in.
    layer = _LAYERS[0]
    for L in _LAYERS:
        if h >= L[0]:
            layer = L
        else:
            break

    h_b, T_b, lapse, P_b = layer

    if lapse == 0.0:
        T = T_b
        P = P_b * np.exp(-G0 * (h - h_b) / (R_AIR * T_b))
    else:
        T = T_b + lapse * (h - h_b)
        P = P_b * (T / T_b) ** (-G0 / (lapse * R_AIR))

    rho = P / (R_AIR * T)
    return rho, P, T


def density(altitude: float) -> float:
    """Convenience: just the density."""
    return atmosphere(altitude)[0]


if __name__ == "__main__":
    print(" alt(km)   rho(kg/m3)   P(kPa)   T(K)")
    for km in [0, 5, 11, 20, 32, 50, 70, 86]:
        rho, P, T = atmosphere(km * 1000)
        print(f"  {km:5d}   {rho:9.4e}  {P/1000:7.2f}  {T:6.1f}")
