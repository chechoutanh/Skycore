"""Astronomical geometry: target / Sun / Moon positions via Astropy (offline).

All times are naive UTC ``pandas.DatetimeIndex``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time
from astropy.utils import iers

# Planning needs ~arcminute accuracy, so the bundled IERS table is enough; downloading
# it at runtime crashed the app whenever the IERS server or SSL was unavailable.
iers.conf.auto_download = False
iers.conf.auto_max_age = None
iers.conf.iers_degraded_accuracy = "warn"

# Deep-sky targets: (RA deg, Dec deg), J2000. Solar-system bodies are computed
# from the built-in ephemeris, so everything works without network access.
DEEP_SKY = {
    "M31 (Andromeda Galaxy)": (10.6847, 41.2687),
    "M42 (Orion Nebula)": (83.8221, -5.3911),
    "M45 (Pleiades)": (56.75, 24.1167),
    "M13 (Hercules Cluster)": (250.4235, 36.4613),
    "M8 (Lagoon Nebula)": (270.92, -24.38),
    "M104 (Sombrero Galaxy)": (189.9976, -11.6231),
}
# Naked-eye constellations: approximate centre of the main star pattern (RA deg, Dec deg).
CONSTELLATIONS = {
    "Orion (constellation)": (83.0, 3.0),
    "Scorpius (constellation)": (253.0, -30.0),
    "Sagittarius - Teapot (constellation)": (283.0, -29.0),
    "Ursa Major - Big Dipper (constellation)": (184.0, 56.0),
    "Cassiopeia (constellation)": (15.0, 60.0),
    "Cygnus (constellation)": (309.0, 40.0),
    "Crux - Southern Cross (constellation)": (187.0, -60.0),
    "Leo (constellation)": (160.0, 17.0),
    "Taurus (constellation)": (68.0, 18.0),
    "Gemini (constellation)": (108.0, 24.0),
}
SOLAR_SYSTEM = ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Sun"]
TARGETS = list(DEEP_SKY) + SOLAR_SYSTEM + list(CONSTELLATIONS)


def sun_limit(name: str) -> float:
    """Highest Sun altitude (deg) at which the target can be observed."""
    if name == "Sun":
        return 90.0  # daytime target: no darkness needed
    if name in ("Mercury", "Venus", "Mars", "Jupiter", "Saturn"):
        return -6.0  # bright planets: visible from the end of civil twilight
    if name in CONSTELLATIONS:
        return -12.0  # naked-eye star patterns: nautical twilight is dark enough
    return -18.0  # faint targets need astronomical night


def _target_coord(name: str, t: Time, loc: EarthLocation):
    if name in DEEP_SKY or name in CONSTELLATIONS:
        ra, dec = DEEP_SKY.get(name) or CONSTELLATIONS[name]
        return SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    if name in SOLAR_SYSTEM:
        return get_body(name.lower(), t, loc)
    raise KeyError(f"Unknown target: {name!r}")


def moon_illumination(t: Time, loc: EarthLocation) -> np.ndarray:
    """Illuminated fraction k = (1 + cos i) / 2 of the Moon's disk."""
    sun = get_sun(t)
    moon = get_body("moon", t, loc)
    elong = sun.separation(moon)
    sun_d = sun.distance.to(u.km)
    moon_d = moon.distance.to(u.km)
    phase_angle = np.arctan2(
        sun_d * np.sin(elong.rad), moon_d - sun_d * np.cos(elong.rad)
    )
    return np.asarray((1 + np.cos(phase_angle)) / 2, dtype=float)


def sky_table(
    lat: float,
    lon: float,
    target: str,
    times: pd.DatetimeIndex,
    elevation_m: float = 0.0,
) -> pd.DataFrame:
    """Return target/Sun/Moon geometry for each timestamp.

    Columns: target_alt, sun_alt, moon_alt, moon_illum, moon_sep (degrees,
    illumination in [0, 1]) and sun_limit (darkness the target needs, degrees).
    """
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elevation_m * u.m)
    t = Time(times.values)
    frame = AltAz(obstime=t, location=loc)

    tgt = _target_coord(target, t, loc)
    tgt_altaz = tgt.transform_to(frame)
    sun_altaz = get_sun(t).transform_to(frame)
    moon = get_body("moon", t, loc)
    moon_altaz = moon.transform_to(frame)

    sep = tgt_altaz.separation(moon_altaz).deg

    return pd.DataFrame(
        {
            "target_alt": np.asarray(tgt_altaz.alt.deg),
            "sun_alt": np.asarray(sun_altaz.alt.deg),
            "moon_alt": np.asarray(moon_altaz.alt.deg),
            "moon_illum": moon_illumination(t, loc),
            "moon_sep": np.asarray(sep),
            "sun_limit": sun_limit(target),
        },
        index=times,
    )
