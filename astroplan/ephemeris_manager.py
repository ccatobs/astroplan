#! /usr/bin/env python

from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
import yaml
from skyfield.api import Loader, wgs84, N, E

# NAIF IDs for the names accepted by astropy.coordinates.get_body.
# Any other body must be specified by naif_id on the SolarSystemTarget.
_NAIF_NAMES = {
    'sun': 10,
    'mercury': 199,             # body is in de430
    'venus': 299,               # body is in de430
    'earth-moon-barycenter': 3,
    'earth': 399,               # body is in de430
    'moon': 301,                # in de430
    'mars': 4,                  # barycenter; body 499 needs mar097.bsp
    'jupiter': 5,               # barycenter; body 599 needs jup365.bsp
    'saturn': 6,                # barycenter; body 699 needs sat441.bsp
    'uranus': 7,                # barycenter; body 799 needs ura111.bsp
    'neptune': 8,               # barycenter; body 899 needs nep095.bsp
}


class EphemerisManager:
    """
    Unified interface to JPL BSP kernels via skyfield.

    Loads the planetary ephemeris (de430/de440) on construction, then lazily
    loads additional kernels for planetary satellites and small bodies as needed.
    All body lookups return a skyfield VectorFunction relative to the Solar
    System Barycenter, ready to be observed from a topocentric observer.

    Parameters
    ----------
    config_path : str or Path
        Path to bsp_config.yaml. Relative paths inside the config are resolved
        relative to the config file's directory.

    Example
    -------
    >>> eph = EphemerisManager('bsp_config.yaml')
    >>> ts  = eph.timescale()
    >>> observer = eph.get_observer(fyst.location)
    >>> mars = eph.get_body(499)
    >>> t = ts.from_astropy(time)
    >>> ra, dec, dist = observer.at(t).observe(mars).apparent().radec()
    """

    def __init__(self, config_path):
        config_path = Path(config_path).resolve()
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self._config_dir = config_path.parent
        self._kernel_cache = {}   # resolved path str -> SpiceKernel

        planetary_path = self._resolve(self.config['planetary'])
        self._planetary = self._load_kernel(planetary_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def timescale(self):
        """Return a skyfield Timescale (convenience wrapper)."""
        loader = Loader(str(self._config_dir))
        return loader.timescale()

    def get_body(self, naif_id):
        """
        Return a skyfield VectorFunction for the given NAIF ID.

        For planetary satellites the result is automatically chained with the
        planet barycenter segment from the planetary kernel, so the returned
        object is always relative to the Solar System Barycenter.

        Parameters
        ----------
        naif_id : int
            NAIF integer ID (e.g. 499 = Mars, 401 = Phobos, 2000001 = Ceres).
        """
        extra_path, is_satellite = self._find_extra_kernel(naif_id)

        if extra_path is None:
            return self._planetary[naif_id]

        extra_kernel = self._load_kernel(extra_path)

        if is_satellite:
            # Satellite BSP gives position relative to planet barycenter.
            # Chain: SSB -> planet barycenter (planetary kernel)
            #             -> satellite        (satellite kernel)
            planet_bcb_id = naif_id // 100
            return self._planetary[planet_bcb_id] + extra_kernel[naif_id]
        else:
            # Small-body BSP gives position relative to SSB directly.
            return extra_kernel[naif_id]

    def resolve_naif_id(self, name):
        """
        Return the NAIF integer ID for a body name string.

        Only the names accepted by ``astropy.coordinates.get_body`` are
        recognised: 'sun', 'mercury', 'venus', 'earth-moon-barycenter',
        'earth', 'moon', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'.
        For any other body set ``naif_id`` explicitly on the SolarSystemTarget.

        Parameters
        ----------
        name : str

        Raises
        ------
        ValueError
            If the name is not in the built-in list.
        """
        key = name.lower().strip()
        if key in _NAIF_NAMES:
            return _NAIF_NAMES[key]
        raise ValueError(
            f"'{name}' is not a recognised body name. "
            f"Recognised names: {sorted(_NAIF_NAMES)}. "
            f"For other bodies set naif_id explicitly on SolarSystemTarget."
        )

    def get_skycoord(self, naif_id_or_name, time, location):
        """
        Return an astropy SkyCoord for a body at the given time(s) from location.

        Parameters
        ----------
        naif_id_or_name : int or str
            NAIF integer ID, or a body name resolved via ``resolve_naif_id``.
        time : `~astropy.time.Time`
            Scalar or array of times.
        location : `~astropy.coordinates.EarthLocation`

        Returns
        -------
        coord : `~astropy.coordinates.SkyCoord`
            ICRS coordinates. Scalar or array matching *time*.
        """
        naif_id = naif_id_or_name if isinstance(naif_id_or_name, int) \
                  else self.resolve_naif_id(naif_id_or_name)

        t = self.timescale().from_astropy(time)
        body = self.get_body(naif_id)
        observer = self.get_observer(location)
        ra, dec, _ = observer.at(t).observe(body).apparent().radec()
        return SkyCoord(ra=ra.hours * u.hourangle, dec=dec.degrees * u.deg, frame='icrs')

    def get_observer(self, location):
        """
        Return a skyfield topocentric observer for an astropy EarthLocation.

        Parameters
        ----------
        location : astropy.coordinates.EarthLocation
        """
        earth = self._planetary['earth']
        return earth + wgs84.latlon(
            location.lat.deg * N,
            location.lon.deg * E,
            elevation_m=location.height.to_value('m'),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, path):
        """Resolve a path that may be relative to the config file."""
        p = Path(path)
        if not p.is_absolute():
            p = self._config_dir / p
        return str(p.resolve())

    def _load_kernel(self, resolved_path):
        if resolved_path not in self._kernel_cache:
            p = Path(resolved_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"BSP kernel not found: {resolved_path}\n"
                    f"Download it from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/"
                )
            loader = Loader(str(p.parent))
            self._kernel_cache[resolved_path] = loader(p.name)
        return self._kernel_cache[resolved_path]

    def _find_extra_kernel(self, naif_id):
        """
        Return (resolved_path, is_satellite) for the BSP file covering naif_id,
        or (None, False) if the planetary kernel is sufficient.
        """
        for entry in self.config.get('satellites', []):
            lo, hi = entry['id_range']
            if lo <= naif_id <= hi:
                return self._resolve(entry['file']), True

        for entry in self.config.get('asteroids', []):
            if naif_id in entry.get('ids', []):
                return self._resolve(entry['file']), False

        return None, False


# ---------------------------------------------------------------------------
# Module-level singleton — call init() once at application startup
# ---------------------------------------------------------------------------

_manager = None


def init(config_path):
    """
    Initialise the module-level EphemerisManager singleton.

    Must be called once before any call to the module-level ``get_skycoord``.

    Parameters
    ----------
    config_path : str or Path
        Path to bsp_config.yaml.
    """
    global _manager
    _manager = EphemerisManager(config_path)


def get_skycoord(naif_id_or_name, time, location):
    """
    Return an astropy SkyCoord using the module-level EphemerisManager.

    Parameters
    ----------
    naif_id_or_name : int or str
        NAIF integer ID or body name (resolved via ``resolve_naif_id``).
    time : `~astropy.time.Time`
    location : `~astropy.coordinates.EarthLocation`
    """
    if _manager is None:
        raise RuntimeError(
            "ephemeris_manager is not initialised. "
            "Call ephemeris_manager.init(config_path) before computing solar system coordinates."
        )
    return _manager.get_skycoord(naif_id_or_name, time, location)
