from __future__ import (absolute_import, division, print_function, unicode_literals)

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from ..ephemeris_manager import EphemerisManager, _NAIF_NAMES, init, get_skycoord
import astroplan.ephemeris_manager as em_module


# ---------------------------------------------------------------------------
# BSP availability marker
# ---------------------------------------------------------------------------

_BSP_CONFIG = Path(__file__).parent.parent / 'bsp_config.yaml'
_DE430 = _BSP_CONFIG.parent / 'bsp' / 'de430.bsp'

bsp_available = pytest.mark.skipif(
    not (_BSP_CONFIG.exists() and _DE430.exists()),
    reason="BSP kernels not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bare_manager():
    """EphemerisManager with __init__ bypassed and empty config."""
    m = object.__new__(EphemerisManager)
    m.config = {}
    return m


@pytest.fixture
def reset_singleton(monkeypatch):
    """Ensure the module-level singleton is None before and after each test."""
    monkeypatch.setattr(em_module, '_manager', None)
    yield
    monkeypatch.setattr(em_module, '_manager', None)


@pytest.fixture
def mock_manager():
    """
    EphemerisManager with all skyfield I/O replaced by mocks.
    get_skycoord and resolve_naif_id are the real implementations;
    timescale / get_body / get_observer are mocked.
    The mock returns RA = 6 h (90 deg) and Dec = 23 deg.
    """
    m = object.__new__(EphemerisManager)
    m.config = {}

    mock_ra = MagicMock()
    mock_dec = MagicMock()
    mock_ra.hours = 6.0     # 6 h = 90 deg
    mock_dec.degrees = 23.0

    mock_radec_result = MagicMock()
    mock_radec_result.radec.return_value = (mock_ra, mock_dec, MagicMock())

    mock_apparent = MagicMock()
    mock_apparent.apparent.return_value = mock_radec_result

    mock_observe = MagicMock()
    mock_observe.observe.return_value = mock_apparent

    mock_at_result = MagicMock()
    mock_at_result.at.return_value = mock_observe

    mock_ts = MagicMock()
    mock_ts.from_astropy.return_value = MagicMock()

    m.timescale = MagicMock(return_value=mock_ts)
    m.get_body = MagicMock(return_value=MagicMock())
    m.get_observer = MagicMock(return_value=mock_at_result)

    return m


# ---------------------------------------------------------------------------
# _NAIF_NAMES table
# ---------------------------------------------------------------------------

class TestNaifNamesTable:
    def test_all_entries(self):
        # Bodies in de430 directly → body IDs; outer planets → barycenter IDs
        expected = {
            'sun': 10, 'mercury': 199, 'venus': 299,
            'earth-moon-barycenter': 3,
            'earth': 399, 'moon': 301,
            'mars': 4, 'jupiter': 5, 'saturn': 6,
            'uranus': 7, 'neptune': 8,
        }
        assert _NAIF_NAMES == expected

    def test_no_satellites_or_extra_barycenters(self):
        # Satellites and extra aliases must not be present
        for name in ('pluto', 'luna', 'emb', 'phobos', 'titan', 'triton',
                     'mars barycenter', 'earth-moon barycenter'):
            assert name not in _NAIF_NAMES


# ---------------------------------------------------------------------------
# resolve_naif_id
# ---------------------------------------------------------------------------

class TestResolveNaifId:
    def test_builtin_lookup(self, bare_manager):
        assert bare_manager.resolve_naif_id('mars') == 499

    def test_case_insensitive(self, bare_manager):
        assert bare_manager.resolve_naif_id('Mars') == 499
        assert bare_manager.resolve_naif_id('MARS') == 499

    def test_strips_whitespace(self, bare_manager):
        assert bare_manager.resolve_naif_id('  mars  ') == 499

    def test_earth_moon_barycenter(self, bare_manager):
        assert bare_manager.resolve_naif_id('earth-moon-barycenter') == 3

    def test_all_recognised_names_resolve(self, bare_manager):
        for name in _NAIF_NAMES:
            assert bare_manager.resolve_naif_id(name) == _NAIF_NAMES[name]

    def test_satellite_name_raises(self, bare_manager):
        with pytest.raises(ValueError, match="naif_id"):
            bare_manager.resolve_naif_id('phobos')

    def test_unknown_name_raises_value_error(self, bare_manager):
        with pytest.raises(ValueError):
            bare_manager.resolve_naif_id('not_a_real_body')

    def test_error_message_mentions_naif_id(self, bare_manager):
        with pytest.raises(ValueError, match="naif_id"):
            bare_manager.resolve_naif_id('titan')

    def test_error_message_lists_recognised_names(self, bare_manager):
        with pytest.raises(ValueError, match="mars"):
            bare_manager.resolve_naif_id('not_a_real_body')


# ---------------------------------------------------------------------------
# _find_extra_kernel
# ---------------------------------------------------------------------------

class TestFindExtraKernel:
    def _manager_with_config(self, config):
        m = object.__new__(EphemerisManager)
        m.config = config
        m._config_dir = Path('/fake')
        return m

    def test_satellite_id_range_matched(self):
        m = self._manager_with_config({
            'satellites': [{'file': 'mar097.bsp', 'id_range': [401, 499]}]
        })
        path, is_satellite, center = m._find_extra_kernel(401)
        assert is_satellite is True
        assert center is None

    def test_asteroid_id_range_matched(self):
        m = self._manager_with_config({
            'asteroids': [{'file': 'codes.bsp', 'id_range': [2000001, 2000953], 'center': 10}]
        })
        path, is_satellite, center = m._find_extra_kernel(2000001)
        assert is_satellite is False
        assert center == 10

    def test_asteroid_id_range_boundary(self):
        m = self._manager_with_config({
            'asteroids': [{'file': 'codes.bsp', 'id_range': [2000001, 2000953], 'center': 10}]
        })
        _, _, center_lo = m._find_extra_kernel(2000001)
        _, _, center_hi = m._find_extra_kernel(2000953)
        assert center_lo == 10
        assert center_hi == 10

    def test_asteroid_id_range_out_of_range(self):
        m = self._manager_with_config({
            'asteroids': [{'file': 'codes.bsp', 'id_range': [2000001, 2000953], 'center': 10}]
        })
        path, is_satellite, center = m._find_extra_kernel(2001000)
        assert path is None

    def test_asteroid_explicit_ids(self):
        m = self._manager_with_config({
            'asteroids': [{'file': 'ceres.bsp', 'ids': [2000001]}]
        })
        path, is_satellite, center = m._find_extra_kernel(2000001)
        assert is_satellite is False
        assert center == 0   # default when 'center' not set

    def test_asteroid_default_center_is_ssb(self):
        m = self._manager_with_config({
            'asteroids': [{'file': 'ssb.bsp', 'id_range': [2000001, 2000001]}]
        })
        _, _, center = m._find_extra_kernel(2000001)
        assert center == 0

    def test_not_found_returns_none(self):
        m = self._manager_with_config({})
        path, is_satellite, center = m._find_extra_kernel(2000001)
        assert path is None
        assert is_satellite is False
        assert center is None


# ---------------------------------------------------------------------------
# get_body — Sun-chaining logic (mocked kernels)
# ---------------------------------------------------------------------------

class TestGetBodyChaining:
    def _manager_with_kernels(self, planetary, extra, extra_path, is_satellite, center):
        m = object.__new__(EphemerisManager)
        m.config = {}
        m._planetary = planetary
        m._kernel_cache = {}
        m._find_extra_kernel = MagicMock(return_value=(extra_path, is_satellite, center))
        m._load_kernel = MagicMock(return_value=extra) if extra_path else MagicMock()
        return m

    def test_planetary_kernel_used_when_no_extra(self):
        planetary = MagicMock()
        m = self._manager_with_kernels(planetary, None, None, False, None)
        m.get_body(499)
        planetary.__getitem__.assert_called_once_with(499)

    def test_satellite_chains_through_barycenter(self):
        extra = MagicMock()
        planet_bcb = MagicMock()
        satellite_vec = MagicMock()
        extra.__getitem__ = MagicMock(return_value=satellite_vec)
        planetary = MagicMock()
        planetary.__getitem__ = MagicMock(return_value=planet_bcb)

        m = self._manager_with_kernels(planetary, extra, '/fake/mar097.bsp', True, None)
        m._load_kernel = MagicMock(return_value=extra)
        result = m.get_body(401)

        planetary.__getitem__.assert_called_once_with(4)   # 401 // 100 = 4
        assert result == planet_bcb + satellite_vec

    def test_sun_relative_asteroid_chains_through_sun(self):
        extra = MagicMock()
        sun_vec = MagicMock()
        asteroid_vec = MagicMock()
        extra.__getitem__ = MagicMock(return_value=asteroid_vec)
        planetary = MagicMock()
        planetary.__getitem__ = MagicMock(return_value=sun_vec)

        m = self._manager_with_kernels(planetary, extra, '/fake/codes.bsp', False, 10)
        m._load_kernel = MagicMock(return_value=extra)
        result = m.get_body(2000001)

        planetary.__getitem__.assert_called_once_with('sun')
        assert result == sun_vec + asteroid_vec

    def test_ssb_relative_asteroid_returned_directly(self):
        extra = MagicMock()
        asteroid_vec = MagicMock()
        extra.__getitem__ = MagicMock(return_value=asteroid_vec)
        planetary = MagicMock()

        m = self._manager_with_kernels(planetary, extra, '/fake/ceres.bsp', False, 0)
        m._load_kernel = MagicMock(return_value=extra)
        result = m.get_body(2000001)

        planetary.__getitem__.assert_not_called()
        assert result == asteroid_vec


# ---------------------------------------------------------------------------
# get_skycoord — unit tests with mocked skyfield
# ---------------------------------------------------------------------------

class TestGetSkycoordUnit:
    def test_returns_icrs_skycoord(self, mock_manager):
        coord = mock_manager.get_skycoord(499, Time("2025-01-01"), _dummy_location())
        assert isinstance(coord, SkyCoord)
        assert coord.frame.name == 'icrs'

    def test_coordinates_match_mock_values(self, mock_manager):
        coord = mock_manager.get_skycoord(499, Time("2025-01-01"), _dummy_location())
        assert abs(coord.ra.deg - 90.0) < 1e-9   # 6 h = 90 deg
        assert abs(coord.dec.deg - 23.0) < 1e-9

    def test_int_naif_id_skips_resolve(self, mock_manager):
        mock_manager.resolve_naif_id = MagicMock()
        mock_manager.get_skycoord(499, Time("2025-01-01"), _dummy_location())
        mock_manager.resolve_naif_id.assert_not_called()

    def test_string_name_calls_resolve(self, mock_manager):
        mock_manager.resolve_naif_id = MagicMock(return_value=499)
        mock_manager.get_skycoord('mars', Time("2025-01-01"), _dummy_location())
        mock_manager.resolve_naif_id.assert_called_once_with('mars')

    def test_get_body_called_with_resolved_id(self, mock_manager):
        mock_manager.resolve_naif_id = MagicMock(return_value=599)
        mock_manager.get_skycoord('jupiter', Time("2025-01-01"), _dummy_location())
        mock_manager.get_body.assert_called_once_with(599)

    def test_get_observer_called_with_location(self, mock_manager):
        loc = _dummy_location()
        mock_manager.get_skycoord(499, Time("2025-01-01"), loc)
        mock_manager.get_observer.assert_called_once_with(loc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestModuleSingleton:
    def test_get_skycoord_before_init_raises(self, reset_singleton):
        with pytest.raises(RuntimeError, match="not initialised"):
            get_skycoord(499, Time("2025-01-01"), _dummy_location())

    def test_error_message_mentions_init(self, reset_singleton):
        with pytest.raises(RuntimeError, match="init"):
            get_skycoord(499, Time("2025-01-01"), _dummy_location())

    def test_init_constructs_manager(self, reset_singleton):
        mock_instance = MagicMock(spec=EphemerisManager)
        with patch('astroplan.ephemeris_manager.EphemerisManager',
                   return_value=mock_instance) as mock_cls:
            init('/fake/bsp_config.yaml')
            mock_cls.assert_called_once_with('/fake/bsp_config.yaml')
            assert em_module._manager is mock_instance

    def test_get_skycoord_delegates_to_manager(self, reset_singleton, monkeypatch):
        expected = SkyCoord(ra=90*u.deg, dec=23*u.deg, frame='icrs')
        mock_instance = MagicMock(spec=EphemerisManager)
        mock_instance.get_skycoord.return_value = expected
        monkeypatch.setattr(em_module, '_manager', mock_instance)

        loc = _dummy_location()
        time = Time("2025-01-01")
        result = get_skycoord(499, time, loc)

        mock_instance.get_skycoord.assert_called_once_with(499, time, loc)
        assert result is expected

    def test_second_init_replaces_manager(self, reset_singleton):
        mock_a = MagicMock(spec=EphemerisManager)
        mock_b = MagicMock(spec=EphemerisManager)
        with patch('astroplan.ephemeris_manager.EphemerisManager',
                   side_effect=[mock_a, mock_b]):
            init('/config_a.yaml')
            assert em_module._manager is mock_a
            init('/config_b.yaml')
            assert em_module._manager is mock_b


# ---------------------------------------------------------------------------
# Integration tests — require real BSP kernels
# ---------------------------------------------------------------------------

@bsp_available
class TestGetSkycoordIntegration:
    @pytest.fixture(scope='class')
    def manager(self):
        return EphemerisManager(_BSP_CONFIG)

    @pytest.fixture
    def location(self):
        # FYST site (Cerro Chajnantor)
        return EarthLocation(lon=-67.759*u.deg, lat=-22.959*u.deg, height=5612*u.m)

    def test_scalar_time_returns_scalar_coord(self, manager, location):
        coord = manager.get_skycoord(499, Time("2025-06-01 00:00"), location)
        assert isinstance(coord, SkyCoord)
        assert coord.isscalar

    def test_array_time_returns_array_coord(self, manager, location):
        times = Time(["2025-06-01", "2025-06-02", "2025-06-03"])
        coord = manager.get_skycoord(499, times, location)
        assert len(coord) == 3

    def test_name_and_barycenter_id_give_same_result(self, manager, location):
        # 'mars' resolves to barycenter (4), not body (499)
        time = Time("2025-06-01 12:00")
        by_id = manager.get_skycoord(4, time, location)
        by_name = manager.get_skycoord('mars', time, location)
        assert by_id.separation(by_name) < 1e-6 * u.arcsec

    def test_result_frame_is_icrs(self, manager, location):
        coord = manager.get_skycoord(10, Time("2025-06-01"), location)
        assert coord.frame.name == 'icrs'

    def test_sun_coords_in_valid_range(self, manager, location):
        coord = manager.get_skycoord('sun', Time("2025-06-21 00:00"), location)
        assert 0 <= coord.ra.deg < 360
        assert -90 <= coord.dec.deg <= 90

    def test_sun_near_summer_solstice_dec(self, manager, location):
        # Around June solstice the Sun's dec should be close to +23.4 deg
        coord = manager.get_skycoord('sun', Time("2025-06-21 00:00"), location)
        assert abs(coord.dec.deg - 23.4) < 1.0

    def test_resolve_naif_id_then_get_skycoord_consistent(self, manager, location):
        naif_id = manager.resolve_naif_id('venus')
        assert naif_id == 299
        coord = manager.get_skycoord(naif_id, Time("2025-01-01"), location)
        assert isinstance(coord, SkyCoord)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dummy_location():
    return EarthLocation(lon=0*u.deg, lat=0*u.deg, height=0*u.m)
