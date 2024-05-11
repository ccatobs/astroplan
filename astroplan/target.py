# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
from abc import ABCMeta

# Third-party
import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS, UnitSphericalRepresentation, AltAz, get_body
import numpy as np

__all__ = ["Target", "FixedTarget", "NonFixedTarget", "ConstantElevationTarget", "SolarSystemTarget"]

# Docstring code examples include printed SkyCoords, but the format changed
# in astropy 1.3. Thus the doctest needs astropy >=1.3 and this is the
# easiest way to make it work.

__doctest_requires__ = {'FixedTarget.*': ['astropy.modeling.Hermite1D']}


class Target(object):
    """
    Abstract base class for target objects.

    This is an abstract base class -- you can't instantiate
    examples of this class, but must work with one of its
    subclasses such as `~astroplan.target.FixedTarget` or
    `~astroplan.target.NonFixedTarget`.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name=None, ra=None, dec=None, marker=None):
        """
        Defines a single observation target.

        Parameters
        ----------
        name : str, optional

        ra : WHAT TYPE IS ra ?

        dec : WHAT TYPE IS dec ?

        marker : str, optional
            User-defined markers to differentiate between different types
            of targets (e.g., guides, high-priority, etc.).
        """
        raise NotImplementedError()

    @property
    def ra(self):
        """
        Right ascension.
        """
        if isinstance(self, FixedTarget):
            return self.coord.ra
        raise NotImplementedError()

    @property
    def dec(self):
        """
        Declination.
        """
        if isinstance(self, FixedTarget):
            return self.coord.dec
        raise NotImplementedError()


class FixedTarget(Target):
    """
    Coordinates and metadata for an object that is "fixed" with respect to the
    celestial sphere.

    Examples
    --------
    Create a `~astroplan.FixedTarget` object for Sirius:

    >>> from astroplan import FixedTarget
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> sirius_coord = SkyCoord(ra=101.28715533*u.deg, dec=16.71611586*u.deg)
    >>> sirius = FixedTarget(coord=sirius_coord, name="Sirius")

    Create an equivalent `~astroplan.FixedTarget` object for Sirius by querying
    for the coordinates of Sirius by name:

    >>> from astroplan import FixedTarget
    >>> sirius = FixedTarget.from_name("Sirius")
    """

    def __init__(self, coord, name=None, **kwargs):
        """
        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            Coordinate of the target

        name : str (optional)
            Name of the target, used for plotting and representing the target
            as a string
        """
        if not (hasattr(coord, 'transform_to') and
                hasattr(coord, 'represent_as')):
            raise TypeError('`coord` must be a coordinate object.')

        self.name = name
        self.coord = coord

    @classmethod
    def from_name(cls, query_name, name=None, **kwargs):
        """
        Initialize a `FixedTarget` by querying for a name from the CDS name
        resolver, using the machinery in
        `~astropy.coordinates.SkyCoord.from_name`.

        This

        Parameters
        ----------
        query_name : str
            Name of the target used to query for coordinates.

        name : string or `None`
            Name of the target to use within astroplan. If `None`, query_name
            is used as ``name``.

        Examples
        --------
        >>> from astroplan import FixedTarget
        >>> sirius = FixedTarget.from_name("Sirius")
        >>> sirius.coord                              # doctest: +FLOAT_CMP
        <SkyCoord (ICRS): (ra, dec) in deg
            ( 101.28715533, -16.71611586)>
        """
        # Allow manual override for name keyword so that the target name can
        # be different from the query name, otherwise assume name=queryname.
        if name is None:
            name = query_name
        return cls(SkyCoord.from_name(query_name), name=name, **kwargs)

    def __repr__(self):
        """
        String representation of `~astroplan.FixedTarget`.

        Examples
        --------
        Show string representation of a `~astroplan.FixedTarget` for Vega:

        >>> from astroplan import FixedTarget
        >>> from astropy.coordinates import SkyCoord
        >>> vega_coord = SkyCoord(ra='279.23473479d', dec='38.78368896d')
        >>> vega = FixedTarget(coord=vega_coord, name="Vega")
        >>> print(vega)                             # doctest: +FLOAT_CMP
        <FixedTarget "Vega" at SkyCoord (ICRS): (ra, dec) in deg ( 279.23473479, 38.78368894)>
        """
        class_name = self.__class__.__name__
        fmt_coord = repr(self.coord).replace('\n   ', '')[1:-1]
        return '<{} "{}" at {}>'.format(class_name, self.name, fmt_coord)

    @classmethod
    def _from_name_mock(cls, query_name, name=None):
        """
        Mock method to replace `FixedTarget.from_name` in tests without
        internet connection.
        """
        # The lowercase method will be run on names, so enter keys in lowercase:
        stars = {
            "rigel": {"ra": 78.63446707*u.deg, "dec": -8.20163837*u.deg},
            "sirius": {"ra": 101.28715533*u.deg, "dec": -16.71611586*u.deg},
            "vega": {"ra": 279.23473479*u.deg, "dec": 38.78368896*u.deg},
            "aldebaran": {"ra": 68.98016279*u.deg, "dec": 16.50930235*u.deg},
            "polaris": {"ra": 37.95456067*u.deg, "dec": 89.26410897*u.deg},
            "deneb": {"ra": 310.35797975*u.deg, "dec": 45.28033881*u.deg},
            "m13": {"ra": 250.423475*u.deg, "dec": 36.4613194*u.deg},
            "altair": {"ra": 297.6958273*u.deg, "dec": 8.8683212*u.deg},
            "hd 209458": {"ra": 330.79*u.deg, "dec": 18.88*u.deg}
        }

        if query_name.lower() in stars:
            return cls(coord=SkyCoord(**stars[query_name.lower()]),
                       name=query_name)
        else:
            raise ValueError("Target named {} not in mocked FixedTarget "
                             "method".format(query_name))


class NonFixedTarget(Target):
    """
    Placeholder for future function.
    """


class ConstantElevationTarget(Target):
    """
    Coordinates and metadata for an object that is "fixed" at the Altitude
    and having a range of Azimuth

    """

    def __init__(self, alt, az_min, az_max,
                 name=None, ramin=None, decc=None, alt_var=False,
                 min_alt=None, max_alt=None, **kwargs):
        """
        Parameters
        ----------
        alt : `~astropy.units.Quantity`
            Constant altitude of the target
        az_min : `~astropy.units.Quantity`
            Minimum azimuth of the scan/target
        az_max : `~astropy.units.Quantity`
            Maximum azimuth of the scan/target

          note - if az_min > az_max, this indicates that the scan goes from
                 az_min -> 360 deg / 0 deg -> az_max

        name : str (optional)
            Name of the target, used for plotting and representing the target
            as a string
        ramin : `~astropy.coordinates.SkyCoord.ra` (optional)
            RA min (first rising edge of the strip) of the target
        decc : `~astropy.coordinates.SkyCoord.dec` (optional)
            DEC center of the target
        alt_var (boolean) -- Flag to specify constant elevation scans 
                          with variable elevations
        min_alt : `~astropy.units.Quantity`
            Minimum altitude of the scan/target
        max_alt : `~astropy.units.Quantity`
            Maximum altitude of the scan/target
        """

        # if az_min <= az_max:
        #     az_c = (az_min + az_max)/2.0
        # else:
        #     az_c = (az_min + 360.*u.deg + az_max)/2.0
        #     if az_c > 360.*u.deg:
        #         az_c = az_c - 360.*u.deg

        self.name = name
        self.az_min = az_min
        self.az_max = az_max
        #self.az_c = az_c
        self.alt = alt
        self.ramin = ramin
        self.decc = decc
        self.alt_var = alt_var
        self.min_alt = min_alt
        self.max_alt = max_alt


class SolarSystemTarget(Target):
    """
    Coordinates and metadata for an solar system object

    """

    def __init__(self, eph_name, name=None, **kwargs):
        """
        Parameters
        ----------
        eph_name : str
            Name of the target, used in get_body

        name : str (optional)
            Name of the target, used for plotting and representing the target
            as a string

        Two names are introduced so that an optional name can have more information
        than a rigid name used in get_body

        """
        self.eph_name = eph_name
        self.name = name


def get_skycoord(targets, times=None, observer=None):
    """
    Return an `~astropy.coordinates.SkyCoord` object.

    When performing calculations it is usually most efficient to have
    a single `~astropy.coordinates.SkyCoord` object, rather than a
    list of `FixedTarget` or `~astropy.coordinates.SkyCoord` objects.

    This is a convenience routine to do that.

    Parameters
    -----------
    targets : list, `~astropy.coordinates.SkyCoord`, `Fixedtarget`
        either a single target or a list of targets

    times:  `~astropy.time.Time`
        Array of times on which to calculate the coordinates. It can be None
        only when all targets are FixedTargets

    If either of targets and times is a single target/time and the other one is a list,
    it is broadcasted. If both are lists, they should have a 1-to-1 association
    (not 2D grid)

    observer -- `~astroplan.Observer`

    Returns
    --------
    coord : `~astropy.coordinates.SkyCoord`
        a single SkyCoord object, which may be non-scalar
    """

    # Note: in case of ConstantElevationTarget, corresponding coordinates for
    #       az_min is returned

    try:
        target_size = targets.size
    except:
        try:
            target_size = len(targets)
        except:
            target_size = 1

    #print(f'target size is {target_size}, time size is {times.size}')
    if not isinstance(targets, list):
        targets = [targets]

    if times is not None:
        if target_size != 1 and times.size != 1 and target_size != times.size:
            raise ValueError('Either targets or times should be scalor, or the lengths of two lists should match')
    
    coords = []

    for itarget, target in enumerate(targets):
        if times is None:
            time = None
        elif times.size == 1 or target_size == 1:
            time = times
        else:
            time = times[itarget]

        if isinstance(target, FixedTarget):
            coord = getattr(target, 'coord')
            if target_size == 1:
                for i in range(times.size):
                    coords.append(coord)
            else:
                coords.append(coord)
        else:
            if isinstance(target, ConstantElevationTarget):
                if time is None:
                    raise ValueError('times should be given to calculate the coordinates for ConstantElevationTarget')
                elif time.size == 1:
                    coord = SkyCoord(target.az_min, target.alt,
                                     frame=AltAz(
                                         obstime=time, location=observer.location))
                    coords.append(coord)
                else:
                    for i in range(times.size):
                        coord = SkyCoord(target.az_min, target.alt,
                                         frame=AltAz(
                                             obstime=time[i],
                                             location=observer.location))
                        coords.append(coord)
            elif isinstance(target, SolarSystemTarget):
                if time is None:
                    raise ValueError('times should be given to calculate the coordinates for SolarSystemTarget')
                coord = get_body(target.eph_name.lower(), time, location=observer.location)
                coords.append(coord)
            else:
                coord = target
                coords.append(coord)

    if target_size == 1:
        if not isinstance(targets[0], ConstantElevationTarget) or times.size == 1:
            return SkyCoord(coords)

    coords = np.array(coords).squeeze().tolist()

    # are all SkyCoordinate's in equivalent frames? If not, convert to ICRS
    convert_to_icrs = not all(
        [coord.frame.is_equivalent_frame(coords[0].frame) for coord in coords[1:]])

    # we also need to be careful about handling mixtures of
    # UnitSphericalRepresentations and others
    targets_is_unitsphericalrep = [x.data.__class__ is
                                   UnitSphericalRepresentation for x in coords]

    longitudes = []
    latitudes = []
    distances = []
    get_distances = not all(targets_is_unitsphericalrep)
    if convert_to_icrs:
        # mixture of frames
        for coordinate in coords:
            icrs_coordinate = coordinate.icrs
            longitudes.append(icrs_coordinate.ra)
            latitudes.append(icrs_coordinate.dec)
            if get_distances:
                distances.append(icrs_coordinate.distance)
        frame = ICRS()
    else:
        # all the same frame, get the longitude and latitude names
        try:
            # from astropy v2.0, keys are classes
            lon_name, lat_name = [
                mapping.framename for mapping in
                coords[0].frame_specific_representation_info[UnitSphericalRepresentation]]
        except BaseException:            # whereas prior to that they were strings.
            lon_name, lat_name = [mapping.framename for mapping in
                                  coords[0].frame_specific_representation_info['spherical']]

        frame = coords[0].frame
        for coordinate in coords:
            longitudes.append(getattr(coordinate, lon_name))
            latitudes.append(getattr(coordinate, lat_name))
            if get_distances:
                distances.append(coordinate.distance)

    # now let's deal with the fact that we may have a mixture of coords with distances and
    # coords with UnitSphericalRepresentations
    if all(targets_is_unitsphericalrep):
        return SkyCoord(longitudes, latitudes, frame=frame)
    elif not any(targets_is_unitsphericalrep):
        return SkyCoord(longitudes, latitudes, distances, frame=frame)
    else:
        """
        We have a mixture of coords with distances and without.
        Since we don't know in advance the origin of the frame where further transformation
        will take place, it's not safe to drop the distances from those coords with them set.

        Instead, let's assign large distances to those objects with none.
        """
        distances = [distance if distance != 1 else 100*u.kpc for distance in distances]
        return SkyCoord(longitudes, latitudes, distances, frame=frame)


class SpecialObjectFlag(object):
    """
    Flag this object as a special non-fixed target, which has a ``get_*`` method
    within astropy (like the Sun or Moon)
    """
    pass


class SunFlag(SpecialObjectFlag):
    """
    Flag for a computation with the Sun
    """
    approx_sidereal_drift = 5 * u.min


class MoonFlag(SpecialObjectFlag):
    """
    Flag for a computation with the Moon
    """
    approx_sidereal_drift = 60 * u.min
