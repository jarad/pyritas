"""Tbd."""
import logging
from math import exp
from typing import Union

import numpy as np
import pyproj
from pyproj import CRS

Number = Union[int, float]
ArrayLike = Union[list[Number], np.ndarray]

MIN_LONGITUDE = -180
MAX_LONGITUDE = 180


UTM_MIN_EASTING = 166000
UTM_MAX_EASTING = 834000
UTM_MIN_NORTHING = 0
UTM_MAX_NORTHING = 10000000
MAX_YIELD_THRESHOLD = 10000


def is_in_utm_range(easting: ArrayLike, northing: ArrayLike) -> bool:
    """
    Check if values (or arrays of values) for easting and northing are in the typical UTM range.
    """
    return (
        UTM_MIN_EASTING < np.min(easting) < UTM_MAX_EASTING and UTM_MIN_NORTHING < np.min(northing) < UTM_MAX_NORTHING
    )


def get_utm_zone(lon: ArrayLike) -> int:
    """
    Given a longitude value (or array of values), determine the appropriate UTM zone.
    """
    if isinstance(lon, (list, np.ndarray)):
        lon = np.mean(lon)
    return int((lon + 180) / 6) + 1


def get_utm_projection(lon: ArrayLike) -> pyproj.Proj:
    """
    Get a pyproj.Proj object for UTM given a longitude value (or array of values).
    """
    zone = get_utm_zone(lon)
    return pyproj.Proj(proj="utm", zone=zone, ellps="WGS84")


def is_in_longitude_range(lon: ArrayLike) -> bool:
    """
    Check if a value (or array of values) is in the typical longitude range.
    """
    return np.min(lon) > MIN_LONGITUDE and np.max(lon) < MAX_LONGITUDE


def convert_to_utm(lon: ArrayLike, lat: ArrayLike, proj4string: str) -> tuple[ArrayLike, ArrayLike]:
    """
    Convert longitude and latitude to UTM only if they are not already in UTM format.
    """
    if is_in_longitude_range(lon):
        if not is_in_utm_range(lon):
            proj_utm = get_utm_projection(lon)
            datum = CRS(proj4string).to_dict().get("datum", "WGS84")
            proj_latlon = pyproj.Proj(proj="latlong", datum=datum)

            transformer = pyproj.Transformer.from_proj(proj_latlon, proj_utm)
            lon, lat = transformer.transform(lon, lat)
    else:
        raise ValueError("Longitude values are out of range.")

    return lon, lat


def yield_equation_mgha(mass_kg: float, area_m2: float) -> float:
    """
    Compute the yield in megagrams per hectare (mg/ha).

    Args:
        mass_kg: The mass in kilograms.
        area_m2: The area in square meters.

    Returns:
        The yield in megagrams per hectare.

    Examples:
        >>> yield_equation_mgha(1000, 10000)
        1.0
    """
    # Conversion factors
    kg_to_mg = 0.001  # 1 kilogram = 0.001 megagram
    m2_to_ha = 0.0001  # 1 square meter = 0.0001 hectare
    yield_values = 10 * (mass_kg * kg_to_mg) / (area_m2 * m2_to_ha)
    too_high_yields = yield_values > MAX_YIELD_THRESHOLD  # or some threshold that is considered too high
    if too_high_yields.any():
        logging.warning("High yields calculated at indices: %s", too_high_yields[too_high_yields].index.tolist())

    return yield_values


def lognormal_to_normal(mean: float, variance: float, value_type: str = "mean") -> Union[float, np.ndarray]:
    """
    Transform the parameter value from lognormal to normal distribution.

    Args:
        mean: The mean of the lognormal distribution, can be a float or a Pandas Series.
        variance: The variance of the lognormal distribution, can be a float or a Pandas Series.
        value_type: The type of the value to return ('mean' or 'var' or 'median').

    Returns:
        The transformed value, will be a float if input is float, or a Pandas Series if input is Series.

    Examples:
        >>> lognormal_to_normal(0, 1, 'mean')
        1.6487212707001282
    """
    if value_type == "mean":
        return np.exp(mean + 0.5 * variance)
    if value_type == "var":
        return (np.exp(variance) - 1) * np.exp(2 * mean + variance)
    if value_type == "median":
        return np.exp(mean)
    raise ValueError("Parameter 'value_type' should be 'mean' or 'var'.")


def grain_market_moisture(crop_string: str) -> list[float]:
    """
    Return the nominal market moisture content of a crop.

    Args:
        crop_string: A string, or a list of strings, with the name of the crop(s).

    Returns:
        The nominal market moisture content in percentage.

    Examples:
        >>> grain_market_moisture(["Corn", "Soybeans"])
        [15.5, 13]
    """
    moisture_content = {
        "Corn": 15.5,
        "Soybeans": 13,
    }
    return [moisture_content[crop] for crop in crop_string if crop in moisture_content]
