"""Tests for the polygons module."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from ritas.polygons import make_bounding_box, make_vehicle_polygons
from shapely.geometry import Polygon


# Test cases for make_bounding_box
@pytest.mark.parametrize(
    ("x", "y", "w", "d", "expected_len"),
    [
        (
            np.array([1, 3, 5]),
            np.array([1, 2, 1]),
            np.array([2, 2, 2]),
            np.array([np.nan, 3, 3]),
            3,
        ),
        (np.array([]), np.array([]), np.array([]), np.array([]), 0),
        (np.array([1]), np.array([1]), np.array([1]), np.array([np.nan]), 1),
    ],
)
def test_make_bounding_box(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    d: np.ndarray,
    expected_len: int,
) -> None:
    df = make_bounding_box(x, y, w, d)
    assert df.shape[0] == expected_len
    assert df.shape[1] == 12


def test_make_bounding_box_input_mismatch() -> None:
    with pytest.raises(ValueError):
        make_bounding_box(
            np.array([1, 2]),
            np.array([1]),
            np.array([2, 2]),
            np.array([np.nan, 3]),
        )


def test_make_bounding_box_empty_arrays() -> None:
    x, y, w, d = np.array([]), np.array([]), np.array([]), np.array([])
    df = make_bounding_box(x, y, w, d)
    assert df.empty


def test_make_bounding_box_invalid_input_types() -> None:
    with pytest.raises(TypeError):
        make_bounding_box([1, 2], [1, 2], [1, 2], [np.nan, 3])


def test_make_bounding_box_zero_width() -> None:
    x = np.array([1, 2])
    y = np.array([1, 2])
    w = np.array([0, 0])
    d = np.array([np.nan, 3])
    make_bounding_box(x, y, w, d)
    # Further assertions can be added here


# Test cases for make_vehicle_polygons
def test_make_vehicle_polygons_basic() -> None:
    df = pd.DataFrame(
        {
            "x": [500000, 500010, 500020],  # UTM coordinates
            "y": [4640000, 4640010, 4640020],
            "swath": [2, 2, 2],
            "d": [1, 3, 3],
        },
    )
    proj4string = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]))

    gdf = make_vehicle_polygons(df, proj4string)
    assert len(gdf) == len(df) - 1
    assert all(isinstance(geom, Polygon) for geom in gdf.geometry)


def test_make_vehicle_polygons_invalid_proj4string() -> None:
    df = pd.DataFrame(
        {
            "x": [1, 3, 5],
            "y": [1, 2, 1],
            "swath": [2, 2, 2],
            "d": [np.nan, 3, 3],
        },
    )
    proj4string = "invalid_proj4string"
    with pytest.raises(ValueError):
        make_vehicle_polygons(df, proj4string)
