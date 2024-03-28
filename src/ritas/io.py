"""RITAS I/O module."""

from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

from ritas import ColNames


def read_input(infile: Path) -> gpd.GeoDataFrame:
    """Read the input file.

    Args:
        infile (Path): The input file to read.

    Returns:
        pd.DataFrame: The input data.
    """
    if infile.suffix == ".shp":
        return gpd.read_file(infile)
    df = pd.read_csv(infile)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]))


def rectify_input(geodf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Rectify the input data to match what RITAS needs for processing.

    Args:
        geodf (gpd.GeoDataFrame): The input data.

    Returns:
        gpd.GeoDataFrame: The rectified data.
    """
    # Ensure the input data has the necessary columns
    for _key, value in asdict(ColNames).items():
        if value not in geodf.columns:
            geodf[value] = np.nan

    return geodf


def write_geotiff(geodf: gpd.GeoDataFrame, outfile: Path) -> None:
    """Write a GeoDataFrame to a GeoTIFF file.

    Args:
        outfile (Path): The output file to create.
        geodf (gpd.GeoDataFrame): The GeoDataFrame to write.
    """
    # The GeoDataFrame contains point values on a grid, so we need to first
    # convert it to a raster.
    bounds = geodf.total_bounds
    # create a 5 meter grid based on the bounds
    x = range(int(bounds[0]), int(bounds[2]), 5)
    y = range(int(bounds[1]), int(bounds[3]), 5)
    # create a raster
    raster = np.zeros((len(y), len(x)))
    # fill the raster with values
    for _i, row in geodf.iterrows():
        pt = row.geometry.centroid
        raster[int((pt.y - y[0]) / 5)][int((pt.x - x[0]) / 5)] = 0  # FIXME
    with rio.open(
        outfile,
        "w",
        driver="GTiff",
        width=raster.shape[1],
        height=raster.shape[0],
        count=1,
        dtype=rio.float32,
        crs=geodf.crs,
        transform=rio.transform.from_origin(bounds[0], bounds[3], 5, 5),
    ) as dst:
        dst.write(np.flipud(raster), 1)
