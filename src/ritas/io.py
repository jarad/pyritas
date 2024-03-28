"""RITAS I/O module."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio


def read_input(infile: Path) -> pd.DataFrame:
    """Read the input file.

    Args:
        infile (Path): The input file to read.

    Returns:
        pd.DataFrame: The input data.
    """
    if infile.suffix == ".shp":
        return gpd.read_file(infile)
    return pd.read_csv(infile)


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
