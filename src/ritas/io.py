"""RITAS I/O module."""

from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.features import rasterize

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
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs="EPSG:26915",
    )


def rectify_input(geodf: gpd.GeoDataFrame, **kwargs: dict) -> gpd.GeoDataFrame:
    """Rectify the input data to match what RITAS needs for processing.

    Args:
        geodf (gpd.GeoDataFrame): The input data.

    Returns:
        gpd.GeoDataFrame: The rectified data.
    """
    # Ensure the input data has a mass field
    if ColNames.MASS not in geodf.columns:
        udf = kwargs.get("mass_field")
        if udf is None or udf not in geodf.columns:
            raise ValueError(
                "No mass field provided, please pass -m <field> to CLI."
            )
        geodf[ColNames.MASS] = geodf[udf]
    if ColNames.SWATH not in geodf.columns:
        udf = kwargs.get("swath_field")
        if udf is None or udf not in geodf.columns:
            raise ValueError(
                "No swath field provided, please pass -s <field> to CLI."
            )
        geodf[ColNames.SWATH] = geodf[udf]

    if ColNames.DISTANCE not in geodf.columns:
        udf = kwargs.get("distance_field")
        if udf is None or udf not in geodf.columns:
            raise ValueError(
                "No distance field provided, please pass -d <field> to CLI."
            )
        geodf[ColNames.DISTANCE] = geodf[udf]

    for _key, value in asdict(ColNames).items():
        if value not in geodf.columns:
            geodf[value] = np.nan

    # Ensure that the input geometry is in UTM 26915
    return geodf.to_crs("EPSG:26915")


def write_geotiff(geodf: gpd.GeoDataFrame, outfile: Path) -> None:
    """Write a GeoDataFrame to a GeoTIFF file.

    Args:
        geodf (gpd.GeoDataFrame): The GeoDataFrame containing polygons.
        outfile (Path): The output file to create.
    """
    # 1. Compute the bounds of the GeoDataFrame
    bounds = geodf.total_bounds
    ny = int((bounds[3] - bounds[1]) / 5)
    nx = int((bounds[2] - bounds[0]) / 5)
    aff = rio.transform.from_bounds(*bounds, nx, ny)
    raster = rasterize(
        geodf[["geometry", ColNames.MASS]],
        out_shape=(ny, nx),
        fill=np.nan,
        transform=aff,
        nodata=np.nan,
        masked=True,
        all_touched=True,  # maybe suboptimal
        dtype=rio.float32,
    )
    with rio.open(
        outfile,
        "w",
        driver="GTiff",
        width=raster.shape[1],
        height=raster.shape[0],
        count=1,
        dtype=rio.float32,
        crs=geodf.crs,
        transform=aff,
    ) as dst:
        dst.write(np.flipud(raster), 1)
