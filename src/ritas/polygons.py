import logging
import warnings
from functools import wraps
from typing import Any, Callable, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from joblib import Parallel, delayed
from pykrige.ok import OrdinaryKriging
from pyproj import CRS
from pyproj.exceptions import CRSError
from rtree import index
from shapely.errors import GEOSException, TopologicalError
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, box
from shapely.ops import transform, unary_union
from tqdm import tqdm

from ritas.utils import convert_to_utm, is_in_utm_range

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)


# =============================================================================
# STEP 1: Create a bounding box around the vehicle
# =============================================================================


def make_bounding_box(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    d: np.ndarray,
) -> pd.DataFrame:
    """
    Find the vertices of the polygon associated with a moving vehicle.

    Args:
        x (np.ndarray): A numeric vector with the latitude of the vehicle
                        locations in UTM.
        y (np.ndarray): A numeric vector with the longitude of the vehicle
                        locations in UTM.
        w (np.ndarray): A numeric vector with the width of the vehicle in meters.
        d (np.ndarray): A numeric vector with the distance traveled by the vehicle
                        in meters. It must have the same length as the other vectors
                        (the first value can be a NA).

    Returns:
        pd.DataFrame: A dataframe with eight columns with the coordinates of the four
                      vertices, where 11 and 12 represent the points closer to the initial
                      location and 21 and 22 represent the pointers closer to the final location.

    Note:
        `w` MUST BE IN METERS.
    """
    if not (len(x) == len(y) == len(w) == len(d)):
        raise ValueError("All input arrays must be of the same length.")

    logging.info("Running make_bounding_box...")

    # prepare the input arrays
    x0, x1 = np.r_[np.NaN, x[:-1]], x
    y0, y1 = np.r_[np.NaN, y[:-1]], y

    # Compute distance of the vertices to the centroid
    h = 0.5 * w  # half width
    m = np.where((x1 - x0) != 0, (y1 - y0) / (x1 - x0), np.inf)  # division by zero
    dx = h / np.sqrt(1 + 1 / m**2)
    dy = -dx / m

    # The slope of the perpendicular line doesn't exist when (y1 - y0) == 0
    # If it moved horizontally, the perpendicular line is vertical
    ind = np.where(m == 0)
    dy[ind] = h[ind]

    # Rescale the rectangle so that the diagonal has length d
    diag = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    d = d.astype(float)
    d = np.where(np.isnan(d), diag, d)
    r = d / diag

    x0 = x0 + (1 - r) * (x1 - x0)
    y0 = y0 + (1 - r) * (y1 - y0)

    # Compute vertices
    x01 = x0 - dx
    y01 = y0 - dy
    x02 = x0 + dx
    y02 = y0 + dy
    x11 = x1 - dx
    y11 = y1 - dy
    x12 = x1 + dx
    y12 = y1 + dy

    # Combine all coordinates into a dataframe
    return pd.DataFrame(
        np.column_stack((x0, y0, x1, y1, x01, y01, x02, y02, x11, y11, x12, y12)),
        columns=[
            "x0",
            "y0",
            "x1",
            "y1",
            "x01",
            "y01",
            "x02",
            "y02",
            "x11",
            "y11",
            "x12",
            "y12",
        ],
    )


def make_vehicle_polygons(df: pd.DataFrame, proj4string: str, is_utm: bool = True) -> gpd.GeoDataFrame:
    """
    Create a gpd.GeoDataFrame object from the dataframe given as an input.
    For each row in the input dataframe, a rectangle is constructed using
    the current and next positions as well as the swath width.

    Args:
        df: A dataframe with columns `x`, `y`, and `swath`.
            The swath represents the box width and must be in meters.
            Optional column `d` can represent box diagonal length in meters.
        proj4string: The proj4string associated with the system of coordinates

    Returns:
        A gpd.GeoDataFrame with one rectangle per row.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a DataFrame.")

    for column in ["x", "y", "swath"]:
        if column not in df.columns:
            raise ValueError(f"Missing required column '{column}' in the dataframe.")

    if not isinstance(proj4string, str):
        raise ValueError("The proj4string must be a string.")

    if "d" not in df.columns:
        df["d"] = np.nan

    try:
        CRS(proj4string)
    except CRSError as e:
        raise ValueError(f"Invalid proj4string: {e}") from e

    if not is_in_utm_range(df["x"].values, df["y"].values):
        raise ValueError("Coordinates are not in UTM range.")

    if not is_utm:
        df["x"], df["y"] = convert_to_utm(df["x"].values, df["y"].values, proj4string)

    # Compute vertices of the rectangles for each coordinate
    bounding_box = make_bounding_box(
        df["x"].values,
        df["y"].values,
        df["swath"].values,
        df["d"].values,
    )

    df = pd.concat([df, bounding_box], axis=1)

    df = df.dropna(subset=["x01", "y01", "x02", "y02", "x11", "y11", "x12", "y12"])

    polygons = [
        Polygon(
            [
                (row["x01"], row["y01"]),
                (row["x02"], row["y02"]),
                (row["x12"], row["y12"]),
                (row["x11"], row["y11"]),
                (row["x01"], row["y01"]),
            ],
        )
        for _, row in df.iterrows()
        if not np.isnan(row["x01"])
    ]

    gdf = gpd.GeoDataFrame(df, geometry=polygons)
    gdf.crs = proj4string

    return gdf


# ============================================================================
# STEP 2: Crop polygons
# ============================================================================


def check_polygon_type(geom: Union[Polygon, MultiPolygon]) -> Union[Polygon, MultiPolygon, None]:
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if geom.is_empty:
        return None
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
        if polys:
            return MultiPolygon(polys) if len(polys) > 1 else polys[0]
    return None


def clean_polygon(poly: Union[Polygon, MultiPolygon]) -> Union[Polygon, MultiPolygon, None]:
    if not poly.is_valid:
        poly = poly.buffer(0)
    return check_polygon_type(poly)


def crop_polygon(sp_from: Polygon, sp_to: Polygon) -> Union[Polygon, MultiPolygon, None]:
    sp_from = clean_polygon(sp_from)
    sp_to = clean_polygon(sp_to)
    if sp_from is None or sp_to is None:
        return None
    result = sp_from.difference(sp_to)
    return check_polygon_type(result)


def process_single_polygon(poly: Polygon, reference_polygons: List[Polygon]) -> List[Polygon]:
    cropped_list = []
    for ref_poly in reference_polygons:
        if poly.within(ref_poly):
            cropped_list.append(poly)
            continue
        cropped = crop_polygon(poly, ref_poly)
        if cropped:
            if isinstance(cropped, MultiPolygon):
                cropped_list.extend(list(cropped.geoms))
            else:
                cropped_list.append(cropped)
    return cropped_list


def reshape_polygons(spdf: gpd.GeoDataFrame, verbose: bool = True) -> gpd.GeoDataFrame:
    spdf = spdf.copy()
    spdf["geometry"] = spdf["geometry"].apply(clean_polygon)
    spdf = spdf.dropna(subset=["geometry"])
    spdf = spdf.explode(index_parts=True).reset_index(drop=True)
    spdf["geometry"] = spdf["geometry"].simplify(tolerance=0.01)
    sindex = spdf.sindex
    n_last = len(spdf) - 1

    for i in tqdm(range(n_last), disable=not verbose, desc="Reshaping polygons..."):
        current_geometry = spdf.at[i, "geometry"]
        overlapping_ids = list(sindex.intersection(current_geometry.bounds))
        overlapping_ids = [idx for idx in overlapping_ids if idx > i]

        for idx in overlapping_ids:
            overlapping_geometry = spdf.at[idx, "geometry"]

            try:
                cropped = crop_polygon(overlapping_geometry, current_geometry)
                if cropped:
                    spdf.at[idx, "geometry"] = cropped
            except (ValueError, AttributeError) as e:
                logging.error("Error processing geometry at index %s with exception: %s", idx, e)
                continue

    # Merge adjacent and overlapping polygons
    spdf["geometry"] = spdf["geometry"].buffer(0)
    spdf = spdf.explode(index_parts=True).reset_index(drop=True)

    # Calculate effective area
    spdf["effectiveArea"] = spdf["geometry"].area
    return spdf


# =============================================================================
# STEP 3: Create a grid
# =============================================================================


def make_grid_by_size(spdf: gpd.GeoDataFrame, width: float, height: float) -> gpd.GeoDataFrame:
    """Create a grid of polygons based on width and height.

    Args:
        spdf (gpd.GeoDataFrame): A GeoDataFrame with polygons that will be covered by the grid.
        width (float): Grid width in meters.
        height (float): Grid height in meters.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing grid polygons.
    """
    minx, miny, maxx, maxy = spdf.total_bounds

    # Create a list to hold the grid's polygons
    grid_polygons = []

    # Generate the grid polygons
    for x0 in np.arange(minx, maxx, width):
        for y0 in np.arange(miny, maxy, height):
            # Calculate the coordinates of the polygon
            x1 = x0 + width
            y1 = y0 + height
            grid_polygons.append(box(x0, y0, x1, y1))

    # Create the GeoDataFrame with the polygons
    return gpd.GeoDataFrame({"geometry": grid_polygons}, crs=spdf.crs)


def make_grid_by_n(geodataframe: gpd.GeoDataFrame, n: int) -> gpd.GeoDataFrame:
    """
    Create a grid of polygons over the bounding box of a GeoDataFrame.

    Args:
    geodataframe (gpd.GeoDataFrame): The input GeoDataFrame.
    n (int): The approximate number of cells in the grid.

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing the grid of polygons.
    """
    minx, miny, maxx, maxy = geodataframe.total_bounds
    grid_width = (maxx - minx) / np.sqrt(n)
    grid_height = (maxy - miny) / np.sqrt(n)

    x_steps = np.arange(minx, maxx, grid_width)
    y_steps = np.arange(miny, maxy, grid_height)

    polygons = [
        Polygon([(x, y), (x + grid_width, y), (x + grid_width, y + grid_height), (x, y + grid_height)])
        for x in x_steps
        for y in y_steps
    ]

    return gpd.GeoDataFrame(geometry=polygons, crs=geodataframe.crs)


def make_grid(
    spdf: gpd.GeoDataFrame,
    n: Optional[int] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    min_area: float = 0,
    regular: bool = True,
) -> gpd.GeoDataFrame:
    """
    Create a grid covering the given polygons.

    Note that `width` and `height` take priority over `n` if all are non-null.

    Args:
        spdf: A GeoDataFrame object with the polygons to be covered by the grid.
        n: The approximate number of polygons that the grid should contain.
        width: The width of the grid polygons in meters.
        height: The height of the grid polygons in meters.
        min_area: A float between 0 and 1 indicating the minimum proportion of area overlay
            between polygons in `spdf` and a pixel. Grid pixels whose area do not cover the
            minimum in proportion will be dropped.
        regular: If True (default), the union of all the elements in the grid will form a
            regular polygons. If False, grid polygons not intercepting with any of the input
            polygons will be dropped. NOTE: this is not about the elements of of the grid
            having a regular shape!

    Returns:
        A GeoDataFrame object with the polygons conforming the grid.
    """
    if n is None and (width is None or height is None):
        raise ValueError("Either `n`, or `width` and `height`, must be non-null.")

    if width is not None and height is not None:
        grid = make_grid_by_size(spdf, width, height)
    else:
        grid = make_grid_by_n(spdf, n)

    if not regular:
        grid = grid[grid.intersects(spdf.unary_union)]

    if min_area > 0:
        # This part of the code calculates the proportion of the area of intersection
        # between each grid cell and the input polygons, and filters the grid based on min_area.
        intersections = gpd.overlay(grid, spdf, how="intersection")
        area_ratios = intersections.area / grid.loc[intersections.index].area
        grid = grid.loc[area_ratios[area_ratios > min_area].index]

    return grid


# =============================================================================
# STEP 4: Chop polygons
# =============================================================================


def chop_polygons(
    spdf: gpd.GeoDataFrame,
    grid_spdf: gpd.GeoDataFrame,
    col_identity: List[str],
    col_weight: List[str],
    tol: float = 1e-8,
) -> gpd.GeoDataFrame:
    # Make sure the grid has an 'id' column
    if "id" not in grid_spdf.columns:
        grid_spdf["id"] = grid_spdf.index.astype(str)

    # Create spatial index for the grid
    grid_sindex = grid_spdf.sindex

    # Initialize an empty list to store the results
    results = []

    # Iterate over input polygons with a progress bar
    for _poly_index, poly_row in tqdm(spdf.iterrows(), total=spdf.shape[0], desc="Chopping polygons..."):
        polygon = poly_row.geometry

        # Fix invalid geometries and simplify with a tolerance
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        polygon = polygon.simplify(tol)

        # Skip if the geometry became empty after simplification
        if polygon.is_empty:
            continue

        # Possible matches with grid cells
        possible_matches_index = list(grid_sindex.intersection(polygon.bounds))
        possible_matches = grid_spdf.iloc[possible_matches_index]

        # Check for actual intersection and perform the crop
        for _grid_index, grid_row in possible_matches.iterrows():
            grid_polygon = grid_row.geometry

            if polygon.intersects(grid_polygon):
                intersection = polygon.intersection(grid_polygon)

                # Ensure that the intersection result is valid and not empty
                if not intersection.is_empty and isinstance(intersection, (Polygon, MultiPolygon)):
                    weight = intersection.area / polygon.area

                    # Gather identity columns and weight columns data
                    data = {col: poly_row[col] for col in col_identity}
                    data.update({f"{col}W": poly_row[col] * weight for col in col_weight})
                    data["originalPolyID"] = poly_row[
                        "record"
                    ]  # Assuming 'record' holds the identifier of original polygons
                    data["gridPolyID"] = grid_row["id"]  # 'id' holds the identifier of grid cells

                    # Add area weight and geometry to the data dictionary
                    data["areaWeight"] = weight
                    data["geometry"] = intersection

                    # Create a unique ID for the resulting polygon based on the original and grid IDs
                    data["outID"] = f"{data['originalPolyID']}-{data['gridPolyID']}"

                    results.append(data)

    # Convert the results to a GeoDataFrame
    chopped_gdf = gpd.GeoDataFrame(
        results,
        columns=list(results[0].keys())
        if results
        else col_identity + col_weight + ["originalPolyID", "gridPolyID", "areaWeight", "geometry", "outID"],
    )

    # Preserve the CRS of the input GeoDataFrame
    chopped_gdf.crs = spdf.crs

    # Return the chopped GeoDataFrame
    return chopped_gdf


# =============================================================================
# STEP 5: Aggregate chopped polygons
# =============================================================================


def aggregate_polygons(
    spdf: gpd.GeoDataFrame,
    grid_spdf: gpd.GeoDataFrame,
    col_names: List[str],
    col_funcs: List[Union[str, Callable]],
    by: List[str],
    min_area: Optional[float] = 0.0,
) -> gpd.GeoDataFrame:
    """
    Aggregates attributes of polygons within a spatial grid,
    grouped by specified criteria and weighted by area overlap.

    This function intersects two GeoDataFrames to find the overlapping areas and aggregates
    specified attributes based on these areas. The aggregation considers only those parts of the
    polygons that meet a minimum area proportion threshold.

    Args:
        spdf (gpd.GeoDataFrame): The GeoDataFrame containing polygons to be aggregated.
        grid_spdf (gpd.GeoDataFrame): The GeoDataFrame containing the grid to which polygons will be aggregated.
        col_names (List[str]): List of column names whose values will be aggregated.
        col_funcs (List[Union[str, Callable]]): List of functions or function names to be used for aggregation.
                                                These should correspond one-to-one with `col_names`.
        by (List[str]): List of column names in `spdf` to group by during aggregation.
        min_area (Optional[float]): The minimum proportion of area overlap for the aggregation to be considered valid.
                          Values should be between 0 and 1.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with the grid polygons as geometries and aggregated values as attributes.
    """
    min_area = max(min(min_area, 1), 0)

    # Reset index to ensure that index columns will be present after overlay
    spdf = spdf.reset_index().rename(columns={"index": "index_spdf"})
    grid_spdf = grid_spdf.reset_index().rename(columns={"index": "index_grid"})

    # Step 1: Calculate the intersections (overlays) between spdf and grid_spdf
    intersections = gpd.overlay(spdf, grid_spdf, how="intersection")
    intersections["area"] = intersections.geometry.area

    # Step 2: Calculate the proportion of the area of spdf that is covered by each intersection
    intersections = intersections.merge(
        spdf[["index_spdf", "geometry"]].rename(columns={"geometry": "geometry_spdf"}),
        left_on="index_spdf",
        right_on="index_spdf",
        how="left",
    )
    intersections["area_proportion"] = intersections["area"] / intersections["geometry_spdf"].area

    # Drop rows that don't meet the minimum area condition
    intersections = intersections[intersections["area_proportion"] >= min_area]

    # Step 3: Calculate the effective area weighted values for each attribute
    aggregation_dict = {col: "sum" for col in col_names}
    aggregation_dict["area_proportion"] = "sum"  # Ensure we sum the 'area' as well

    for col, _func in zip(col_names, col_funcs):
        intersections[col] = intersections[col] * intersections["area_proportion"]

    # Step 4: Aggregate the weighted values by the specified grouping columns
    aggregated_data = intersections.groupby(by).agg(aggregation_dict).reset_index()

    # Step 5: Merge the aggregated data back to the grid
    aggregated_gdf = grid_spdf.merge(aggregated_data, how="left", left_on="id", right_on=by)

    # Step 6: Scale the aggregated values up if necessary
    mean_grid_area = grid_spdf.geometry.area.mean()
    for col in col_names:
        upscaled_col_name = f"{col}Up"
        aggregated_gdf[upscaled_col_name] = aggregated_gdf[col] * (
            mean_grid_area / (aggregated_gdf["area_proportion"] * mean_grid_area)
        )

    # Drop columns that are no longer needed to clean up the DataFrame
    drop_cols = ["id", "index_grid", "gridPolyID"]
    aggregated_gdf.drop(columns=drop_cols, inplace=True)
    return aggregated_gdf.dropna(subset=col_names)


# =============================================================================
# STEP 6: Smooth polygons
# =============================================================================


def smooth_polygons(
    spdf: gpd.GeoDataFrame,
    formula: str,
    spdf_pred: gpd.GeoDataFrame = None,
    col_identity: Optional[list] = None,
) -> gpd.GeoDataFrame:
    """
    Smooths the polygons using Kriging.

    Args:
    spdf (gpd.GeoDataFrame): The input GeoDataFrame.
    formula (str): The formula for the variogram.
    spdf_pred (gpd.GeoDataFrame, optional): The GeoDataFrame for prediction. Defaults to None.
    col_identity (list, optional): Columns to preserve in the output. Defaults to None.
    n_cores (int, optional): Number of cores to use for parallelization. Defaults to 1.
    **kwargs: Additional arguments for the Kriging function.

    Returns:
    gpd.GeoDataFrame: The smoothed polygons.
    """
    if spdf_pred is None:
        spdf_pred = spdf

    crs = spdf.crs

    # Transform to WGS 84 for Kriging
    spdf = spdf.to_crs(CRS.from_epsg(4326))
    spdf_pred = spdf_pred.to_crs(CRS.from_epsg(4326))

    # Extracting the dependent and independent variables from the formula
    y_var, x_var = formula.split("~")
    y_var = y_var.strip()
    x_vars = [var.strip() for var in x_var.split("+")]

    # Prepare data for Kriging
    x = spdf.geometry.x
    y = spdf.geometry.y
    z = spdf[y_var].values
    x_pred = spdf_pred.geometry.x
    y_pred = spdf_pred.geometry.y

    # Perform Ordinary Kriging
    ok = OrdinaryKriging(x, y, z, variogram_model="spherical", **kwargs)
    z_pred, ss = ok.execute("points", x_pred, y_pred)

    # Constructing the result GeoDataFrame
    geometry = [Point(xp, yp) for xp, yp in zip(x_pred, y_pred)]
    result_df = pd.DataFrame({y_var: z_pred}, index=spdf_pred.index)
    result_gdf = gpd.GeoDataFrame(result_df, geometry=geometry, crs=CRS.from_epsg(4326))

    # Transform back to original projection
    result_gdf = result_gdf.to_crs(crs)
    spdf = spdf.to_crs(crs)
    spdf_pred = spdf_pred.to_crs(crs)

    if col_identity is not None:
        for col in col_identity:
            result_gdf[col] = spdf_pred[col]

    return result_gdf


# =============================================================================
# STEP 7: post-processing polygons
# =============================================================================
