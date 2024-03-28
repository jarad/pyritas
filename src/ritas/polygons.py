import warnings
from typing import Callable, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import (
    MultiPolygon,
    Polygon,
    box,
)
from tqdm import tqdm

from ritas import LOG, ColNames
from ritas.utils import (
    lognormal_to_normal,
    yield_equation_mgha,
)

warnings.filterwarnings("ignore")


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

    LOG.info("Running make_bounding_box...")

    # prepare the input arrays
    x0, x1 = np.r_[np.nan, x[:-1]], x
    y0, y1 = np.r_[np.nan, y[:-1]], y

    # Compute distance of the vertices to the centroid
    h = 0.5 * w  # half width
    m = np.where(
        (x1 - x0) != 0, (y1 - y0) / (x1 - x0), np.inf
    )  # division by zero
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
        np.column_stack(
            (x0, y0, x1, y1, x01, y01, x02, y02, x11, y11, x12, y12)
        ),
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


def make_vehicle_polygons(
    geodf: gpd.GeoDataFrame,
    proj4string: str,
) -> gpd.GeoDataFrame:
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
    if not isinstance(proj4string, str):
        raise ValueError("The proj4string must be a string.")

    try:
        CRS(proj4string)
    except CRSError as e:
        raise ValueError(f"Invalid proj4string: {e}") from e

    # Compute vertices of the rectangles for each coordinate
    bounding_box = make_bounding_box(
        geodf.geometry.x,
        geodf.geometry.y,
        geodf[ColNames.SWATH].values,
        geodf[ColNames.DISTANCE].values,
    )

    df = pd.concat([geodf, bounding_box], axis=1)

    df = df.dropna(
        subset=["x01", "y01", "x02", "y02", "x11", "y11", "x12", "y12"]
    )

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


def check_polygon_type(
    geom: Union[Polygon, MultiPolygon],
) -> Union[Polygon, MultiPolygon, None]:
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if geom.is_empty:
        return None
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
        if polys:
            return MultiPolygon(polys) if len(polys) > 1 else polys[0]
    return None


def clean_polygon(
    poly: Union[Polygon, MultiPolygon],
) -> Union[Polygon, MultiPolygon, None]:
    if not poly.is_valid:
        poly = poly.buffer(0)
    return check_polygon_type(poly)


def crop_polygon(
    sp_from: Polygon,
    sp_to: Polygon,
) -> Union[Polygon, MultiPolygon, None]:
    sp_from = clean_polygon(sp_from)
    sp_to = clean_polygon(sp_to)
    if sp_from is None or sp_to is None:
        return None
    result = sp_from.difference(sp_to)
    return check_polygon_type(result)


def process_single_polygon(
    poly: Polygon,
    reference_polygons: list[Polygon],
) -> list[Polygon]:
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


def reshape_polygons(
    spdf: gpd.GeoDataFrame, verbose: bool = True
) -> gpd.GeoDataFrame:
    spdf = spdf.copy()
    spdf["geometry"] = spdf["geometry"].apply(clean_polygon)
    spdf = spdf.dropna(subset=["geometry"])
    spdf = spdf.explode(index_parts=True).reset_index(drop=True)
    spdf["geometry"] = spdf["geometry"].simplify(tolerance=0.01)
    sindex = spdf.sindex
    n_last = len(spdf) - 1

    for i in tqdm(
        range(n_last), disable=not verbose, desc="Reshaping polygons..."
    ):
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
                LOG.error(
                    "Error processing geometry at index %s with exception: %s",
                    idx,
                    e,
                )
                continue

    # Merge adjacent and overlapping polygons
    spdf["geometry"] = spdf["geometry"].buffer(0)
    spdf = spdf.explode(index_parts=True).reset_index(drop=True)

    # Calculate effective area
    spdf["effectiveArea"] = spdf["geometry"].area
    return spdf


def make_grid_by_size(
    spdf: gpd.GeoDataFrame,
    width: float,
    height: float,
) -> gpd.GeoDataFrame:
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
        Polygon(
            [
                (x, y),
                (x + grid_width, y),
                (x + grid_width, y + grid_height),
                (x, y + grid_height),
            ],
        )
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
        raise ValueError(
            "Either `n`, or `width` and `height`, must be non-null."
        )

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


def chop_polygons(
    spdf: gpd.GeoDataFrame,
    grid_spdf: gpd.GeoDataFrame,
    col_identity: list[str],
    col_weight: list[str],
    tol: float = 1e-8,
    min_intersection_area: float = 1e-4,  # Minimum intersection area threshold
) -> gpd.GeoDataFrame:
    # Make sure the grid has an 'id' column
    if "id" not in grid_spdf.columns:
        grid_spdf["id"] = grid_spdf.index.astype(str)

    # Create spatial index for the grid
    grid_sindex = grid_spdf.sindex

    # Initialize an empty list to store the results
    results = []

    # Iterate over input polygons with a progress bar
    for _poly_index, poly_row in tqdm(
        spdf.iterrows(),
        total=spdf.shape[0],
        desc="Chopping polygons...",
    ):
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

                if not intersection.is_empty and isinstance(
                    intersection, (Polygon, MultiPolygon)
                ):
                    intersection_area = intersection.area

                    if intersection_area < min_intersection_area:
                        continue

                    weight = intersection_area / polygon.area
                    data = {col: poly_row[col] for col in col_identity}
                    data.update(
                        {
                            f"{col}W": poly_row[col] * weight
                            for col in col_weight
                        }
                    )
                    data["originalPolyID"] = poly_row["record"]
                    data["gridPolyID"] = grid_row["id"]
                    data["areaWeight"] = weight
                    data["geometry"] = intersection
                    data["outID"] = (
                        f"{data['originalPolyID']}-{data['gridPolyID']}"
                    )

                    results.append(data)

    chopped_gdf = gpd.GeoDataFrame(
        results,
        columns=list(results[0].keys())
        if results
        else col_identity
        + col_weight
        + ["originalPolyID", "gridPolyID", "areaWeight", "geometry", "outID"],
    )
    chopped_gdf.crs = spdf.crs

    # Return the chopped GeoDataFrame
    return chopped_gdf


def aggregate_polygons(
    spdf: gpd.GeoDataFrame,
    grid_spdf: gpd.GeoDataFrame,
    col_names: list[str],
    col_funcs: list[Union[str, Callable]],
    by: list[str],
    min_area: Optional[float] = 0.0,
    min_area_proportion_threshold: float = 1e-6,  # New parameter to avoid too small area proportions
) -> gpd.GeoDataFrame:
    """
    Aggregates attributes of polygons within a spatial grid, grouped by specified criteria
    and weighted by area overlap. Adjusted to handle small area proportions.

    Args:
        spdf (gpd.GeoDataFrame): The GeoDataFrame containing polygons to be aggregated.
        grid_spdf (gpd.GeoDataFrame): The GeoDataFrame containing the grid for aggregation.
        col_names (list[str]): List of column names for aggregation.
        col_funcs (list[Union[str, Callable]]): Functions for aggregation.
        by (list[str]): Column names to group by during aggregation.
        min_area (Optional[float]): Minimum area overlap for valid aggregation.
        min_area_proportion_threshold (float): Threshold to avoid too small area proportions.

    Returns:
        gpd.GeoDataFrame: Aggregated GeoDataFrame.
    """
    min_area = max(min(min_area, 1), 0)

    spdf = spdf.reset_index().rename(columns={"index": "index_spdf"})
    grid_spdf = grid_spdf.reset_index().rename(columns={"index": "index_grid"})

    # Step 1: Calculate intersections
    intersections = gpd.overlay(spdf, grid_spdf, how="intersection")
    intersections["area"] = intersections.geometry.area

    # Step 2: Calculate area proportion
    intersections = intersections.merge(
        spdf[["index_spdf", "geometry"]].rename(
            columns={"geometry": "geometry_spdf"}
        ),
        left_on="index_spdf",
        right_on="index_spdf",
        how="left",
    )
    intersections["area_proportion"] = (
        intersections["area"] / intersections["geometry_spdf"].area
    )
    intersections["area_proportion"] = intersections["area_proportion"].clip(
        lower=min_area_proportion_threshold
    )

    # Exclude small intersections
    intersections = intersections[intersections["area_proportion"] >= min_area]

    # Step 3: Weighted value calculation
    aggregation_dict = {col: "sum" for col in col_names}
    aggregation_dict["area_proportion"] = "sum"

    for col, _func in zip(col_names, col_funcs):
        intersections[col] = (
            intersections[col] * intersections["area_proportion"]
        )

    # Step 4: Aggregate values
    aggregated_data = (
        intersections.groupby(by).agg(aggregation_dict).reset_index()
    )

    # Step 5: Merge with grid
    aggregated_gdf = grid_spdf.merge(
        aggregated_data,
        how="left",
        left_on="id",
        right_on=by,
    )

    # Step 6: Scale the aggregated values up if necessary
    mean_grid_area = grid_spdf.geometry.area.mean()
    for col in col_names:
        upscaled_col_name = f"{col}Up"
        aggregated_gdf[upscaled_col_name] = aggregated_gdf[col] * (
            mean_grid_area
            / (aggregated_gdf["area_proportion"] * mean_grid_area)
        )

    # Drop unnecessary columns
    drop_cols = ["id", "index_grid", "gridPolyID"]
    aggregated_gdf.drop(columns=drop_cols, inplace=True)

    return aggregated_gdf.dropna(subset=col_names)


def smooth_polygons(
    spdf: gpd.GeoDataFrame,
    formula: str,
    spdf_pred: gpd.GeoDataFrame = None,
) -> gpd.GeoDataFrame:
    """
    Smooths the polygons using Kriging with an adapted variogram model.

    Args:
        spdf (gpd.GeoDataFrame): The input GeoDataFrame.
        formula (str): The formula for the variogram.
        spdf_pred (gpd.GeoDataFrame, optional): The GeoDataFrame for prediction. Defaults to None.
        col_identity (list, optional): Columns to preserve in the output. Defaults to None.

    Returns:
        gpd.GeoDataFrame: The smoothed polygons.
    """
    if spdf_pred is None:
        spdf_pred = spdf

    # Extracting the dependent and independent variables from the formula
    y_var, _ = formula.split("~")
    y_var = y_var.strip()

    # Apply transformation if necessary
    if "log(" in y_var:
        variable_name = y_var.split("(")[1].strip(")")

        if variable_name not in spdf.columns:
            raise ValueError(
                f"Column '{variable_name}' does not exist in the input GeoDataFrame."
            )

        if spdf[variable_name].isna().any():
            raise ValueError(f"Column '{variable_name}' contains NA values.")

        spdf[variable_name] = np.log(spdf[variable_name])
        y_var = variable_name

    # Prepare data for Kriging using centroids of polygons
    centroids = spdf.geometry.centroid
    x = centroids.x
    y = centroids.y
    z = spdf[y_var].values

    centroids_pred = spdf_pred.geometry.centroid
    x_pred = centroids_pred.x
    y_pred = centroids_pred.y

    # Fit the variogram model and perform Ordinary Kriging
    # NOTE: This is a simple approach. For more complex data, consider using automated variogram fitting
    # Assuming 'z' contains log-transformed values
    ok = OrdinaryKriging(
        x,
        y,
        z,
        variogram_model="spherical",
        enable_plotting=False,
        nlags=20,
        verbose=True,
        weight=True,
        coordinates_type="geographic",
    )

    z_pred, ss = ok.execute("points", x_pred, y_pred)

    # Create the result GeoDataFrame
    result_gdf = spdf_pred.copy()
    result_gdf["logMassKgMean"] = z_pred
    result_gdf["logMassKgVar"] = ss  # Semivariance as a proxy for variance

    # Convert lognormal mean and variance to normal scale
    result_gdf["massKgMean"] = lognormal_to_normal(
        result_gdf["logMassKgMean"],
        result_gdf["logMassKgVar"],
        "mean",
    )
    result_gdf["massKgVar"] = lognormal_to_normal(
        result_gdf["logMassKgMean"],
        result_gdf["logMassKgVar"],
        "var",
    )

    # Calculate yield in mg/ha
    result_gdf["yieldMgHaMean"] = yield_equation_mgha(
        result_gdf["massKgMean"],
        result_gdf["effectiveAreaWUp"],
    )
    result_gdf["yieldMgHaVar"] = (
        yield_equation_mgha(1, result_gdf["effectiveAreaWUp"]) ** 2
        * result_gdf["massKgVar"]
    )

    return result_gdf
