"""Definition of workflows used by this project."""

from pathlib import Path

from ritas import ColNames
from ritas.io import read_input, rectify_input, write_geotiff
from ritas.polygons import make_vehicle_polygons, reshape_polygons


def simple_workflow(infile: Path, outfile: Path, **kwargs: dict) -> None:
    """A simple workflow converting the input file to the output file.

    Args:
        infile (Path): The input file to copy.
        outfile (Path): The output file to create.
        **kwargs (dict): Additional keyword arguments.
        swath_width (float): The width (m) of the swath.
    """
    # Read input
    df = read_input(infile)
    if (sw := kwargs.get("swath_width")) is not None:
        df[ColNames.SWATH] = sw
    # Rectify input
    df = rectify_input(df, **kwargs)
    # Create vehicle polygons
    geodf = make_vehicle_polygons(df, "EPSG:26915")
    # Reshape polygons
    reshaped_geodf = reshape_polygons(geodf)
    # output is based on filename extension
    if outfile.suffix == ".tiff":
        write_geotiff(reshaped_geodf, outfile)
    else:
        reshaped_geodf.to_file(outfile)
