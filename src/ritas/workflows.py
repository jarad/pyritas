"""Definition of workflows used by this project."""

from pathlib import Path

from ritas import ColNames
from ritas.io import read_input, write_geotiff
from ritas.polygons import make_grid, make_vehicle_polygons, reshape_polygons


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
    # Create vehicle polygons
    geodf = make_vehicle_polygons(df, "EPSG:26915")
    # Reshape polygons
    reshaped_geodf = reshape_polygons(geodf)
    # Write output
    grid = make_grid(reshaped_geodf, width=5, height=5)
    # output is based on filename extension
    if outfile.endswith(".tiff"):
        write_geotiff(grid, outfile)
    else:
        grid.to_file(outfile)
