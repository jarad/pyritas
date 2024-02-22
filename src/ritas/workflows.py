"""Definition of workflows used by this project."""
from pathlib import Path

from ritas.io import read_input
from ritas.polygons import make_grid, make_vehicle_polygons, reshape_polygons


def simple_workflow(infile: Path, outfile: Path) -> None:
    """A simple workflow converting the input file to the output file.

    Args:
        infile (Path): The input file to copy.
        outfile (Path): The output file to create.
    """
    # Read input
    df = read_input(infile)
    # Create vehicle polygons
    geodf = make_vehicle_polygons(df, "EPSG:26915")
    # Reshape polygons
    reshaped_geodf = reshape_polygons(geodf)
    # Write output
    grid = make_grid(reshaped_geodf, width=5, height=5)
    grid.to_file(outfile)
