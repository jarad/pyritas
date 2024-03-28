import argparse
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ritas import LOG
from ritas.polygons import make_vehicle_polygons, reshape_polygons
from ritas.viz import plot_coordinates, plot_map


def main(args: Any) -> None:
    LOG.info("Reading CSV file from %s", args.input_csv)
    df = pd.read_csv(args.input_csv)

    LOG.info("Using Proj4 string: %s", args.proj4string)
    proj4string = args.proj4string

    LOG.info("Plotting coordinates.")
    plot_coordinates(df)

    LOG.info("Creating vehicle polygons.")
    geodf = make_vehicle_polygons(df, proj4string=proj4string)

    LOG.info("Reshaping polygons.")
    reshaped_geodf = reshape_polygons(geodf)

    LOG.info("Creating plots.")
    # Start by creating a single figure and two axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # 1 row, 2 columns

    # Plot on the first axes, ignore the returned figure
    _, ax1 = plot_map(geodf, ax=ax1, title="Original Polygons")
    # Plot on the second axes, ignore the returned figure
    _, ax2 = plot_map(reshaped_geodf, ax=ax2, title="Reshaped Polygons")

    plt.tight_layout()
    LOG.info("Displaying plots.")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and visualize vehicle polygons."
    )
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument(
        "proj4string",
        help="Proj4 string for the coordinate reference system",
    )
    args = parser.parse_args()
    main(args)
