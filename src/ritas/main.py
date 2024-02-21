import argparse
import logging
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ritas.polygons import make_vehicle_polygons, reshape_polygons
from ritas.viz import plot_coordinates, plot_map


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main(args: Any) -> None:
    logging.info("Reading CSV file from %s", args.input_csv)
    df = pd.read_csv(args.input_csv)

    logging.info("Using Proj4 string: %s", args.proj4string)
    proj4string = args.proj4string

    logging.info("Plotting coordinates.")
    plot_coordinates(df)

    logging.info("Creating vehicle polygons.")
    geodf = make_vehicle_polygons(df, proj4string=proj4string)

    logging.info("Reshaping polygons.")
    reshaped_geodf = reshape_polygons(geodf)

    logging.info("Creating plots.")
    # Start by creating a single figure and two axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # 1 row, 2 columns

    # Plot on the first axes, ignore the returned figure
    _, ax1 = plot_map(geodf, ax=ax1, title="Original Polygons")
    # Plot on the second axes, ignore the returned figure
    _, ax2 = plot_map(reshaped_geodf, ax=ax2, title="Reshaped Polygons")

    plt.tight_layout()
    logging.info("Displaying plots.")
    plt.show()


if __name__ == "__main__":
    setup_logging()

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
