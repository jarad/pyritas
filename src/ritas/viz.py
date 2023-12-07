import logging
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

project_dir = Path(__file__).resolve().parents[0].parents[0].parents[0]

green_cm = LinearSegmentedColormap.from_list("green_cm", ["lightgreen", "green", "darkgreen"], N=256)


def plot_map(
    gdf: gpd.GeoDataFrame,
    column: Optional[str] = None,
    cmap: str = green_cm,
    fname: Optional[str] = None,
    axis_off: bool = True,
    edge_color: str = "green",
    figsize: Tuple[float, float] = (20, 20),
    dpi: int = 300,
    title: Optional[str] = None,
    number: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot and save a GeoDataFrame boundary map with filled colors based on specified column values.

    Parameters:
    - column (str): The column name to base the colors on.
    - cmap (str): The colormap for the plot.
    - ... [other parameters are unchanged]
    """
    # Ensure the input is a GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("gdf must be a GeoDataFrame.")

    # Log the beginning of the plotting process
    logging.info("Creating plot...")

    # Create the plot if no Axes is passed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig_created = True
    else:
        fig = ax.figure
        fig_created = False

    # Plot the GeoDataFrame with color based on the specified column
    if column is None:
        gdf.plot(ax=ax, edgecolor=edge_color, linewidth=0.8, facecolor="none")
    else:
        gdf.plot(ax=ax, column=column, cmap=cmap, edgecolor=edge_color, linewidth=0.8)

    # Annotate each polygon with its index
    if number:
        for idx, row in gdf.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(
                text=str(idx),
                xy=(centroid.x, centroid.y),
                ha="center",
                fontsize=9,
                color="white",  # Changed to white for visibility on potentially dark colors
            )

    # Optional title
    if title:
        ax.set_title(title, fontdict={"fontsize": "15", "fontweight": "3"})

    # Remove axis if required
    if axis_off:
        ax.set_axis_off()

    # Save the figure if a filename is provided and a new figure was created
    if fname is not None and fig_created:
        fig_path = project_dir / "plots" / fname
        fig.savefig(fig_path, dpi=dpi)
        logging.info("Plot saved to %s", fig_path)

    return fig, ax


def plot_coordinates(data: pd.DataFrame) -> plt:
    """
    Plots latitude and longitude coordinates.

    Args:
        data (list of dict): List of dictionaries containing lat and long coordinates.

    Returns:
        None
    """
    # Extract latitude and longitude data from the input data, while checking for 'x' and 'y' keys
    latitudes = data["x"]
    longitudes = data["y"]

    # Create a scatter plot of the coordinates
    plt.figure(figsize=(10, 8))
    plt.scatter(
        longitudes,
        latitudes,
        marker="o",
        c="blue",
        label="Coordinates",
        alpha=0.5,
    )

    # Set labels and title
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Latitude vs Longitude")

    # Add a grid
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    return plt
