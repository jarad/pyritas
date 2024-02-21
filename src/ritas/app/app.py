import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ritas.polygons import (
    chop_polygons,
    make_grid,
    make_vehicle_polygons,
    reshape_polygons,
)
from ritas.viz import plot_coordinates, plot_map


def main() -> None:
    st.title("RITAS")

    with st.form("upload_form"):
        st.header("Upload Data File")
        data_file = st.file_uploader("Choose a CSV file", type="csv")
        proj4 = st.text_input("Enter the proj4 string")
        submit = st.form_submit_button("Submit")

        if submit:
            if proj4 == "":
                st.error("Please enter a proj4 string.")
                return

            # Load data
            data = pd.read_csv(data_file)

            if len(data) == 0:
                st.error("Please upload a non-empty CSV file.")
                return

            # Plot coordinates
            st.header("Plot Coordinates")
            fig = plot_coordinates(data)
            st.pyplot(fig)

            # polygons
            st.header("Polygons")
            with st.spinner("Creating polygons..."):
                polygons = make_vehicle_polygons(data, proj4string=proj4)
                fig = plot_map(polygons)
                st.pyplot(fig)

            st.header("Polygons (After reshaping)")
            with st.spinner("Reshaping polygons..."):
                polygons = reshape_polygons(polygons)
                fig = plot_map(polygons)
                st.pyplot(fig)

            # grid
            st.header("Grid")
            with st.spinner("Creating grid..."):
                grid = make_grid(reshaped_geodf, width=5, height=5)
                fig, ax = plt.subplots(figsize=(20, 10))
                _, ax = plot_map(grid, ax=ax)
                fig, ax = plot_map(reshaped_geodf, ax=ax)
                st.pyplot(fig)

            with st.spinner("Chopping grid..."):
                col_identity = [
                    col
                    for col in reshaped_geodf.columns
                    if col not in ["x", "y"]
                ]
                col_identity.append("effectiveArea")
                col_weight = ["mass", "effectiveArea"]

                chopped = chop_polygons(
                    reshaped_geodf, grid, col_identity, col_weight
                )
                fig = plot_map(chopped)
                st.pyplot(fig)

            with st.spinner("Aggregating..."):
                col_weight = ["mass", "effectiveArea"]
                col_names = [f"{col}W" for col in col_weight]
                col_funcs = [np.sum, np.sum]

                agg_df = aggregate_polygons(
                    chopped, grid, col_names, col_funcs, by=["gridPolyID"]
                )

                fig = plot_map(agg_df, column="massWUp")
                st.pyplot(fig)


if __name__ == "__main__":
    main()
