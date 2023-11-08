import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ritas.polygons import make_vehicle_polygons
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


if __name__ == "__main__":
    main()
