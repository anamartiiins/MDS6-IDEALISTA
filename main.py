import pandas as pd
pd.set_option("display.max_columns", None)
from src.data_extraction.data_extraction import extract_initial_data


if __name__ == "__main__":

    #Extract all dataframes available
    df_assets, df_ine, df_osm, df_pois, df_polygons = extract_initial_data(
        root_dir="data"
    )
