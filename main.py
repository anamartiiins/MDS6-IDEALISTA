import pandas as pd
import sys
import warnings
from src.data_extraction.data_extraction import extract_initial_data
from src.preprocessing.preprocessing_o import dataset_preprocessing

print(sys.executable)
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    # Extract all dataframes available
    df_assets, df_ine, df_osm, df_pois, df_polygons = extract_initial_data(
        root_dir="data"
    )

    df = dataset_preprocessing(df_assets=df_assets, df_polygons=df_polygons)
