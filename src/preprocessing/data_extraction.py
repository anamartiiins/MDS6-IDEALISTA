import pandas as pd
import os
import numpy as np

# Specific imports
import geopandas
from geopandas import GeoDataFrame
from shapely import wkt
from shapely.geometry import Point


def read_csv_zip(root_dir: str, subfolder_name: str, is_zip: bool) -> pd.DataFrame:
    """
    Read CSV files from a specified subfolder within a root directory.

    Parameters:
        root_dir (str): The root directory containing the subfolder.
        subfolder_name (str): The name of the subfolder containing CSV files.
        is_zip (bool): Indicates whether the CSV files are gzip-compressed or not.

    Returns:
        pd.DataFrame: A DataFrame containing concatenated data from all CSV files.

    Example:
        # Read CSV files from the 'assets' subfolder, where files are gzip-compressed
        df_assets = read_csv_zip(root_dir='data', subfolder_name='assets', is_zip=True)
    """
    # Create an empty list to store DataFrames
    dataframes = []

    # Traverse through each subfolder starting from 'assets' folder
    for subdir, dirs, files in os.walk(root_dir):
        # Check if the current subdirectory is the specified subfolder
        if os.path.basename(subdir) == subfolder_name:
            # Iterate through each file in the current subfolder
            for file in files:
                # Check if the file is a gzipped CSV file based on is_zip flag
                if is_zip and file.endswith(".csv.gz"):
                    # Construct the full path to the file
                    file_path = os.path.join(subdir, file)

                    # Read the gzipped CSV file into a Pandas DataFrame
                    df = pd.read_csv(file_path, compression="gzip", sep=";")

                    # Append the DataFrame to the list
                    dataframes.append(df)
                elif not is_zip and not file.endswith(".csv.gz"):
                    # Construct the full path to the file
                    file_path = os.path.join(subdir, file)

                    # Read the CSV file into a Pandas DataFrame
                    df = pd.read_csv(file_path, sep=";")

                    # Append the DataFrame to the list
                    dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame if needed
    result_df = pd.concat(dataframes, ignore_index=True)

    return result_df


def extract_initial_data(root_dir: str):
    """
    Extract initial data from CSV files in specified subfolders within a root directory.

    Parameters:
        root_dir (str): The root directory containing the subfolders.

    Returns:
        tuple: A tuple containing DataFrames extracted from each subfolder.
            The order of DataFrames is: (df_assets, df_ine, df_osm, df_pois, df_polygons).

    Example:
        # Extract initial data from CSV files in subfolders under the 'data' directory
        df_assets, df_ine, df_osm, df_pois, df_polygons = extract_initial_data(root_dir='data')
    """
    subfolder_info = [
        ("assets", True),
        ("ine", True),
        ("osm", True),
        ("pois", False),
        ("polygons", True),
    ]

    # Extract data for each subfolder and store in separate variables
    dataframes_by_subfolder = {}
    for subfolder_name, is_zip in subfolder_info:
        df_name = f"df_{subfolder_name}"
        dataframes_by_subfolder[df_name] = read_csv_zip(
            root_dir=root_dir, subfolder_name=subfolder_name, is_zip=is_zip
        )
    df_assets, df_ine, df_osm, df_pois, df_polygons = dataframes_by_subfolder.values()

    return df_assets, df_ine, df_osm, df_pois, df_polygons


def get_info_from_polygons_and_ine(df_polygons, df_ine, df):
    # Convert WKT strings to Shapely geometries and create a GeoDataFrame
    df_polygons["geometry"] = df_polygons["WKT"].apply(wkt.loads)
    gdf_polygons = geopandas.GeoDataFrame(df_polygons["geometry"], crs="epsg:4326")

    # Add additional columns to the GeoDataFrame
    gdf_polygons["barrio_id"] = df_polygons["LOCATIONID"]
    gdf_polygons["barrio"] = df_polygons["LOCATIONNAME"]

    # Create Point geometries using longitude and latitude coordinates from df_train
    geometry = [Point(xy) for xy in zip(df.longitud, df.latitud)]

    # Create a GeoDataFrame gdf_ads with df_prices data and geometry column
    gdf_train_train = GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)

    # Apply a logarithmic scale transformation to the 'precio' column in gdf_ads
    gdf_train_train["precio_logaritmico"] = np.log(gdf_train_train["precio"])

    # Convert WKT strings to Shapely geometries and create a GeoDataFrame for census polygons
    df_ine["geometry"] = df_ine["WKT"].apply(wkt.loads)
    gdf_polygons_census = geopandas.GeoDataFrame(df_ine["geometry"], crs="epsg:4326")

    # Add additional column 'CUSEC' to the GeoDataFrame representing census polygons
    gdf_polygons_census["cusec"] = df_ine["CUSEC"]

    # Add the census codes (CUSEC)
    gdf_train_train = geopandas.sjoin(gdf_train_train, gdf_polygons_census, how="inner")

    # Drop index_right
    gdf_train_train = gdf_train_train.drop(columns=["index_right"])

    # Now add the idealista zones (LOCATIONID, LOCATIONNAME)
    gdf_train_train = geopandas.sjoin(gdf_train_train, gdf_polygons, how="inner")

    # Drop index_right
    gdf_train_train = gdf_train_train.drop(columns=["index_right"])

    return gdf_train_train
