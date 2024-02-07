import pandas as pd
import os


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


def extract_initial_data(root_dir: str) -> pd.DataFrame:
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
