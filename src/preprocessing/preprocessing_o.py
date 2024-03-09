import pandas as pd
from src.constants import (
    NEW_COLUMNS_NAMES,
    REMOVE_COLUMNS_BY_INPUT,
    NUM_VARIABLES_TO_SEE_DISTRIBUTION,
    BINARY_VARIABLES,
)

from ydata_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def generate_pandas_profiling_report(df: pd.DataFrame, save_path=None):
    # Get the directory of the currently executing script
    script_dir = os.path.dirname(__file__)

    # Set the default save path
    if save_path is None:
        save_path = os.path.join(script_dir, "..", "data_visualization")

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Generate the profile report
    profile = ProfileReport(df)

    # Save the report to an HTML file in the specified path
    report_path = os.path.join(save_path, "profile_report.html")
    profile.to_file(report_path)
    print(f"Profile report saved to: {report_path}")


def find_single_value_columns(df: pd.DataFrame) -> list:
    single_value_columns = []
    for column in df.columns:
        if df[column].nunique() == 1:
            single_value_columns.append(column)

    print("Columns with only one distinct value:", single_value_columns)

    return single_value_columns


def visualize_distribution(
    df: pd.DataFrame, save_path=None, numerical_columns: list = None
):
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = df.select_dtypes(include=["object"]).columns

    # Grid of distribution plots for numerical variables
    if len(numerical_columns) > 0:
        f = pd.melt(df, value_vars=numerical_columns)
        g = sns.FacetGrid(f, col="variable", col_wrap=4, sharex=False, sharey=False)
        g.map(sns.histplot, "value")

        if save_path:
            plt.savefig(os.path.join(save_path, "distribution_plots.png"))
        else:
            plt.show()
        plt.clf()  # Clear the plot after saving or displaying

    # Plot counts of categorical variables
    for column in categorical_columns:
        plt.figure(figsize=(12, 8))
        sns.countplot(data=df, x=column)
        plt.title(f"Count of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")

        if save_path:
            plt.savefig(os.path.join(save_path, f"{column}_count.png"))
        else:
            plt.show()
        plt.clf()  # Clear the plot after saving or displaying


def convert_binary_to_categorical(df: pd.DataFrame, binary_columns: list):
    for column in binary_columns:
        # Check if the column exists in the DataFrame and is binary
        if column in df.columns and len(df[column].unique()) == 2:
            # Convert the column to categorical dtype if it's not already categorical
            if df[column].dtype != "category":
                df[column] = df[column].astype("category")
            else:
                print(f"Column '{column}' is already categorical.")
        else:
            print(
                f"Column '{column}' is not binary or does not exist in the DataFrame."
            )
    return df


def visualize_binary_distribution(df, binary_columns, save_path=None):
    # Create a single FacetGrid for all binary columns with adjusted size and spacing
    g = sns.FacetGrid(
        df.melt(value_vars=binary_columns),
        col="variable",
        col_wrap=3,
        height=5,
        aspect=1.5,
    )

    # Map count plots onto the FacetGrid
    g.map(sns.countplot, "value", palette="Set2")

    # Set titles above each plot and adjust font size
    g.set_titles(row_template="{row_name}", fontsize=8, pad=10)

    # Set x and y labels for each plot and adjust font size
    g.set_axis_labels("Class", "Perc", fontsize=8)

    # Iterate over each subplot to annotate the percentage of each class and set x-axis labels
    for ax, title in zip(g.axes.flat, binary_columns):
        # Calculate the total count for the column
        total = len(df[title])

        # Calculate and annotate the percentage of each class
        for p in ax.patches:
            percentage = "{:.1f}%".format(100 * p.get_height() / total)
            rounded_percentage = round(100 * p.get_height() / total)
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.text(
                x, y, f"{rounded_percentage}%", ha="center", va="bottom", fontsize=8
            )

    # Adjust layout to prevent overlap and increase space between rows
    plt.tight_layout(pad=3.0)

    # Save the plot if save_path is provided
    if save_path is None:
        script_dir = os.path.dirname(__file__)
        save_path = os.path.join(script_dir, "..", "data_visualization")

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the plot
    save_file_path = os.path.join(save_path, "binary_distribution_plots.png")
    plt.savefig(save_file_path)
    plt.close()
    print("Image saved successfully at:", save_file_path)


def correlation_values(df: pd.DataFrame, save_path: str = None, threshold: float = 0.8):
    # Select numerical columns
    df_numerical_columns = df.select_dtypes(include=["int64", "float64"])

    # Calculate correlation matrix
    correlation_matrix = df_numerical_columns.corr()

    # Find variable pairs with correlation greater than the threshold
    correlated_variables = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                correlated_variables.append(
                    (
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j],
                    )
                )

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    # Increase the size of the heatmap
    plt.figure(figsize=(10, 8))  # Adjust dimensions as needed

    # Plot the heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        mask=mask,
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # FIXME: não está a guardar direito. validar
    # Save the plot if save_path is provided
    if save_path is None:
        script_dir = os.path.dirname(__file__)
        save_path = os.path.join(script_dir, "..", "data_visualization")

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the plot
    save_file_path = os.path.join(save_path, "correlation_matrix.png")
    plt.savefig(save_file_path)
    plt.close()

    # Print correlated variable pairs with their correlation coefficients
    for pair in correlated_variables:
        print(f"Correlated variables: {pair[0]}, {pair[1]}, Correlation: {pair[2]}")

    print(
        "After evaluate which columns remove by coorelations, update list in constants REMOVE_COLUMNS_BY_CORRELATIONS"
    )

    return correlation_matrix, correlated_variables


def detect_outliers(df, column_name, threshold=10):
    """
    Detect outliers in a DataFrame based on a specific column using the IQR (Interquartile Range) method.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to detect outliers.
        threshold (float): The threshold multiplier for determining outliers. Default is 10.

    Returns:
        pandas.DataFrame: A DataFrame containing the outlier observations.
    """
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)

    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds for outlier detection
    lower_bound = Q1 - (threshold * IQR)
    upper_bound = Q3 + (threshold * IQR)

    # Detect and return the outliers
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]

    return outliers


def standardize_variables(df: pd.DataFrame, columns: list):
    ## TODO: define which variables do we should standardize and apply it in preprocessing.

    # Step 1: Standardize the target variable "precio" using Min-Max scaling
    scaler = MinMaxScaler()
    for col in columns:
        df[col + "_scaled"] = scaler.fit_transform(df[[col]])

    df = df.drop(colums=[columns])

    # # Step 2: Apply the inverse transformation to revert the standardized values back to the original scale
    # for col in columns:
    #     df[col] = scaler.inverse_transform(df[[col+'_scaled']])

    return df


##FIXME
def get_location_name_w_gdf(df_assets: pd.DataFrame, df_polygons: pd.DataFrame):
    from shapely import wkt
    from shapely.geometry import Point
    import geopandas as gpd

    # Convert the column WKT to geometry using geopandas
    df_polygons["geometry"] = df_polygons["WKT"].apply(wkt.loads)
    gdf_polygons = gpd.GeoDataFrame(df_polygons, geometry="geometry")

    # Convert the LATITUDE and LONGITUDE coordinates from df_assets into geometric points
    df_assets["geometry"] = df_assets.apply(
        lambda row: Point(row["longitud"], row["latitud"]), axis=1
    )

    # Create a GeoDataFrame for df_assets
    gdf_sales = gpd.GeoDataFrame(df_assets, geometry="geometry")


def hist_plot_outliers(df, name_variable):
    # Create a new figure and axis for each plot
    fig, ax = plt.subplots()

    # Plot histogram with automatic bins
    ax.hist(df[name_variable], bins="auto", edgecolor="black")

    # Adding labels and title
    ax.set_xlabel(name_variable)
    ax.set_ylabel("Count")
    ax.set_title("Histogram of " + name_variable)

    # Display the plot
    plt.show()


def dataset_preprocessing(
    df_assets: pd.DataFrame,
    df_polygons: pd.DataFrame,
    pandas_profiling: bool = False,
    distributions: bool = False,
) -> pd.DataFrame:
    print("Start")

    # FIXME
    aux = pd.read_csv(
        r"C:\Users\aimartins\OneDrive - Parfois, SA\Documents\GitHub\MDS6-IDEALISTA\data_spatial.csv"
    )
    aux = aux.drop(
        columns=["geometry", "index_right", "LOCATIONID", "WKT", "ZONELEVELID"]
    )

    df_assets = aux.copy()

    # Change columns names to friendly ones
    df_assets = df_assets.drop(columns=["ADTYPOLOGY", "ADOPERATION"])
    df_assets.columns = NEW_COLUMNS_NAMES

    # See general statistics of df
    description_df = df_assets.describe().transpose()

    # Remove duplicated anuncios_id
    df_assets = remove_duplicated_anuncios_id(df_assets=df_assets, criteria="last")

    # Remove columns by input
    df_assets = df_assets.drop(columns=REMOVE_COLUMNS_BY_INPUT)

    # Remove columns that only have one different value
    remove_unique_value_columns = find_single_value_columns(df=df_assets)
    df_assets = df_assets.drop(columns=remove_unique_value_columns)

    # Missing values
    remove_columns_by_missing_values, df_assets = treatment_missing_values(df=df_assets)

    # Feature Engineering: add new variables
    (
        add_columns,
        remove_columns_by_creating_new_variables,
        df_assets,
    ) = feature_engineering(df=df_assets)

    # Correlation values
    (
        remove_columns_by_correlations,
        df_assets,
        correlation_matrix,
        correlated_variables,
    ) = correlation_values(df=df_assets)

    # See distributions
    if distributions:
        visualize_distribution(
            df=df_assets, numerical_columns=NUM_VARIABLES_TO_SEE_DISTRIBUTION
        )

        visualize_binary_distribution(df=df_assets, binary_columns=BINARY_VARIABLES)

    # Run pandas profiling
    if pandas_profiling:
        generate_pandas_profiling_report(df=df_assets)

    # Export csv to help in PBI
    df_assets.to_csv("df_assets_final.csv")

    ## FIXME: code section to validate some questions with group
    # Validate if one house has multiple status: in theory, a house only can be in one of this status.
    (
        df_assets["nueva_construccion"]
        + df_assets["a_reformar"]
        + df_assets["buen_estado"]
    ).value_counts()

    # Validate if a house has multiple orientations, and if there are anuncios with no orientation.
    # 0 - 34811, 1-20903, 2-11457, 3-1030, 4-524
    (
        df_assets["orientacion_n"]
        + df_assets["orientacion_s"]
        + df_assets["orientacion_o"]
        + df_assets["orientacion_e"]
    ).value_counts()

    # Validate: does it make sense that: if parking = 0, parking price > 0 ?
    df_assets[(df_assets["parking"] == 0) & (df_assets["precio_parking"] > 0)][
        "precio_parking"
    ].count()

    # Validate maximum value of antiguity
    a = df_assets[df_assets["antiguidade"] == df_assets["antiguidade"].max()]
    df_assets["antiguidade"].plot.hist(bins=30, color="skyblue", edgecolor="black")

    # Validate maximum value of cat_n_vecinos
    a = df_assets[df_assets["cat_n_vecinos"] == df_assets["cat_n_vecinos"].max()]

    # Validate minimum and maximum value of n_habitaciones
    a = df_assets[
        (df_assets["n_habitaciones"] == df_assets["n_habitaciones"].min())
        | (df_assets["n_habitaciones"] == df_assets["n_habitaciones"].max())
    ]

    # Validate minimum and maximum value of n_banos
    a = df_assets[
        (df_assets["n_banos"] == df_assets["n_banos"].min())
        | (df_assets["n_banos"] == df_assets["n_banos"].max())
    ]

    return df_assets
