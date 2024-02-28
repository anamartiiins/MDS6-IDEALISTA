import pandas as pd
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

    

def remove_duplicated_anuncios_id(
    df_assets: pd.DataFrame, criteria: str = "last"
) -> pd.DataFrame:
    # Identify unique values that are duplicated
    unique_duplicated_anuncio_id = df_assets.loc[
        df_assets.duplicated(subset=["id_anuncio"], keep=False), "id_anuncio"
    ].unique()

    # Create a new DataFrame containing only the duplicated rows, ordered by association ID
    df_duplicates_anuncio_id = df_assets.loc[
        df_assets["id_anuncio"].isin(unique_duplicated_anuncio_id)
    ].sort_values(by="id_anuncio")

    # Validate if duplicated 'id_anuncio' values are associated with the same 'fecha' value
    validation_result = df_duplicates_anuncio_id.groupby("id_anuncio")[
        "fecha"
    ].nunique()

    # Check if all 'id_anuncio' values have only one unique 'fecha'
    if validation_result.max() == 1:
        print("All duplicated id_anuncio values are associated with the same fecha.")
        df_assets_unique = df_assets.drop_duplicates(subset="id_anuncio", keep=criteria)
        # You can further validate if there are still duplicated 'id_anuncio' values
        duplicated_anuncios_after_drop = df_assets_unique[
            df_assets_unique.duplicated(subset="id_anuncio", keep=False)
        ]

        if not duplicated_anuncios_after_drop.empty:
            print(
                "Warning: There are still duplicated 'id_anuncio' values after dropping duplicates."
            )
        return df_assets_unique
    else:
        print(
            "There are some id_anuncio values associated with different fechas. Found a different criteria to remove duplicates"
        )


def find_single_value_columns(df: pd.DataFrame) -> list:
    single_value_columns = []
    for column in df.columns:
        if df[column].nunique() == 1:
            single_value_columns.append(column)

    print("Columns with only one distinct value:", single_value_columns)

    return single_value_columns


def find_null_columns(df: pd.DataFrame):
    null_columns = df.columns[df.isnull().any()]
    null_columns_percentage = {}
    total_rows = len(df)

    for column in null_columns:
        null_count = df[column].isnull().sum()
        null_percentage = (null_count / total_rows) * 100
        null_columns_percentage[column] = null_percentage

    print("Columns with missing values:", null_columns)

    for column, percentage in null_columns_percentage.items():
        print(f"{column}: {percentage:.2f}%")

    return null_columns


def treatment_missing_values(df: pd.DataFrame):
    # Only to see in which variables do we have nulls
    null_columns = find_null_columns(df=df)

    rows_before = len(df)
    
    # Drop this NaN values as we do not have a coherent way to input missing values
    df = df.dropna(subset=["n_piso", "exterior_interior", "cat_calidad"])

    rows_after = len(df)

    print(
        'Percentage of rows affected by dropping NaN values:",',
        (rows_before - rows_after) / rows_before,
    )

    return df

def visualize_distribution(df: pd.DataFrame, save_path=None, numerical_columns: list = None):
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
        correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1, mask=mask
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    #FIXME: não está a guardar direito. validar
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
    
    print('After evaluate which columns remove by coorelations, update list in constants REMOVE_COLUMNS_BY_CORRELATIONS')

    return correlation_matrix, correlated_variables


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # It assigns 1 when flatlocationid is 1 (internal), otherwise 0.
    df["interior"] = (df["exterior_interior"] == 1).astype(int)


    # Create a new variable that adds information to year construction.
    df["antiguidade"] = 2018 - df["cat_ano_construccion"]

    # Regarding the context of the project, we will only keep the houses that are not new construction
    df = df[df.nueva_construccion==0].drop(columns=['nueva_construccion'])
    
    #Once this variables are complementar ones, we can drop one of them
    df[['buen_estado', 'a_reformar']].value_counts()

    df.loc[df['parking'] == 0, 'precio_parking'] = 1

    new_columns = ["interior", "antiguidade"]

    columns_to_drop = [
        "cat_ano_construccion",
        "ano_construccion",
        "exterior_interior",
        "a_reformar"
    ]

    df = df.drop(columns=columns_to_drop)

    return new_columns, columns_to_drop, df


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
def get_location_name_w_gdf(df_assets:pd.DataFrame, df_polygons:pd.DataFrame):
    from shapely import wkt
    from shapely.geometry import Point
    import geopandas as gpd

    # Convert the column WKT to geometry using geopandas
    df_polygons['geometry'] = df_polygons['WKT'].apply(wkt.loads)
    gdf_polygons = gpd.GeoDataFrame(df_polygons, geometry='geometry')

    # Convert the LATITUDE and LONGITUDE coordinates from df_assets into geometric points
    df_assets['geometry'] = df_assets.apply(lambda row: Point(row['longitud'], row['latitud']), axis=1)

    # Create a GeoDataFrame for df_assets
    gdf_sales = gpd.GeoDataFrame(df_assets, geometry='geometry')



def hist_plot_outliers(df, name_variable):
    # Create a new figure and axis for each plot
    fig, ax = plt.subplots()

    # Plot histogram with automatic bins
    ax.hist(df[name_variable], bins='auto', edgecolor='black')

    # Adding labels and title
    ax.set_xlabel(name_variable)
    ax.set_ylabel('Count')
    ax.set_title('Histogram of ' + name_variable)

    # Display the plot
    plt.show()