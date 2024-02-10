import pandas as pd
from ydata_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt
import os


def generate_pandas_profiling_report(df: pd.DataFrame, save_path=None):
    # Get the directory of the currently executing script
    script_dir = os.path.dirname(__file__)

    # Set the default save path
    if save_path is None:
        save_path = os.path.join(script_dir, '..', 'data_visualization')

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Generate the profile report
    profile = ProfileReport(df)

    # Save the report to an HTML file in the specified path
    report_path = os.path.join(save_path, 'profile_report.html')
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

        print("df_assets contains only the last occurrence of each 'id_anuncio'")
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

    for column, percentage in null_columns_percentage.items():
        print(f"{column}: {percentage:.2f}%")

    return null_columns


def treatment_missing_values(df: pd.DataFrame):
    # Only to see in which variables do we have nulls
    null_columns = find_null_columns(df=df)

    rows_before = len(df)

    # Drop this column because we have another column without nulls with same information
    df = df.drop(columns=["ano_construccion"])

    # Drop this NaN values as we does not have a coherent way to input missing values
    df = df.dropna(subset=["n_piso", "exterior_interior"])

    rows_after = len(df)

    print(
        'Percentage of rows affected by dropping NaN values:",',
        (rows_before - rows_after) / rows_before,
    )

    return df


def visualize_distribution(
    df: pd.DataFrame, save_path=None, numerical_columns: list = None
):
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = df.select_dtypes(include=["object"]).columns

    # Get the directory of the currently executing script
    script_dir = os.path.dirname(__file__)

    # Set the default save path
    if save_path is None:
        save_path = os.path.join(script_dir, "..", "data_visualization")

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

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
        plt.figure()
        sns.countplot(data=df, x=column)
        plt.title(f"Count of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")

        if save_path:
            plt.savefig(os.path.join(save_path, f"{column}_count.png"))
        else:
            plt.show()
        plt.clf()  # Clear the plot after saving or displaying


def convert_binary_to_categorical(df:pd.DataFrame, binary_columns:list):
    for column in binary_columns:
        # Check if the column exists in the DataFrame and is binary
        if column in df.columns and len(df[column].unique()) == 2:
            # Convert the column to categorical dtype if it's not already categorical
            if df[column].dtype != 'category':
                df[column] = df[column].astype('category')
            else:
                print(f"Column '{column}' is already categorical.")
        else:
            print(f"Column '{column}' is not binary or does not exist in the DataFrame.")
    return df

def visualize_binary_distribution(df, binary_columns, save_path=None):
    # Create a single FacetGrid for all binary columns with adjusted size and spacing
    g = sns.FacetGrid(df.melt(value_vars=binary_columns), col='variable', col_wrap=3, height=5, aspect=1.5)

    # Map count plots onto the FacetGrid
    g.map(sns.countplot, 'value', palette='Set2')

    # Set titles above each plot and adjust font size
    g.set_titles(row_template='{row_name}', fontsize=8, pad=10)

    # Set x and y labels for each plot and adjust font size
    g.set_axis_labels('Class', 'Perc', fontsize=8)

    # Iterate over each subplot to annotate the percentage of each class and set x-axis labels
    for ax, title in zip(g.axes.flat, binary_columns):
        # Calculate the total count for the column
        total = len(df[title])

        # Calculate and annotate the percentage of each class
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height() / total)
            rounded_percentage = round(100 * p.get_height() / total)
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.text(x, y, f'{rounded_percentage}%', ha='center', va='bottom', fontsize=8)

    # Adjust layout to prevent overlap and increase space between rows
    plt.tight_layout(pad=3.0)

    # Save the plot if save_path is provided
    if save_path is None:
        script_dir = os.path.dirname(__file__)
        save_path = os.path.join(script_dir, '..', 'data_visualization')

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the plot
    save_file_path = os.path.join(save_path, 'binary_distribution_plots.png')
    plt.savefig(save_file_path)
    plt.close()
    print("Image saved successfully at:", save_file_path)

