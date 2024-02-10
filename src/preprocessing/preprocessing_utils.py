import pandas as pd
from ydata_profiling import ProfileReport


def generate_pandas_profiling_report(df: pd.DataFrame):
    # Generate the profile report
    profile = ProfileReport(df)
    # Save the report to an HTML file
    profile.to_file("profile_report.html")


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

    print('Percentage of rows affected by dropping NaN values:",', (rows_before - rows_after)/rows_before)

    return df
