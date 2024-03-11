import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupKFold

from src.preprocessing.data_extraction import (
    extract_initial_data,
    get_info_from_polygons_and_ine,
)

# from src.preprocessing.preprocessing import (
#     remove_duplicated_assets_id,
#     find_single_value_columns,
#     treatment_missing_values,
#     feature_engineering,
#     imputation_values_not_nulls,
#     detect_outliers_by_percentile,
#     add_aggregated_features,
#     correlation_values,
# )

from src.constants import (
    NEW_COLUMNS_NAMES,
    REMOVE_COLUMNS_BY_INPUT,
    REMOVE_COLUMNS_BY_CORRELATIONS,
    PATH_TRAIN,
    PATH_TEST,
)


def remove_duplicated_assets_id(
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

    # Validate if duplicated 'id_anuncio' values are associated with the same 'date' value
    validation_result = df_duplicates_anuncio_id.groupby("id_anuncio")[
        "fecha"
    ].nunique()

    # Check if all 'id_anuncio' values have only one unique 'fecha'
    if validation_result.max() == 1:
        print("All duplicated id_anuncio values are associated with the same date.")
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
            "There are some id_anuncio values associated with different dates. "
            "Found a different criteria to remove duplicates"
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


def treatment_missing_values(df: pd.DataFrame, columns_to_drop_null_values: list):
    rows_before = len(df)

    # Drop this NaN values as we do not have a coherent way to input missing values
    df = df.dropna(subset=columns_to_drop_null_values)

    rows_after = len(df)

    print(
        'Percentage of rows affected by dropping NaN values:",',
        (rows_before - rows_after) / rows_before,
    )

    return df


def feature_engineering(df: pd.DataFrame):
    # It assigns 1 when flatlocationid is 1 (internal), otherwise 0.
    df["interior"] = (df["exterior_interior"] == 1).astype(int)

    # Create a new variable that adds information to year construction.
    df["antiguidade"] = 2018 - df["cat_ano_construccion"]

    # Regarding the context of the project, we will only keep the houses that are not new construction
    df = df[df.nueva_construccion == 0].drop(columns=["nueva_construccion"])

    new_columns = ["interior", "antiguidade"]

    columns_to_drop_by_creation_of_new_ones = [
        "cat_ano_construccion",
        "ano_construccion",
        "exterior_interior",
    ]

    return new_columns, columns_to_drop_by_creation_of_new_ones, df


def imputation_values_not_nulls(df: pd.DataFrame):
    # Filter rows where n_banos > 0
    filtered_df = df[df["n_banos"] > 0]

    # Calculate number of bathrooms per square meter
    filtered_df["n_banos_m2"] = filtered_df["n_banos"] / filtered_df["area_construida"]

    # Calculate mean number of bathrooms per square meter
    median_bathrooms_per_sqm = filtered_df["n_banos_m2"].median()

    # Impute number of bathrooms for rows where n_banos == 0: mean number of bath by m^2 * m^2, rounded, and minimum 1
    df["n_banos_m2"] = (
        (np.maximum(median_bathrooms_per_sqm * df["area_construida"], 1))
        .round()
        .astype(int)
    )
    # Validate if it is a good way to values, calculating mape comparing with the real n_banos
    df_train_aux = df[df["n_banos"] > 0]
    absolute_percentage_errors = np.abs(
        (df_train_aux["n_banos"] - df_train_aux["n_banos_m2"]) / df_train_aux["n_banos"]
    )
    mape = np.mean(absolute_percentage_errors) * 100
    print("MAPE", mape)

    # Delete filtered_df, df_train_aux as they are only auxiliar
    del filtered_df, df_train_aux

    # Assign the imputed value to n_banos where n_banos == 0. All houses with 0 bathrooms are houses to renovate
    df.loc[df["n_banos"] == 0, "n_banos"] = df.loc[df["n_banos"] == 0, "n_banos_m2"]

    # Drop the n_banos_m2 column as it is no longer needed
    df = df.drop(columns=["n_banos_m2"])

    # Does not make sense to have parking = 0 and precio_parking > 0.
    # Assuming that all prices of parking are 1 when the assets do not have parking
    df.loc[df["parking"] == 0, "precio_parking"] = 1

    return df, median_bathrooms_per_sqm


def detect_outliers_by_percentile(
    df: pd.DataFrame,
    variables_most_correlated_w_target: list,
    percentile: float = 0.995,
):
    nr_row_before = df.shape[0]

    percentile_995_values = {}

    for var in variables_most_correlated_w_target:
        percentile_995_values[var] = df[var].quantile(percentile)

    condition_to_exclude_outliers = (
        (df["n_banos"] > percentile_995_values["n_banos"])
        | (df["n_habitaciones"] > percentile_995_values["n_habitaciones"])
        | (df["area_construida"] > percentile_995_values["area_construida"])
        | (df["distancia_castellana"] > percentile_995_values["distancia_castellana"])
    )

    df = df[~condition_to_exclude_outliers]

    nr_row_after = df.shape[0]

    print("Percentage of rows deleted: ", 1 - nr_row_after / nr_row_before)

    return df


def add_aggregated_features(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    df_metrics = (
        df.groupby([variable])
        .agg(
            {
                "precio": ["median", "mean", "std"],
                "precio_unitario_m2": ["median", "mean", "std"],
                "precio_logaritmico": ["median", "mean", "std"],
            }
        )
        .reset_index()
    )

    df_metrics.columns = [
        "barrio",
        f"precio_median_{variable}",
        f"precio_mean_{variable}",
        f"precio_std_{variable}",
        f"precio_unitario_m2_median_{variable}",
        f"precio_unitario_m2_mean_{variable}",
        f"precio_unitario_m2_std_{variable}",
        f"precio_logaritmico_median_{variable}",
        f"precio_logaritmico_mean_{variable}",
        f"precio_logaritmico_std_{variable}",
    ]

    df = df.merge(
        df_metrics[
            [
                variable,
                f"precio_mean_{variable}",
                f"precio_unitario_m2_mean_{variable}",
                f"precio_logaritmico_mean_{variable}",
            ]
        ],
        on=[variable],
        how="inner",
    )

    return df


def correlation_values(df: pd.DataFrame, threshold: float = 0.8):
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

    # Print correlated variable pairs with their correlation coefficients
    for pair in correlated_variables:
        print(f"Correlated variables: {pair[0]}, {pair[1]}, Correlation: {pair[2]}")

    print(
        "After evaluate which columns remove by correlations, update list in constants REMOVE_COLUMNS_BY_CORRELATIONS"
    )

    return correlation_matrix, correlated_variables


def preprocessing(
    df: pd.DataFrame,
    df_polygons: pd.DataFrame,
    df_ine: pd.DataFrame,
    predict: bool = True,
):
    # Change columns names to friendly ones
    df.columns = NEW_COLUMNS_NAMES

    df = remove_duplicated_assets_id(df_assets=df, criteria="last")

    # Add columns 'geometry', 'precio_logaritmico', 'cusec', 'barrio_id', 'barrio'
    df = get_info_from_polygons_and_ine(df_polygons=df_polygons, df_ine=df_ine, df=df)

    # Add variables interior (1/0) and antiguidade.
    # Remove assets that are "nueva_construccion"
    _, columns_to_drop_by_creation_of_new_ones, df = feature_engineering(df=df)
    df = treatment_missing_values(
        df=df, columns_to_drop_null_values=["n_piso", "cat_calidad", "interior"]
    )
    # Impute number of bathrooms and price_parking when asset does not have parking
    df, median_bathrooms_per_sqm = imputation_values_not_nulls(df=df)

    # Remove columns that only have one different value
    remove_unique_value_columns = find_single_value_columns(df=df)
    df = df.drop(columns=remove_unique_value_columns)

    # Remove columns by input (team decision)
    df = df.drop(columns=REMOVE_COLUMNS_BY_INPUT)

    # Remove columns by creation of new ones (team decision)
    df = df.drop(columns=columns_to_drop_by_creation_of_new_ones)

    if not predict:
        # Remove columns by correlations (team decision)
        df = df.drop(columns=REMOVE_COLUMNS_BY_CORRELATIONS)

        # Identifiy and remove outliers by percentile 995 for most correlated variables with target
        df = detect_outliers_by_percentile(
            df=df,
            percentile=0.995,
            variables_most_correlated_w_target=[
                "n_banos",
                "n_habitaciones",
                "area_construida",
                "distancia_castellana",
            ],
        )

    correlation_matrix, _ = correlation_values(df=df, threshold=0.8)

    df = add_aggregated_features(df=df, variable="barrio")

    if predict:
        df.to_csv(PATH_TEST)
    else:
        df.to_csv(PATH_TRAIN)

    return df


def split_data_train_test(df: pd.DataFrame):
    # Remove target
    X = df.drop(columns=["PRICE"])
    y = df["PRICE"]

    # Split train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Keep predictors variables and target together to do preprocessing
    df_train = pd.concat([y_train, X_train], axis=1)
    df_test = pd.concat([y_test, X_test], axis=1)

    return df_train, df_test
