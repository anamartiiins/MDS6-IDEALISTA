import pandas as pd
from src.constants import (
    NEW_COLUMNS_NAMES,
    REMOVE_COLUMNS_BY_INPUT,
    NUM_VARIABLES_TO_SEE_DISTRIBUTION,
    BINARY_VARIABLES,
)
from src.preprocessing.preprocessing_utils import (
    generate_pandas_profiling_report,
    remove_duplicated_anuncios_id,
    find_single_value_columns,
    treatment_missing_values,
    visualize_distribution,
    visualize_binary_distribution,
    correlation_values,
    feature_engineering,
    get_location_name_w_gdf,
)


def dataset_preprocessing(
    df_assets: pd.DataFrame,
    df_polygons: pd.DataFrame,
    pandas_profiling: bool = False,
    distributions: bool = False,
) -> pd.DataFrame:
    print("Start")

    #FIXME
    aux=pd.read_csv(r'C:\Users\aimartins\OneDrive - Parfois, SA\Documents\GitHub\MDS6-IDEALISTA\data_spatial.csv')
    aux = aux.drop(columns=['geometry', 'index_right', 'LOCATIONID', 'WKT', 'ZONELEVELID'])

    df_assets=aux.copy()

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
