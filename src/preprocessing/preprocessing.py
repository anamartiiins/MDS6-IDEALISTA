import pandas as pd
from src.constants import (
    NEW_COLUMNS_NAMES,
    REMOVE_COLUMNS_BY_INPUT,
    NUM_VARIABLES_TO_SEE_DISTRIBUTION,
    BINARY_VARIABLES
)
from src.preprocessing.preprocessing_utils import (
    generate_pandas_profiling_report,
    remove_duplicated_anuncios_id,
    find_single_value_columns,
    treatment_missing_values,
    visualize_distribution,
    visualize_binary_distribution
)
import seaborn as sns
import matplotlib.pyplot as plt


def dataset_preprocessing(df_assets: pd.DataFrame) -> pd.DataFrame:
    print("Start")
    # Change columns names to friendly ones
    df_assets = df_assets.drop(columns=["ADTYPOLOGY", "ADOPERATION"])
    df_assets.columns = NEW_COLUMNS_NAMES

    # Run pandas profiling
    # generate_pandas_profiling_report(df=df_assets)

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
    df_assets = treatment_missing_values(df=df_assets)

    # See distributions
    # visualize_distribution(
    #     df=df_assets, numerical_columns=NUM_VARIABLES_TO_SEE_DISTRIBUTION
    # )

    visualize_binary_distribution(df=df_assets, binary_columns=BINARY_VARIABLES)

    print(df_assets.shape)
    return df_assets
