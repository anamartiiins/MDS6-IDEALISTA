import matplotlib as plt
from typing import Any, List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def plot_predictions(df_evaluacion, macro_features, columna = 'precio'):
    """
    Plotea las predicciones contra los valores reales y muestra los datos macro.

    Parámetros:
    - df_evaluacion: DataFrame con las predicciones y métricas.
    - macro_features: DataFrame con métricas macro.
    - columna: Nombre de la columna objetivo.

    Uso:
    >>> plot_predictions(df_evaluacion, macro_features, columna='precio')
    """
    # Obtener el nombre de la variable
    df_name = [name for name, var in globals().items() if var is df_evaluacion][0]

    # Imprimir datos macro
    print("Datos Macro:")
    print(macro_features.to_string(index=False))

    # Plotear las predicciones
    plt.scatter(df_evaluacion[columna], df_evaluacion['predicciones'])
    plt.plot([df_evaluacion[columna].min(), df_evaluacion[columna].max()],
             [df_evaluacion[columna].min(), df_evaluacion[columna].max()],
             linestyle='--', color='red', linewidth=2)
    plt.xlabel('Valor Real')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs. Valores Reales')
    plt.show()




def get_pipeline(
    base_model: Any,
    impute: bool,
    scale: bool,
    encode: bool,
    impute_model_cat: Any = SimpleImputer(strategy="most_frequent"),
    impute_model_num: Any = SimpleImputer(strategy="median"),
    scale_model: Any = StandardScaler(),
    encode_model: Any = OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    cat_features: List[str] = None,
    num_features: List[str] = None,
) -> Pipeline:
    """
    Build a sklearn pipeline for preprocessing and modeling.
    :param base_model: Base model to be added at the end of the pipeline.
    :param impute: Indicates whether missing values imputation should be performed.
    :param scale: Indicates whether numerical feature scaling should be performed.
    :param encode: Indicates whether categorical feature encoding should be performed.
    :param impute_model_cat: Imputation model for categorical features.
      Default is SimpleImputer with strategy "most_frequent".
    :param impute_model_num: Imputation model for numerical features.
      Default is SimpleImputer with strategy "median".
    :param scale_model: Scaling model for numerical features.
      Default is StandardScaler.
    :param encode_model: Encoding model for categorical features.
      Default is OneHotEncoder with "ignore" handling for unknown values.
    :param cat_features: List of names of categorical features.
      Used to apply specific transformations to these features.
    :param num_features: List of names of numerical features.
      Used to apply specific transformations to these features.
    :Return: Sklearn pipeline that includes preprocessing steps and the base model.

    """
    # Create the list of transformers
    cat_transformers = []
    num_transformers = []
    if impute and cat_features:
        cat_transformers.append(("imputer_cat", impute_model_cat))
    if impute and num_features:
        num_transformers.append(("imputer_num", impute_model_num))
    if scale and num_features:
        num_transformers.append(("scaler", scale_model))
    if encode and cat_features:
        cat_transformers.append(("encoder", encode_model))

    if not cat_transformers:
        cat_transformers.append(("passthrough", "passthrough"))
    if not num_transformers:
        num_transformers.append(("passthrough", "passthrough"))

    numeric_transformer = Pipeline(
        steps=num_transformers,
    )
    categorical_transformer = Pipeline(
        steps=cat_transformers,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            # ("cat", categorical_transformer, cat_features),
        ],
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", base_model)])

    return pipeline