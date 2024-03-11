from typing import Any, List

import matplotlib as plt
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    cross_val_predict,
    GridSearchCV,
    RandomizedSearchCV,
    GroupKFold,
)
from src.evaluation.evaluation import calculate_metrics


def plot_predictions(df_evaluacion, macro_features, columna="precio"):
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
    plt.scatter(df_evaluacion[columna], df_evaluacion["predicciones"])
    plt.plot(
        [df_evaluacion[columna].min(), df_evaluacion[columna].max()],
        [df_evaluacion[columna].min(), df_evaluacion[columna].max()],
        linestyle="--",
        color="red",
        linewidth=2,
    )
    plt.xlabel("Valor Real")
    plt.ylabel("Predicciones")
    plt.title("Predicciones vs. Valores Reales")
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
            ("cat", categorical_transformer, cat_features),
        ],
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", base_model)])

    return pipeline


def train_model_with_pipeline(X_train, y_train, model):
    """
    Train the model until the pipeline.
    """
    pipeline = get_pipeline(
        base_model=model,
        impute=True,
        scale=True,
        encode=True,
        encode_model=TargetEncoder(),
        num_features=[col for col in X_train.columns if col != "barrio"],
        cat_features=["barrio"],
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def get_predictions_and_evaluate(
    pipeline, X_train, y_train, target, cv_spatial: bool = False
):
    """
    Get predictions and evaluate the model until metrics.
    """
    if cv_spatial:
        barrio = X_train["barrio"].values
        group_kfold = GroupKFold(n_splits=5)
        city_kfold = group_kfold.split(X_train, y_train, barrio)
        train_indices, test_indices = [
            list(traintest) for traintest in zip(*city_kfold)
        ]
        city_cv = [*zip(train_indices, test_indices)]
        y_pred = cross_val_predict(pipeline, X_train, y_train, cv=city_cv)
    else:
        y_pred = cross_val_predict(pipeline, X_train, y_train, cv=5)
    model_name = type(pipeline.named_steps["model"]).__name__
    metrics, df_metrics = calculate_metrics(
        y_train, y_pred, X_train, model_name, target
    )
    return metrics, df_metrics


# FIXME
def get_best_model_with_hyperparameter(
    pipeline, X_train, y_train, target, param_random
):
    """
    Hyperparameter tuning with both options: grid and random.
    """
    model_name = type(pipeline.named_steps["model"]).__name__

    # # Grid Search
    # search_cv = GridSearchCV(
    #     estimator=pipeline,
    #     param_grid=param_grid,
    #     scoring="neg_mean_squared_error",
    #     cv=5,
    # )
    # search_cv.fit(X_train, y_train)
    # best_model_grid = search_cv.best_estimator_
    # y_pred_grid = cross_val_predict(best_model_grid, X_train, y_train, cv=5)
    # best_model_metrics_grid, _ = calculate_metrics(
    #     y_train, y_pred_grid, X_train, model_name + " (tuned_grid_search)", target)

    # Randomized Search
    search_cv_1 = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_random,
        random_state=42,
        verbose=1,
    )
    search_cv_1.fit(X_train, y_train)
    print("random search finished")
    best_model_random = search_cv_1.best_estimator_
    y_pred_random = cross_val_predict(best_model_random, X_train, y_train, cv=5)
    print("predictions finished")
    best_model_metrics_random, _ = calculate_metrics(
        y_train,
        y_pred_random,
        X_train,
        model_name + " (tuned_randomized_search)",
        target,
    )
    print("metrics finished")
    return best_model_metrics_random
