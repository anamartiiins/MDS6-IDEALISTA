import numpy as np
import pandas as pd
import json
import os
import pickle
import shutil
from datetime import datetime
from enum import Enum
from typing import Dict, Any
import sklearn.base
from catboost import CatBoost
from lightgbm import LGBMModel
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    confusion_matrix,
    precision_score,
    f1_score,
    recall_score,
)


def calculate_metrics(
    y_true, y_pred, df_test, model_name, target: str, confidence_interval=0.05
):
    """
    Calculate various evaluation metrics including MAE, MAPE, RMSE, APE, R-squared (R2),
    standard deviation of errors, and confidence interval.

    Parameters:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.
        df_test (pd.DataFrame): DataFrame containing original data.
        model_name (str): Name of the model.
        confidence_interval (float): Confidence interval (default is 0.05).

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # Calculate Mean Error
    me = np.mean(y_true - y_pred)

    # Calculate Median Error
    med = np.median(y_true - y_pred)

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))

    # Calculate Median Absolute Error (MAED)
    maed = np.median(np.abs(y_true - y_pred))

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate Median Absolute Percentage Error (MAPED)
    maped = np.median(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Calculate R-squared (R2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    # Calculate Standard Deviation of Errors
    std_errors = np.std(y_true - y_pred)

    # Calculate Confidence Interval
    ci_lower = np.percentile(y_true - y_pred, (confidence_interval / 2) * 100)
    ci_upper = np.percentile(y_true - y_pred, 100 - (confidence_interval / 2) * 100)

    # Create a DataFrame to hold the original data along with APE and absolute error
    df_test_new = df_test.copy()
    df_test_new["ape"] = (abs(y_true - y_pred) / y_true) * 100
    df_test_new["absolute_error"] = abs(y_true - y_pred)
    df_test_new["error"] = y_true - y_pred
    df_test_new["percentage_error"] = (abs((y_true - y_pred)) / y_true) * 100
    percentage_error_lower_5 = (
        len(df_test_new[df_test_new["percentage_error"] <= 5]) / len(df_test_new) * 100
    )
    percentage_error_lower_10 = (
        len(df_test_new[df_test_new["percentage_error"] <= 10]) / len(df_test_new) * 100
    )
    percentage_error_lower_25 = (
        len(df_test_new[df_test_new["percentage_error"] <= 25]) / len(df_test_new) * 100
    )

    metrics = {
        "Model": model_name,
        "Target": target,
        "RMSE - Root Mean Squared Error": "{:,.0f}".format(round(float(rmse), 2)),
        "MAPE - Mean Absolute Percentage Error": round(float(mape), 2),
        "R2 - Coefficient of Determination": round(float(r2), 2),
        "ME - Mean Error": "{:,.0f}".format(round(float(me), 0)),
        "MED - Median Error": "{:,.0f}".format(round(float(med), 0)),
        "MAE - Mean Absolute Error": "{:,.0f}".format(round(float(mae), 0)),
        "MAED - Median Absolute Error": "{:,.0f}".format(round(float(maed), 2)),
        "MAPED - Median Absolute Percentage Error": round(float(maped), 0),
        "Percentage error lower_5": percentage_error_lower_5,
        "Percentage error lower_10": percentage_error_lower_10,
        "Percentage error lower_25": percentage_error_lower_25,
        "Standard Deviation of Errors": round(float(std_errors), 2),
        "Confidence Interval Lower": round(float(ci_lower), 2),
        "Confidence Interval Upper": round(float(ci_upper), 2),
        "Model Folder": f"experiment_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    }

    metrics_df = pd.DataFrame(metrics, index=[0])

    return metrics_df, df_test_new


def save_graph_feature_importance(model, X_train, folder):
    """
    Extracts feature importances from the model (if available) and visualizes them.

    Args:
    - model: The trained machine learning model.
    - X_train: The training data features.
    - folder: The folder where the visualization will be saved.
    """
    # Check if the model supports feature importances
    if hasattr(model, "feature_importances_"):
        # Extract feature importances
        feature_importances = model.feature_importances_

        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame(
            {"Feature": X_train.columns, "Importance": feature_importances}
        )

        # Sort the DataFrame by importance in descending order
        feature_importance_df = feature_importance_df.sort_values(
            by="Importance", ascending=False
        )

        # Create a plot to visualize feature importances
        plt.figure(figsize=(15, 8))
        plt.barh(
            range(len(feature_importances)),
            feature_importance_df["Importance"],
            tick_label=feature_importance_df["Feature"],
        )
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top

        # Save the plot as an image file in the specified folder
        plt.savefig(f"src/evaluation/{folder}/feature_importance.png")
        plt.close()


def calculate_metrics_by_row(y_true, y_pred):
    """
    Esta funcion esta pensada analizar detenidamente un unico modelo, calculara las siguientes metricas:

    Error (E): Muestra plana del error (acepta tanto positivos como negativos)
    Absolute error (AE): Muestra plana del error en números positivos absolutos
    Percentage error (PE): Porcentaje del error en funcion del precio real (por abajo y por arriba de 100%)
    Absolute Percentage error (APE): Porcentaje del error en funcion del precio en numeros absolutos

    Parameters:
        y_true (array-like): The true value.
        y_pred (array-like): The predicted value.

    Returns:
        dict: A dictionary containing the calculated metrics.


    # Ejemplo de uso:
    y_true_example = [100, 150, 200]
    y_pred_example = [90, 160, 180]
    metrics_example = calculate_metrics_by_row(y_true_example, y_pred_example)

    """

    # Convertir listas a arrays de numpy si no están en formato array
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Calcular el error (E)
    error = y_pred - y_true

    # Calcular el Absolute error (AE)
    absolute_error = abs(error)

    # Calcular el Percentage error (PE)
    percentage_error = (error / y_true) * 100

    # Calcular el Absolute Percentage error (APE)
    absolute_percentage_error = (absolute_error / y_true) * 100

    # Crear un diccionario con las métricas calculadas
    metrics = {
        "Error (E)": error,
        "Absolute Error (AE)": absolute_error,
        "Percentage Error (PE)": percentage_error,
        "Absolute Percentage Error (APE)": absolute_percentage_error,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    return metrics


def calculate_metrics_macro(y_true, y_pred):
    """
    Esta función devuelve métricas macro del rendimiento de los modelos:

    Mean Error (ME): Muestra plana del error promedio.
    Root Mean Squared Error (RMSE): El RMSE penaliza más los errores grandes, ya que eleva al cuadrado las diferencias entre los valores predichos y reales, y luego aplica la raíz cuadrada.
    R-squared (R2): R2 compara que tanto puede el modelo explicar la realidad o la variable a predcir.
    Standard Deviation of Errors (STDE): La desviación estándar de los errores mide la dispersión de los errores, indicando qué tan variable es el modelo.
    Mean Absolute Error (MAE): El MAE proporciona la diferencia promedio absoluta entre los valores predichos y reales.
    Mean Percentage Error (MPE): El MPE muestra el porcentaje promedio del error en función de los valores reales.
    Mean Absolute Percentage Error (MAPE): El MAPE indica el porcentaje promedio absoluto del error en función de los valores reales.

    Parameters:
        y_true (array-like): Los valores reales.
        y_pred (array-like): Los valores predichos.

    Returns:
        dict: Un diccionario con métricas macro.


    # Ejemplo de datos reales y predichos
    y_true_example = np.array([100, 170, 180, 120, 180])
    y_pred_example = np.array([90, 160, 190, 110, 170])

    # Calcular métricas macro utilizando la función
    macro_metrics_example = calculate_metrics_macro(y_true_example, y_pred_example)

    # Mostrar las métricas macro en un DataFrame
    df_metrics = pd.DataFrame([macro_metrics_example])
    df_metrics

    """

    # Convertir listas a arrays de numpy si no están en formato array
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Calcular métricas macro adicionales
    mean_error = np.mean(y_pred - y_true)
    mean_absolute_error = np.mean(abs(y_pred - y_true))
    mean_percentage_error = np.mean(((y_pred - y_true) / y_true) * 100)
    mean_absolute_percentage_error = np.mean((abs(y_pred - y_true) / y_true) * 100)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r_squared = r2_score(y_true, y_pred)
    stde = np.std(y_pred - y_true)

    # Crear un nuevo diccionario para las métricas macro
    macro_metrics = {
        "ME": mean_error,
        "MAE": mean_absolute_error,
        "MPE": mean_percentage_error,
        "MAPE": mean_absolute_percentage_error,
        "STDE": stde,
        "RMSE": rmse,
        "R2": r_squared,
    }

    return macro_metrics


def evaluate_regresion_model(model, df_test, columna="precio"):
    """
    Evalua un modelo de regresión utilizando un conjunto de prueba y calcula métricas de rendimiento.

      Parámetros:
      - model: Modelo de regresión entrenado.
      - df_test: DataFrame de prueba que contiene características donde nos se ha entrenado.
      - columna: Nombre de la columna objetivo en el DataFrame (por defecto, 'precio').

      Salida:
      Retorna un par de DataFrames:
      1. df_test_evaluacion: DataFrame con las predicciones y métricas para cada muestra en el conjunto de prueba.
      2. macro_features: DataFrame con métricas macro calculadas a partir de las predicciones.

      Uso:
      Suponiendo que tienes un modelo de regresión 'modelo_regresion' y un DataFrame de prueba 'df_test':
      >>> df_evaluacion, macro_features = evaluate_model(modelo_regresion, df_test=df_test_util, columna='precio')

    Esto proporcionará un DataFrame 'df_evaluacion' con las predicciones y métricas para cada muestra,
    así como un DataFrame 'macro_features' con métricas macro calculadas a partir de las predicciones.
    """

    # Hacemos una copia, que nos conocemos
    df_test_evaluacion = df_test.copy()

    # Dividir el DataFrame de prueba en características
    X_test = df_test_evaluacion.drop(columna, axis=1)

    # Agregar las predicciones al DataFrame de prueba
    df_test_evaluacion["predicciones"] = model.predict(X_test)
    # Create a DataFrame to hold the original data along with APE and absolute error
    df_test_evaluacion["ape"] = (
        abs(df_test_evaluacion["predicciones"] - df_test_evaluacion[columna])
        / df_test_evaluacion[columna]
    ) * 100
    df_test_evaluacion["absolute_error"] = abs(
        df_test_evaluacion["predicciones"] - df_test_evaluacion[columna]
    )
    df_test_evaluacion["error"] = (
        df_test_evaluacion["predicciones"] - df_test_evaluacion[columna]
    )
    df_test_evaluacion["percentage_error"] = (
        (df_test_evaluacion["predicciones"] - df_test_evaluacion[columna])
        / df_test_evaluacion[columna]
    ) * 100

    # Crear un DataFrame para hold las características macro de las predicciones
    macro_features = pd.DataFrame()

    macro_features["ME"] = [df_test_evaluacion["error"].mean()]
    macro_features["MAE"] = [df_test_evaluacion["absolute_error"].mean()]
    macro_features["MPE"] = [df_test_evaluacion["percentage_error"].mean()]
    macro_features["MAPE"] = [df_test_evaluacion["ape"].mean()]
    macro_features["R2"] = [model.score(X_test, df_test_evaluacion[columna])]
    macro_features["STDE"] = [df_test_evaluacion["error"].std()]
    macro_features["RMSE"] = [((df_test_evaluacion["error"] ** 2).mean()) ** 0.5]

    # Lo ajusto a 6 los decimales que si no hay que pensar mucho
    macro_features = macro_features.round(6)

    return df_test_evaluacion, macro_features


class ProblemType(Enum):
    """
    Basic enum to represent the problem type.
    """

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class ModelLibrary(Enum):
    """
    Basic enum to represent the model library.
    """

    SCIKIT_LEARN = "scikit-learn"
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"


#
# def export_model(
#     model: Any,
#     X_train: pd.DataFrame,
#     y_train: pd.Series,
#     # X_test: pd.DataFrame,
#     y_test: pd.Series,
#     base_path: str,
#     save_model: bool = True,
#     save_datasets: bool = False,
#     zip_files: bool = True,
# ):
#     """
#     Register model and metrics in a json file and save the model and datasets in a folder.
#     :param model: A model from one of the supported libraries. Currently supported libraries are:
#             - scikit-learn
#             - catboost
#             - lightgbm
#     :param X_train: Training data as a pandas dataframe.
#     :param y_train: Training target as a pandas series.
#     :param X_test: Test data as a pandas dataframe.
#     :param y_test: Test target as a pandas series.
#     :param base_path: Path to the base folder where the model and datasets will be saved in a subfolder structure.
#     :param zip_files: Whether to zip the files or not.
#     :param save_datasets: Whether to save the datasets or not.
#     :param save_model: Whether to save the model or not.
#     :return: The path to the subfolder inside base_path where the model and datasets have been saved.
#     Usage
#     -----
#     # >>> from sklearn.datasets import fetch_california_housing
#     # >>> from sklearn.linear_model import LogisticRegression
#     # >>> from sklearn.model_selection import train_test_split
#     # >>> from mango.models.experiment_tracking import export_model
#     # >>>
#     # >>>
#     # >>> X, y = fetch_california_housing(return_X_y=True, as_frame=True)
#     # >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
#     # >>> model = LogisticRegression()
#     # >>> model.fit(X_train, y_train)
#     # >>> output_folder = export_model(model, X_train, y_train, X_test, y_test, "/my_experiments_folder")
#     # >>> print(output_folder) # /my_experiments_folder/experiment_LogisticRegression_YYYYMMDD-HHMMSS
#     Subfolder structure
#     -------------------
#     The subfolder structure will be the following:
#     |- base_path
#         |- experiment_{model_name}_{datetime}
#             |- model
#                 |- model.pkl
#                 |- hyperparameters.json
#             |- data
#                 |- X_train.csv
#                 |- y_train.csv
#                 |- X_test.csv
#                 |- y_test.csv
#             |- summary.json
#
#     Intended structure
#     summary = {
#         "model": {
#             "name": "",
#             "problem_type": "",
#             # Optional "num_classes": 0, if classification
#             "input": "",
#             "target": "",
#             "hyperparameters": {},
#             "library": "",
#         },
#         "results": {},
#         Optional
#         "files": {
#             "model": {
#                 "zip": "",
#                 "model.pkl": "",
#                 "hyperparameters.json": "",
#             },
#             "data": {
#                 "zip": "",
#                 "X_train.csv": "",
#                 "y_train.csv": "",
#                 "X_test.csv": "",
#                 "y_test.csv": "",
#             },
#         },
#     }
#     """
#     _SUPPORTED_LIBRARIES_CLASSES = {
#         ModelLibrary.SCIKIT_LEARN: sklearn.base.BaseEstimator,
#         ModelLibrary.CATBOOST: CatBoost,
#         ModelLibrary.LIGHTGBM: LGBMModel,
#     }
#     if not os.path.exists(base_path):
#         raise FileNotFoundError(f"Folder {base_path} does not exist.")
#     model_name = model.__class__.__name__
#     model_library = None
#     for library, class_name in _SUPPORTED_LIBRARIES_CLASSES.items():
#         if isinstance(model, class_name):
#             model_library = library
#     if model_library is None:
#         raise ValueError(f"Model {model_name} is not supported.")
#
#     # Detect if it is a classification or regression model
#     if hasattr(model, "predict_proba"):
#         problem_type = ProblemType.CLASSIFICATION
#     else:
#         problem_type = ProblemType.REGRESSION
#
#     summary = {}
#     extra_params = []
#     # Fill structure
#     summary["model"] = {}
#     summary["model"]["name"] = model_name
#     summary["model"]["problem_type"] = problem_type.value
#     summary["model"]["target"] = y_train.name
#     summary["model"]["library"] = model_library.value
#     if model_library == ModelLibrary.CATBOOST:
#         summary["model"]["input"] = list(model.feature_names_)
#         summary["model"]["hyperparameters"] = model.get_all_params()
#     elif model_library == ModelLibrary.SCIKIT_LEARN:
#         # summary["model"]["input"] = list(model.feature_names_in_)
#         summary["model"]["input"] = list()
#         summary["model"]["hyperparameters"] = model.get_params(deep=True)
#     elif model_library == ModelLibrary.LIGHTGBM:
#         summary["model"]["input"] = list(model.feature_name_)
#         summary["model"]["hyperparameters"] = model.get_params(deep=True)
#
#     # Sort keys in summary["model"]
#     if problem_type == ProblemType.CLASSIFICATION:
#         summary["model"]["num_classes"] = len(y_train.unique())
#         # Sort keys in summary["model"] to be: name, problem_type, num_classes, input, target, hyperparameters, library
#         summary["model"] = {
#             k: summary["model"][k]
#             for k in [
#                 "name",
#                 "problem_type",
#                 "num_classes",
#                 "input",
#                 "target",
#                 "hyperparameters",
#                 "library",
#             ]
#         }
#     else:
#         # Sort keys in summary["model"] to be: name, problem_type, input, target, hyperparameters, library
#         summary["model"] = {
#             k: summary["model"][k]
#             for k in [
#                 "name",
#                 "problem_type",
#                 "input",
#                 "target",
#                 "hyperparameters",
#                 "library",
#             ]
#         }
#
#     # Prepare environment to save files
#     folder_name = os.path.join(
#         base_path,
#         f"experiment_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
#     )
#
#     # Compress model and save
#     if save_model:
#         os.makedirs(os.path.join(folder_name, "model"))
#         if not "files" in summary:
#             summary["files"] = {}
#         if not "model" in summary["files"]:
#             summary["files"]["model"] = {}
#         # Save hyperparameters
#         hyperparameters_path = os.path.join(
#             folder_name, "model", "hyperparameters.json"
#         )
#
#         summary["files"]["model"]["hyperparameters.json"] = os.path.abspath(
#             hyperparameters_path
#         )
#
#         if isinstance(hyperparameters, sklearn.compose.ColumnTransformer):
#             hyperparameters = hyperparameters.get_params()
#
#         with open(hyperparameters_path, "w") as f:
#             json.dump(summary["model"]["hyperparameters"], f, indent=4)
#         # Save the model
#         model_path = os.path.join(folder_name, "model", "model.pkl")
#         summary["files"]["model"]["model.pkl"] = os.path.abspath(model_path)
#         with open(model_path, "wb") as f:
#             pickle.dump(model, f)
#         if zip_files:
#             zip_path = os.path.join(folder_name, "model.zip")
#             summary["files"]["model"]["zip"] = os.path.abspath(zip_path)
#             shutil.make_archive(
#                 zip_path.rstrip(".zip"), "zip", os.path.join(folder_name, "model")
#             )
#             shutil.rmtree(os.path.join(folder_name, "model"))
#
#     if save_datasets:
#         os.makedirs(os.path.join(folder_name, "data"))
#         if not "files" in summary:
#             summary["files"] = {}
#         if not "data" in summary["files"]:
#             summary["files"]["data"] = {}
#         X_train_path = os.path.join(folder_name, "data", "X_train.csv")
#         summary["files"]["data"]["X_train.csv"] = os.path.abspath(X_train_path)
#         X_train.to_csv(X_train_path, index=False)
#         y_train_path = os.path.join(folder_name, "data", "y_train.csv")
#         summary["files"]["data"]["y_train.csv"] = os.path.abspath(y_train_path)
#         y_train.to_csv(y_train_path, index=False)
#         # X_test_path = os.path.join(folder_name, "data", "X_test.csv")
#         # summary["files"]["data"]["X_test.csv"] = os.path.abspath(X_test_path)
#         # X_test.to_csv(X_test_path, index=False)
#         y_test_path = os.path.join(folder_name, "data", "y_test.csv")
#         summary["files"]["data"]["y_test.csv"] = os.path.abspath(y_test_path)
#         y_test.to_csv(y_test_path, index=False)
#         if zip_files:
#             # Compress data and save
#             zip_path = os.path.join(folder_name, "data.zip")
#             summary["files"]["data"]["zip"] = os.path.abspath(zip_path)
#             shutil.make_archive(
#                 zip_path.rstrip(".zip"), "zip", os.path.join(folder_name, "data")
#             )
#             shutil.rmtree(os.path.join(folder_name, "data"))
#
#     # Save json
#     json_path = os.path.join(folder_name, "summary.json")
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=4, ensure_ascii=False)
#
#     return folder_name
#
# import os
# import json
# import shutil
# import pickle
# from datetime import datetime
# from typing import Any
# import pandas as pd
#
#
# class ModelLibrary:
#     SCIKIT_LEARN = "scikit-learn"
#     CATBOOST = "catboost"
#     LIGHTGBM = "lightgbm"
#
#
# class ProblemType:
#     CLASSIFICATION = "classification"
#     REGRESSION = "regression"
#
#
# def export_model(
#     model: Any,
#     y_train: pd.Series,
#     base_path: str,
#     save_model: bool = True,
#     save_datasets: bool = False,
#     zip_files: bool = True,
# ):
#     """
#     Register model and metrics in a json file and save the model and datasets in a folder.
#     :param model: A model from one of the supported libraries. Currently supported libraries are:
#             - scikit-learn
#             - catboost
#             - lightgbm
#     :param X_train: Training data as a pandas dataframe.
#     :param y_train: Training target as a pandas series.
#     :param X_test: Test data as a pandas dataframe.
#     :param y_test: Test target as a pandas series.
#     :param base_path: Path to the base folder where the model and datasets will be saved in a subfolder structure.
#     :param zip_files: Whether to zip the files or not.
#     :param save_datasets: Whether to save the datasets or not.
#     :param save_model: Whether to save the model or not.
#     :return: The path to the subfolder inside base_path where the model and datasets have been saved.
#     """
#     _SUPPORTED_LIBRARIES_CLASSES = {
#         ModelLibrary.SCIKIT_LEARN: sklearn.base.BaseEstimator,
#         ModelLibrary.CATBOOST: CatBoost,
#         ModelLibrary.LIGHTGBM: LGBMModel,
#     }
#     if not os.path.exists(base_path):
#         raise FileNotFoundError(f"Folder {base_path} does not exist.")
#     model_name = model.__class__.__name__
#     model_library = None
#     for library, class_name in _SUPPORTED_LIBRARIES_CLASSES.items():
#         if isinstance(model, class_name):
#             model_library = library
#     if model_library is None:
#         raise ValueError(f"Model {model_name} is not supported.")
#
#     # Detect if it is a classification or regression model
#     if hasattr(model, "predict_proba"):
#         problem_type = ProblemType.CLASSIFICATION
#     else:
#         problem_type = ProblemType.REGRESSION
#
#     summary = {}
#     extra_params = []
#     # Fill structure
#     summary["model"] = {}
#     summary["model"]["name"] = model_name
#     summary["model"]["problem_type"] = problem_type.value
#     summary["model"]["target"] = y_train.name
#     summary["model"]["library"] = model_library.value
#     if model_library == ModelLibrary.CATBOOST:
#         summary["model"]["input"] = list(model.feature_names_)
#         summary["model"]["hyperparameters"] = model.get_all_params()
#     elif model_library == ModelLibrary.SCIKIT_LEARN:
#         # summary["model"]["input"] = list(model.feature_names_in_)
#         summary["model"]["input"] = list()
#         summary["model"]["hyperparameters"] = model.get_params(deep=True)
#     elif model_library == ModelLibrary.LIGHTGBM:
#         summary["model"]["input"] = list(model.feature_name_)
#         summary["model"]["hyperparameters"] = model.get_params(deep=True)
#
#     # Sort keys in summary["model"]
#     if problem_type == ProblemType.CLASSIFICATION:
#         summary["model"]["num_classes"] = len(y_train.unique())
#         # Sort keys in summary["model"] to be: name, problem_type, num_classes, input, target, hyperparameters, library
#         summary["model"] = {
#             k: summary["model"][k]
#             for k in [
#                 "name",
#                 "problem_type",
#                 "num_classes",
#                 "input",
#                 "target",
#                 "hyperparameters",
#                 "library",
#             ]
#         }
#     else:
#         # Sort keys in summary["model"] to be: name, problem_type, input, target, hyperparameters, library
#         summary["model"] = {
#             k: summary["model"][k]
#             for k in [
#                 "name",
#                 "problem_type",
#                 "input",
#                 "target",
#                 "hyperparameters",
#                 "library",
#             ]
#         }
#
#     # Prepare environment to save files
#     folder_name = os.path.join(
#         base_path,
#         f"experiment_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
#     )
#
#     # Compress model and save
#     if save_model:
#         os.makedirs(os.path.join(folder_name, "model"))
#         if not "files" in summary:
#             summary["files"] = {}
#         if not "model" in summary["files"]:
#             summary["files"]["model"] = {}
#         # Save hyperparameters
#         hyperparameters_path = os.path.join(
#             folder_name, "model", "hyperparameters.json"
#         )
#
#         summary["files"]["model"]["hyperparameters.json"] = os.path.abspath(
#             hyperparameters_path
#         )
#
#         if isinstance(
#             summary["model"]["hyperparameters"], sklearn.compose.ColumnTransformer
#         ):
#             summary["model"]["hyperparameters"] = summary["model"][
#                 "hyperparameters"
#             ].get_params()
#
#         with open(hyperparameters_path, "w") as f:
#             json.dump(summary["model"]["hyperparameters"], f, indent=4)
#         # Save the model
#         model_path = os.path.join(folder_name, "model", "model.pkl")
#         summary["files"]["model"]["model.pkl"] = os.path.abspath(model_path)
#         with open(model_path, "wb") as f:
#             pickle.dump(model, f)
#         if zip_files:
#             zip_path = os.path.join(folder_name, "model.zip")
#             summary["files"]["model"]["zip"] = os.path.abspath(zip_path)
#             shutil.make_archive(
#                 zip_path.rstrip(".zip"), "zip", os.path.join(folder_name, "model")
#             )
#             shutil.rmtree(os.path.join(folder_name, "model"))
#
#     # if save_datasets:
#     #     os.makedirs(os.path.join(folder_name, "data"))
#     #     if not "files" in summary:
#     #         summary["files"] = {}
#     #     if not "data" in summary["files"]:
#     #         summary["files"]["data"] = {}
#     #     X_train_path = os.path.join(folder_name, "data", "X_train.csv")
#     #     summary["files"]["data"]["X_train.csv"] = os.path.abspath(X_train_path)
#     #     X_train.to_csv(X_train_path, index=False)
#     #     y_train_path = os.path.join(folder_name, "data", "y_train.csv")
#     #     summary["files"]["data"]["y_train.csv"] = os.path.abspath(y_train_path)
#     #     y_train.to_csv(y_train_path, index=False)
#     #     # X_test_path = os.path.join(folder_name, "data", "X_test.csv")
#     #     # summary["files"]["data"]["X_test.csv"] = os.path.abspath(X_test_path)
#     #     # X_test.to_csv(X_test_path, index=False)
#     #     y_test_path = os.path.join(folder_name, "data", "y_test.csv")
#     #     summary["files"]["data"]["y_test.csv"] = os.path.abspath(y_test_path)
#     #     y_test.to_csv(y_test_path, index=False)
#     #     if zip_files:
#     #         zip_path = os.path.join(folder_name, "data.zip")
#     #         summary["files"]["data"]["zip"] = os.path.abspath(zip_path)
#     #         shutil.make_archive(
#     #             zip_path.rstrip(".zip"), "zip", os.path.join(folder_name, "data")
#     #         )
#     #         shutil.rmtree(os.path.join(folder_name, "data"))
#
#     # Save json
#     json_path = os.path.join(folder_name, "summary.json")
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=4, ensure_ascii=False)
#
#     return folder_name
import os
import json
import shutil
import pickle
from datetime import datetime
from typing import Any
import pandas as pd
import sklearn


def convert_to_serializable(obj):
    if isinstance(obj, sklearn.compose.ColumnTransformer):
        return obj.get_params()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    else:
        return obj


def export_model_simple(
    model: Any,
    y_train: pd.Series,
    base_path: str,
    save_model: bool = True,
):
    """
    Save the model as a pickle file and its hyperparameters as a JSON file.
    :param model: A model to be saved.
    :param y_train: Training target as a pandas series.
    :param base_path: Path to the base folder where the model and datasets will be saved in a subfolder structure.
    :param save_model: Whether to save the model or not.
    :return: The path to the subfolder inside base_path where the model and datasets have been saved.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    model_name = model.__class__.__name__
    folder_name = os.path.join(
        base_path,
        f"experiment_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )

    if save_model:
        os.makedirs(os.path.join(folder_name, "model"))

        # Save hyperparameters as JSON
        hyperparameters_path = os.path.join(
            folder_name, "model", "hyperparameters.json"
        )
        hyperparameters = model.get_params()

        # Convert ColumnTransformer parameters and other non-serializable objects
        serializable_hyperparameters = convert_to_serializable(hyperparameters)

        with open(hyperparameters_path, "w") as f:
            json.dump(serializable_hyperparameters, f, indent=4)

        # Save the model as a pickle file
        model_path = os.path.join(folder_name, "model", "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    return folder_name
