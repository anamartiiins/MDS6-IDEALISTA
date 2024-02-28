import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def calculate_metrics(y_true, y_pred, df_test):
    """
    Calculate various evaluation metrics including MAE, MAPE, RMSE, APE, and R-squared (R2).
    
    Parameters:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.
    
    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # Calculate Mean Error
    me = np.mean(y_true - y_pred)

    # Calculate Median Error
    med = np.median(y_true - y_pred)

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    # mae = mean_absolute_error(y_true, y_pred)

    # Calculate Median Absolute Error (MAED)
    maed = np.median(np.abs(y_true - y_pred))

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    # mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate Median Absolute Percentage Error (MAPED)
    maped = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    # rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate R-squared (R2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    # r2 = r2_score(y_true, y_pred)

   
    # Create a DataFrame to hold the original data along with APE and absolute error
    df_test_new=df_test.copy()
    df_test_new['ape'] = (abs(y_true - y_pred) / y_true) * 100
    df_test_new['absolute_error'] = abs(y_true - y_pred)
    df_test_new['error']= y_true - y_pred
    df_test_new['percentage_error'] = (abs((y_true - y_pred)) / y_true) * 100
    percentage_error_higher_5 = len(df_test_new[df_test_new["percentage_error"] >= 5])/len(df_test_new)*100    
    percentage_error_higher_10 = len(df_test_new[df_test_new["percentage_error"] >= 10])/len(df_test_new)*100    
    percentage_error_higher_25 = len(df_test_new[df_test_new["percentage_error"] >= 25])/len(df_test_new)*100
    
        
    metrics = {
        'Model' : 'Precio medio por Barrio',
        'ME - Mean Error': round(me.astype(int),0),
        'MED - Median Error': round(med.astype(int),0),
        'MAE - Mean Absolute Error': round(mae.astype(int),0),
        'MAED - Median Absolute Error': round(maed.astype(int),2),
        'MAPE - Mean Absolute Percentage Error': round(mape.astype(float),2),
        'MAPED - Median Absolute Percentage Error': round(maped.astype(float),0),
        'RMSE - Root Mean Squared Error': round(rmse.astype(int),2),
        'R2 - Coefficient of Determination': round(r2.astype(float),2),
        "Percentage error higher_5":percentage_error_higher_5,
        "Percentage error higher_10":percentage_error_higher_10,
        "Percentage error higher_25":percentage_error_higher_25
    }

    metrics_df = pd.DataFrame(metrics, index=['Model'])
    metrics_df = metrics_df.transpose()

    return metrics_df, df_test_new


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
        'Error (E)': error,
        'Absolute Error (AE)': absolute_error,
        'Percentage Error (PE)': percentage_error,
        'Absolute Percentage Error (APE)': absolute_percentage_error,
        'y_true': y_true,
        'y_pred': y_pred
    }

    return metrics

"""
# Ejemplo de uso:
y_true_example = [100, 150, 200]
y_pred_example = [90, 160, 180]
metrics_example = calculate_metrics_by_row(y_true_example, y_pred_example)
"""

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
        'ME': mean_error,
        'MAE': mean_absolute_error,
        'MPE': mean_percentage_error,
        'MAPE': mean_absolute_percentage_error,
        'STDE': stde,
        'RMSE': rmse,
        'R2': r_squared,
    }

    return macro_metrics

"""
# Ejemplo de datos reales y predichos
y_true_example = np.array([100, 170, 180, 120, 180])
y_pred_example = np.array([90, 160, 190, 110, 170])

# Calcular métricas macro utilizando la función
macro_metrics_example = calculate_metrics_macro(y_true_example, y_pred_example)

# Mostrar las métricas macro en un DataFrame
df_metrics = pd.DataFrame([macro_metrics_example])
df_metrics
"""

import pandas as pd
import numpy as np

def evaluate_regresion_model(model, df_test=df_test_util, columna = 'precio'):
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
  df_test_evaluacion['predicciones'] = model.predict(X_test)
  # Create a DataFrame to hold the original data along with APE and absolute error
  df_test_evaluacion['ape'] = (abs(df_test_evaluacion['predicciones'] - df_test_evaluacion[columna]) / df_test_evaluacion[columna]) * 100
  df_test_evaluacion['absolute_error'] = abs(df_test_evaluacion['predicciones'] - df_test_evaluacion[columna])
  df_test_evaluacion['error']= df_test_evaluacion['predicciones'] - df_test_evaluacion[columna]
  df_test_evaluacion['percentage_error'] = ((df_test_evaluacion['predicciones'] - df_test_evaluacion[columna]) / df_test_evaluacion[columna]) * 100

  # Crear un DataFrame para hold las características macro de las predicciones
  macro_features = pd.DataFrame()

  macro_features['ME'] = [df_test_evaluacion['error'].mean()]
  macro_features['MAE'] = [df_test_evaluacion['absolute_error'].mean()]
  macro_features['MPE'] = [df_test_evaluacion['percentage_error'].mean()]
  macro_features['MAPE'] = [df_test_evaluacion['ape'].mean()]
  macro_features['R2'] = [model.score(X_test, df_test_evaluacion[columna])]
  macro_features['STDE'] = [df_test_evaluacion['error'].std()]
  macro_features['RMSE'] = [((df_test_evaluacion['error']**2).mean())**0.5]

  # Lo ajusto a 6 los decimales que si no hay que pensar mucho
  macro_features = macro_features.round(6)

  return df_test_evaluacion, macro_features
