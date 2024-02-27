import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred, df_test):
    """
    Calculate various evaluation metrics including MAE, MAPE, RMSE, APE, and R-squared (R2).
    
    Parameters:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.
    
    Returns:
        dict: A dictionary containing the calculated metrics.
    """

    me = np.mean(y_true - y_pred)

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    # mae = mean_absolute_error(y_true, y_pred)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    # mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    # rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate R-squared (R2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    # r2 = r2_score(y_true, y_pred)

    
    metrics = {
        'ME - Mean Error': me,
        'MAE - Mean Absolute Error': mae,
        'MAPE - Mean Absolute Percentage Error': mape,
        'RMSE - Root Mean Squared Error': rmse,
        'R2 - Coefficient of Determination': r2
    }
    
    # Create a DataFrame to hold the original data along with APE and absolute error
    df_test_new=df_test.copy()
    df_test_new['ape'] = (abs(y_true - y_pred) / y_true) * 100
    df_test_new['absolute_error'] = abs(y_true - y_pred)
    df_test_new['error']= y_true - y_pred
    df_test_new['percentage_error'] = ((y_true - y_pred) / y_true) * 100

    return metrics, df_test_new

# Está bien la base, pero vamos a ir row a row para tener más facilidad de clusterizar y entender:

import numpy as np

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

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

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
