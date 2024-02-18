
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

    return metrics, df_test_new
