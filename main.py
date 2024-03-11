import pandas as pd
import sys
import warnings
from src.preprocessing.data_extraction import extract_initial_data
from src.preprocessing.preprocessing import preprocessing, split_data_train_test
from src.constants import BASE_PATH_EXPERIMENTS
from src.modelization.models_utils import (
    train_model_with_pipeline,
    get_predictions_and_evaluate,
    get_best_model_with_hyperparameter,
)
from src.evaluation.evaluation import calculate_metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from scipy.stats import randint
import os
from datetime import datetime
import pickle

print(sys.executable)
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Extract all dataframes available
    df, df_ine, df_osm, df_pois, df_polygons = extract_initial_data(
        root_dir="input_data"
    )

    df_train, df_test = split_data_train_test(df=df)

    df_train = preprocessing(
        df=df_train, df_ine=df_ine, df_polygons=df_polygons, predict=False
    )

    df_test = preprocessing(
        df=df_test, df_ine=df_ine, df_polygons=df_polygons, predict=True
    )

    # Define your list of models
    models = [
        RandomForestRegressor(),
        DecisionTreeRegressor(),
        GradientBoostingRegressor(),
        XGBRegressor(),
    ]

    # Define target variable
    target = "precio_unitario_m2"

    # Prepare data
    X_train = df_train.drop(
        columns=["precio", "precio_unitario_m2", "precio_logaritmico"]
    )
    y_train = df_train[target]

    param_randoms = {
        RandomForestRegressor: {
            "model__n_estimators": randint(50, 200),
            "model__max_depth": randint(3, 20),
            "model__min_samples_split": randint(2, 10),
            "model__min_samples_leaf": randint(1, 4),
        },
        DecisionTreeRegressor: {
            "model__max_depth": randint(3, 20),
            "model__min_samples_split": randint(2, 10),
            "model__min_samples_leaf": randint(1, 4),
        },
        GradientBoostingRegressor: {
            "model__n_estimators": randint(50, 200),
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7],
        },
        XGBRegressor: {
            "model__n_estimators": randint(50, 200),
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7],
        },
    }

    # Initialize an empty DataFrame to store metrics
    metrics_df = pd.DataFrame()
    df_metrics_complete = pd.DataFrame()

    # Loop over each model
    for model in models:
        # Train the model
        pipeline = train_model_with_pipeline(X_train, y_train, model)
        # Get predictions and evaluate
        model_metrics, df_metrics = get_predictions_and_evaluate(
            pipeline, X_train, y_train, target, cv_spatial=False
        )
        metrics_df = pd.concat([metrics_df, model_metrics])
        df_metrics_complete = pd.concat([df_metrics_complete, df_metrics])

        # # Get best model with hyperparameter tuning
        best_model_metrics_random = get_best_model_with_hyperparameter(
            pipeline,
            X_train,
            y_train,
            target,
            param_random=param_randoms[type(model)],
        )
        metrics_df = pd.concat([metrics_df, best_model_metrics_random])

        model_name = model.__class__.__name__
        folder_name = os.path.join(
            BASE_PATH_EXPERIMENTS,
            f"experiment_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )

        model_dir = os.path.join(folder_name, "model")
        os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists

        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

    metrics_df.to_csv("metrics_df.csv")

    metrics_df_barrios, _ = calculate_metrics(
        df_test["precio_unitario_m2"],
        df_test["precio_unitario_m2_mean_barrio"],
        df_test,
        model_name="Baseline Barrios",
        target=target,
    )
