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
