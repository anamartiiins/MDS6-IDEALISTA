NEW_COLUMNS_NAMES = [
    "precio",
    "id_anuncio",
    "fecha",
    "precio_unitario_m2",
    "tipologia_imueble",
    "operacion",
    "area_construida",
    "n_habitaciones",
    "n_banos",
    "terraza",
    "ascensor",
    "aire_acondicionado",
    "amueblado",
    "parking",
    "parking_incluido_precio",
    "precio_parking",
    "orientacion_n",
    "orientacion_s",
    "orientacion_e",
    "orientacion_o",
    "trastero",
    "armarios",
    "piscina",
    "portero",
    "jardin",
    "duplex",
    "estudio",
    "arico",
    "ano_construccion",
    "n_piso",
    "exterior_interior",
    "cat_ano_construccion",
    "cat_n_max_pisos",
    "cat_n_vecinos",
    "cat_calidad",
    "nueva_construccion",
    "a_reformar",
    "buen_estado",
    "distancia_puerta_sol",
    "distancia_metro",
    "distancia_castellana",
    "longitud",
    "latitud",
    "ciudad",
    "ADTYPOLOGY",
    "ADOPERATION",
    # "barrio"
]


REMOVE_COLUMNS_BY_INPUT = ["fecha", "id_anuncio"]

REMOVE_COLUMNS_BY_CORRELATIONS = ["parking_incluido_precio", "a_reformar"]

BASE_PATH_EXPERIMENTS = r"src\evaluation"
PATH_EVALUATION_CSV = r"src\evaluation\evaluation.csv"
PATH_EVALUATION_DF_WITH_METRICS_CSV = r"src\evaluation\evaluation_df_with_metrics.csv"
PATH_TRAIN = r"output_data/df_train_util.csv"
PATH_TEST = r"output_data/df_test_util.csv"
