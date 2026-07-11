# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
"""
Entrenamiento de un modelo de clasificación para predecir el default
de pago de tarjetas de crédito.
"""

import gzip
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLUMNS = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]


def load_datasets():
    """Carga los conjuntos de entrenamiento y prueba desde los csv comprimidos"""
    train_df = pd.read_csv("files/input/train_data.csv.zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip")
    return train_df, test_df


def clean_dataset(df):
    """
    - Renombra la columna objetivo a "default"
    - Elimina la columna ID
    - Elimina registros con EDUCATION=0 o MARRIAGE=0 (información no disponible)
    - Agrupa EDUCATION > 4 en la categoría "others" (4)
    """
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.loc[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda value: 4 if value > 4 else value)
    df = df.dropna()
    return df


def split_features_target(df):
    """Separa las variables explicativas (x) de la variable objetivo (y)"""
    x = df.drop(columns=["default"])
    y = df["default"]
    return x, y


def build_pipeline():
    """
    Pipeline con:
    - One-hot-encoding para las variables categóricas
    - Modelo de bosques aleatorios (Random Forest)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "one_hot_encoding",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            )
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    return pipeline


def optimize_hyperparameters(pipeline, x_train, y_train):
    """
    Búsqueda de hiperparámetros usando 10 splits de validación cruzada
    y precisión balanceada como métrica de evaluación.
    """
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(x_train, y_train)

    return grid_search


def save_model(model, output_path="files/models/model.pkl.gz"):
    """Guarda el modelo entrenado, comprimido con gzip"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_path, "wb") as file:
        pickle.dump(model, file)


def compute_metrics(model, x, y, dataset_name):
    """Calcula precision, precision balanceada, recall y f1-score"""
    predictions = model.predict(x)

    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "recall": recall_score(y, predictions),
        "f1_score": f1_score(y, predictions),
    }


def compute_confusion_matrix(model, x, y, dataset_name):
    """Calcula la matriz de confusión en el formato pedido"""
    predictions = model.predict(x)
    cm = confusion_matrix(y, predictions)

    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1]),
        },
    }


def save_results(results, output_path="files/output/metrics.json"):
    """Guarda los resultados como un archivo con un diccionario JSON por línea"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for entry in results:
            file.write(json.dumps(entry) + "\n")


def main():
    """Ejecuta el flujo completo del taller"""

    train_df, test_df = load_datasets()
    train_df = clean_dataset(train_df)
    test_df = clean_dataset(test_df)

    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    pipeline = build_pipeline()
    model = optimize_hyperparameters(pipeline, x_train, y_train)

    save_model(model)

    results = [
        compute_metrics(model, x_train, y_train, "train"),
        compute_metrics(model, x_test, y_test, "test"),
        compute_confusion_matrix(model, x_train, y_train, "train"),
        compute_confusion_matrix(model, x_test, y_test, "test"),
    ]

    save_results(results)


if __name__ == "__main__":
    main()