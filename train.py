from pathlib import Path
import pandas as pd
import argparse
import mlflow
import json
from tempfile import TemporaryDirectory

import platform
import sklearn
import openpyxl

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def cargar_datos(ruta, hoja="Full_new"):
    ruta = Path(ruta)

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta.resolve()}"
        )

    datos = pd.read_excel(ruta, sheet_name=hoja)

    return datos

def limpiar_datos(datos):
    limpios = datos.copy()

    # Quitar espacios al inicio y al final de los nombres.
    limpios.columns = limpios.columns.str.strip()

    # Eliminar identificadores y la columna sin información útil.
    columnas_descartadas = [
        "Sl. No",
        "Patient File No.",
        "Unnamed: 44",
    ]

    limpios = limpios.drop(columns=columnas_descartadas)

    columnas_categoricas = [
        "Blood Group",
        "Cycle(R/I)",
        "Pregnant(Y/N)",
        "Weight gain(Y/N)",
        "hair growth(Y/N)",
        "Skin darkening (Y/N)",
        "Hair loss(Y/N)",
        "Pimples(Y/N)",
        "Fast food (Y/N)",
        "Reg.Exercise(Y/N)",
    ]

    for columna in columnas_categoricas:
        limpios[columna] = limpios[columna].astype("category")

    columnas_numericas = [
        "II    beta-HCG(mIU/mL)",
        "AMH(ng/mL)",
    ]

    for columna in columnas_numericas:
        limpios[columna] = pd.to_numeric(
            limpios[columna],
            errors="coerce",
        )

    return limpios

def dividir_datos(datos, test_size=0.20, seed=42):
    objetivo = "PCOS (Y/N)"

    X = datos.drop(columns=[objetivo])
    y = datos[objetivo].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test

def construir_pipeline(
    X_train,
    C=1.0,
    class_weight=None,
    fit_intercept=True,
    seed=42,
):
    columnas_num = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    columnas_cat = X_train.select_dtypes(
        include="category"
    ).columns.tolist()

    pipeline_num = Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalamiento", StandardScaler()),
    ])

    pipeline_cat = Pipeline([
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("codificacion", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocesador = ColumnTransformer([
        ("num", pipeline_num, columnas_num),
        ("cat", pipeline_cat, columnas_cat),
    ])

    modelo = Pipeline([
        ("preprocesamiento", preprocesador),
        ("clasificador", LogisticRegression(
            C=C,
            class_weight=class_weight,
            fit_intercept=fit_intercept,
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )),
    ])

    return modelo

def evaluar_modelo(modelo, X, y):
    predicciones = modelo.predict(X)

    metricas = {
        "accuracy": accuracy_score(y, predicciones),
        "precision": precision_score(
            y, predicciones, zero_division=0
        ),
        "recall": recall_score(
            y, predicciones, zero_division=0
        ),
        "f1": f1_score(
            y, predicciones, zero_division=0
        ),
    }

    matriz = confusion_matrix(
        y,
        predicciones,
        labels=[0, 1],
    )

    return metricas, matriz

def leer_argumentos():
    parser = argparse.ArgumentParser()

    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-weight",
        choices=["none","balanced"],
        default="none"
    )
    parser.add_argument(
        "--fit-intercept",
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument("--run-name", default=None)

    args = parser.parse_args()

    return args

def validar_modelo(modelo, X_train, y_train, seed):
    particiones = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    resultados = cross_val_score(
        modelo,
        X_train,
        y_train,
        cv=particiones,
        scoring="f1",
    )

    return resultados

def main():
    args = leer_argumentos()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("PCOS_Logistica_Parametrizada")

    print("C recibido:", args.C)

    ruta = Path(__file__).resolve().parent / "PCOS_data_without_infertility.xlsx"

    datos = cargar_datos(ruta)

    print("Dimensiones:", datos.shape)
    print("Primeras columnas:", datos.columns[:5].tolist())

    datos_limpios = limpiar_datos(datos)

    print("Dimensiones después de limpiar:", datos_limpios.shape)
    print("Primeras columnas limpias:", datos_limpios.columns[:5].tolist())

    print(
        datos_limpios[
            ["Blood Group", "II    beta-HCG(mIU/mL)", "AMH(ng/mL)"]
        ].dtypes
    )

    faltantes = datos_limpios.isna().sum()
    print("Columnas con faltantes:")
    print(faltantes[faltantes > 0])

    X_train, X_test, y_train, y_test = dividir_datos(
        datos_limpios,
        seed=args.seed
    )

    print("Dimensiones de entrenamiento:", X_train.shape)
    print("Dimensiones de prueba:", X_test.shape)

    print("Proporciones de clases en entrenamiento:")
    print(y_train.value_counts(normalize=True).sort_index())

    print("Proporciones de clases en prueba:")
    print(y_test.value_counts(normalize=True).sort_index())

    print(
        "¿El objetivo está entre los predictores?",
        "PCOS (Y/N)" in X_train.columns,
    )

    peso_clases = (None if args.class_weight == "none" else args.class_weight)

    modelo = construir_pipeline(
        X_train,
        C=args.C,
        class_weight=peso_clases,
        fit_intercept=args.fit_intercept,
        seed=args.seed
    )

    print(
        "Peso de clases del clasificador:",
        modelo.named_steps["clasificador"].class_weight
    )

    print(
        "Aprender intercepto:",
        modelo.named_steps["clasificador"].fit_intercept
    )

    print("Pasos del modelo:", list(modelo.named_steps))
    print(
        "C del clasificador:",
        modelo.named_steps["clasificador"].C,
    )

    preprocesador = modelo.named_steps["preprocesamiento"]

    for nombre, pipeline, columnas in preprocesador.transformers:
        print(f"Rama {nombre}: {len(columnas)} columnas")

    with mlflow.start_run(run_name=args.run_name) as run:
        clasificador = modelo.named_steps["clasificador"]

        mlflow.log_param("C", clasificador.C)
        mlflow.log_param("class_weight", clasificador.class_weight)
        mlflow.log_param("fit_intercept", clasificador.fit_intercept)
        mlflow.log_param("seed", args.seed)

        resultados_cv = validar_modelo(modelo, X_train, y_train, args.seed)

        print("F1 de cada partición:", resultados_cv)

        media_cv = float(resultados_cv.mean())
        desviacion_cv = float(resultados_cv.std())

        mlflow.log_metric("cv_f1_mean", media_cv)
        mlflow.log_metric("cv_f1_std", desviacion_cv)
        mlflow.log_param("cv_folds", 5)

        print("F1 medio de validación:", media_cv)
        print("Desviación del F1:", desviacion_cv)

        modelo.fit(X_train, y_train)

        metricas, matriz = evaluar_modelo(
            modelo,
            X_test,
            y_test,
        )

        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)

        with TemporaryDirectory() as carpeta:
            ruta_resultados = Path(carpeta) / "resultados.json"

            resultados = {
                "metricas": metricas,
                "matriz_confusion": matriz.tolist(),
                "orden_clases": [0,1]
            }

            with ruta_resultados.open(
                "w", encoding = "utf-8"
            ) as archivo:
                json.dump(resultados, archivo, indent = 2)

            mlflow.log_artifact(str(ruta_resultados))

            ruta_entorno = Path(carpeta) / "entorno.json"

            entorno = {
                "python":platform.python_version(),
                "pandas":pd.__version__,
                "scikit-learn": sklearn.__version__,
                "mlflow": mlflow.__version__,
                "openpyxl": openpyxl.__version__
            }

            with ruta_entorno.open("w", encoding="utf-8") as archivo:
                json.dump(entorno, archivo, indent= 2)

            print("Archivo de entorno creado:", ruta_entorno.is_file())

            mlflow.log_artifact(str(ruta_entorno))

            print("Registro de entorno completado")

        informacion_modelo = mlflow.sklearn.log_model(
            sk_model=modelo,
            name="pipeline_logistica",
            skops_trusted_types=["numpy.dtype"]
        )

        print("Direccion del modelo:", informacion_modelo.model_uri)

        modelo_recuperado = mlflow.sklearn.load_model(informacion_modelo.model_uri)

        pred_originales = modelo.predict(X_test)
        pred_recuperadas = modelo_recuperado.predict(X_test)

        coinciden = (pred_originales == pred_recuperadas).all()

        print("¿Coinciden las predicciones?", coinciden)

        print("ID de la corrida:", run.info.run_id)

    print("Métricas de prueba:")
    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor:.6f}")

    print("Matriz de confusión:")
    print(matriz)

if __name__ == "__main__":
    main()