# Experimentación de regresión logística con MLflow

**Autora:** Ariadna Meliza Martínez Anaya  
**Matrícula:** A01797975  
**Materia:** MLOps

Este proyecto convierte el modelo base de un notebook en un script de entrenamiento parametrizado. Cada ejecución registra parámetros, métricas, artefactos y el pipeline entrenado en MLflow.

## Archivos

| Archivo | Contenido |
|---|---|
| `train.py` | Carga, limpieza, validación cruzada, entrenamiento, evaluación y registro en MLflow |
| `requirements.txt` | Versiones de los paquetes del entorno utilizado |
| `PCOS_data_without_infertility.xlsx` | Dataset utilizado |
| `README.md` | Instrucciones de ejecución y atribución de los datos |

## Preparación

El proyecto se ejecutó con **Python 3.14.7**.

Descarga el repositorio mediante **Code → Download ZIP**, descomprímelo y abre una terminal en la carpeta que contiene `train.py`.

Instala las dependencias:

```bash
python -m pip install -r requirements.txt
```

El Excel debe permanecer junto a `train.py`. El script lee la hoja `Full_new`.

## Ejecución

Para entrenar con los valores predeterminados:

```bash
python train.py
```

Para consultar los argumentos:

```bash
python train.py --help
```

| Argumento | Valor predeterminado | Propósito |
|---|---|---|
| `--C` | `1.0` | Inverso de la intensidad de regularización; debe ser positivo |
| `--class-weight` | `none` | Ponderación de clases: `none` o `balanced` |
| `--fit-intercept` / `--no-fit-intercept` | Activado | Permitir o impedir el aprendizaje del intercepto |
| `--seed` | `42` | Semilla para la división de datos y validación cruzada |
| `--run-name` | Nombre automático | Nombre de la corrida en MLflow |

## Configuraciones comparadas

Ejecuta los siguientes comandos uno por uno:

```bash
python train.py --run-name A_baseline --C 1 --class-weight none --fit-intercept --seed 42
python train.py --run-name B_C_0.1 --C 0.1 --class-weight none --fit-intercept --seed 42
python train.py --run-name C_C_10 --C 10 --class-weight none --fit-intercept --seed 42
python train.py --run-name D_balanced --C 1 --class-weight balanced --fit-intercept --seed 42
python train.py --run-name E_sin_intercepto --C 1 --class-weight none --no-fit-intercept --seed 42
```

Cada alternativa modifica un hiperparámetro respecto del baseline.

## Evaluación

Se utiliza una división estratificada de **80 % entrenamiento y 20 % prueba**. Dentro del entrenamiento se aplica validación cruzada estratificada de **cinco particiones**.

El pipeline incluye imputación por mediana y escalamiento para variables numéricas, imputación por categoría más frecuente y codificación One-Hot para variables categóricas, y regresión logística. En cada partición, el preprocesamiento se ajusta únicamente con los datos que entrenan.

Las métricas registradas son:

- `cv_f1_mean`: promedio del F1 de validación cruzada.
- `cv_f1_std`: desviación estándar del F1 entre particiones.
- `accuracy`, `precision`, `recall` y `f1`: resultados del conjunto de prueba.

La selección se basa en el mayor `cv_f1_mean`. La configuración B obtuvo aproximadamente **0.829**, frente a **0.813** del baseline. En prueba, B obtuvo un F1 de **0.824**, mientras el baseline alcanzó **0.865**. Estos resultados no permiten afirmar que B sea superior en cualquier muestra futura.

## Visualización en MLflow

Desde la misma carpeta, abre otra terminal y ejecuta:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Abre **http://127.0.0.1:5001** y selecciona el experimento **`PCOS_Logistica_Parametrizada`**.

Selecciona las corridas A–E y utiliza **Compare** para contrastar sus parámetros y métricas.

Cada corrida completa incluye:

- Los hiperparámetros, la semilla y el número de particiones.
- Las seis métricas de evaluación.
- `resultados.json`: métricas de prueba, matriz de confusión y orden de clases.
- `entorno.json`: versiones de Python y principales librerías.
- `pipeline_logistica`: pipeline entrenado, disponible en **Logged models**.

La base `mlflow.db` y los artefactos se generan localmente al ejecutar el script. Ejecuta los comandos desde la misma carpeta para utilizar el mismo almacenamiento.

## Reproducibilidad

Para repetir la configuración B:

```bash
python train.py --run-name B_repeticion --C 0.1 --class-weight none --fit-intercept --seed 42
```

Compara sus métricas con las de `B_C_0.1`. En el entorno utilizado, las seis métricas coincidieron al conservar los datos, su orden, los argumentos y las dependencias. Los identificadores de corrida y modelo son distintos en cada ejecución.

El script también carga el pipeline guardado y comprueba que sus predicciones sobre el conjunto de prueba coinciden con las del pipeline original.

## Datos y atribución

Este proyecto utiliza el dataset *Polycystic ovary syndrome (PCOS)*, publicado por **Prasoon Kottarathil** en [Kaggle](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos).

El archivo `PCOS_data_without_infertility.xlsx` se incluye bajo la licencia [Creative Commons Atribución–NoComercial–CompartirIgual 4.0 Internacional (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

El Excel se conserva sin modificaciones. La limpieza y las transformaciones se aplican desde el código, sin sobrescribirlo.

Mi contribución consiste en el código de procesamiento, entrenamiento parametrizado, evaluación y seguimiento de experimentos. La atribución no implica que el autor del dataset respalde este proyecto.

La reutilización del dataset debe respetar su licencia: atribuir la fuente, limitarse a usos no comerciales, indicar las modificaciones y distribuir las adaptaciones bajo la misma licencia. Esta declaración corresponde al dataset y no asigna automáticamente una licencia al código del repositorio.
