
import numpy as np
from sklearn.metrics import log_loss


def riesgo_cuadratico(y_real, y_predicho):
    """
    Calcula el riesgo empírico cuadrático:
    promedio de (y_real - y_predicho)^2.
    """
    y_real = np.asarray(y_real, dtype=float)
    y_predicho = np.asarray(y_predicho, dtype=float)

    if y_real.shape != y_predicho.shape:
        raise ValueError("Las variables deben tener la misma dimensión.")

    return np.mean((y_real - y_predicho) ** 2)


def riesgo_logistico(y_real, probabilidades):
    """
    Calcula el riesgo empírico logístico.
    Las probabilidades deben corresponder a la clase positiva.
    """
    y_real = np.asarray(y_real, dtype=int)
    probabilidades = np.asarray(probabilidades, dtype=float)

    if y_real.shape != probabilidades.shape:
        raise ValueError("Las variables deben tener la misma dimensión.")

    probabilidades = np.clip(probabilidades, 1e-15, 1 - 1e-15)

    return log_loss(
        y_real,
        probabilidades,
        labels=[0, 1]
    )


def errores_clasificacion(y_real, probabilidades, umbral=0.5):
    """
    Devuelve el número y la proporción de errores.
    """
    y_real = np.asarray(y_real, dtype=int)
    probabilidades = np.asarray(probabilidades, dtype=float)

    predicciones = (probabilidades >= umbral).astype(int)
    errores = int(np.sum(y_real != predicciones))
    proporcion = errores / len(y_real)

    return errores, proporcion
