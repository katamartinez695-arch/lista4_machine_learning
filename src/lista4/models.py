
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


def ajustar_regresion_lineal(X, y):
    """
    Ajusta una regresión lineal con intercepto.
    """
    modelo = LinearRegression(fit_intercept=True)
    modelo.fit(X, y)
    return modelo


def ajustar_regresion_logistica(X, y):
    """
    Ajusta regresión logística con los parámetros
    exigidos en el ejercicio 4.
    """
    modelo = LogisticRegression(
        solver="lbfgs",
        C=1e6,
        max_iter=10000
    )
    modelo.fit(X, y)
    return modelo
