
from pathlib import Path
import pandas as pd


def cargar_datos(nombre_archivo):
    """
    Carga un archivo CSV ubicado en la carpeta data del proyecto.
    """
    ruta = Path("data") / nombre_archivo

    if not ruta.exists():
        ruta = Path("..") / "data" / nombre_archivo

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {nombre_archivo}. "
            "Verifica que esté dentro de la carpeta data."
        )

    return pd.read_csv(ruta)
