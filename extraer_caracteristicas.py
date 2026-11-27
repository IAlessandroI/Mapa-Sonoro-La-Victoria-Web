"""Genera el CSV de características desde carpetas de audios (una carpeta por clase).
Usa la misma función que la página web (caracteristicas.py), así los números coinciden.
Necesita: audios/<clase>/*.wav  y  metadatos.csv (archivo, zona, hora)
"""
import os
import pandas as pd
from caracteristicas import extraer

CARPETA_AUDIOS = "audios"
METADATOS = "metadatos.csv"
SALIDA = "Data_Sonidos_Nuevos.csv"

meta = pd.read_csv(METADATOS)
filas = []
for clase in sorted(os.listdir(CARPETA_AUDIOS)):
    carpeta = os.path.join(CARPETA_AUDIOS, clase)
    if not os.path.isdir(carpeta):
        continue
    for archivo in sorted(os.listdir(carpeta)):
        if not archivo.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
            continue
        datos = meta[meta["archivo"] == archivo]
        if datos.empty:
            print(f"AVISO: {archivo} no está en {METADATOS}, se omite"); continue
        fila = {"archivo": archivo, "zona": datos.iloc[0]["zona"], "hora": int(datos.iloc[0]["hora"])}
        fila.update(extraer(os.path.join(carpeta, archivo)))
        fila["clase"] = clase
        filas.append(fila); print("Procesado:", archivo)

pd.DataFrame(filas).round(4).to_csv(SALIDA, index=False)
print(f"Listo: {len(filas)} audios en {SALIDA}")
