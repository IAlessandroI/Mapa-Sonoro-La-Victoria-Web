"""Calcula las 36 características de un audio (misma función para entrenar y para la página)."""
import numpy as np
import librosa

SR = 22050          # todos los audios se pasan a esta frecuencia, vengan como vengan
MAX_SEG = 30        # se usan como máximo 30 segundos

def extraer(ruta):
    y, sr = librosa.load(ruta, sr=SR, mono=True, duration=MAX_SEG)
    if len(y) < SR * 0.5:
        raise ValueError("El audio es muy corto (menos de medio segundo).")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    series = {
        "centroide": librosa.feature.spectral_centroid(y=y, sr=sr)[0],
        "ancho_banda": librosa.feature.spectral_bandwidth(y=y, sr=sr)[0],
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr)[0],
        "zcr": librosa.feature.zero_crossing_rate(y)[0],
        "rms": librosa.feature.rms(y=y)[0],
    }
    fila = {}
    for i in range(13):
        fila[f"mfcc{i+1}_media"] = float(mfcc[i].mean())
        fila[f"mfcc{i+1}_std"] = float(mfcc[i].std())
    for nombre, s in series.items():
        fila[f"{nombre}_media"] = float(s.mean())
        fila[f"{nombre}_std"] = float(s.std())
    return fila
