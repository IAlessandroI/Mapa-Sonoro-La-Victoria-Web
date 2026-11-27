"""Entrena la SVM (igual que en Orange) con un CSV de características y crea modelo_svm.json.
Uso:  python entrenar_modelo.py Data_Sonidos.csv
Necesita: pip install scikit-learn pandas numpy
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

df = pd.read_csv(sys.argv[1])
cols = [c for c in df.columns if c.endswith("_media") or c.endswith("_std")]
assert len(cols) == 36, f"Se esperaban 36 características y hay {len(cols)}"
clases = sorted(df["clase"].unique())
X = df[cols].values
y = df["clase"].map({c: i for i, c in enumerate(clases)}).values

offset = X.mean(axis=0); factor = 1.0 / X.std(axis=0)     # estandarizar, como Orange
Z = (X - offset) * factor
svm = SVC(kernel="rbf", C=1.0, gamma="auto", probability=True, random_state=0)

cv = cross_val_score(svm, Z, y, cv=StratifiedKFold(10, shuffle=True, random_state=0))
print(f"Exactitud en validación cruzada (10 pliegues): {cv.mean():.3f}")
svm.fit(Z, y)

json.dump(dict(clases=clases, columnas=cols, offset=offset.tolist(), factor=factor.tolist(),
               gamma=float(svm._gamma), sv=svm.support_vectors_.tolist(),
               dual_coef=svm.dual_coef_.tolist(), intercept=svm.intercept_.tolist(),
               n_support=svm.n_support_.tolist(), probA=svm.probA_.tolist(),
               probB=svm.probB_.tolist()),
          open("modelo_svm.json", "w", encoding="utf-8"))
print("Listo: modelo_svm.json creado. Cámbialo por el de la carpeta de la página.")
