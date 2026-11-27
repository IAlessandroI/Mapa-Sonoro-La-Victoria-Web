"""Predictor de la SVM de Orange sin necesitar Orange ni scikit-learn (solo numpy)."""
import json, os
import numpy as np

class ModeloSonidos:
    def __init__(self, ruta=None):
        ruta = ruta or os.path.join(os.path.dirname(__file__), "modelo_svm.json")
        m = json.load(open(ruta, encoding="utf-8"))
        self.clases = m["clases"]
        self.columnas = m["columnas"]
        self.offset = np.array(m["offset"])
        self.factor = np.array(m["factor"])
        self.gamma = m["gamma"]
        self.sv = np.array(m["sv"])
        self.dual = np.array(m["dual_coef"])
        self.inter = np.array(m["intercept"])
        self.nsv = np.array(m["n_support"])
        self.probA = np.array(m["probA"]); self.probB = np.array(m["probB"])
        self.k = len(self.clases)
        self.ini = np.concatenate([[0], np.cumsum(self.nsv)])

    def _decisiones(self, z):
        kern = np.exp(-self.gamma * ((self.sv - z) ** 2).sum(axis=1))
        dec, p = [], 0
        for i in range(self.k):
            for j in range(i + 1, self.k):
                si, ei = self.ini[i], self.ini[i + 1]
                sj, ej = self.ini[j], self.ini[j + 1]
                v = (self.dual[j - 1, si:ei] * kern[si:ei]).sum() \
                    + (self.dual[i, sj:ej] * kern[sj:ej]).sum() + self.inter[p]
                dec.append(v); p += 1
        return np.array(dec)

    def _probabilidades(self, dec):
        k = self.k
        r = np.zeros((k, k)); p = 0
        for i in range(k):
            for j in range(i + 1, k):
                f = dec[p] * self.probA[p] + self.probB[p]
                pr = np.exp(-f) / (1 + np.exp(-f)) if f >= 0 else 1 / (1 + np.exp(f))
                pr = min(max(pr, 1e-7), 1 - 1e-7)
                r[i, j] = pr; r[j, i] = 1 - pr; p += 1
        Q = np.zeros((k, k))
        for t in range(k):
            Q[t, t] = sum(r[j, t] ** 2 for j in range(k) if j != t)
            for j in range(k):
                if j != t: Q[t, j] = -r[j, t] * r[t, j]
        pv = np.full(k, 1.0 / k); eps = 0.005 / k
        for _ in range(max(100, k)):
            Qp = Q @ pv; pQp = pv @ Qp
            if np.abs(Qp - pQp).max() < eps: break
            for t in range(k):
                diff = (-Qp[t] + pQp) / Q[t, t]
                pv[t] += diff
                pQp = (pQp + diff * (diff * Q[t, t] + 2 * Qp[t])) / (1 + diff) ** 2
                Qp = (Qp + diff * Q[t]) / (1 + diff)
                pv /= (1 + diff)
        return pv

    def predecir(self, x):
        """x: 36 valores en el orden de self.columnas. Devuelve (clase, {clase: prob})."""
        z = (np.asarray(x, float) - self.offset) * self.factor
        dec = self._decisiones(z)
        votos = np.zeros(self.k); p = 0
        for i in range(self.k):
            for j in range(i + 1, self.k):
                votos[i if dec[p] > 0 else j] += 1; p += 1
        pv = self._probabilidades(dec)
        return self.clases[int(votos.argmax())], dict(zip(self.clases, map(float, pv)))
