# -*- coding: utf-8 -*-
import sys, numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from scipy.stats import norm   # usamos scipy para obtener z según nivel de confianza

def mc_integrate(f, a, b, N, seed=None, conf=95):
    """
    Integral de f(x) en [a,b] con Monte Carlo.
    Parámetros:
        f    : función a integrar
        a,b  : límites de integración (ej: 0, 1)
        N    : número de muestras (ej: 100_000)
        seed : opcional, semilla aleatoria (ej: 42)
        conf : nivel de confianza en % (default 95)
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=N)
    fx = np.asarray(f(x), dtype=float)
    mask = np.isfinite(fx)
    if not np.all(mask):
        fx = fx[mask]
        if fx.size == 0:
            raise ValueError("Todas las evaluaciones de f(x) fueron no finitas.")

    mean_f = np.mean(fx)
    sigma = np.std(fx, ddof=1)
    volume = (b - a)
    estimate = volume * mean_f
    se = volume * sigma / np.sqrt(fx.size)

    # valor crítico z según el nivel de confianza
    z = norm.ppf(0.5 + conf/200.0)
    ci = (estimate - z*se, estimate + z*se)

    return {"I": estimate, "SE": se, "CI": ci, "N": fx.size}

# ================================
# 👇 EJEMPLO DE USO: CAMBIÁ ESTOS VALORES 👇
# ================================
if __name__ == "__main__":
    import math

    # 1) DEFINÍ TU FUNCIÓN ACÁ 👇
    def f(x):
        return np.exp(-x**2)   # <--- CAMBIÁ ESTA FUNCIÓN

    # 2) DEFINÍ LOS PARÁMETROS DE ENTRADA 👇
    a = 0        # <--- CAMBIÁ: límite inferior del intervalo
    b = 1        # <--- CAMBIÁ: límite superior del intervalo
    N = 100000   # <--- CAMBIÁ: número de muestras
    seed = 42    # <--- CAMBIÁ (opcional): semilla para reproducibilidad
    conf = 95    # <--- CAMBIÁ (opcional): nivel de confianza (%)

    # 3) EJECUTAR MONTE CARLO 👇
    res = mc_integrate(f, a=a, b=b, N=N, seed=seed, conf=conf)

    # 4) MOSTRAR RESULTADOS 👇
    print(f"Integral ∫_{a}^{b} f(x) dx")
    print(f"  I_MC ≈ {res['I']:.8f}")
    print(f"  SE   ≈ {res['SE']:.8f}")
    print(f"  IC   ≈ [{res['CI'][0]:.8f}, {res['CI'][1]:.8f}]  (nivel {conf}%)")
    print(f"  N efectivo = {res['N']}")
