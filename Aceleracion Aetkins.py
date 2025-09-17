# -*- coding: utf-8 -*-
# Punto Fijo con Aceleracion de Aitken - Algoritmo Generico

from math import cos, sqrt, pi, isfinite

# ===========================
# CONFIGURACION DEL PROBLEMA
# ===========================
x0 = 0.5   # Valor inicial
tol = 1e-10
ftol = 1e-12
max_iter = 100

def f(x):
    return cos(x) - x

def g(x):
    # Despeje elegido: x = cos(x)   (modificar segun problema)
    return cos(x)

# ===========================
# FUNCIONES GENERICAS
# ===========================
def aitken_delta2(x):
    """
    Aplica Aitken Delta^2 a la secuencia de punto fijo g(x)
    Devuelve: (x_acelerado, x1, x2, aplicado)
    """
    x1 = g(x)
    x2 = g(x1)
    denom = x2 - 2.0*x1 + x
    if denom == 0.0 or not isfinite(denom):
        return x2, x1, x2, False
    xA = x - (x1 - x)**2 / denom
    return xA, x1, x2, True

def solve_with_aitken(x0, tol, ftol, max_iter):
    x = float(x0)
    print(f"{'it':>3} | {'x':>18} | {'g(x)':>18} | {'x_aitken':>18} | {'|dx|':>10} | {'|f(x)|':>10} | Aitken")
    print("-"*95)
    for it in range(1, max_iter+1):
        xA, x1, x2, used = aitken_delta2(x)
        dx = abs(xA - x)
        fx = abs(f(xA))
        print(f"{it:3d} | {x:18.12f} | {x1:18.12f} | {xA:18.12f} | {dx:10.3e} | {fx:10.3e} | {'ON' if used else 'OFF'}")
        if dx < tol or fx < ftol:
            return xA, it
        x = xA
    raise RuntimeError("No convergio en el maximo de iteraciones")

# ===========================
# EJECUCION
# ===========================
if __name__ == "__main__":
    root, iters = solve_with_aitken(x0, tol, ftol, max_iter)
    print("\nResultado final:")
    print(f"  Raiz aprox: {root:.12f} en {iters} iteraciones")
    print(f"  f(raiz) = {f(root):.3e}")
    