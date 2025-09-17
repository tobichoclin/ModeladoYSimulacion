import sympy as sp

def lagrange_interpolacion(puntos):
    x = sp.Symbol('x')
    P = 0
    for i, (xi, yi) in enumerate(puntos):
        Li = 1
        for j, (xj, _) in enumerate(puntos):
            if i != j:
                Li *= (x - xj) / (xi - xj)
        P += yi * Li
    return sp.expand(P)

# Ejemplo de nodos
puntos = [(0, 0), (0.6, 0.47), (0.9, 0.64)]  # sin(x) en [0, pi]

P = lagrange_interpolacion(puntos)

# Convierte el polinomio en forma decimal con 6 cifras
P_decimal = sp.N(P, 6)

print("Polinomio de Lagrange:")
print(P_decimal)
