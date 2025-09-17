import numpy as np

# =====================================================
# 🔧 CONFIGURACION: CAMBIAR SOLO ESTAS LINEAS
def f(x):
    return (np.sin(x) / x)          # Función
a = 0                                 # límite inferior
b = np.pi / 2                         # límite superior
n = 4                                 # intervalos
metodo = "trapecio"                  # opciones: "rectangulo", "trapecio", "simpson13", "simpson38"
# =====================================================

def rectangulo(f, a, b, n):
    h = (b - a) / n
    xm = a + (np.arange(n) + 0.5) * h
    return h * np.sum(f(xm))

def trapecio(f, a, b, n):
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f(x)
    return (h / 2.0) * (y[0] + 2.0 * np.sum(y[1:-1]) + y[-1])

def simpson_13(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("Simpson 1/3 requiere n par")
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f(x)
    return (h / 3.0) * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]))

def simpson_38(f, a, b, n):
    if n % 3 != 0:
        raise ValueError("Simpson 3/8 requiere n multiplo de 3")
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f(x)
    idx = np.arange(1, n)
    coef = np.where(idx % 3 == 0, 2.0, 3.0)
    return (3*h/8) * (y[0] + y[-1] + np.sum(coef * y[1:-1]))

# Seleccion del metodo
if metodo == "rectangulo":
    I = rectangulo(f, a, b, n)
elif metodo == "trapecio":
    I = trapecio(f, a, b, n)
elif metodo == "simpson13":
    I = simpson_13(f, a, b, n)
elif metodo == "simpson38":
    I = simpson_38(f, a, b, n)
else:
    raise ValueError("Metodo no valido")

print(f"Metodo: {metodo}")
print(f"Integral aproximada en [{a},{b}] con n={n}: {I:.12f}")
