import numpy as np

x0 = 1
tol = 1e-10       
ftol = 1e-12      
max_iter = 100

def f(x):
    return x**4 - 16

def df(x):
    return 4*x**3

def newton_tabla(x0, tol, ftol, max_iter):
    rows = []
    x_prev = None
    x = float(x0)

    for n in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError(f"Derivada nula en iteracion {n}, x={x}")

        x_new = x - fx / dfx

        Ea = abs(x_new - x) * 100 if x_prev is not None else None  # porcentaje
        Er = (Ea / abs(x_new)) if (Ea is not None and x_new != 0) else None

        rows.append((n, x, fx, dfx, Ea, Er))

        if (Ea is not None and Ea/100 < tol) or abs(f(x_new)) < ftol:
            return x_new, n+1, rows

        x_prev = x
        x = x_new

    raise RuntimeError("No convergio en el maximo de iteraciones")

if __name__ == "__main__":
    root, iters, table = newton_tabla(x0, tol, ftol, max_iter)

    print(f"{'n':<3} {'x_n':<15} {'f(x_n)':<15} {'f\'(x_n)':<15} {'Ea (%)':<15} {'Er (%)':<15}")
    print("-"*83)
    for n, xn, fxn, dfxn, Ea, Er in table:
        Ea_s = f"{Ea:.6g}" if Ea is not None else "-"
        Er_s = f"{Er:.6g}" if Er is not None else "-"
        print(f"{n:<3} {xn:<15.8f} {fxn:<15.8f} {dfxn:<15.8f} {Ea_s:<15} {Er_s:<15}")

    print("\nResultado final:")
    print(f"  Raiz aprox: {root:.12f} en {iters} iteraciones")
    print(f"  f(raiz) = {f(root):.3e}")
