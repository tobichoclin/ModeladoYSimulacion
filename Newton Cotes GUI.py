# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# =========================
# Helpers: parser de f(x)
# =========================
def make_funcs(expr_str):
    """
    Devuelve (f_numpy(x), f_sympy(x), sympy_symbol_x).
    Acepta sin, cos, tan, exp, log, sqrt, etc. con pi y e.
    """
    x = sp.Symbol('x')
    try:
        f_sym = sp.sympify(expr_str, locals={
            'pi': sp.pi, 'e': sp.E,
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'exp': sp.exp, 'log': sp.log, 'log10': sp.log, 'sqrt': sp.sqrt,
        })
    except Exception as e:
        raise ValueError(f"Expresion invalida: {e}")
    f_np = sp.lambdify(x, f_sym, modules=["numpy"])
    return f_np, f_sym, x

# =========================
# Metodos de integracion
# =========================
def rectangulo(f_np, a, b, n):
    h = (b - a) / n
    xm = a + (np.arange(n) + 0.5) * h
    y = f_np(xm)
    return h * np.sum(y), list(range(n)), xm.tolist(), y.tolist(), h

def trapecio(f_np, a, b, n):
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f_np(x)
    I = (h / 2.0) * (y[0] + 2.0 * np.sum(y[1:-1]) + y[-1])
    return I, list(range(n + 1)), x.tolist(), y.tolist(), h

def simpson_13(f_np, a, b, n):
    if n % 2 != 0:
        raise ValueError("Simpson 1/3 requiere n par")
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f_np(x)
    I = (h / 3.0) * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]))
    return I, list(range(n + 1)), x.tolist(), y.tolist(), h

def simpson_38(f_np, a, b, n):
    if n % 3 != 0:
        raise ValueError("Simpson 3/8 requiere n multiplo de 3")
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f_np(x)
    idx = np.arange(1, n)
    coef = np.where(idx % 3 == 0, 2.0, 3.0)
    I = (3*h/8.0) * (y[0] + y[-1] + np.sum(coef * y[1:-1]))
    return I, list(range(n + 1)), x.tolist(), y.tolist(), h

# "Real": intenta integral analitica; si no, Simpson 1/3 muy fino
def integral_real(f_np, f_sym, x, a, b):
    try:
        F = sp.integrate(f_sym, (x, a, b))
        val = float(F.evalf())
        return val, "Exacta (Simbolica)"
    except Exception:
        n_fino = 2000
        h = (b - a) / n_fino
        xs = a + h * np.arange(n_fino + 1)
        ys = f_np(xs)
        I = (h/3.0)*(ys[0] + ys[-1] + 4*np.sum(ys[1:-1:2]) + 2*np.sum(ys[2:-1:2]))
        return float(I), f"Numerica fina (n={n_fino})"

# =========================
# GUI
# =========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Integracion Numerica - Newton-Cotes")
        self.geometry("1100x700")

        # Paneles: izquierda (inputs) / derecha (resultados, tabla, grafico)
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, width=300)
        right = ttk.Frame(paned)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        # ----- Columna izquierda -----
        box = ttk.LabelFrame(left, text="Entradas")
        box.pack(fill="x", padx=10, pady=10)

        ttk.Label(box, text="f(x):").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.ent_fx = ttk.Entry(box, width=24); self.ent_fx.grid(row=0, column=1, padx=5, pady=5)
        self.ent_fx.insert(0, "sin(x)")

        ttk.Label(box, text="a:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.ent_a = ttk.Entry(box, width=12); self.ent_a.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.ent_a.insert(0, "0")

        ttk.Label(box, text="b:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.ent_b = ttk.Entry(box, width=12); self.ent_b.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.ent_b.insert(0, "pi")

        ttk.Label(box, text="n (subintervalos):").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.ent_n = ttk.Entry(box, width=12); self.ent_n.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        self.ent_n.insert(0, "12")

        ttk.Label(box, text="Metodo:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.cmb_met = ttk.Combobox(box, width=20, state="readonly",
                                    values=["Rectangulo", "Trapecio", "Simpson 1/3", "Simpson 3/8"])
        self.cmb_met.grid(row=4, column=1, padx=5, pady=5)
        self.cmb_met.current(2)

        btns = ttk.Frame(left); btns.pack(fill="x", padx=10, pady=(0,10))
        ttk.Button(btns, text="Calcular", command=self.calcular).pack(fill="x", pady=4)
        ttk.Button(btns, text="Graficar", command=self.graficar).pack(fill="x", pady=4)

        # ----- Derecha: panel de resultados + tabla + grafico -----
        results_frame = ttk.LabelFrame(right, text="Resultados")
        results_frame.pack(fill="x", padx=10, pady=(10,6))

        self.lbl_real = ttk.Label(results_frame, text="Resultado real: -",
                                  font=("Arial", 14, "bold"), foreground="blue")
        self.lbl_real.pack(fill="x", pady=6)

        self.lbl_met  = ttk.Label(results_frame, text="Resultado metodo: -",
                                  font=("Arial", 14, "bold"), foreground="darkgreen")
        self.lbl_met.pack(fill="x", pady=6)

        top = ttk.LabelFrame(right, text="Tabla (i, xi, f(xi))")
        top.pack(fill="both", expand=True, padx=10, pady=6)
        self.tree = ttk.Treeview(top, columns=("i","xi","fxi"), show="headings", height=8)
        self.tree.heading("i", text="i")
        self.tree.heading("xi", text="xi")
        self.tree.heading("fxi", text="f(xi)")
        self.tree.column("i", width=60, anchor="center")
        self.tree.column("xi", width=160, anchor="e")
        self.tree.column("fxi", width=160, anchor="e")
        self.tree.pack(fill="both", expand=True)

        bottom = ttk.LabelFrame(right, text="Grafico")
        bottom.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.fig, self.ax = plt.subplots(figsize=(7,3.5), dpi=110)
        self.canvas = FigureCanvasTkAgg(self.fig, master=bottom)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._last_plot = None  # (f_np, a, b, metodo, xs, ys, h)

    # ----------------------------
    def _leer_entradas(self):
        fx_str = self.ent_fx.get().strip().replace(",", ".")
        a_str = self.ent_a.get().strip().replace(",", ".")
        b_str = self.ent_b.get().strip().replace(",", ".")
        n_str = self.ent_n.get().strip()
        metodo = self.cmb_met.get()

        f_np, f_sym, x = make_funcs(fx_str)
        a_val = float(sp.N(sp.sympify(a_str, locals={'pi': sp.pi, 'e': sp.E})))
        b_val = float(sp.N(sp.sympify(b_str, locals={'pi': sp.pi, 'e': sp.E})))
        n_val = int(n_str)
        if n_val <= 0:
            raise ValueError("n debe ser positivo")
        if a_val == b_val:
            raise ValueError("a y b no pueden ser iguales")
        return f_np, f_sym, x, a_val, b_val, n_val, metodo

    def _aplicar_metodo(self, f_np, a, b, n, metodo):
        if metodo == "Rectangulo":
            return rectangulo(f_np, a, b, n)
        elif metodo == "Trapecio":
            return trapecio(f_np, a, b, n)
        elif metodo == "Simpson 1/3":
            return simpson_13(f_np, a, b, n)
        elif metodo == "Simpson 3/8":
            return simpson_38(f_np, a, b, n)
        else:
            raise ValueError("Metodo no valido")

    def calcular(self):
        try:
            f_np, f_sym, x, a, b, n, metodo = self._leer_entradas()
            I_real, _ = integral_real(f_np, f_sym, x, a, b)
            I_met, idxs, xs, ys, h = self._aplicar_metodo(f_np, a, b, n, metodo)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return

        self.lbl_real.config(text=f"Resultado real: {I_real:.12g}")
        self.lbl_met.config(text=f"Resultado metodo ({metodo}): {I_met:.12g}")

        # tabla
        for row in self.tree.get_children():
            self.tree.delete(row)
        for i, (xi, yi) in enumerate(zip(xs, ys)):
            self.tree.insert("", "end", values=(i, f"{xi:.10g}", f"{yi:.10g}"))

        self._last_plot = (f_np, a, b, metodo, xs, ys, h)

    # ---- Dibujo de areas por metodo ----
    def _shade_rectangulo(self, f_np, a, b, n, h):
        # rectangulos de punto medio
        for k in range(n):
            x0 = a + k*h
            xm = x0 + h/2
            y  = float(f_np(xm))
            self.ax.fill([x0, x0, x0+h, x0+h], [0, y, y, 0], alpha=0.2, color="C1", edgecolor="C1")

    def _shade_trapecio(self, xs, ys):
        # trapezoides por subintervalo
        for i in range(len(xs)-1):
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = ys[i], ys[i+1]
            self.ax.fill([x0, x0, x1, x1], [0, y0, y1, 0], alpha=0.2, color="C1", edgecolor="C1")

    def _shade_simpson13(self, xs, ys, h):
        # bloques de 2 subintervalos: puntos i, i+1, i+2 -> parabolica
        for i in range(0, len(xs)-1, 2):
            if i+2 >= len(xs): break
            x0, x1, x2 = xs[i], xs[i+1], xs[i+2]
            y0, y1, y2 = ys[i], ys[i+1], ys[i+2]
            coeff = np.polyfit([x0, x1, x2], [y0, y1, y2], deg=2)
            xx = np.linspace(x0, x2, 80)
            yy = np.polyval(coeff, xx)
            self.ax.fill_between(xx, yy, 0, alpha=0.2, color="C1", step=None)

    def _shade_simpson38(self, xs, ys, h):
        # bloques de 3 subintervalos: puntos i..i+3 -> cubica
        for i in range(0, len(xs)-1, 3):
            if i+3 >= len(xs): break
            x0, x1, x2, x3 = xs[i], xs[i+1], xs[i+2], xs[i+3]
            y0, y1, y2, y3 = ys[i], ys[i+1], ys[i+2], ys[i+3]
            coeff = np.polyfit([x0, x1, x2, x3], [y0, y1, y2, y3], deg=3)
            xx = np.linspace(x0, x3, 100)
            yy = np.polyval(coeff, xx)
            self.ax.fill_between(xx, yy, 0, alpha=0.2, color="C1")

    def graficar(self):
        if self._last_plot is None:
            self.calcular()
            if self._last_plot is None:
                return
        f_np, a, b, metodo, xs_nodes, ys_nodes, h = self._last_plot

        self.ax.clear()
        X = np.linspace(a, b, 1000)
        Y = f_np(X)
        self.ax.plot(X, Y, label="f(x)")

        # sombrear segun metodo
        if metodo == "Rectangulo":
            n = len(xs_nodes)  # nodos de medios no se guardan; reconstruimos n
            # para rectangulo guardamos xs_nodes como midpoints; reconstruimos n por longitud
            # en calcular usamos list(range(n)) -> len(xs_nodes) == n
            self._shade_rectangulo(f_np, a, b, n, h)
        elif metodo == "Trapecio":
            self._shade_trapecio(xs_nodes, ys_nodes)
        elif metodo == "Simpson 1/3":
            self._shade_simpson13(xs_nodes, ys_nodes, h)
        elif metodo == "Simpson 3/8":
            self._shade_simpson38(xs_nodes, ys_nodes, h)

        # puntos (nodos o medios)
        label_pts = "puntos (nodos)" if metodo != "Rectangulo" else "puntos (medios)"
        self.ax.scatter(xs_nodes, ys_nodes, color="red", s=25, zorder=5, label=label_pts)

        self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.ax.grid(True, ls="--", lw=0.6)
        self.ax.legend()
        self.canvas.draw()

# =========================
if __name__ == "__main__":
    App().mainloop()
