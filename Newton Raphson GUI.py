# newton_gui_compacto.py  — GUI Newton-Raphson (derivada automática)
import tkinter as tk
from tkinter import ttk, messagebox
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# fuentes más grandes
plt.rcParams.update({"font.size": 12})
DEFAULT_FONT = ("Segoe UI", 12)

def newton_tabla_sympy(f_str, x0, tol, ftol, max_iter):
    x = sp.Symbol('x')
    f  = sp.sympify(f_str)
    df = sp.diff(f, x)
    f_num, df_num = sp.lambdify(x, f, "math"), sp.lambdify(x, df, "math")
    rows, hist = [], []
    x_prev, xn = None, float(x0)
    for n in range(max_iter):
        fx, dfx = float(f_num(xn)), float(df_num(xn))
        if dfx == 0.0:
            raise ZeroDivisionError(f"Derivada nula en iteración {n}, x={xn}")
        x_new = xn - fx/dfx
        Ea = abs(x_new - xn) if x_prev is not None else None
        Er = (Ea/abs(x_new)) if (Ea is not None and x_new != 0) else None
        rows.append((n, xn, fx, dfx, Ea, Er))
        hist.append(xn)
        if (Ea is not None and Ea < tol) or abs(f_num(x_new)) < ftol:
            hist.append(x_new)
            return float(x_new), n+1, rows, hist, f_num
        x_prev, xn = xn, x_new
    raise RuntimeError("No convergió en el máximo de iteraciones")

class NewtonGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Newton–Raphson (derivada automática)")
        self.geometry("1100x700")
        self.option_add("*Font", DEFAULT_FONT)

        frm = ttk.LabelFrame(self, text="Parámetros")
        frm.pack(fill="x", padx=10, pady=8)

        # etiquetas más claras
        self._row(frm, 0, "Función f(x):", "x**3 - 2*x - 5", "Punto inicial (x0):", "1.5")
        self._row(frm, 1, "Error total (Ea):", "1e-10", "Error en |f(x)|:", "1e-12")
        self._row(frm, 2, "Iteraciones máx.:", "100")

        btns = ttk.Frame(self); btns.pack(fill="x", padx=10, pady=5)
        ttk.Button(btns, text="Calcular", command=self.run).pack(side="left", padx=5)
        ttk.Button(btns, text="Limpiar", command=self.clear).pack(side="left", padx=5)

        self.lbl_res = ttk.Label(self, text="Resultado: -")
        self.lbl_res.pack(fill="x", padx=10, pady=5)

        cols = ("n", "x_n", "f(x_n)", "f'(x_n)", "Ea", "Er")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=170 if c!="n" else 60, anchor="center")
        self.tree.pack(fill="both", expand=False, padx=10, pady=8)

        self.fig, self.ax = plt.subplots(figsize=(8.5, 3.8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0,10))
        self._style_axes()

    def _row(self, parent, r, l1, v1, l2=None, v2=None):
        ttk.Label(parent, text=l1).grid(row=r, column=0, sticky="e", padx=6, pady=6)
        e1 = ttk.Entry(parent); e1.grid(row=r, column=1, sticky="we", padx=6, pady=6); e1.insert(0, v1)
        if r == 0: self.e_fx, self.e_x0 = e1, None
        if r == 1: self.e_tol, self.e_ftol = e1, None
        if r == 2: self.e_max = e1
        parent.columnconfigure(1, weight=1)
        if l2:
            ttk.Label(parent, text=l2).grid(row=r, column=2, sticky="e", padx=6, pady=6)
            e2 = ttk.Entry(parent, width=14); e2.grid(row=r, column=3, padx=6, pady=6, sticky="we"); e2.insert(0, v2)
            if r == 0: self.e_x0 = e2
            if r == 1: self.e_ftol = e2

    def _style_axes(self):
        self.ax.clear()
        self.ax.set_xlabel("x"); self.ax.set_ylabel("f(x)")
        self.ax.grid(True, linestyle="--", linewidth=0.6)
        # eje x SIEMPRE visible
        self.ax.axhline(0, color="black", linewidth=0.9)
        self.canvas.draw()

    def clear(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        self.lbl_res.config(text="Resultado: -")
        self._style_axes()

    def run(self):
        try:
            f_str = self.e_fx.get().strip()
            x0   = float(self.e_x0.get().replace(",", "."))
            tol  = float(self.e_tol.get().replace(",", "."))
            ftol = float(self.e_ftol.get().replace(",", "."))
            maxi = int(self.e_max.get())
        except Exception:
            messagebox.showerror("Error", "Parámetros numéricos inválidos."); return

        try:
            root, iters, rows, hist, f_num = newton_tabla_sympy(f_str, x0, tol, ftol, maxi)
        except Exception as e:
            messagebox.showerror("Error en cálculo", str(e)); return

        # tabla
        self.clear()
        for n, xn, fx, dfx, Ea, Er in rows:
            fmt = lambda v: "-" if v is None else f"{v:.8g}"
            self.tree.insert("", "end", values=(n, f"{xn:.12g}", f"{fx:.12g}", f"{dfx:.12g}", fmt(Ea), fmt(Er)))
        self.lbl_res.config(text=f"Resultado: raíz ~ {root:.12g} en {iters} it. | f(raíz) ~ {f_num(root):.3e}")

        # gráfico con eje x visible y límites claros
        xmin, xmax = min(hist + [root]), max(hist + [root])
        if xmin == xmax: xmin -= 1; xmax += 1
        pad = 0.25 * (xmax - xmin if xmax != xmin else 1.0)
        xs = np.linspace(xmin - pad, xmax + pad, 600)
        ys = f_num(xs)

        self._style_axes()
        self.ax.plot(xs, ys, label="f(x)")
        self.ax.scatter(hist, [f_num(h) for h in hist], s=35, label="iteraciones")
        self.ax.scatter([root], [0], marker="x", s=70, label="raíz")
        # aseguro que el 0 de y quede dentro del rango
        ymin, ymax = float(np.min(ys)), float(np.max(ys))
        if not (ymin <= 0 <= ymax):
            extra = 0.1 * max(1.0, abs(ymin) + abs(ymax))
            self.ax.set_ylim(min(ymin, 0) - extra, max(ymax, 0) + extra)
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    NewtonGUI().mainloop()
