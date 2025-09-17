# lagrange_gui_final_fix2.py
import sympy as sp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ---------- Lagrange ----------
def lagrange_interpolacion(puntos):
    x = sp.Symbol('x')
    P = 0
    for i, (xi, yi) in enumerate(puntos):
        Li = 1
        for j, (xj, _) in enumerate(puntos):
            if i != j:
                Li *= (x - xj) / (xi - xj)
        P += yi * Li
    # fracciones limpias
    P = sp.expand(P)
    P = sp.nsimplify(P, rational=True)
    return sp.together(sp.simplify(P))

# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interpolación de Lagrange (fracciones + gráfico)")
        self.geometry("1100x680")

        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, width=360)
        right = ttk.Frame(paned)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        box = ttk.LabelFrame(left, text="Puntos")
        box.pack(fill="x", padx=10, pady=(10,6))
        ttk.Label(box, text="x:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.e_x = ttk.Entry(box, width=10); self.e_x.grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(box, text="y:").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        self.e_y = ttk.Entry(box, width=10); self.e_y.grid(row=0, column=3, sticky="w", padx=2)
        ttk.Button(box, text="Agregar", command=self.agregar_punto).grid(row=0, column=4, padx=6)

        ttk.Label(box, text="Xmáx (opcional):").grid(row=1, column=0, columnspan=2, sticky="e", padx=4)
        self.e_xmax = ttk.Entry(box, width=10); self.e_xmax.grid(row=1, column=2, sticky="w")
        ttk.Label(box, text="Ymáx (opcional):").grid(row=1, column=3, sticky="e", padx=4)
        self.e_ymax = ttk.Entry(box, width=10); self.e_ymax.grid(row=1, column=4, sticky="w")

        lst_frame = ttk.Frame(left); lst_frame.pack(fill="x", padx=10, pady=(0,6))
        self.lst = tk.Listbox(lst_frame, height=7); self.lst.pack(side="left", fill="x", expand=True)
        sb = ttk.Scrollbar(lst_frame, orient="vertical", command=self.lst.yview)
        sb.pack(side="right", fill="y"); self.lst.config(yscrollcommand=sb.set)

        btns = ttk.Frame(left); btns.pack(fill="x", padx=10, pady=6)
        ttk.Button(btns, text="Calcular y graficar", command=self.calcular_y_graficar).pack(fill="x", pady=2)
        ttk.Button(btns, text="Eliminar seleccionado", command=self.eliminar_sel).pack(fill="x", pady=2)
        ttk.Button(btns, text="Limpiar puntos", command=self.limpiar_puntos).pack(fill="x", pady=2)

        poly_frame = ttk.LabelFrame(left, text="Polinomio (fracciones)")
        poly_frame.pack(fill="both", expand=True, padx=10, pady=(6,10))
        self.txt_poly = tk.Text(poly_frame, height=10, wrap="word")
        self.txt_poly.pack(fill="both", expand=True, padx=6, pady=6)

        self.fig, self.ax = plt.subplots(figsize=(8,6), dpi=110)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        self._puntos = []
        self._P = None
        self._init_plot()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_title("Gráfico P(x) y puntos")
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.ax.axhline(0, color="black", lw=0.8)
        self.ax.axvline(0, color="black", lw=0.8)
        self.ax.grid(True, ls="--", lw=0.6)
        self.canvas.draw()

    def agregar_punto(self):
        x_str = self.e_x.get().strip().replace(",", ".")
        y_str = self.e_y.get().strip().replace(",", ".")
        try:
            xi = sp.nsimplify(x_str, rational=True)
            yi = sp.nsimplify(y_str, rational=True)
        except Exception:
            messagebox.showerror("Error", "x o y inválidos."); return
        self._puntos.append((xi, yi))
        self.lst.insert("end", f"({xi}, {yi})")
        self.e_x.delete(0, "end"); self.e_y.delete(0, "end")

    def eliminar_sel(self):
        sel = self.lst.curselection()
        if not sel: return
        idx = sel[0]; self.lst.delete(idx); del self._puntos[idx]

    def limpiar_puntos(self):
        self.lst.delete(0, "end"); self._puntos.clear(); self._P = None
        self.txt_poly.delete("1.0", "end"); self._init_plot()

    def calcular_y_graficar(self):
        if len(self._puntos) < 2:
            messagebox.showwarning("Aviso", "Agrega al menos 2 puntos."); return

        # 1) Polinomio
        try:
            P = lagrange_interpolacion(self._puntos)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        self._P = P

        # 2) Mostrar polinomio (fracciones)
        self.txt_poly.delete("1.0", "end")
        self.txt_poly.insert("end", sp.pretty(P) + "\n")

        # 3) Preparar gráfico
        self.ax.clear()
        self.ax.set_title(r"$\displaystyle " + sp.latex(P) + r"$", fontsize=14, pad=14)
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.ax.grid(True, ls="--", lw=0.6)
        self.ax.axhline(0, color="black", lw=0.8); self.ax.axvline(0, color="black", lw=0.8)

        # 4) Datos y rangos
        xs_pts = [float(p[0]) for p in self._puntos]
        ys_pts = [float(p[1]) for p in self._puntos]
        xmin, xmax = min(xs_pts), max(xs_pts)
        pad_x = 0.2 * (xmax - xmin if xmax > xmin else 1.0)

        try:
            xmax_user = float(self.e_xmax.get().replace(",", ".")) if self.e_xmax.get().strip() else None
            ymax_user = float(self.e_ymax.get().replace(",", ".")) if self.e_ymax.get().strip() else None
        except Exception:
            messagebox.showerror("Error", "Xmáx/Ymáx inválidos."); return

        if xmax_user is not None:
            x_left, x_right = -abs(xmax_user), abs(xmax_user)
        else:
            x_left, x_right = xmin - pad_x, xmax + pad_x
        if x_left == x_right:
            x_left -= 1; x_right += 1

        # 5) Evaluar P(x) con numpy (robusto)
        x = sp.Symbol('x')
        f = sp.lambdify(x, P, modules=["numpy"])
        xs = np.linspace(x_left, x_right, 800)
        try:
            ys = np.asarray(f(xs), dtype=float)
        except Exception:
            # Fallback si falla la vectorización
            ys = np.array([float(sp.N(P.subs(x, t))) for t in xs], dtype=float)

        # 6) Dibujar SIEMPRE límites y datos
        self.ax.set_xlim(x_left, x_right)
        if ymax_user is not None:
            self.ax.set_ylim(-abs(ymax_user), abs(ymax_user))

        # Curva y puntos (aunque la curva falle, los puntos se ven)
        try:
            self.ax.plot(xs, ys, label="P(x)")
        except Exception:
            pass
        self.ax.scatter(xs_pts, ys_pts, c="red", zorder=5, label="Puntos")

        # Ajuste Y automático si no se pidió Ymáx
        if ymax_user is None:
            try:
                y_min, y_max = np.nanmin(ys), np.nanmax(ys)
                if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
                    raise ValueError
                pad_y = 0.1 * (y_max - y_min)
                self.ax.set_ylim(y_min - pad_y, y_max + pad_y)
            except Exception:
                self.ax.set_ylim(-1, 1)

        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    App().mainloop()
