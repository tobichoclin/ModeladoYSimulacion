# -*- coding: utf-8 -*-
"""
GUI Monte Carlo 1D y 2D
- 1D y 2D con barra de resultados en 2 renglones:
  1) I_MC y Media muestral
  2) Error estándar (SE), Desviación (σ_f), IC y Valor (Gauss)
"""
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad, dblquad   # <-- Gauss–Kronrod (quadpack)

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ========= Utilidades =========
def safe_eval_function_1d(expr):
    allowed = {
        "np": np,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
        "abs": np.abs, "floor": np.floor, "ceil": np.ceil,
        "pi": np.pi, "e": np.e,
    }
    code = compile(expr, "<fx-1d>", "eval")
    def f(x):
        return eval(code, {"__builtins__": {}}, {**allowed, "x": x})
    return f

def safe_eval_function_2d(expr):
    allowed = {
        "np": np,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
        "abs": np.abs, "floor": np.floor, "ceil": np.ceil,
        "pi": np.pi, "e": np.e,
    }
    code = compile(expr, "<fxy-2d>", "eval")
    def f(x, y):
        return eval(code, {"__builtins__": {}}, {**allowed, "x": x, "y": y})
    return f

def mc_mean_value_1d(f, a, b, N, seed=None, conf=95):
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=N)
    fx = np.asarray(f(x), dtype=float)
    mask = np.isfinite(fx)
    if not np.all(mask):
        x = x[mask]; fx = fx[mask]
        if fx.size == 0:
            raise ValueError("Todas las evaluaciones f(x) fueron no finitas.")
    mean_f = np.mean(fx)  # media muestral
    sigma = np.std(fx, ddof=1) if fx.size > 1 else 0.0
    vol = (b - a)
    I = vol * mean_f
    SE = vol * sigma / np.sqrt(fx.size) if fx.size > 0 else float("nan")
    z = norm.ppf(0.5 + conf/200.0)
    CI = (I - z*SE, I + z*SE)
    return {
        "I": I, "SE": SE, "sigma_f": sigma, "mean_f": mean_f,
        "CI": CI, "N": fx.size, "x": x, "fx": fx
    }

def mc_mean_value_2d(f, ax, bx, ay, by, N, seed=None, conf=95):
    rng = np.random.default_rng(seed)
    x = rng.uniform(ax, bx, size=N)
    y = rng.uniform(ay, by, size=N)
    fxy = np.asarray(f(x, y), dtype=float)
    mask = np.isfinite(fxy)
    if not np.all(mask):
        x = x[mask]; y = y[mask]; fxy = fxy[mask]
        if fxy.size == 0:
            raise ValueError("Todas las evaluaciones f(x,y) fueron no finitas.")
    mean_f = np.mean(fxy)  # media muestral 2D
    sigma = np.std(fxy, ddof=1) if fxy.size > 1 else 0.0
    vol = (bx - ax) * (by - ay)
    I = vol * mean_f
    SE = vol * sigma / np.sqrt(fxy.size) if fxy.size > 0 else float("nan")
    z = norm.ppf(0.5 + conf/200.0)
    CI = (I - z*SE, I + z*SE)
    return {
        "I": I, "SE": SE, "sigma_f": sigma, "mean_f": mean_f,
        "CI": CI, "N": fxy.size, "x": x, "y": y, "fxy": fxy
    }

def hit_miss_samples_1d(f, a, b, N, seed=None, grid=1000):
    xs = np.linspace(a, b, grid)
    vals = np.asarray(f(xs), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        ymin, ymax = -1.0, 1.0
    else:
        ymin = float(min(np.min(vals), 0.0))
        ymax = float(max(np.max(vals), 0.0))
        if abs(ymax - ymin) < 1e-12:
            pad = 1.0 if abs(ymax) < 1e-6 else 0.1*abs(ymax)
            ymin -= pad; ymax += pad
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=N)
    y = rng.uniform(ymin, ymax, size=N)
    fx = np.asarray(f(x), dtype=float)
    success = np.isfinite(fx) & np.isfinite(y)
    same_side = (y * fx) >= 0
    within = np.abs(y) <= np.abs(fx)
    success &= same_side & within
    return x, y, fx, success, (ymin, ymax)

def hit_miss_samples_2d(f, ax, bx, ay, by, N, seed=None, grid=50):
    gx = np.linspace(ax, bx, grid)
    gy = np.linspace(ay, by, grid)
    XX, YY = np.meshgrid(gx, gy)
    ZZ = np.asarray(f(XX, YY), dtype=float)
    ZZ = ZZ[np.isfinite(ZZ)]
    if ZZ.size == 0:
        zmin, zmax = -1.0, 1.0
    else:
        zmin = float(min(np.min(ZZ), 0.0))
        zmax = float(max(np.max(ZZ), 0.0))
        if abs(zmax - zmin) < 1e-12:
            pad = 1.0 if abs(zmax) < 1e-6 else 0.1*abs(zmax)
            zmin -= pad; zmax += pad
    rng = np.random.default_rng(seed)
    x = rng.uniform(ax, bx, size=N)
    y = rng.uniform(ay, by, size=N)
    z = rng.uniform(zmin, zmax, size=N)
    fxy = np.asarray(f(x, y), dtype=float)
    success = np.isfinite(fxy) & np.isfinite(z)
    same_side = (z * fxy) >= 0
    within = np.abs(z) <= np.abs(fxy)
    success &= same_side & within
    return x, y, z, fxy, success, (zmin, zmax)

# ========= GUI 1D =========
class MonteCarloGUI1D(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Monte Carlo 1D - Integración")
        self.geometry("1100x700")

        top = ttk.Frame(self, padding=8); top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="f(x):").grid(row=0, column=0, sticky="w")
        self.fx_entry = ttk.Entry(top, width=40); self.fx_entry.grid(row=0, column=1, padx=4, sticky="we")
        self.fx_entry.insert(0, "exp(-x**2)")
        ttk.Label(top, text="a:").grid(row=0, column=2, sticky="w")
        self.a_entry = ttk.Entry(top, width=10); self.a_entry.grid(row=0, column=3, padx=4); self.a_entry.insert(0, "0")
        ttk.Label(top, text="b:").grid(row=0, column=4, sticky="w")
        self.b_entry = ttk.Entry(top, width=10); self.b_entry.grid(row=0, column=5, padx=4); self.b_entry.insert(0, "1")
        ttk.Label(top, text="Muestras:").grid(row=0, column=6, sticky="w")
        self.N_entry = ttk.Entry(top, width=12); self.N_entry.grid(row=0, column=7, padx=4); self.N_entry.insert(0, "100000")
        ttk.Label(top, text="Seed (opcional):").grid(row=0, column=8, sticky="w")
        self.seed_entry = ttk.Entry(top, width=10); self.seed_entry.grid(row=0, column=9, padx=4); self.seed_entry.insert(0, "")
        ttk.Label(top, text="Confianza % (opcional):").grid(row=0, column=10, sticky="w")
        self.conf_entry = ttk.Entry(top, width=8); self.conf_entry.grid(row=0, column=11, padx=4); self.conf_entry.insert(0, "95")

        self.sim_btn = ttk.Button(top, text="Simular", command=self.simular); self.sim_btn.grid(row=0, column=12, padx=8)
        self.btn_2d = ttk.Button(top, text="Integral doble", command=self.abrir_2d); self.btn_2d.grid(row=0, column=13, padx=4)

        # Resultados (dos renglones)
        self.result_label1 = ttk.Label(self, text="Integral: —", font=("Segoe UI", 14, "bold"), wraplength=1050)
        self.result_label1.pack(side=tk.TOP, anchor="w", padx=12, pady=(6, 0))
        self.result_label2 = ttk.Label(self, text="", font=("Segoe UI", 13), wraplength=1050)
        self.result_label2.pack(side=tk.TOP, anchor="w", padx=12, pady=(0, 6))

        body = ttk.Frame(self); body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(body, padding=(8, 8)); left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="Muestras Monte Carlo").pack(anchor="w")
        cols = ("x", "y", "fx", "exito")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=25)
        for c, w in zip(cols, (110, 110, 110, 70)):
            self.tree.heading(c, text=c); self.tree.column(c, width=w, anchor="center")
        yscroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.Y); yscroll.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(body, padding=(8, 8)); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def abrir_2d(self):
        MonteCarloGUI2D(self)

    def simular(self):
        fx_text = self.fx_entry.get().strip()
        try:
            a = float(self.a_entry.get().strip()); b = float(self.b_entry.get().strip())
            if not (np.isfinite(a) and np.isfinite(b)) or not (b > a): raise ValueError("Intervalo inválido: a < b.")
        except Exception as e:
            messagebox.showerror("Error en intervalos", str(e)); return

        try:
            N = int(self.N_entry.get().replace("_", "").strip())
            if N <= 0: raise ValueError("Muestras debe ser > 0.")
        except Exception as e:
            messagebox.showerror("Error en muestras", f"Valor inválido de Muestras. {e}"); return

        conf = 95
        ct = self.conf_entry.get().strip()
        if ct:
            try:
                conf = float(ct)
                if not (0 < conf < 100): raise ValueError
            except Exception:
                messagebox.showerror("Error en confianza", "Ingrese 0–100 o vacío.")
                return

        seed = None
        st = self.seed_entry.get().strip()
        if st:
            try:
                seed = int(st)
            except Exception:
                messagebox.showerror("Error en seed", "Seed debe ser entero o vacío.")
                return

        try:
            f = safe_eval_function_1d(fx_text)
            test_x = np.array([(a + b) / 2.0], dtype=float)   # probamos en el medio
            _ = np.asarray(f(test_x), dtype=float)
        except Exception as e:
            messagebox.showerror(
                "Error en f(x)",
                f"No se pudo interpretar f(x).\nEj.: exp(-x**2), sin(x), etc.\n{e}"
            )
            return

        # Monte Carlo
        res = mc_mean_value_1d(f, a, b, N, seed=seed, conf=conf)
        I, SE, sigma, mean_f, (lo, hi) = res["I"], res["SE"], res["sigma_f"], res["mean_f"], res["CI"]

        # Gauss–Kronrod (quad)
        try:
            # wrapper escalar para quad (f es vectorizada)
            g = lambda t: float(np.asarray(f(np.array([t]))[0]))
            gauss_val, gauss_err = quad(g, a, b, epsabs=1e-10, epsrel=1e-10, limit=200)
            gauss_text = f"{gauss_val:.8f} (± {gauss_err:.1e})"
        except Exception:
            gauss_text = "n/d"

        self.result_label1.config(
            text=(f"Integral estimada: I_MC = {I:.8f}  |  Media muestral = {mean_f:.8f}")
        )
        self.result_label2.config(
            text=(f"Error estándar (SE) = {SE:.8f}  |  Desviación σ_f = {sigma:.8f}  |  "
                  f"IC {conf:.0f}%: [{lo:.8f}, {hi:.8f}]  |  Valor (Gauss) ≈ {gauss_text}")
        )

        # Visual + tabla
        xhm, yhm, fxhm, success, (ymin, ymax) = hit_miss_samples_1d(f, a, b, N=min(N, 5000), seed=seed)

        for r in self.tree.get_children(): self.tree.delete(r)
        max_rows = 2000
        shown = min(xhm.size, max_rows)
        for i in range(shown):
            self.tree.insert("", tk.END, values=(f"{xhm[i]:.6f}", f"{yhm[i]:.6f}", f"{fxhm[i]:.6f}", "✓" if success[i] else "✗"))

        self.ax.clear(); self.ax.grid(True, alpha=0.3); self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        xx = np.linspace(a, b, 800)
        try: yy = np.asarray(f(xx), dtype=float)
        except Exception: yy = np.zeros_like(xx)
        self.ax.fill_between(xx, yy, 0.0, where=(yy >= 0), alpha=0.2, color="green")
        self.ax.fill_between(xx, yy, 0.0, where=(yy < 0),  alpha=0.2, color="green")
        self.ax.plot(xx, yy, lw=2, color="blue", label="f(x)")
        self.ax.scatter(xhm[success],  yhm[success],  s=10, c="green",  alpha=0.6, label="Éxito")
        self.ax.scatter(xhm[~success], yhm[~success], s=10, c="orange", alpha=0.6, label="Fracaso")
        self.ax.set_xlim(a, b); self.ax.set_ylim(ymin, ymax); self.ax.legend(loc="upper right")
        self.canvas.draw_idle()

# ========= GUI 2D =========
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class MonteCarloGUI2D(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Monte Carlo 2D - Integral doble")
        self.geometry("1200x800")

        top = ttk.Frame(self, padding=8); top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="f(x,y):").grid(row=0, column=0, sticky="w")
        self.fxy_entry = ttk.Entry(top, width=40); self.fxy_entry.grid(row=0, column=1, padx=4, sticky="we")
        self.fxy_entry.insert(0, "sin(x) + cos(y)")
        ttk.Label(top, text="ax:").grid(row=0, column=2, sticky="w")
        self.ax_entry = ttk.Entry(top, width=8); self.ax_entry.grid(row=0, column=3, padx=4); self.ax_entry.insert(0, "0")
        ttk.Label(top, text="bx:").grid(row=0, column=4, sticky="w")
        self.bx_entry = ttk.Entry(top, width=8); self.bx_entry.grid(row=0, column=5, padx=4); self.bx_entry.insert(0, "3.1415926535")
        ttk.Label(top, text="ay:").grid(row=0, column=6, sticky="w")
        self.ay_entry = ttk.Entry(top, width=8); self.ay_entry.grid(row=0, column=7, padx=4); self.ay_entry.insert(0, "0")
        ttk.Label(top, text="by:").grid(row=0, column=8, sticky="w")
        self.by_entry = ttk.Entry(top, width=8); self.by_entry.grid(row=0, column=9, padx=4); self.by_entry.insert(0, "1.5707963268")
        ttk.Label(top, text="Muestras:").grid(row=0, column=10, sticky="w")
        self.N_entry = ttk.Entry(top, width=12); self.N_entry.grid(row=0, column=11, padx=4); self.N_entry.insert(0, "200000")
        ttk.Label(top, text="Seed (opcional):").grid(row=0, column=12, sticky="w")
        self.seed_entry = ttk.Entry(top, width=10); self.seed_entry.grid(row=0, column=13, padx=4); self.seed_entry.insert(0, "")
        ttk.Label(top, text="Confianza % (opcional):").grid(row=0, column=14, sticky="w")
        self.conf_entry = ttk.Entry(top, width=8); self.conf_entry.grid(row=0, column=15, padx=4); self.conf_entry.insert(0, "95")
        ttk.Button(top, text="Simular", command=self.simular).grid(row=0, column=16, padx=8)

        # Resultados en dos renglones
        self.result_label1 = ttk.Label(self, text="Integral doble: —", font=("Segoe UI", 14, "bold"), wraplength=1150)
        self.result_label1.pack(side=tk.TOP, anchor="w", padx=12, pady=(6, 0))
        self.result_label2 = ttk.Label(self, text="", font=("Segoe UI", 13), wraplength=1150)
        self.result_label2.pack(side=tk.TOP, anchor="w", padx=12, pady=(0, 6))

        body = ttk.Frame(self); body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(body, padding=(8, 8)); left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="Muestras Monte Carlo (2D)").pack(anchor="w")
        cols = ("x", "y", "fxy", "exito")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=28)
        for c, w in zip(cols, (110, 110, 120, 70)):
            self.tree.heading(c, text=c); self.tree.column(c, width=w, anchor="center")
        yscroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.Y); yscroll.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(body, padding=(8, 8)); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self.ax3d.set_xlabel("x"); self.ax3d.set_ylabel("y"); self.ax3d.set_zlabel("z")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def simular(self):
        fxy_text = self.fxy_entry.get().strip()
        try:
            ax = float(self.ax_entry.get().strip()); bx = float(self.bx_entry.get().strip())
            ay = float(self.ay_entry.get().strip()); by = float(self.by_entry.get().strip())
            if not (bx > ax and by > ay): raise ValueError("Se requiere ax < bx y ay < by.")
        except Exception as e:
            messagebox.showerror("Error en límites", str(e)); return

        try:
            N = int(self.N_entry.get().replace("_", "").strip())
            if N <= 0: raise ValueError("Muestras debe ser > 0.")
        except Exception as e:
            messagebox.showerror("Error en muestras", f"Valor inválido: {e}"); return

        seed = None
        st = self.seed_entry.get().strip()
        if st:
            try: seed = int(st)
            except Exception: messagebox.showerror("Error en seed", "Seed debe ser entero o vacío."); return

        conf = 95
        ct = self.conf_entry.get().strip()
        if ct:
            try:
                conf = float(ct)
                if not (0 < conf < 100): raise ValueError
            except Exception: messagebox.showerror("Error en confianza", "Confianza 0–100 o vacío."); return

        try:
            f = safe_eval_function_2d(fxy_text)
            _ = float(np.asarray(f(np.array([ax]), np.array([ay]))[0]))
        except Exception as e:
            messagebox.showerror("Error en f(x,y)", f"No se pudo interpretar f(x,y).\nEj.: sin(x)+cos(y), exp(-(x**2+y**2)).\n{e}")
            return

        # Monte Carlo
        res = mc_mean_value_2d(f, ax, bx, ay, by, N, seed=seed, conf=conf)
        I, SE, sigma, mean_f, (lo, hi) = res["I"], res["SE"], res["sigma_f"], res["mean_f"], res["CI"]

        # Gauss–Kronrod 2D (dblquad integra primero en y, luego en x)
        try:
            # wrapper escalar para dblquad (f es vectorizada). Orden: integrando(y, x).
            def g(y, x):
                return float(np.asarray(f(np.array([x]), np.array([y]))[0]))
            gauss_val, gauss_err = dblquad(g, ax, bx, lambda _: ay, lambda _: by, epsabs=1e-9, epsrel=1e-9)
            gauss_text = f"{gauss_val:.8f} (± {gauss_err:.1e})"
        except Exception:
            gauss_text = "n/d"

        self.result_label1.config(
            text=(f"Integral doble: I_MC = {I:.8f}  |  Media muestral = {mean_f:.8f}")
        )
        self.result_label2.config(
            text=(f"Error estándar (SE) = {SE:.8f}  |  Desviación σ_f = {sigma:.8f}  |  "
                  f"IC {conf:.0f}%: [{lo:.8f}, {hi:.8f}]  |  Valor (Gauss) ≈ {gauss_text}")
        )

        # Visual + tabla
        x, y, z, fxy, success, (zmin, zmax) = hit_miss_samples_2d(f, ax, bx, ay, by, N=min(N, 8000), seed=seed)

        for r in self.tree.get_children(): self.tree.delete(r)
        max_rows = 2500
        shown = min(x.size, max_rows)
        for i in range(shown):
            self.tree.insert("", tk.END, values=(f"{x[i]:.6f}", f"{y[i]:.6f}", f"{fxy[i]:.6f}", "✓" if success[i] else "✗"))

        self.ax3d.clear()
        self.ax3d.set_xlabel("x"); self.ax3d.set_ylabel("y"); self.ax3d.set_zlabel("z")
        gx, gy = np.linspace(ax, bx, 60), np.linspace(ay, by, 60)
        XX, YY = np.meshgrid(gx, gy)
        ZZ = np.asarray(f(XX, YY), dtype=float)
        self.ax3d.plot_surface(XX, YY, ZZ, linewidth=0, antialiased=True, alpha=0.35)
        self.ax3d.scatter(x[success], y[success], z[success], s=6, c="green",  alpha=0.8, label="Éxito")
        self.ax3d.scatter(x[~success], y[~success], z[~success], s=6, c="orange", alpha=0.8, label="Fracaso")
        self.ax3d.set_xlim(ax, bx); self.ax3d.set_ylim(ay, by); self.ax3d.set_zlim(zmin, zmax)
        self.ax3d.legend(loc="upper left")
        self.canvas.draw_idle()

# ========= Lanzador =========
def main():
    app = MonteCarloGUI1D()
    app.mainloop()

if __name__ == "__main__":
    main()
