# -*- coding: utf-8 -*-
"""
Suite Numérica - Launcher (Tkinter + ttk)
Incluye:
  • Monte Carlo (1D/2D) con Gauss-Kronrod de referencia
  • Newton–Cotes (Rectángulo, Trapecio, Simpson 1/3, 3/8)
  • Newton–Raphson (derivada automática)
  • Interpolación de Lagrange (fracciones + gráfico)
  • Punto Fijo (x_{n+1}=g(x))  [nuevo]
  • Búsqueda Binaria / Bisección [nuevo]
  • Aceleración Aitken (Δ²)      [nuevo]

Requiere: numpy, sympy, scipy, matplotlib
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import numpy as np
import sympy as sp
from scipy.stats import norm
from scipy.integrate import quad, dblquad
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# =========================
# Estilos / Tema ttk
# =========================
APP_ACCENT = "#3b82f6"   # azul
APP_BG     = "#0b1220"   # fondo oscuro
APP_CARD   = "#111827"   # cards
APP_TEXT   = "#e5e7eb"   # gris claro

def setup_style(root):
    style = ttk.Style(root)
    # Forzar tema que respeta background/foreground en ttk
    style.theme_use("clam")

    style.configure(".", font=("Segoe UI", 11))
    style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"),
                    foreground=APP_TEXT, background=APP_BG)
    style.configure("Sub.TLabel", font=("Segoe UI", 12),
                    foreground="#9ca3af", background=APP_BG)
    style.configure("Card.TFrame", background=APP_CARD)
    style.configure("App.TFrame", background=APP_BG)

    # Botones grandes con color de acento
    style.configure("Big.TButton",
                    font=("Segoe UI", 12, "bold"),
                    padding=12,
                    background=APP_ACCENT,
                    foreground="#ffffff")
    style.map("Big.TButton",
              background=[("pressed", "#1d4ed8"), ("active", "#2563eb")],
              foreground=[("disabled", "#9ca3af")])

    # Botón por defecto legible en tema claro
    style.configure("TButton", foreground="#111827")

# =========================
#  Utilities comunes
# =========================
def safe_eval_function_1d(expr):
    allowed = {
        "np": np,
        # trig / hiper
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        # expo / logs
        "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "sqrt": np.sqrt,
        # otros
        "abs": np.abs, "floor": np.floor, "ceil": np.ceil, "where": np.where,
        "sinc": (lambda x: np.sinc(x/np.pi)),  # sin(x)/x, con sinc(0)=1
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
        "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "sqrt": np.sqrt,
        "abs": np.abs, "floor": np.floor, "ceil": np.ceil, "where": np.where,
        "sinc": (lambda x: np.sinc(x/np.pi)),
        "pi": np.pi, "e": np.e,
    }
    code = compile(expr, "<fxy-2d>", "eval")
    def f(x, y):
        return eval(code, {"__builtins__": {}}, {**allowed, "x": x, "y": y})
    return f

# ===== Monte Carlo (1D/2D) =====
def mc_mean_value_1d(f, a, b, N, seed=None, conf=95):
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=N)
    fx = np.asarray(f(x), dtype=float)
    mask = np.isfinite(fx)
    if not np.all(mask):
        x = x[mask]; fx = fx[mask]
        if fx.size == 0: raise ValueError("Todas las evaluaciones f(x) fueron no finitas.")
    mean_f = np.mean(fx)
    sigma = np.std(fx, ddof=1) if fx.size > 1 else 0.0
    vol = (b - a)
    I = vol * mean_f
    SE = vol * sigma / np.sqrt(fx.size) if fx.size > 0 else float("nan")
    z = norm.ppf(0.5 + conf/200.0)
    CI = (I - z*SE, I + z*SE)
    return {"I": I, "SE": SE, "sigma_f": sigma, "mean_f": mean_f, "CI": CI, "N": fx.size, "x": x, "fx": fx}

def mc_mean_value_2d(f, ax, bx, ay, by, N, seed=None, conf=95):
    rng = np.random.default_rng(seed)
    x = rng.uniform(ax, bx, size=N)
    y = rng.uniform(ay, by, size=N)
    fxy = np.asarray(f(x, y), dtype=float)
    mask = np.isfinite(fxy)
    if not np.all(mask):
        x = x[mask]; y = y[mask]; fxy = fxy[mask]
        if fxy.size == 0: raise ValueError("Todas las evaluaciones f(x,y) fueron no finitas.")
    mean_f = np.mean(fxy)
    sigma = np.std(fxy, ddof=1) if fxy.size > 1 else 0.0
    vol = (bx - ax) * (by - ay)
    I = vol * mean_f
    SE = vol * sigma / np.sqrt(fxy.size) if fxy.size > 0 else float("nan")
    z = norm.ppf(0.5 + conf/200.0)
    CI = (I - z*SE, I + z*SE)
    return {"I": I, "SE": SE, "sigma_f": sigma, "mean_f": mean_f, "CI": CI, "N": fxy.size, "x": x, "y": y, "fxy": fxy}

def hit_miss_samples_1d(f, a, b, N, seed=None, grid=1000):
    xs = np.linspace(a, b, grid)
    vals = np.asarray(f(xs), dtype=float); vals = vals[np.isfinite(vals)]
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
    ZZ = np.asarray(f(XX, YY), dtype=float); ZZ = ZZ[np.isfinite(ZZ)]
    if ZZ.size == 0:
        zmin, zmax = -1.0, 1.0
    else:
        zmin = float(min(np.min(ZZ), 0.0)); zmax = float(max(np.max(ZZ), 0.0))
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

# ========= Ventanas Monte Carlo =========
class MonteCarlo1DWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("🎲 Monte Carlo 1D – Integración")
        self.geometry("1150x720")

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
        self.seed_entry = ttk.Entry(top, width=10); self.seed_entry.grid(row=0, column=9, padx=4)
        ttk.Label(top, text="Confianza %:").grid(row=0, column=10, sticky="w")
        self.conf_entry = ttk.Entry(top, width=8); self.conf_entry.grid(row=0, column=11, padx=4); self.conf_entry.insert(0, "95")
        ttk.Button(top, text="Simular", style="Big.TButton", command=self.simular).grid(row=0, column=12, padx=10)
        ttk.Button(top, text="Integral doble →", command=self.abrir_2d).grid(row=0, column=13, padx=4)

        self.result_label1 = ttk.Label(self, text="Integral: —", font=("Segoe UI", 14, "bold"), wraplength=1100)
        self.result_label1.pack(side=tk.TOP, anchor="w", padx=12, pady=(8, 0))
        self.result_label2 = ttk.Label(self, text="", font=("Segoe UI", 12), wraplength=1100)
        self.result_label2.pack(side=tk.TOP, anchor="w", padx=12, pady=(0, 8))

        body = ttk.Panedwindow(self, orient="horizontal"); body.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(body, padding=(8, 8), style="Card.TFrame"); body.add(left, weight=0)
        right = ttk.Frame(body, padding=(8, 8)); body.add(right, weight=1)

        ttk.Label(left, text="Muestras Monte Carlo", foreground=APP_TEXT, background=APP_CARD).pack(anchor="w")
        cols = ("x", "y", "fx", "exito")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=26)
        for c, w in zip(cols, (120, 120, 120, 80)):
            self.tree.heading(c, text=c); self.tree.column(c, width=w, anchor="center")
        yscroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.Y); yscroll.pack(side=tk.LEFT, fill=tk.Y)

        self.fig = Figure(figsize=(6, 4.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def abrir_2d(self):
        MonteCarlo2DWindow(self)

    def simular(self):
        fx_text = self.fx_entry.get().strip()
        try:
            a = float(self.a_entry.get().strip()); b = float(self.b_entry.get().strip())
            if not (np.isfinite(a) and np.isfinite(b)) or not (b > a):
                raise ValueError("Intervalo inválido: a < b.")
        except Exception as e:
            messagebox.showerror("Error en intervalos", str(e)); return

        try:
            N = int(self.N_entry.get().replace("_", "").strip());  assert N > 0
        except Exception as e:
            messagebox.showerror("Error en muestras", f"Valor inválido de Muestras. {e}"); return

        conf = 95
        ct = self.conf_entry.get().strip()
        if ct:
            try:
                conf = float(ct);  assert 0 < conf < 100
            except Exception:
                messagebox.showerror("Error en confianza", "Ingrese 0–100 o vacío."); return

        seed = None
        st = self.seed_entry.get().strip()
        if st:
            try: seed = int(st)
            except Exception: messagebox.showerror("Error en seed", "Seed debe ser entero o vacío."); return

        try:
            f = safe_eval_function_1d(fx_text)
            test_x = np.array([(a + b) / 2.0], dtype=float)
            _ = np.asarray(f(test_x), dtype=float)
        except Exception as e:
            messagebox.showerror("Error en f(x)", f"No se pudo interpretar f(x).\nEj.: exp(-x**2), sin(x), ln(x), sinc(x).\n{e}")
            return

        res = mc_mean_value_1d(f, a, b, N, seed=seed, conf=conf)
        I, SE, sigma, mean_f, (lo, hi) = res["I"], res["SE"], res["sigma_f"], res["mean_f"], res["CI"]

        try:
            g = lambda t: float(np.asarray(f(np.array([t]))[0]))
            gauss_val, gauss_err = quad(g, a, b, epsabs=1e-10, epsrel=1e-10, limit=200)
            gauss_text = f"{gauss_val:.8f} (± {gauss_err:.1e})"
        except Exception:
            gauss_text = "n/d"

        self.result_label1.config(
            text=(f"Integral estimada: I_MC = {I:.8f}  |  Valor (Gauss) ≈ {gauss_text}  |  Media muestral = {mean_f:.8f}")
        )
        self.result_label2.config(
            text=(f"Error estándar (SE) = {SE:.8f}  |  Desviación σ_f = {sigma:.8f}  |  IC {conf:.0f}%: [{lo:.8f}, {hi:.8f}]")
        )

        xhm, yhm, fxhm, success, (ymin, ymax) = hit_miss_samples_1d(f, a, b, N=min(N, 5000), seed=seed)
        for r in self.tree.get_children(): self.tree.delete(r)
        shown = min(xhm.size, 2000)
        for i in range(shown):
            self.tree.insert("", tk.END, values=(f"{xhm[i]:.6f}", f"{yhm[i]:.6f}", f"{fxhm[i]:.6f}", "✓" if success[i] else "✗"))

        self.ax.clear(); self.ax.grid(True, alpha=0.3); self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        xx = np.linspace(a, b, 800)
        try: yy = np.asarray(f(xx), dtype=float)
        except Exception: yy = np.zeros_like(xx)
        self.ax.fill_between(xx, yy, 0.0, where=(yy >= 0), alpha=0.2, color="green")
        self.ax.fill_between(xx, yy, 0.0, where=(yy < 0),  alpha=0.2, color="green")
        self.ax.plot(xx, yy, lw=2, color="tab:blue", label="f(x)")
        self.ax.scatter(xhm[success],  yhm[success],  s=10, c="green",  alpha=0.7, label="Éxito")
        self.ax.scatter(xhm[~success], yhm[~success], s=10, c="orange", alpha=0.7, label="Fracaso")
        self.ax.set_xlim(a, b); self.ax.set_ylim(ymin, ymax); self.ax.legend(loc="upper right")
        self.canvas.draw_idle()

class MonteCarlo2DWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("🎲 Monte Carlo 2D – Integral doble")
        self.geometry("1250x820")

        top = ttk.Frame(self, padding=8); top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="f(x,y):").grid(row=0, column=0, sticky="w")
        self.fxy_entry = ttk.Entry(top, width=42); self.fxy_entry.grid(row=0, column=1, padx=4, sticky="we")
        self.fxy_entry.insert(0, "sin(x) + cos(y)")
        ttk.Label(top, text="ax:").grid(row=0, column=2, sticky="w")
        self.ax_entry = ttk.Entry(top, width=8); self.ax_entry.grid(row=0, column=3, padx=4); self.ax_entry.insert(0, "0")
        ttk.Label(top, text="bx:").grid(row=0, column=4, sticky="w")
        self.bx_entry = ttk.Entry(top, width=8); self.bx_entry.grid(row=0, column=5, padx=4); self.bx_entry.insert(0, "pi")
        ttk.Label(top, text="ay:").grid(row=0, column=6, sticky="w")
        self.ay_entry = ttk.Entry(top, width=8); self.ay_entry.grid(row=0, column=7, padx=4); self.ay_entry.insert(0, "0")
        ttk.Label(top, text="by:").grid(row=0, column=8, sticky="w")
        self.by_entry = ttk.Entry(top, width=8); self.by_entry.grid(row=0, column=9, padx=4); self.by_entry.insert(0, "pi/2")
        ttk.Label(top, text="Muestras:").grid(row=0, column=10, sticky="w")
        self.N_entry = ttk.Entry(top, width=12); self.N_entry.grid(row=0, column=11, padx=4); self.N_entry.insert(0, "200000")
        ttk.Label(top, text="Seed:").grid(row=0, column=12, sticky="w")
        self.seed_entry = ttk.Entry(top, width=10); self.seed_entry.grid(row=0, column=13, padx=4)
        ttk.Label(top, text="Conf %:").grid(row=0, column=14, sticky="w")
        self.conf_entry = ttk.Entry(top, width=8); self.conf_entry.grid(row=0, column=15, padx=4); self.conf_entry.insert(0, "95")
        ttk.Button(top, text="Simular", style="Big.TButton", command=self.simular).grid(row=0, column=16, padx=10)

        self.result_label1 = ttk.Label(self, text="Integral doble: —", font=("Segoe UI", 14, "bold"), wraplength=1180)
        self.result_label1.pack(side=tk.TOP, anchor="w", padx=12, pady=(8, 0))
        self.result_label2 = ttk.Label(self, text="", font=("Segoe UI", 12), wraplength=1180)
        self.result_label2.pack(side=tk.TOP, anchor="w", padx=12, pady=(0, 8))

        body = ttk.Panedwindow(self, orient="horizontal"); body.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(body, padding=(8, 8), style="Card.TFrame"); body.add(left, weight=0)
        right = ttk.Frame(body, padding=(8, 8)); body.add(right, weight=1)

        ttk.Label(left, text="Muestras Monte Carlo (2D)", foreground=APP_TEXT, background=APP_CARD).pack(anchor="w")
        cols = ("x", "y", "fxy", "exito")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=28)
        for c, w in zip(cols, (110, 110, 120, 80)):
            self.tree.heading(c, text=c); self.tree.column(c, width=w, anchor="center")
        yscroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.Y); yscroll.pack(side=tk.LEFT, fill=tk.Y)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self.ax3d.set_xlabel("x"); self.ax3d.set_ylabel("y"); self.ax3d.set_zlabel("z")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def simular(self):
        fxy_text = self.fxy_entry.get().strip()
        try:
            ax = float(sp.N(sp.sympify(self.ax_entry.get().strip(), locals={'pi':sp.pi})))
            bx = float(sp.N(sp.sympify(self.bx_entry.get().strip(), locals={'pi':sp.pi})))
            ay = float(sp.N(sp.sympify(self.ay_entry.get().strip(), locals={'pi':sp.pi})))
            by = float(sp.N(sp.sympify(self.by_entry.get().strip(), locals={'pi':sp.pi})))
            if not (bx > ax and by > ay): raise ValueError("Se requiere ax < bx y ay < by.")
        except Exception as e:
            messagebox.showerror("Error en límites", str(e)); return

        try:
            N = int(self.N_entry.get().replace("_", "").strip());  assert N > 0
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
                conf = float(ct); assert 0 < conf < 100
            except Exception: messagebox.showerror("Error en confianza", "Confianza 0–100 o vacío."); return

        try:
            f = safe_eval_function_2d(fxy_text)
            _ = float(np.asarray(f(np.array([ax]), np.array([ay]))[0]))
        except Exception as e:
            messagebox.showerror("Error en f(x,y)", f"No se pudo interpretar f(x,y).\nEj.: sin(x)+cos(y), exp(-(x**2+y**2)).\n{e}")
            return

        res = mc_mean_value_2d(f, ax, bx, ay, by, N, seed=seed, conf=conf)
        I, SE, sigma, mean_f, (lo, hi) = res["I"], res["SE"], res["sigma_f"], res["mean_f"], res["CI"]

        try:
            def g(y, x):  # orden que espera dblquad
                return float(np.asarray(f(np.array([x]), np.array([y]))[0]))
            gauss_val, gauss_err = dblquad(g, ax, bx, lambda _: ay, lambda _: by, epsabs=1e-9, epsrel=1e-9)
            gauss_text = f"{gauss_val:.8f} (± {gauss_err:.1e})"
        except Exception:
            gauss_text = "n/d"

        self.result_label1.config(
            text=(f"Integral doble: I_MC = {I:.8f}  |  Valor (Gauss) ≈ {gauss_text}  |  Media muestral = {mean_f:.8f}")
        )
        self.result_label2.config(
            text=(f"Error estándar (SE) = {SE:.8f}  |  Desviación σ_f = {sigma:.8f}  |  IC {conf:.0f}%: [{lo:.8f}, {hi:.8f}]")
        )

        x, y, z, fxy, success, (zmin, zmax) = hit_miss_samples_2d(f, ax, bx, ay, by, N=min(N, 8000), seed=seed)
        for r in self.tree.get_children(): self.tree.delete(r)
        shown = min(x.size, 2500)
        for i in range(shown):
            self.tree.insert("", tk.END, values=(f"{x[i]:.6f}", f"{y[i]:.6f}", f"{fxy[i]:.6f}", "✓" if success[i] else "✗"))

        self.ax3d.clear()
        self.ax3d.set_xlabel("x"); self.ax3d.set_ylabel("y"); self.ax3d.set_zlabel("z")
        gx, gy = np.linspace(ax, bx, 60), np.linspace(ay, by, 60)
        XX, YY = np.meshgrid(gx, gy)
        ZZ = np.asarray(f(XX, YY), dtype=float)
        self.ax3d.plot_surface(XX, YY, ZZ, linewidth=0, antialiased=True, alpha=0.35)
        self.ax3d.scatter(x[success], y[success], z[success], s=6, c="green",  alpha=0.85, label="Éxito")
        self.ax3d.scatter(x[~success], y[~success], z[~success], s=6, c="orange", alpha=0.85, label="Fracaso")
        self.ax3d.set_xlim(ax, bx); self.ax3d.set_ylim(ay, by); self.ax3d.set_zlim(zmin, zmax)
        self.ax3d.legend(loc="upper left")
        self.canvas.draw_idle()

# ===== Newton–Cotes =====
def make_funcs(expr_str):
    x = sp.Symbol('x')
    f_sym = sp.sympify(expr_str, locals={
        'pi': sp.pi, 'e': sp.E,
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
        'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
        'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
    })
    f_np = sp.lambdify(x, f_sym, modules=["numpy"])
    return f_np, f_sym, x

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
    if n % 2 != 0: raise ValueError("Simpson 1/3 requiere n par")
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f_np(x)
    I = (h / 3.0) * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]))
    return I, list(range(n + 1)), x.tolist(), y.tolist(), h

def simpson_38(f_np, a, b, n):
    if n % 3 != 0: raise ValueError("Simpson 3/8 requiere n múltiplo de 3")
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f_np(x)
    idx = np.arange(1, n)
    coef = np.where(idx % 3 == 0, 2.0, 3.0)
    I = (3*h/8.0) * (y[0] + y[-1] + np.sum(coef * y[1:-1]))
    return I, list(range(n + 1)), x.tolist(), y.tolist(), h

def integral_real(f_np, f_sym, x, a, b):
    try:
        F = sp.integrate(f_sym, (x, a, b))
        val = float(F.evalf())
        return val, "Exacta (Simbólica)"
    except Exception:
        n_fino = 2000
        h = (b - a) / n_fino
        xs = a + h * np.arange(n_fino + 1)
        ys = f_np(xs)
        I = (h/3.0)*(ys[0] + ys[-1] + 4*np.sum(ys[1:-1:2]) + 2*np.sum(ys[2:-1:2]))
        return float(I), f"Num. fina (n={n_fino})"

class NewtonCotesWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("📐 Newton–Cotes")
        self.geometry("1150x740")

        paned = ttk.Panedwindow(self, orient="horizontal"); paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, width=320, padding=10, style="Card.TFrame")
        right = ttk.Frame(paned, padding=10)
        paned.add(left, weight=0); paned.add(right, weight=1)

        box = ttk.LabelFrame(left, text="Entradas")
        box.pack(fill="x")
        ttk.Label(box, text="f(x):").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.ent_fx = ttk.Entry(box, width=28); self.ent_fx.grid(row=0, column=1, padx=5, pady=5); self.ent_fx.insert(0, "sin(x)")
        ttk.Label(box, text="a:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.ent_a = ttk.Entry(box, width=12); self.ent_a.grid(row=1, column=1, sticky="w", padx=5, pady=5); self.ent_a.insert(0, "0")
        ttk.Label(box, text="b:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.ent_b = ttk.Entry(box, width=12); self.ent_b.grid(row=2, column=1, sticky="w", padx=5, pady=5); self.ent_b.insert(0, "pi")
        ttk.Label(box, text="n (subintervalos):").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.ent_n = ttk.Entry(box, width=12); self.ent_n.grid(row=3, column=1, sticky="w", padx=5, pady=5); self.ent_n.insert(0, "12")
        ttk.Label(box, text="Método:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.cmb_met = ttk.Combobox(box, width=22, state="readonly",
                                    values=["Rectángulo", "Trapecio", "Simpson 1/3", "Simpson 3/8"])
        self.cmb_met.grid(row=4, column=1, padx=5, pady=5); self.cmb_met.current(2)

        btns = ttk.Frame(left); btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Calcular", style="Big.TButton", command=self.calcular).pack(fill="x", pady=4)
        ttk.Button(btns, text="Graficar", command=self.graficar).pack(fill="x", pady=4)

        results = ttk.LabelFrame(right, text="Resultados"); results.pack(fill="x")
        self.lbl_real = ttk.Label(results, text="Resultado real: -", font=("Segoe UI", 14, "bold"))
        self.lbl_real.pack(fill="x", pady=6)
        self.lbl_met  = ttk.Label(results, text="Resultado método: -", font=("Segoe UI", 14, "bold"))
        self.lbl_met.pack(fill="x", pady=6)

        top = ttk.LabelFrame(right, text="Tabla (i, xi, f(xi))"); top.pack(fill="both", expand=True, pady=6)
        self.tree = ttk.Treeview(top, columns=("i","xi","fxi"), show="headings", height=8)
        self.tree.heading("i", text="i"); self.tree.heading("xi", text="xi"); self.tree.heading("fxi", text="f(xi)")
        self.tree.column("i", width=60, anchor="center"); self.tree.column("xi", width=160, anchor="e"); self.tree.column("fxi", width=160, anchor="e")
        self.tree.pack(fill="both", expand=True)

        bottom = ttk.LabelFrame(right, text="Gráfico"); bottom.pack(fill="both", expand=True)
        self.fig, self.ax = plt.subplots(figsize=(7.5,3.8), dpi=110)
        self.canvas = FigureCanvasTkAgg(self.fig, master=bottom); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._last_plot = None  # (f_np, a, b, metodo, xs, ys, h)

    def _leer(self):
        fx_str = self.ent_fx.get().strip().replace(",", ".")
        a_val = float(sp.N(sp.sympify(self.ent_a.get().strip(), locals={'pi': sp.pi, 'e': sp.E})))
        b_val = float(sp.N(sp.sympify(self.ent_b.get().strip(), locals={'pi': sp.pi, 'e': sp.E})))
        n_val = int(self.ent_n.get().strip());  assert n_val > 0 and a_val != b_val
        metodo = self.cmb_met.get()
        f_np, f_sym, x = make_funcs(fx_str)
        return f_np, f_sym, x, a_val, b_val, n_val, metodo

    def _aplicar(self, f_np, a, b, n, metodo):
        if metodo == "Rectángulo": return rectangulo(f_np, a, b, n)
        if metodo == "Trapecio":   return trapecio(f_np, a, b, n)
        if metodo == "Simpson 1/3":return simpson_13(f_np, a, b, n)
        if metodo == "Simpson 3/8":return simpson_38(f_np, a, b, n)
        raise ValueError("Método inválido")

    def calcular(self):
        try:
            f_np, f_sym, x, a, b, n, metodo = self._leer()
            I_real, _ = integral_real(f_np, f_sym, x, a, b)
            I_met, idxs, xs, ys, h = self._aplicar(f_np, a, b, n, metodo)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return

        self.lbl_real.config(text=f"Resultado real: {I_real:.12g}")
        self.lbl_met.config(text=f"Resultado método ({metodo}): {I_met:.12g}")

        for row in self.tree.get_children(): self.tree.delete(row)
        for i, (xi, yi) in enumerate(zip(xs, ys)):
            self.tree.insert("", "end", values=(i, f"{xi:.10g}", f"{yi:.10g}"))
        self._last_plot = (f_np, a, b, metodo, xs, ys, h)

    def _shade_rectangulo(self, f_np, a, b, n, h):
        for k in range(n):
            x0 = a + k*h; xm = x0 + h/2; y = float(f_np(xm))
            self.ax.fill([x0, x0, x0+h, x0+h], [0, y, y, 0], alpha=0.2, color="C1", edgecolor="C1")

    def _shade_trapecio(self, xs, ys):
        for i in range(len(xs)-1):
            x0, x1 = xs[i], xs[i+1]; y0, y1 = ys[i], ys[i+1]
            self.ax.fill([x0, x0, x1, x1], [0, y0, y1, 0], alpha=0.2, color="C1", edgecolor="C1")

    def _shade_simpson13(self, xs, ys, h):
        for i in range(0, len(xs)-1, 2):
            if i+2 >= len(xs): break
            x0, x1, x2 = xs[i], xs[i+1], xs[i+2]
            y0, y1, y2 = ys[i], ys[i+1], ys[i+2]
            coeff = np.polyfit([x0, x1, x2], [y0, y1, y2], deg=2)
            xx = np.linspace(x0, x2, 80); yy = np.polyval(coeff, xx)
            self.ax.fill_between(xx, yy, 0, alpha=0.2, color="C1")

    def _shade_simpson38(self, xs, ys, h):
        for i in range(0, len(xs)-1, 3):
            if i+3 >= len(xs): break
            x0, x1, x2, x3 = xs[i], xs[i+1], xs[i+2], xs[i+3]
            y0, y1, y2, y3 = ys[i], ys[i+1], ys[i+2], ys[i+3]
            coeff = np.polyfit([x0, x1, x2, x3], [y0, y1, y2, y3], deg=3)
            xx = np.linspace(x0, x3, 100); yy = np.polyval(coeff, xx)
            self.ax.fill_between(xx, yy, 0, alpha=0.2, color="C1")

    def graficar(self):
        if self._last_plot is None:
            self.calcular()
            if self._last_plot is None: return
        f_np, a, b, metodo, xs_nodes, ys_nodes, h = self._last_plot
        self.ax.clear()
        X = np.linspace(a, b, 1000); Y = f_np(X)
        self.ax.plot(X, Y, label="f(x)")
        if metodo == "Rectángulo":
            n = len(xs_nodes); self._shade_rectangulo(f_np, a, b, n, h)
        elif metodo == "Trapecio":
            self._shade_trapecio(xs_nodes, ys_nodes)
        elif metodo == "Simpson 1/3":
            self._shade_simpson13(xs_nodes, ys_nodes, h)
        elif metodo == "Simpson 3/8":
            self._shade_simpson38(xs_nodes, ys_nodes, h)
        self.ax.scatter(xs_nodes, ys_nodes, color="red", s=25, zorder=5, label="puntos")
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.grid(True, ls="--", lw=0.6); self.ax.legend()
        self.canvas.draw()

# ===== Newton–Raphson =====
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

class NewtonRaphsonWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("🔎 Newton–Raphson (derivada automática)")
        self.geometry("1150x740")

        frm = ttk.LabelFrame(self, text="Parámetros"); frm.pack(fill="x", padx=10, pady=8)
        self._row(frm, 0, "Función f(x):", "x**3 - 2*x - 5", "Punto inicial (x0):", "1.5")
        self._row(frm, 1, "Error total (Ea):", "1e-10", "Error en |f(x)|:", "1e-12")
        self._row(frm, 2, "Iteraciones máx.:", "100")

        btns = ttk.Frame(self); btns.pack(fill="x", padx=10, pady=5)
        ttk.Button(btns, text="Calcular", style="Big.TButton", command=self.run).pack(side="left", padx=5)
        ttk.Button(btns, text="Limpiar", command=self.clear).pack(side="left", padx=5)

        self.lbl_res = ttk.Label(self, text="Resultado: -"); self.lbl_res.pack(fill="x", padx=10, pady=5)

        cols = ("n", "x_n", "f(x_n)", "f'(x_n)", "Ea", "Er")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=170 if c!="n" else 60, anchor="center")
        self.tree.pack(fill="both", expand=False, padx=10, pady=8)

        self.fig, self.ax = plt.subplots(figsize=(8.8, 3.9), dpi=100)
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

        self.clear()
        for n, xn, fx, dfx, Ea, Er in rows:
            fmt = lambda v: "-" if v is None else f"{v:.8g}"
            self.tree.insert("", "end", values=(n, f"{xn:.12g}", f"{fx:.12g}", f"{dfx:.12g}", fmt(Ea), fmt(Er)))
        self.lbl_res.config(text=f"Resultado: raíz ≈ {root:.12g} en {iters} it.  |  f(raíz) ≈ {f_num(root):.3e}")

        xmin, xmax = min(hist + [root]), max(hist + [root])
        if xmin == xmax: xmin -= 1; xmax += 1
        pad = 0.25 * (xmax - xmin if xmax != xmin else 1.0)
        xs = np.linspace(xmin - pad, xmax + pad, 600); ys = [f_num(u) for u in xs]

        self._style_axes()
        self.ax.plot(xs, ys, label="f(x)")
        self.ax.scatter(hist, [f_num(h) for h in hist], s=35, label="iteraciones")
        self.ax.scatter([root], [0], marker="x", s=70, label="raíz")
        ymin, ymax = float(np.min(ys)), float(np.max(ys))
        if not (ymin <= 0 <= ymax):
            extra = 0.1 * max(1.0, abs(ymin) + abs(ymax))
            self.ax.set_ylim(min(ymin, 0) - extra, max(ymax, 0) + extra)
        self.ax.legend(); self.canvas.draw()

# ===== Punto fijo =====
class FixedPointWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("🧭 Punto Fijo (x_{n+1} = g(x_n))")
        self.geometry("1150x740")

        frm = ttk.LabelFrame(self, text="Parámetros"); frm.pack(fill="x", padx=10, pady=8)
        ttk.Label(frm, text="g(x):").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        self.e_g = ttk.Entry(frm); self.e_g.grid(row=0, column=1, sticky="we", padx=6, pady=6); self.e_g.insert(0, "cos(x)")
        ttk.Label(frm, text="x0:").grid(row=0, column=2, sticky="e", padx=6, pady=6)
        self.e_x0 = ttk.Entry(frm, width=14); self.e_x0.grid(row=0, column=3, padx=6, pady=6); self.e_x0.insert(0, "1.0")
        ttk.Label(frm, text="Ea (|x_{n+1}-x_n|):").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        self.e_tol = ttk.Entry(frm); self.e_tol.grid(row=1, column=1, sticky="we", padx=6, pady=6); self.e_tol.insert(0, "1e-8")
        ttk.Label(frm, text="Iter. máx:").grid(row=1, column=2, sticky="e", padx=6, pady=6)
        self.e_max = ttk.Entry(frm, width=14); self.e_max.grid(row=1, column=3, padx=6, pady=6); self.e_max.insert(0, "100")
        frm.columnconfigure(1, weight=1)

        btns = ttk.Frame(self); btns.pack(fill="x", padx=10, pady=5)
        ttk.Button(btns, text="Iterar", style="Big.TButton", command=self.run).pack(side="left", padx=5)
        ttk.Button(btns, text="Limpiar", command=self.clear).pack(side="left", padx=5)

        self.lbl_res = ttk.Label(self, text="Resultado: -"); self.lbl_res.pack(fill="x", padx=10, pady=5)

        cols = ("n", "x_n", "g(x_n)", "Ea", "Er")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=180 if c!="n" else 60, anchor="center")
        self.tree.pack(fill="both", expand=False, padx=10, pady=8)

        self.fig, self.ax = plt.subplots(figsize=(8.8, 3.9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0,10))
        self._style_axes()

    def _style_axes(self):
        self.ax.clear()
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.ax.grid(True, linestyle="--", linewidth=0.6)
        self.ax.axhline(0, color="black", linewidth=0.9)
        self.ax.axline((0,0),(1,1), color="gray", linestyle=":", linewidth=1.0, label="y = x")
        self.canvas.draw()

    def clear(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        self.lbl_res.config(text="Resultado: -")
        self._style_axes()

    def run(self):
        try:
            g_str = self.e_g.get().strip()
            g = safe_eval_function_1d(g_str)
            x0 = float(self.e_x0.get().replace(",", "."))
            tol = float(self.e_tol.get().replace(",", "."))
            maxi = int(self.e_max.get())
        except Exception:
            messagebox.showerror("Error", "Parámetros inválidos."); return

        self.clear()
        xs = [x0]
        for n in range(maxi):
            x_old = xs[-1]
            x_new = float(np.asarray(g(np.array([x_old])))[0])
            Ea = abs(x_new - x_old)
            Er = Ea/abs(x_new) if x_new != 0 else np.nan
            self.tree.insert("", "end", values=(n, f"{x_old:.12g}", f"{x_new:.12g}", f"{Ea:.3e}", "-" if np.isnan(Er) else f"{Er:.3e}"))
            xs.append(x_new)
            if Ea < tol: break

        root = xs[-1]
        self.lbl_res.config(text=f"Resultado: x* ≈ {root:.12g}  (iteraciones = {len(xs)-1})")

        # Gráfico: y=g(x) y diagonal
        self._style_axes()
        xmin, xmax = min(xs), max(xs)
        if xmin == xmax: xmin -= 1; xmax += 1
        pad = 0.25 * (xmax - xmin if xmax != xmin else 1.0)
        X = np.linspace(xmin - pad, xmax + pad, 600)
        try:
            Y = np.asarray(g(X), dtype=float)
        except Exception:
            Y = np.zeros_like(X)
        self.ax.plot(X, Y, label="g(x)")
        # Cobweb simple
        for i in range(len(xs)-1):
            x_i, x_ip1 = xs[i], xs[i+1]
            self.ax.plot([x_i, x_i], [x_i, x_ip1], color="C1", alpha=0.7)     # vertical a g(x_i)
            self.ax.plot([x_i, x_ip1], [x_ip1, x_ip1], color="C1", alpha=0.7) # horizontal a y=x
        self.ax.scatter(xs[-1], xs[-1], color="red", zorder=5, label="x*")
        self.ax.legend(); self.canvas.draw()

# ===== Búsqueda binaria (Bisección) =====
class BisectionWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("🪚 Búsqueda Binaria de Raíces (Bisección)")
        self.geometry("1150x740")

        frm = ttk.LabelFrame(self, text="Parámetros"); frm.pack(fill="x", padx=10, pady=8)
        ttk.Label(frm, text="f(x):").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        self.e_f = ttk.Entry(frm); self.e_f.grid(row=0, column=1, sticky="we", padx=6, pady=6); self.e_f.insert(0, "x**3 - x - 2")
        ttk.Label(frm, text="a:").grid(row=0, column=2, sticky="e", padx=6, pady=6)
        self.e_a = ttk.Entry(frm, width=12); self.e_a.grid(row=0, column=3, padx=6, pady=6); self.e_a.insert(0, "1")
        ttk.Label(frm, text="b:").grid(row=0, column=4, sticky="e", padx=6, pady=6)
        self.e_b = ttk.Entry(frm, width=12); self.e_b.grid(row=0, column=5, padx=6, pady=6); self.e_b.insert(0, "2")
        ttk.Label(frm, text="tol intervalo:").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        self.e_tol = ttk.Entry(frm); self.e_tol.grid(row=1, column=1, sticky="we", padx=6, pady=6); self.e_tol.insert(0, "1e-10")
        ttk.Label(frm, text="tol |f(c)|:").grid(row=1, column=2, sticky="e", padx=6, pady=6)
        self.e_ftol = ttk.Entry(frm, width=12); self.e_ftol.grid(row=1, column=3, padx=6, pady=6); self.e_ftol.insert(0, "1e-12")
        ttk.Label(frm, text="Iter. máx:").grid(row=1, column=4, sticky="e", padx=6, pady=6)
        self.e_max = ttk.Entry(frm, width=12); self.e_max.grid(row=1, column=5, padx=6, pady=6); self.e_max.insert(0, "200")
        frm.columnconfigure(1, weight=1)

        btns = ttk.Frame(self); btns.pack(fill="x", padx=10, pady=5)
        ttk.Button(btns, text="Ejecutar", style="Big.TButton", command=self.run).pack(side="left", padx=5)
        ttk.Button(btns, text="Limpiar", command=self.clear).pack(side="left", padx=5)

        self.lbl_res = ttk.Label(self, text="Resultado: -"); self.lbl_res.pack(fill="x", padx=10, pady=5)

        cols = ("n", "a_n", "b_n", "c_n", "f(c_n)", "ancho/2")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=160 if c!="n" else 60, anchor="center")
        self.tree.pack(fill="both", expand=False, padx=10, pady=8)

        self.fig, self.ax = plt.subplots(figsize=(8.8, 3.9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0,10))
        self._style_axes()

    def _style_axes(self):
        self.ax.clear()
        self.ax.set_xlabel("x"); self.ax.set_ylabel("f(x)")
        self.ax.grid(True, linestyle="--", linewidth=0.6)
        self.ax.axhline(0, color="black", linewidth=0.9)
        self.canvas.draw()

    def clear(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        self.lbl_res.config(text="Resultado: -")
        self._style_axes()

    def run(self):
        try:
            f_str = self.e_f.get().strip()
            f = safe_eval_function_1d(f_str)
            a = float(sp.N(sp.sympify(self.e_a.get().strip(), locals={'pi': sp.pi})))
            b = float(sp.N(sp.sympify(self.e_b.get().strip(), locals={'pi': sp.pi})))
            tol  = float(self.e_tol.get().replace(",", "."))
            ftol = float(self.e_ftol.get().replace(",", "."))
            maxi = int(self.e_max.get())
        except Exception:
            messagebox.showerror("Error", "Parámetros inválidos."); return
        if a >= b:
            messagebox.showerror("Error", "Se requiere a < b."); return

        f_s = lambda x: float(np.asarray(f(np.array([x])))[0])
        fa, fb = f_s(a), f_s(b)
        if np.sign(fa) * np.sign(fb) >= 0:
            messagebox.showerror("Bolzano", "f(a) y f(b) deben tener signos opuestos en [a,b]."); return

        self.clear()
        an, bn = a, b
        for n in range(1, maxi+1):
            c = 0.5*(an+bn); fc = f_s(c)
            half = 0.5*(bn - an)
            self.tree.insert("", "end",
                             values=(n, f"{an:.12g}", f"{bn:.12g}", f"{c:.12g}", f"{fc:.3e}", f"{half:.3e}"))
            if abs(fc) < ftol or half < tol:
                root = c
                break
            if np.sign(fa) * np.sign(fc) < 0:
                bn, fb = c, fc
            else:
                an, fa = c, fc
        else:
            root = c

        self.lbl_res.config(text=f"Resultado: raíz ≈ {root:.12g}  (iteraciones ≤ {maxi})")

        # Gráfico
        self._style_axes()
        xmin, xmax = min(a, b, root), max(a, b, root)
        pad = 0.2*(xmax - xmin if xmax > xmin else 1.0)
        X = np.linspace(xmin - pad, xmax + pad, 800)
        Y = np.asarray(f(X), dtype=float)
        self.ax.plot(X, Y, label="f(x)")
        self.ax.axvline(a, color="C1", linestyle=":", label="a")
        self.ax.axvline(b, color="C2", linestyle=":", label="b")
        self.ax.axvline(root, color="C3", linestyle="--", label="raíz")
        self.ax.legend(); self.canvas.draw()

# ===== Aceleración Aitken (Δ²) =====
class AitkenWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("⚡ Aceleración Aitken (sobre g(x))")
        self.geometry("1150x740")

        frm = ttk.LabelFrame(self, text="Parámetros"); frm.pack(fill="x", padx=10, pady=8)
        ttk.Label(frm, text="g(x):").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        self.e_g = ttk.Entry(frm); self.e_g.grid(row=0, column=1, sticky="we", padx=6, pady=6); self.e_g.insert(0, "cos(x)")
        ttk.Label(frm, text="x0:").grid(row=0, column=2, sticky="e", padx=6, pady=6)
        self.e_x0 = ttk.Entry(frm, width=14); self.e_x0.grid(row=0, column=3, padx=6, pady=6); self.e_x0.insert(0, "1.0")
        ttk.Label(frm, text="Ea (|x*_{n}-x_n|):").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        self.e_tol = ttk.Entry(frm); self.e_tol.grid(row=1, column=1, sticky="we", padx=6, pady=6); self.e_tol.insert(0, "1e-8")
        ttk.Label(frm, text="Iter. máx:").grid(row=1, column=2, sticky="e", padx=6, pady=6)
        self.e_max = ttk.Entry(frm, width=14); self.e_max.grid(row=1, column=3, padx=6, pady=6); self.e_max.insert(0, "100")
        frm.columnconfigure(1, weight=1)

        btns = ttk.Frame(self); btns.pack(fill="x", padx=10, pady=5)
        ttk.Button(btns, text="Acelerar", style="Big.TButton", command=self.run).pack(side="left", padx=5)
        ttk.Button(btns, text="Limpiar", command=self.clear).pack(side="left", padx=5)

        self.lbl_res = ttk.Label(self, text="Resultado: -"); self.lbl_res.pack(fill="x", padx=10, pady=5)

        cols = ("n", "x_n", "x_{n+1}", "x_{n+2}", "x*_n (Aitken)", "Ea")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=170 if c!="n" else 60, anchor="center")
        self.tree.pack(fill="both", expand=False, padx=10, pady=8)

        self.fig, self.ax = plt.subplots(figsize=(8.8, 3.9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0,10))
        self._style_axes()

    def _style_axes(self):
        self.ax.clear()
        self.ax.set_xlabel("iteración n"); self.ax.set_ylabel("valor")
        self.ax.grid(True, linestyle="--", linewidth=0.6)
        self.canvas.draw()

    def clear(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        self.lbl_res.config(text="Resultado: -")
        self._style_axes()

    def run(self):
        try:
            g_str = self.e_g.get().strip()
            g = safe_eval_function_1d(g_str)
            x0 = float(self.e_x0.get().replace(",", "."))
            tol = float(self.e_tol.get().replace(",", "."))
            maxi = int(self.e_max.get())
        except Exception:
            messagebox.showerror("Error", "Parámetros inválidos."); return

        self.clear()
        seq_plain = [x0]
        seq_accel = []

        # Generar los primeros dos para poder aplicar Δ²
        def gx(x): return float(np.asarray(g(np.array([x])))[0])

        x = x0
        for n in range(maxi):
            x1 = gx(x)
            x2 = gx(x1)
            denom = (x2 - 2*x1 + x)
            if denom != 0 and np.isfinite(denom):
                x_star = x - (x1 - x)**2 / denom
            else:
                x_star = x2
            Ea = abs(x_star - x)
            self.tree.insert("", "end",
                             values=(n, f"{x:.12g}", f"{x1:.12g}", f"{x2:.12g}", f"{x_star:.12g}", f"{Ea:.3e}"))
            seq_plain.extend([x1, x2])
            seq_accel.append(x_star)
            x = x_star
            if Ea < tol: break

        self.lbl_res.config(text=f"Resultado: x* (Aitken) ≈ {x:.12g}  (iteraciones = {len(seq_accel)})")

        # Gráfico: comparar secuencia acelerada vs no acelerada (submuestreada)
        self._style_axes()
        n_acc = np.arange(len(seq_accel))
        self.ax.plot(n_acc, seq_accel, marker="o", label="Aitken Δ²")
        # mostrar también los x_n (cada paso original)
        seq_plain_n = np.arange(len(seq_plain))
        self.ax.plot(seq_plain_n, seq_plain, alpha=0.5, label="Secuencia g(x) (referencia)")
        self.ax.legend(); self.canvas.draw()

# ===== Diferencias finitas =====
class FiniteDifferencesWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("📉 Diferencias Finitas")
        self.geometry("1120x760")

        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)

        # --- Derivadas 1D ---
        frm_1d = ttk.LabelFrame(container, text="Derivadas numéricas (datos discretos)")
        frm_1d.pack(fill="x", pady=6)

        ttk.Label(frm_1d, text="f(x):").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        self.e_fx = ttk.Entry(frm_1d)
        self.e_fx.grid(row=0, column=1, sticky="we", padx=6, pady=6)
        self.e_fx.insert(0, "sin(x)")

        ttk.Label(frm_1d, text="x_i:").grid(row=0, column=2, sticky="e", padx=6, pady=6)
        self.e_xi = ttk.Entry(frm_1d, width=12)
        self.e_xi.grid(row=0, column=3, sticky="we", padx=6, pady=6)
        self.e_xi.insert(0, "0.8")

        ttk.Label(frm_1d, text="h:").grid(row=0, column=4, sticky="e", padx=6, pady=6)
        self.e_h = ttk.Entry(frm_1d, width=12)
        self.e_h.grid(row=0, column=5, sticky="we", padx=6, pady=6)
        self.e_h.insert(0, "0.1")

        frm_1d.columnconfigure(1, weight=1)
        frm_1d.columnconfigure(3, weight=1)
        frm_1d.columnconfigure(5, weight=1)

        ttk.Button(frm_1d, text="Calcular diferencias", style="Big.TButton", command=self.run_1d)\
            .grid(row=0, column=6, padx=8, pady=6)

        cols = ("Esquema", "f'(x_i)", "f''(x_i)")
        self.tree_1d = ttk.Treeview(frm_1d, columns=cols, show="headings", height=4)
        for c in cols:
            self.tree_1d.heading(c, text=c)
            self.tree_1d.column(c, anchor="center", width=160 if c != "Esquema" else 180)
        self.tree_1d.grid(row=1, column=0, columnspan=7, sticky="nsew", padx=6, pady=6)
        frm_1d.rowconfigure(1, weight=1)

        self.lbl_samples = ttk.Label(frm_1d, text="Puntos evaluados: -")
        self.lbl_samples.grid(row=2, column=0, columnspan=7, sticky="w", padx=6, pady=(0,6))

        # --- Derivadas parciales ---
        frm_partial = ttk.LabelFrame(container, text="Derivadas parciales numéricas (mallado rectangular)")
        frm_partial.pack(fill="x", pady=6)

        ttk.Label(frm_partial, text="f(x,y):").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        self.e_fxy = ttk.Entry(frm_partial)
        self.e_fxy.grid(row=0, column=1, sticky="we", padx=6, pady=6)
        self.e_fxy.insert(0, "x**2 + y**2")

        ttk.Label(frm_partial, text="x_i:").grid(row=0, column=2, sticky="e", padx=6, pady=6)
        self.e_xp = ttk.Entry(frm_partial, width=12)
        self.e_xp.grid(row=0, column=3, sticky="we", padx=6, pady=6)
        self.e_xp.insert(0, "1.0")

        ttk.Label(frm_partial, text="y_j:").grid(row=0, column=4, sticky="e", padx=6, pady=6)
        self.e_yp = ttk.Entry(frm_partial, width=12)
        self.e_yp.grid(row=0, column=5, sticky="we", padx=6, pady=6)
        self.e_yp.insert(0, "-0.5")

        ttk.Label(frm_partial, text="h_x:").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        self.e_hx = ttk.Entry(frm_partial, width=12)
        self.e_hx.grid(row=1, column=1, sticky="we", padx=6, pady=6)
        self.e_hx.insert(0, "0.1")

        ttk.Label(frm_partial, text="h_y:").grid(row=1, column=2, sticky="e", padx=6, pady=6)
        self.e_hy = ttk.Entry(frm_partial, width=12)
        self.e_hy.grid(row=1, column=3, sticky="we", padx=6, pady=6)
        self.e_hy.insert(0, "0.1")

        frm_partial.columnconfigure(1, weight=1)
        frm_partial.columnconfigure(3, weight=1)
        frm_partial.columnconfigure(5, weight=1)

        ttk.Button(frm_partial, text="Calcular parciales", style="Big.TButton", command=self.run_partial)\
            .grid(row=1, column=4, columnspan=2, padx=8, pady=6, sticky="we")

        self.lbl_partial = ttk.Label(frm_partial, text="∂f/∂x ≈ -,   ∂f/∂y ≈ -")
        self.lbl_partial.grid(row=2, column=0, columnspan=6, sticky="w", padx=6, pady=(0,6))

        info = (
            "Las fórmulas empleadas siguen las diferencias finitas progresivas, regresivas y centrales "
            "para f'(x_i) y f''(x_i); las derivadas parciales usan un esquema central sobre una malla."
        )
        ttk.Label(container, text=info, wraplength=980, justify="left").pack(fill="x", pady=(8,0))

    def clear_tree(self):
        for item in self.tree_1d.get_children():
            self.tree_1d.delete(item)

    def run_1d(self):
        try:
            f_str = self.e_fx.get().strip()
            fx = safe_eval_function_1d(f_str)
            xi = float(self.e_xi.get().replace(",", "."))
            h = float(self.e_h.get().replace(",", "."))
            if h == 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Error", "Verifique f(x), x_i y h.")
            return

        def f_eval(x):
            val = fx(np.array([x]))
            val = np.asarray(val, dtype=float)
            if val.size == 0 or not np.isfinite(val[0]):
                raise ValueError("Evaluación de f(x) no finita.")
            return float(val[0])

        try:
            f_x = f_eval(xi)
            f_xph = f_eval(xi + h)
            f_xmh = f_eval(xi - h)
            f_xp2h = f_eval(xi + 2*h)
            f_xm2h = f_eval(xi - 2*h)
        except Exception as e:
            messagebox.showerror("Evaluación", str(e))
            return

        self.clear_tree()

        rows = [
            ("Progresiva",
             (f_xph - f_x) / h,
             (f_xp2h - 2*f_xph + f_x) / (h**2)),
            ("Regresiva",
             (f_x - f_xmh) / h,
             (f_x - 2*f_xmh + f_xm2h) / (h**2)),
            ("Central",
             (f_xph - f_xmh) / (2*h),
             (f_xph - 2*f_x + f_xmh) / (h**2)),
        ]

        for esquema, d1, d2 in rows:
            self.tree_1d.insert("", "end", values=(esquema, f"{d1:.10g}", f"{d2:.10g}"))

        puntos = (
            f"f(x_i) = {f_x:.10g}, f(x_i±h) = [{f_xmh:.10g}, {f_xph:.10g}], "
            f"f(x_i±2h) = [{f_xm2h:.10g}, {f_xp2h:.10g}]"
        )
        self.lbl_samples.config(text="Puntos evaluados: " + "  ".join(puntos))

    def run_partial(self):
        try:
            f_str = self.e_fxy.get().strip()
            fxy = safe_eval_function_2d(f_str)
            xi = float(self.e_xp.get().replace(",", "."))
            yj = float(self.e_yp.get().replace(",", "."))
            hx = float(self.e_hx.get().replace(",", "."))
            hy = float(self.e_hy.get().replace(",", "."))
            if hx == 0 or hy == 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Error", "Verifique f(x,y), (x_i,y_j) y los pasos h_x, h_y.")
            return

        def f_eval(x, y):
            val = fxy(np.array([x]), np.array([y]))
            val = np.asarray(val, dtype=float)
            if val.size == 0 or not np.isfinite(val[0]):
                raise ValueError("Evaluación de f(x,y) no finita.")
            return float(val[0])

        try:
            df_dx = (f_eval(xi + hx, yj) - f_eval(xi - hx, yj)) / (2 * hx)
            df_dy = (f_eval(xi, yj + hy) - f_eval(xi, yj - hy)) / (2 * hy)
        except Exception as e:
            messagebox.showerror("Evaluación", str(e))
            return

        self.lbl_partial.config(text=f"∂f/∂x ≈ {df_dx:.10g},   ∂f/∂y ≈ {df_dy:.10g}")


# ===== Runge-Kutta =====
class RungeKuttaWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("🧮 Métodos de Runge–Kutta")
        self.geometry("1150x760")

        frm = ttk.LabelFrame(self, text="Problema de valor inicial: y' = f(x,y)")
        frm.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm, text="f(x,y):").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        self.e_f = ttk.Entry(frm)
        self.e_f.grid(row=0, column=1, sticky="we", padx=6, pady=6)
        self.e_f.insert(0, "x + y")

        ttk.Label(frm, text="x₀:").grid(row=0, column=2, sticky="e", padx=6, pady=6)
        self.e_x0 = ttk.Entry(frm, width=12)
        self.e_x0.grid(row=0, column=3, sticky="we", padx=6, pady=6)
        self.e_x0.insert(0, "0.0")

        ttk.Label(frm, text="y₀:").grid(row=0, column=4, sticky="e", padx=6, pady=6)
        self.e_y0 = ttk.Entry(frm, width=12)
        self.e_y0.grid(row=0, column=5, sticky="we", padx=6, pady=6)
        self.e_y0.insert(0, "1.0")

        ttk.Label(frm, text="h:").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        self.e_h = ttk.Entry(frm, width=12)
        self.e_h.grid(row=1, column=1, sticky="we", padx=6, pady=6)
        self.e_h.insert(0, "0.2")

        ttk.Label(frm, text="Pasos N:").grid(row=1, column=2, sticky="e", padx=6, pady=6)
        self.e_n = ttk.Entry(frm, width=12)
        self.e_n.grid(row=1, column=3, sticky="we", padx=6, pady=6)
        self.e_n.insert(0, "10")

        ttk.Label(frm, text="Método:").grid(row=1, column=4, sticky="e", padx=6, pady=6)
        self.cmb_method = ttk.Combobox(frm, state="readonly", values=[
            "Euler (RK1)",
            "Heun (RK2)",
            "Kutta clásico (RK3)",
            "Clásico (RK4)",
        ])
        self.cmb_method.grid(row=1, column=5, sticky="we", padx=6, pady=6)
        self.cmb_method.current(3)

        frm.columnconfigure(1, weight=1)
        frm.columnconfigure(3, weight=1)
        frm.columnconfigure(5, weight=1)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=(0,6))
        ttk.Button(btns, text="Integrar", style="Big.TButton", command=self.run).pack(side="left", padx=5)
        ttk.Button(btns, text="Limpiar", command=self.clear).pack(side="left", padx=5)

        self.lbl_res = ttk.Label(self, text="Resultado: -")
        self.lbl_res.pack(fill="x", padx=12, pady=6)

        cols = ("n", "x_n", "y_n", "k1", "k2", "k3", "k4")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=10)
        for idx, c in enumerate(cols):
            self.tree.heading(c, text=c)
            width = 90 if c == "n" else 150
            self.tree.column(c, width=width, anchor="center")
        self.tree.pack(fill="both", expand=False, padx=10, pady=8)

        self.fig, self.ax = plt.subplots(figsize=(8.6, 4.0), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0,10))
        self._style_axes()

    def _style_axes(self):
        self.ax.clear()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y(x)")
        self.ax.grid(True, linestyle="--", linewidth=0.6)
        self.canvas.draw()

    def clear(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.lbl_res.config(text="Resultado: -")
        self._style_axes()

    def run(self):
        try:
            f_str = self.e_f.get().strip()
            f = safe_eval_function_2d(f_str)
            x0 = float(self.e_x0.get().replace(",", "."))
            y0 = float(self.e_y0.get().replace(",", "."))
            h = float(self.e_h.get().replace(",", "."))
            if h == 0:
                raise ValueError
            N = int(self.e_n.get())
            if N <= 0:
                raise ValueError
            method = self.cmb_method.get()
        except Exception:
            messagebox.showerror("Error", "Revise la función, condiciones iniciales, h y N.")
            return

        def f_eval(x, y):
            val = f(np.array([x]), np.array([y]))
            val = np.asarray(val, dtype=float)
            if val.size == 0 or not np.isfinite(val[0]):
                raise ValueError("Evaluación de f(x,y) no finita.")
            return float(val[0])

        self.clear()

        def step(xn, yn):
            if method == "Euler (RK1)":
                k1 = f_eval(xn, yn)
                yn1 = yn + h * k1
                return yn1, {"k1": k1, "k2": None, "k3": None, "k4": None}
            if method == "Heun (RK2)":
                k1 = f_eval(xn, yn)
                k2 = f_eval(xn + h, yn + h * k1)
                yn1 = yn + h * 0.5 * (k1 + k2)
                return yn1, {"k1": k1, "k2": k2, "k3": None, "k4": None}
            if method == "Kutta clásico (RK3)":
                k1 = f_eval(xn, yn)
                k2 = f_eval(xn + 0.5 * h, yn + 0.5 * h * k1)
                k3 = f_eval(xn + h, yn - h * k1 + 2 * h * k2)
                yn1 = yn + (h / 6.0) * (k1 + 4 * k2 + k3)
                return yn1, {"k1": k1, "k2": k2, "k3": k3, "k4": None}
            if method == "Clásico (RK4)":
                k1 = f_eval(xn, yn)
                k2 = f_eval(xn + 0.5 * h, yn + 0.5 * h * k1)
                k3 = f_eval(xn + 0.5 * h, yn + 0.5 * h * k2)
                k4 = f_eval(xn + h, yn + h * k3)
                yn1 = yn + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                return yn1, {"k1": k1, "k2": k2, "k3": k3, "k4": k4}
            raise ValueError("Método desconocido")

        xs = [x0]
        ys = [y0]

        try:
            for n in range(N):
                xn, yn = xs[-1], ys[-1]
                y_next, ks = step(xn, yn)
                ks_row = []
                for key in ("k1", "k2", "k3", "k4"):
                    val = ks.get(key)
                    ks_row.append("-" if val is None else f"{val:.8g}")
                self.tree.insert("", "end", values=(n, f"{xn:.8g}", f"{yn:.8g}", *ks_row))
                xs.append(xn + h)
                ys.append(y_next)
            # último punto sin ks
            self.tree.insert("", "end", values=(N, f"{xs[-1]:.8g}", f"{ys[-1]:.8g}", "-", "-", "-", "-"))
        except Exception as e:
            messagebox.showerror("Evaluación", str(e))
            return

        self.lbl_res.config(text=f"Resultado: y(x_N) ≈ {ys[-1]:.10g} (x_N = {xs[-1]:.10g})")

        self._style_axes()
        self.ax.plot(xs, ys, marker="o", label=method)
        self.ax.legend()
        self.canvas.draw()

# ===== Lagrange =====
def lagrange_interpolacion(puntos):
    x = sp.Symbol('x')
    P = 0
    for i, (xi, yi) in enumerate(puntos):
        Li = 1
        for j, (xj, _) in enumerate(puntos):
            if i != j:
                Li *= (x - xj) / (xi - xj)
        P += yi * Li
    P = sp.expand(P)
    P = sp.nsimplify(P, rational=True)
    return sp.together(sp.simplify(P))

class LagrangeWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("📈 Interpolación de Lagrange (fracciones + gráfico)")
        self.geometry("1150x720")

        paned = ttk.Panedwindow(self, orient="horizontal"); paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, width=360, padding=10, style="Card.TFrame")
        right = ttk.Frame(paned, padding=8)
        paned.add(left, weight=0); paned.add(right, weight=1)

        box = ttk.LabelFrame(left, text="Puntos"); box.pack(fill="x")
        ttk.Label(box, text="x:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.e_x = ttk.Entry(box, width=10); self.e_x.grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(box, text="y:").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        self.e_y = ttk.Entry(box, width=10); self.e_y.grid(row=0, column=3, sticky="w", padx=2)
        ttk.Button(box, text="Agregar", command=self.agregar_punto).grid(row=0, column=4, padx=6)

        ttk.Label(box, text="Xmáx (opt):").grid(row=1, column=0, columnspan=2, sticky="e", padx=4)
        self.e_xmax = ttk.Entry(box, width=10); self.e_xmax.grid(row=1, column=2, sticky="w")
        ttk.Label(box, text="Ymáx (opt):").grid(row=1, column=3, sticky="e", padx=4)
        self.e_ymax = ttk.Entry(box, width=10); self.e_ymax.grid(row=1, column=4, sticky="w")

        lst_frame = ttk.Frame(left); lst_frame.pack(fill="x", pady=6)
        self.lst = tk.Listbox(lst_frame, height=7); self.lst.pack(side="left", fill="x", expand=True)
        sb = ttk.Scrollbar(lst_frame, orient="vertical", command=self.lst.yview); sb.pack(side="right", fill="y")
        self.lst.config(yscrollcommand=sb.set)

        btns = ttk.Frame(left); btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Calcular y graficar", style="Big.TButton", command=self.calcular_y_graficar).pack(fill="x", pady=2)
        ttk.Button(btns, text="Eliminar seleccionado", command=self.eliminar_sel).pack(fill="x", pady=2)
        ttk.Button(btns, text="Limpiar puntos", command=self.limpiar_puntos).pack(fill="x", pady=2)

        poly_frame = ttk.LabelFrame(left, text="Polinomio (fracciones)")
        poly_frame.pack(fill="both", expand=True)
        self.txt_poly = tk.Text(poly_frame, height=10, wrap="word")
        self.txt_poly.pack(fill="both", expand=True, padx=6, pady=6)

        self.fig, self.ax = plt.subplots(figsize=(8,6), dpi=110)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right); self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        self._puntos = []; self._P = None; self._init_plot()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_title("Gráfico P(x) y puntos")
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.ax.axhline(0, color="black", lw=0.8); self.ax.axvline(0, color="black", lw=0.8)
        self.ax.grid(True, ls="--", lw=0.6); self.canvas.draw()

    def agregar_punto(self):
        x_str = self.e_x.get().strip().replace(",", "."); y_str = self.e_y.get().strip().replace(",", ".")
        try:
            xi = sp.nsimplify(x_str, rational=True); yi = sp.nsimplify(y_str, rational=True)
        except Exception:
            messagebox.showerror("Error", "x o y inválidos."); return
        self._puntos.append((xi, yi)); self.lst.insert("end", f"({xi}, {yi})")
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
            messagebox.showwarning("Aviso", "Agregá al menos 2 puntos."); return
        try:
            P = lagrange_interpolacion(self._puntos)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        self._P = P
        self.txt_poly.delete("1.0", "end"); self.txt_poly.insert("end", sp.pretty(P) + "\n")

        self.ax.clear()
        self.ax.set_title(r"$\displaystyle " + sp.latex(P) + r"$", fontsize=14, pad=14)
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.ax.grid(True, ls="--", lw=0.6)
        self.ax.axhline(0, color="black", lw=0.8); self.ax.axvline(0, color="black", lw=0.8)

        xs_pts = [float(p[0]) for p in self._puntos]; ys_pts = [float(p[1]) for p in self._puntos]
        xmin, xmax = min(xs_pts), max(xs_pts); pad_x = 0.2 * (xmax - xmin if xmax > xmin else 1.0)

        try:
            xmax_user = float(self.e_xmax.get().replace(",", ".")) if self.e_xmax.get().strip() else None
            ymax_user = float(self.e_ymax.get().replace(",", ".")) if self.e_ymax.get().strip() else None
        except Exception:
            messagebox.showerror("Error", "Xmáx/Ymáx inválidos."); return

        if xmax_user is not None:
            x_left, x_right = -abs(xmax_user), abs(xmax_user)
        else:
            x_left, x_right = xmin - pad_x, xmax + pad_x
        if x_left == x_right: x_left -= 1; x_right += 1

        x = sp.Symbol('x'); f = sp.lambdify(x, P, modules=["numpy"])
        xs = np.linspace(x_left, x_right, 800)
        try:
            ys = np.asarray(f(xs), dtype=float)
        except Exception:
            ys = np.array([float(sp.N(P.subs(x, t))) for t in xs], dtype=float)

        self.ax.set_xlim(x_left, x_right)
        if ymax_user is not None: self.ax.set_ylim(-abs(ymax_user), abs(ymax_user))
        try:
            self.ax.plot(xs, ys, label="P(x)")
        except Exception:
            pass
        self.ax.scatter(xs_pts, ys_pts, c="red", zorder=5, label="Puntos")

        if ymax_user is None:
            try:
                y_min, y_max = np.nanmin(ys), np.nanmax(ys)
                if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max: raise ValueError
                pad_y = 0.1 * (y_max - y_min); self.ax.set_ylim(y_min - pad_y, y_max + pad_y)
            except Exception:
                self.ax.set_ylim(-1, 1)
        self.ax.legend(); self.canvas.draw()

# =========================
# Launcher principal
# =========================
class SuiteNumericaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Suite Numérica — UADE")
        self.geometry("1080x720")
        setup_style(self)

        header = ttk.Frame(self, style="App.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Suite Numérica", style="Title.TLabel").pack(anchor="w", padx=16, pady=(16,2))
        subtitle = (
            "Monte Carlo · Newton–Cotes · Newton–Raphson · Lagrange · Punto Fijo · "
            "Bisección · Aitken Δ² · Diferencias Finitas · Runge-Kutta"
        )
        ttk.Label(header, text=subtitle, style="Sub.TLabel").pack(anchor="w", padx=16, pady=(0,16))

        grid = ttk.Frame(self, padding=16)
        grid.pack(fill="both", expand=True)

        def card(parent, emoji, title, desc, cmd):
            frame = ttk.Frame(parent, padding=16, style="Card.TFrame")
            lbl_t = tk.Label(frame, text=f"{emoji}  {title}", font=("Segoe UI", 14, "bold"),
                             fg=APP_TEXT, bg=APP_CARD)
            lbl_t.pack(anchor="w")
            lbl_d = tk.Label(frame, text=desc, fg="#cbd5e1", bg=APP_CARD, wraplength=360, justify="left")
            lbl_d.pack(anchor="w", pady=(6,8))
            ttk.Button(frame, text="Abrir", style="Big.TButton", command=cmd).pack(anchor="e")
            return frame

        cards = [
            ("🎲", "Monte Carlo (1D/2D)",
             "Valor medio + hit-or-miss. Muestra I_MC, Gauss, media muestral, SE, σ_f e IC.",
             lambda: MonteCarlo1DWindow(self)),
            ("📐", "Newton–Cotes",
             "Rectángulo, Trapecio, Simpson 1/3 y 3/8 con tabla de nodos y sombreado.",
             lambda: NewtonCotesWindow(self)),
            ("🔎", "Newton–Raphson",
             "Raíces con derivada automática (Sympy), tabla y gráfico de iteraciones.",
             lambda: NewtonRaphsonWindow(self)),
            ("📈", "Lagrange",
             "Interpolación polinómica (fracciones exactas), muestra P(x) y grafica.",
             lambda: LagrangeWindow(self)),
            ("🧭", "Punto Fijo",
             "Iteración x_{n+1}=g(x_n). Diagonal y=g(x), cobweb y tabla.",
             lambda: FixedPointWindow(self)),
            ("🪚", "Bisección",
             "Búsqueda binaria de raíces con tabla (a,b,c) y gráfico de f(x).",
             lambda: BisectionWindow(self)),
            ("⚡", "Aitken Δ²",
             "Acelera una secuencia tipo punto fijo: compara g(x) vs acelerada.",
             lambda: AitkenWindow(self)),
            ("📉", "Diferencias Finitas",
             "Aproxima derivadas primera/segunda y parciales con fórmulas progresivas, regresivas y centrales.",
             lambda: FiniteDifferencesWindow(self)),
            ("🧮", "Runge–Kutta",
             "Resuelve y' = f(x,y) con Euler, Heun, RK-3 y RK-4; tabla de pendientes k_i y gráfica.",
             lambda: RungeKuttaWindow(self)),
        ]

        # Layout 3x3 (ajustable)
        rows, cols = 3, 3
        for i in range(rows):
            grid.rowconfigure(i, weight=1)
        for j in range(cols):
            grid.columnconfigure(j, weight=1)

        for idx, (emj, title, desc, cmd) in enumerate(cards):
            r, c = divmod(idx, cols)
            w = card(grid, emj, title, desc, cmd)
            w.grid(row=r, column=c, sticky="nsew", padx=10, pady=10)

        footer = ttk.Frame(self, style="App.TFrame")
        footer.pack(fill="x")
        ttk.Label(footer, text="UADE · Modelado y Simulación — by Tobias", style="Sub.TLabel")\
           .pack(anchor="e", padx=16, pady=10)

# =========================
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
    app = SuiteNumericaApp()
    app.mainloop()
