# -*- coding: utf-8 -*-
"""
Topgal / SL - Generador de gráficos q(L) (GUI) - VERSIÓN MULTIMODO
- Modos: Fachadas y Techumbres (Dinámico).
- Filtro físico: No se calcula montante para 0 tomas intermedias.
- Aluminio: FS=1.65 en tensión (Integrado en sigma_adm).
- Policarbonato: FS=3.0 estricto en el cálculo global.
- Curva del Sistema: Fusión (Envolvente mínima) entre Panel y Montante.
- Incorporación de Tensión Admisible (Resistencia) para Policarbonato.
- Interfaz Gráfica Rediseñada (Horizontal) para pantallas pequeñas.
- Módulo Seccional (W) explícito e editable para cálculo real de resistencia.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ==========================================================
# UTILIDAD RUTAS (script vs exe)
# ==========================================================
def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# ==========================================================
# MOTOR DE GIF ANIMADO PARA TKINTER
# ==========================================================
class AnimatedGifLabel(tk.Label):
    def __init__(self, master, path, delay=100, **kwargs):
        super().__init__(master, **kwargs)
        self.delay = delay
        self.frames = []
        self.is_animating = False
        self.current_frame = 0

        if os.path.exists(path):
            try:
                i = 0
                while True:
                    frame = tk.PhotoImage(file=path, format=f"gif -index {i}")
                    self.frames.append(frame)
                    i += 1
            except tk.TclError:
                pass

        if self.frames:
            self.config(image=self.frames[0])
            self.start_animation()
        else:
            self.config(text="[ Imagen DVP.gif no encontrada ]", foreground="red", background="white", width=40, height=5)

    def start_animation(self):
        self.is_animating = True
        self._animate()

    def stop_animation(self):
        self.is_animating = False

    def _animate(self):
        if not self.is_animating or not self.frames:
            return
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self.config(image=self.frames[self.current_frame])
        self.after(self.delay, self._animate)


# ==========================================================
# CONSTANTES DE DOMINIO
# ==========================================================
g = 9.80665  # [m/s²]

L_MIN_MM = 500
L_MAX_MM = 4000
PASO_MM = 10  

PRESIONES_DVP_KGF = np.arange(50.0, 601.0, 50.0)
PLACA_INCLUDE_END_SUPPORTS = True

B_POR_ESPESOR_MM = {16: 998.0, 20: 998.0}
I_POR_ESPESOR_CM4 = {16: 12.71, 20: 19.91}

IMAGENES_TOMAS = {
    1: "esquema_1.png",
    2: "esquema_2.png",
    3: "esquema_3.png",
    4: "esquema_4.png",
}

# ==========================================================
# BEAM FE (Euler-Bernoulli) 1D
# ==========================================================
def _beam_element_stiffness(EI: float, L: float) -> np.ndarray:
    k = EI / (L ** 3)
    return k * np.array([
        [12,    6*L,   -12,    6*L],
        [6*L, 4*L**2,  -6*L, 2*L**2],
        [-12,   -6*L,   12,   -6*L],
        [6*L, 2*L**2,  -6*L, 4*L**2],
    ], dtype=float)

def _beam_element_consistent_load(w: float, L: float) -> np.ndarray:
    return np.array([w*L/2.0, w*(L**2)/12.0, w*L/2.0, -w*(L**2)/12.0], dtype=float)

def _assemble_beam(nodes_x: np.ndarray, EI: float, w_line: float):
    nodes_x = np.array(nodes_x, dtype=float)
    n = len(nodes_x)
    dof = 2*n
    K = np.zeros((dof, dof), dtype=float)
    F = np.zeros(dof, dtype=float)
    for e in range(n-1):
        Le = float(nodes_x[e+1] - nodes_x[e])
        ke = _beam_element_stiffness(EI, Le)
        fe = _beam_element_consistent_load(w_line, Le)
        idx = np.array([2*e, 2*e+1, 2*(e+1), 2*(e+1)+1], dtype=int)
        F[idx] += fe
        K[np.ix_(idx, idx)] += ke
    return K, F

def _apply_constraints(K: np.ndarray, F: np.ndarray, constrained_dofs: list[int]) -> np.ndarray:
    dof = K.shape[0]
    constrained = np.unique(np.array(constrained_dofs, dtype=int))
    all_dofs = np.arange(dof, dtype=int)
    free = np.setdiff1d(all_dofs, constrained, assume_unique=False)
    Kff = K[np.ix_(free, free)]
    Ff = F[free]
    uf = np.linalg.solve(Kff, Ff)
    u = np.zeros(dof, dtype=float)
    u[free] = uf
    return u

def _shape_functions_v(s: float, L: float) -> np.ndarray:
    return np.array([1 - 3*s**2 + 2*s**3, L * (s - 2*s**2 + s**3), 3*s**2 - 2*s**3, L * (-s**2 + s**3)], dtype=float)

def _support_node_indices(n_nodes: int, include_end_supports: bool) -> list[int]:
    idx = [0, n_nodes-1] if include_end_supports else []
    idx += list(range(1, n_nodes-1))
    return sorted(set(idx))

def beam_response_max(nodes_x: np.ndarray, E: float, I: float, B: float, p_area: float, support_node_ids: list[int]) -> tuple[float, float]:
    EI = E * I
    w_line = p_area * B 
    K, F = _assemble_beam(nodes_x, EI, w_line)
    constrained = [2*i for i in support_node_ids] 
    if len(support_node_ids) < 2: constrained.append(2*support_node_ids[0] + 1)
    u = _apply_constraints(K, F, constrained)

    vmax, Mmax = 0.0, 0.0
    for e in range(len(nodes_x)-1):
        Le = float(nodes_x[e+1] - nodes_x[e])
        de = u[np.array([2*e, 2*e+1, 2*(e+1), 2*(e+1)+1], dtype=int)]
        for s in np.linspace(0.0, 1.0, 25):
            v = float(_shape_functions_v(s, Le) @ de)
            vpp_vec = (1.0/(Le**2)) * np.array([-6+12*s, Le*(-4+6*s), 6-12*s, Le*(-2+6*s)], dtype=float)
            M = float(EI * (vpp_vec @ de))
            vmax, Mmax = max(vmax, abs(v)), max(Mmax, abs(M))
    return vmax, Mmax

# ==========================================================
# INTERPOLACIÓN INVERSA PARA DVP (EXCEL)
# ==========================================================
def invertir_curva_para_excel(L_mm: np.ndarray, q_vals: np.ndarray) -> pd.DataFrame:
    mask = np.isfinite(q_vals)
    L_c = L_mm[mask]
    q_c = q_vals[mask]
    
    if len(q_c) == 0:
        return pd.DataFrame({"Presión Viento (kgf/m²)": PRESIONES_DVP_KGF, "Longitud Máx Admisible L (mm)": np.nan})

    sort_idx = np.argsort(q_c)
    q_asc = q_c[sort_idx]
    L_asc = L_c[sort_idx]
    
    L_targets = np.interp(PRESIONES_DVP_KGF, q_asc, L_asc, left=np.nan, right=np.nan)
    return pd.DataFrame({
        "Presión Viento (kgf/m²)": PRESIONES_DVP_KGF,
        "Longitud Máx Admisible L (mm)": np.round(L_targets, 1)
    })

# ==========================================================
# CURVAS q(L) - LÓGICA DE NEGOCIO FACHADAS/TECHUMBRE
# ==========================================================
def placas_q_vs_L(n_tomas: int, E_Pa: float, B_m: float, I_m4: float, ratio_L_over_delta: float, sigma_adm_Pa: float, c_m: float) -> pd.DataFrame:
    L_mm = np.arange(L_MIN_MM, L_MAX_MM + PASO_MM, PASO_MM)
    q_def_list, q_ten_list, q_min_list = [], [], []
    
    for Ltot_mm in L_mm:
        Ltot = Ltot_mm / 1000.0
        a = Ltot / (n_tomas + 1)
        delta_adm = a / ratio_L_over_delta
        nodes_x = np.linspace(0.0, Ltot, n_tomas + 2)
        v1, M1 = beam_response_max(nodes_x, E_Pa, I_m4, B_m, 1.0, _support_node_indices(len(nodes_x), PLACA_INCLUDE_END_SUPPORTS))
        
        p_allow_def = delta_adm / v1 if v1 > 0 else np.nan
        M_allow = sigma_adm_Pa * I_m4 / c_m if sigma_adm_Pa else np.inf
        p_allow_ten = M_allow / M1 if M1 > 0 else np.nan
        
        q_def = (p_allow_def / g)
        q_ten = (p_allow_ten / g)
        q_min = np.nanmin([q_def, q_ten])
        
        q_def_list.append(q_def)
        q_ten_list.append(q_ten)
        q_min_list.append(q_min)
        
    return pd.DataFrame({
        "L_total (mm)": L_mm,
        "q_deflexion (kgf/m2)": np.array(q_def_list),
        "q_tension (kgf/m2)": np.array(q_ten_list),
        "q_admisible (kgf/m2)": np.array(q_min_list)
    })

def conector_q_vs_L(n_tomas: int, E: float, Iyy: float, W_m3: float, sigma_adm: float, ratio_montante: float, fs_aplicado: float, B_m: float) -> pd.DataFrame:
    L_mm = np.arange(L_MIN_MM, L_MAX_MM + PASO_MM, PASO_MM)
    q_def_list, q_ten_list, q_min_list = [], [], []
    
    for Ltot_mm in L_mm:
        Ltot = Ltot_mm / 1000.0
        
        delta_adm = min(Ltot / ratio_montante, 50.0 / 1000.0) 
        
        nodes_x = np.linspace(0.0, Ltot, n_tomas + 2)
        v1, M1 = beam_response_max(nodes_x, E, Iyy, B_m, 1.0, [0, len(nodes_x)-1])

        p_allow_def = delta_adm / v1 if v1 > 0 else np.nan
        M_allow = sigma_adm * W_m3 if sigma_adm else np.inf
        p_allow_ten = M_allow / M1 if M1 > 0 else np.nan

        q_def = (p_allow_def / g) / fs_aplicado
        q_ten = (p_allow_ten / g) / fs_aplicado
        q_min = np.nanmin([q_def, q_ten])

        q_def_list.append(q_def)
        q_ten_list.append(q_ten)
        q_min_list.append(q_min)

    return pd.DataFrame({
        "L_total (mm)": L_mm,
        "q_deflexion (kgf/m2)": np.array(q_def_list),
        "q_tension (kgf/m2)": np.array(q_ten_list),
        "q_admisible (kgf/m2)": np.array(q_min_list)
    })

# ==========================================================
# MOTOR DE GRAFICACIÓN DE ÉLITE
# ==========================================================
def _ajustar_eje_x_por_qmax(ax, L_mm: np.ndarray, q_vals: np.ndarray):
    Q_MAX_GRAFICO = 200.0
    MARGEN_X_MM = 150
    mask = np.isfinite(q_vals) & (q_vals <= Q_MAX_GRAFICO)
    if np.any(mask):
        L_vis = L_mm[mask]
        xmin = max(float(L_vis.min()) - MARGEN_X_MM, float(L_mm.min()))
        xmax = min(float(L_vis.max()) + MARGEN_X_MM, float(L_mm.max()))
        if xmax - xmin < 300: xmax = min(xmin + 300, float(L_mm.max()))
        ax.set_xlim(xmin, xmax)
        return
    xmax = float(L_mm.max())
    xmin = max(xmax - 600.0, float(L_mm.min()))
    ax.set_xlim(xmin, xmax)

def _plot_esquema(ax_esq, n_tomas: int):
    img_file = IMAGENES_TOMAS.get(n_tomas, "")
    img_path = resource_path(img_file) if img_file else ""

    if img_file and os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax_esq.imshow(img)
        ax_esq.axis("off")
        tomas_text = f"{n_tomas-1} tomas int." if n_tomas > 1 else "Sin tomas int."
        ax_esq.set_title(tomas_text, fontsize=10, fontweight="bold")
    else:
        ax_esq.text(0.5, 0.5, f"Esquema {n_tomas}\n(Imagen no encontrada)", ha="center", va="center")
        ax_esq.axis("off")

def _anotar_li(ax, L_mm, q_vals, q_des, color, y_text):
    ax.axhline(q_des, linestyle="--", color=color, label=f"q viento = {q_des:.0f} kgf/m²")
    qmin = float(np.nanmin(q_vals))
    qmax = float(np.nanmax(q_vals))
    if np.isfinite(qmin) and np.isfinite(qmax) and (qmin <= q_des <= qmax):
        idx = int(np.nanargmin(np.abs(q_vals - q_des)))
        Li = float(L_mm[idx])
        ax.axvline(Li, linestyle="--", color=color)
        ax.text(Li + 30, y_text, f"L ≈ {Li:.0f} mm", color=color, fontweight="bold")

def guardar_grafico(L_mm, q_vals, titulo, filename, n_tomas, q_disenos=None, q_def=None, q_ten=None):
    fig, (ax_esq, ax) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 2.5]})
    
    _plot_esquema(ax_esq, n_tomas)
    
    ax.plot(L_mm, q_vals, linewidth=2.5, color='blue', label="q(L) Admisible")
    
    if q_def is not None and q_ten is not None:
        ax.plot(L_mm, q_def, linestyle=":", color='orange', linewidth=1.5, label="Límite por Flecha")
        ax.plot(L_mm, q_ten, linestyle="--", color='red', linewidth=1.5, label="Límite por Tensión")

    _ajustar_eje_x_por_qmax(ax, L_mm, q_vals)
    
    ymax = 600.0
    xmin_val, xmax_val = ax.get_xlim()
    mask_vis = (L_mm >= xmin_val) & (L_mm <= xmax_val) & np.isfinite(q_vals)
    if np.any(mask_vis):
        ymax = min(float(np.nanmax(q_vals[mask_vis])) * 1.15, 600.0)
    ax.set_ylim(0.0, max(ymax, 100.0))

    if q_disenos:
        colores = ["red", "orange", "magenta", "purple"]
        y_max_ax = ax.get_ylim()[1]
        for i, q_des in enumerate(q_disenos):
            if q_des > 0:  
                _anotar_li(ax, L_mm, q_vals, q_des, colores[i % len(colores)], y_max_ax * (0.90 - 0.07 * i))

    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel("Altura L [mm]", fontweight="bold")
    ax.set_ylabel("Presión de viento q(L) [kgf/m²]", fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.7)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig) # Importante para liberar memoria

def guardar_grafico_sistema(L_mm, q_sistema, titulo, filename, n_tomas, q_disenos=None):
    fig, (ax_esq, ax) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 2.5]})
    
    _plot_esquema(ax_esq, n_tomas)
    
    ax.plot(L_mm, q_sistema, linewidth=3.0, color='black', label="q(L) SISTEMA (Envolvente Mínima)")

    _ajustar_eje_x_por_qmax(ax, L_mm, q_sistema)
    
    ymax = 600.0
    xmin_val, xmax_val = ax.get_xlim()
    mask_vis = (L_mm >= xmin_val) & (L_mm <= xmax_val) & np.isfinite(q_sistema)
    if np.any(mask_vis):
        ymax = min(float(np.nanmax(q_sistema[mask_vis])) * 1.15, 600.0)
    ax.set_ylim(0.0, max(ymax, 100.0))

    if q_disenos:
        colores = ["red", "orange", "magenta", "purple"]
        y_max_ax = ax.get_ylim()[1]
        for i, q_des in enumerate(q_disenos):
            if q_des > 0:  
                _anotar_li(ax, L_mm, q_sistema, q_des, colores[i % len(colores)], y_max_ax * (0.90 - 0.07 * i))

    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel("Altura L [mm]", fontweight="bold")
    ax.set_ylabel("Presión de viento q(L) [kgf/m²]", fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.7)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig) # Importante para liberar memoria

# ==========================================================
# GUI - FACHADAS / TECHUMBRES
# ==========================================================
class TopgalFachadasApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SL - Generador de Gráficos FACHADA y TECHUMBRE - Versión 11.1")
        self.geometry("850x750") 
        
        # VARIABLES GLOBALES ESTRUCTURALES
        self.var_modo = tk.StringVar(value="Fachada")  
        
        self.var_espesor = tk.StringVar(value="16")
        self.var_E_placa = tk.StringVar(value="2.4")
        self.var_B = tk.StringVar(value="998")
        self.var_I_placa = tk.StringVar(value="18.0")
        self.var_sigma_placa = tk.StringVar(value="15.0")  
        self.var_ratio_placa = tk.StringVar(value="50")
        
        self.var_tipo_montante = tk.StringVar(value="Aluminio")
        self.var_E_montante = tk.StringVar(value="70.0")
        self.var_I_montante = tk.StringVar(value="2.75") 
        self.var_W_montante = tk.StringVar(value="1.77") 
        self.var_sigma_montante = tk.StringVar(value="142.4") 
        self.var_ratio_montante = tk.StringVar(value="50")

        # VARIABLES NUEVAS PARA TECHUMBRE
        self.var_peso_pc = tk.StringVar(value="1.0")
        self.var_carga_nieve = tk.StringVar(value="0.0")

        self.var_q1 = tk.StringVar(value="50")
        self.var_q2 = tk.StringVar(value="100")
        self.var_q3 = tk.StringVar(value="150")

        self._build()

    def _on_espesor_change(self, _evt=None):
        e = int(self.var_espesor.get())
        self.var_B.set(str(B_POR_ESPESOR_MM.get(e, 998.0)))
        self.var_I_placa.set(str(I_POR_ESPESOR_CM4.get(e, 18.0)))
        
        if e == 16:
            self.var_peso_pc.set("1.0")
        elif e == 20:
            self.var_peso_pc.set("1.25")

    def _on_modo_change(self):
        modo = self.var_modo.get()
        if modo == "Techumbre":
            self.frm_techumbre.grid()  
        else:
            self.frm_techumbre.grid_remove() 

    def _on_montante_change(self):
        tipo = self.var_tipo_montante.get()
        if tipo == "Aluminio":
            self.var_E_montante.set("70.0")
            self.var_I_montante.set("2.75")
            self.var_W_montante.set("1.77") 
            self.var_sigma_montante.set("142.4") 
            self.entry_E_mont.config(state="readonly")
            self.entry_I_mont.config(state="normal") 
            self.entry_W_mont.config(state="normal") 
            self.entry_sigma_mont.config(state="normal") 
        else:
            self.var_E_montante.set("2.4")
            self.var_I_montante.set("25.0")
            self.var_W_montante.set("12.5") 
            self.var_sigma_montante.set("15.0") 
            self.entry_E_mont.config(state="normal")
            self.entry_I_mont.config(state="normal")
            self.entry_W_mont.config(state="normal")
            self.entry_sigma_mont.config(state="normal")

    def _build(self):
        frm_main = ttk.Frame(self)
        frm_main.pack(fill="both", expand=True, padx=10, pady=10)
        
        left_col = ttk.Frame(frm_main)
        left_col.pack(side="left", fill="y", expand=False, padx=(0, 15))
        
        ttk.Separator(frm_main, orient='vertical').pack(side="left", fill="y", padx=5)

        right_col = ttk.Frame(frm_main)
        right_col.pack(side="left", fill="both", expand=True, padx=(15, 0))

        pad = {"padx": 5, "pady": 4}

        # ==================== COLUMNA IZQUIERDA ====================
        ttk.Label(left_col, text=" PANEL TOPGAL ", foreground="blue", font=("Arial", 10, "bold"), relief="solid", borderwidth=1, padding=4).grid(row=0, column=0, sticky="w", **pad)
        
        gif_path = resource_path("DVP.gif")
        gif_label = AnimatedGifLabel(left_col, gif_path, delay=50)
        gif_label.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")
        
        ttk.Label(left_col, text="Espesor (mm):").grid(row=2, column=0, sticky="w", **pad)
        cb = ttk.Combobox(left_col, textvariable=self.var_espesor, values=["16", "20"], state="readonly", width=10)
        cb.grid(row=2, column=1, sticky="w", **pad)
        cb.bind("<<ComboboxSelected>>", self._on_espesor_change)

        ttk.Label(left_col, text="B tributario (mm):").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(left_col, textvariable=self.var_B, width=12).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(left_col, text="Tensión adm. placa (MPa):").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(left_col, textvariable=self.var_sigma_placa, width=12).grid(row=4, column=1, sticky="w", **pad)

        ttk.Label(left_col, text="E placa (GPa):").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(left_col, textvariable=self.var_E_placa, state="readonly", width=12).grid(row=5, column=1, sticky="w", **pad)

        ttk.Label(left_col, text="I placa (cm⁴):").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(left_col, textvariable=self.var_I_placa, state="readonly", width=12).grid(row=6, column=1, sticky="w", **pad)
        
        ttk.Label(left_col, text="Deformación Panel (L/δ):").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(left_col, textvariable=self.var_ratio_placa, width=12).grid(row=7, column=1, sticky="w", **pad)

        # ==================== COLUMNA DERECHA ====================
        
        # --- SECCIÓN: MODO DE CÁLCULO ---
        ttk.Label(right_col, text=" MODO DE CÁLCULO ", foreground="navy", font=("Arial", 10, "bold"), relief="solid", borderwidth=1, padding=4).grid(row=0, column=0, columnspan=2, sticky="w", **pad)
        
        frm_modo = ttk.Frame(right_col)
        frm_modo.grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 10))
        ttk.Radiobutton(frm_modo, text="Fachada (Vertical)", variable=self.var_modo, value="Fachada", command=self._on_modo_change).pack(side="left", padx=(5,15))
        ttk.Radiobutton(frm_modo, text="Techumbre (Cubierta)", variable=self.var_modo, value="Techumbre", command=self._on_modo_change).pack(side="left", padx=5)

        # --- SECCIÓN DINÁMICA: TECHUMBRE (Oculta por defecto) ---
        self.frm_techumbre = ttk.Frame(right_col)
        self.frm_techumbre.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        ttk.Label(self.frm_techumbre, text=" PARÁMETROS DE TECHUMBRE ", foreground="#d35400", font=("Arial", 9, "bold"), relief="groove").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        
        ttk.Label(self.frm_techumbre, text="Peso Policarbonato (kgf/m²):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(self.frm_techumbre, textvariable=self.var_peso_pc, width=10).grid(row=1, column=1, sticky="w", **pad)
        
        ttk.Label(self.frm_techumbre, text="Carga Nieve (kgf/m²):").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(self.frm_techumbre, textvariable=self.var_carga_nieve, width=10).grid(row=2, column=1, sticky="w", **pad)
        
        ttk.Label(self.frm_techumbre, text="Pendiente: Fija al 5%").grid(row=3, column=0, columnspan=2, sticky="w", **pad)
        
        # Espacio para la imagen de la pendiente
        self.lbl_img_pendiente = tk.Label(self.frm_techumbre, text="[esquema_pendiente.png]", fg="gray")
        self.lbl_img_pendiente.grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        img_pend_path = resource_path("esquema_pendiente.png")
        if os.path.exists(img_pend_path):
            img_original = tk.PhotoImage(file=img_pend_path)
            img_escalada = img_original.subsample(4, 4) 
            self.lbl_img_pendiente.config(image=img_escalada, text="")
            self.lbl_img_pendiente.image = img_escalada

        # Ocultar inicialmente porque arranca en Fachada
        self.frm_techumbre.grid_remove()

        ttk.Separator(right_col, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky="ew", pady=(5, 10))

        # --- SECCIÓN: MATERIAL MONTANTE ---
        ttk.Label(right_col, text=" MATERIAL DEL MONTANTE ", foreground="green", font=("Arial", 10, "bold"), relief="solid", borderwidth=1, padding=4).grid(row=4, column=0, columnspan=2, sticky="w", **pad)
        
        rb_frame = ttk.Frame(right_col)
        rb_frame.grid(row=5, column=0, columnspan=2, sticky="w", pady=(5, 5))
        
        ttk.Radiobutton(rb_frame, text="Aluminio AA6061-T6", variable=self.var_tipo_montante, value="Aluminio", command=self._on_montante_change).pack(side="left", padx=(5,15))
        ttk.Radiobutton(rb_frame, text="Policarbonato", variable=self.var_tipo_montante, value="Policarbonato", command=self._on_montante_change).pack(side="left", padx=5)

        ttk.Label(right_col, text="Tensión adm. (MPa):").grid(row=6, column=0, sticky="w", **pad)
        self.entry_sigma_mont = ttk.Entry(right_col, textvariable=self.var_sigma_montante, width=12, state="normal")
        self.entry_sigma_mont.grid(row=6, column=1, sticky="w", **pad)

        ttk.Label(right_col, text="E montante (GPa):").grid(row=7, column=0, sticky="w", **pad)
        self.entry_E_mont = ttk.Entry(right_col, textvariable=self.var_E_montante, width=12, state="readonly")
        self.entry_E_mont.grid(row=7, column=1, sticky="w", **pad)
        
        ttk.Label(right_col, text="I montante (cm⁴):").grid(row=8, column=0, sticky="w", **pad)
        self.entry_I_mont = ttk.Entry(right_col, textvariable=self.var_I_montante, width=12, state="normal")
        self.entry_I_mont.grid(row=8, column=1, sticky="w", **pad)

        ttk.Label(right_col, text="Módulo Sec. W (cm³):").grid(row=9, column=0, sticky="w", **pad)
        self.entry_W_mont = ttk.Entry(right_col, textvariable=self.var_W_montante, width=12, state="normal")
        self.entry_W_mont.grid(row=9, column=1, sticky="w", **pad)

        ttk.Label(right_col, text="Deformación (L/δ):").grid(row=10, column=0, sticky="w", **pad)
        ttk.Entry(right_col, textvariable=self.var_ratio_montante, width=12).grid(row=10, column=1, sticky="w", **pad)

        ttk.Separator(right_col, orient='horizontal').grid(row=11, column=0, columnspan=2, sticky="ew", pady=(10, 10))

        # --- SECCIÓN PRESIONES DE VIENTO ---
        ttk.Label(right_col, text=" PRESIONES DE DISEÑO ", foreground="purple", font=("Arial", 10, "bold"), relief="solid", borderwidth=1, padding=4).grid(row=12, column=0, columnspan=2, sticky="w", **pad)
        
        ttk.Label(right_col, text="Presiones ref. (kgf/m²):").grid(row=13, column=0, sticky="w", **pad)
        q_frame = ttk.Frame(right_col)
        q_frame.grid(row=13, column=1, sticky="w", **pad)
        ttk.Entry(q_frame, textvariable=self.var_q1, width=6).pack(side="left", padx=(0, 5))
        ttk.Entry(q_frame, textvariable=self.var_q2, width=6).pack(side="left", padx=(0, 5))
        ttk.Entry(q_frame, textvariable=self.var_q3, width=6).pack(side="left")

        # --- BOTONES DE ACCIÓN ---
        btn_frame = ttk.Frame(right_col)
        btn_frame.grid(row=14, column=0, columnspan=2, pady=15, sticky="w")
        
        btn_gen = tk.Button(btn_frame, text="Generar Reporte Excel", command=self._run, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), padx=10, pady=5)
        btn_gen.pack(side="left", padx=(0, 15))
        
        btn_salir = tk.Button(btn_frame, text="Salir", command=self.destroy, bg="#f44336", fg="white", font=("Arial", 10, "bold"), padx=10, pady=5)
        btn_salir.pack(side="left")
        
        self._on_espesor_change()
        self._on_montante_change()

    def _run(self):
        try:
            # 1. RECUPERAR VARIABLES GLOBALES
            modo = self.var_modo.get()
            
            E_placa = float(self.var_E_placa.get()) * 1e9
            B_m = float(self.var_B.get()) / 1000.0
            I_placa = float(self.var_I_placa.get()) * 1e-8
            sigma_adm_placa_Pa = float(self.var_sigma_placa.get()) * 1e6
            ratio_placa = float(self.var_ratio_placa.get())
            
            tipo_mont = self.var_tipo_montante.get()
            E_mont = float(self.var_E_montante.get()) * 1e9
            I_mont = float(self.var_I_montante.get()) * 1e-8
            W_montante = float(self.var_W_montante.get()) * 1e-6 
            sigma_adm_montante_Pa = float(self.var_sigma_montante.get()) * 1e6
            ratio_montante = float(self.var_ratio_montante.get())

            q_disenos = (
                float(self.var_q1.get() or 0), 
                float(self.var_q2.get() or 0), 
                float(self.var_q3.get() or 0)
            )

            # Extraer variables de techumbre
            if modo == "Techumbre":
                peso_pc_kgf = float(self.var_peso_pc.get())
                carga_nieve_kgf = float(self.var_carga_nieve.get())
                pendiente_rad = np.arctan(5.0 / 100.0) # Pendiente 5%
                print(f"Modo Techumbre Activo: Nieve={carga_nieve_kgf}, Peso={peso_pc_kgf}, Ángulo={np.degrees(pendiente_rad):.2f}°")

            out_xlsx = os.path.abspath(f"Tabla_{modo}_{tipo_mont}.xlsx")
            writer = pd.ExcelWriter(out_xlsx)

            graficos_generados = 0

            # BUCLE PRINCIPAL (Abarca desde 0 tomas hasta 3 tomas)
            for n in [1, 2, 3, 4]:
                tomas_str = f"{n-1} Tomas Int."
                
                # ==================== 1. CÁLCULO Y GRÁFICO DEL PANEL ====================
                c_placa = (float(self.var_espesor.get()) / 2.0) / 1000.0
                
                df_p = placas_q_vs_L(n, E_placa, B_m, I_placa, ratio_placa, sigma_adm_placa_Pa, c_placa)
                q_panel_array = df_p["q_admisible (kgf/m2)"].values
                L_mm_array = df_p["L_total (mm)"].values
                
                df_p_inv = invertir_curva_para_excel(L_mm_array, q_panel_array)
                df_p_inv.to_excel(writer, sheet_name=f"Panel_{n}t", index=False)
                
                guardar_grafico(
                    L_mm=L_mm_array,
                    q_vals=q_panel_array,
                    titulo=f"PANEL TOPGAL - {modo.upper()} - {tomas_str}",
                    filename=f"Grafico_{modo}_Panel_{n}t.png",
                    n_tomas=n,
                    q_disenos=q_disenos,
                    q_def=df_p["q_deflexion (kgf/m2)"].values,
                    q_ten=df_p["q_tension (kgf/m2)"].values
                )
                graficos_generados += 1

                # ==================== 2. CÁLCULO DEL MONTANTE Y SISTEMA ====================
                if n > 1: # Solo calcula montante si hay tomas intermedias
                    if tipo_mont == "Aluminio":
                        fs_material = 1.0  
                    else:
                        fs_material = 3.0  
                    
                    df_mont = conector_q_vs_L(n, E_mont, I_mont, W_montante, sigma_adm_montante_Pa, ratio_montante, fs_material, B_m)
                    q_montante_array = df_mont["q_admisible (kgf/m2)"].values
                    
                    df_mont_inv = invertir_curva_para_excel(L_mm_array, q_montante_array)
                    df_mont_inv.to_excel(writer, sheet_name=f"Montante_{n}t", index=False)
                    
                    guardar_grafico(
                        L_mm=L_mm_array,
                        q_vals=q_montante_array,
                        titulo=f"MONTANTE {tipo_mont.upper()} - AISLADO - {tomas_str}",
                        filename=f"Grafico_{modo}_Montante_{tipo_mont}_{n}t.png",
                        n_tomas=n,
                        q_disenos=q_disenos,
                        q_def=df_mont["q_deflexion (kgf/m2)"].values,
                        q_ten=df_mont["q_tension (kgf/m2)"].values
                    )
                    graficos_generados += 1

                    # CREACIÓN DEL SISTEMA (ENVOLVENTE)
                    q_sistema_array = np.minimum(q_panel_array, q_montante_array)

                    df_sistema_inv = invertir_curva_para_excel(L_mm_array, q_sistema_array)
                    df_sistema_inv.to_excel(writer, sheet_name=f"SISTEMA_{n}t", index=False)

                    guardar_grafico_sistema(
                        L_mm=L_mm_array,
                        q_sistema=q_sistema_array,
                        titulo=f"SISTEMA TOTAL ({tipo_mont}) - ENVOLVENTE - {tomas_str}",
                        filename=f"Grafico_{modo}_SISTEMA_{tipo_mont}_{n}t.png",
                        n_tomas=n,
                        q_disenos=q_disenos
                    )
                    graficos_generados += 1

            # FIN DEL BUCLE
            writer.close()
            messagebox.showinfo(
                "Operación Exitosa", 
                f"Archivos generados en la carpeta actual:\n\n"
                f"1. Excel: Tabla_{modo}_{tipo_mont}.xlsx (Hoja SISTEMA añadida)\n"
                f"2. Imágenes: {graficos_generados} gráficos PNG guardados y desplegados."
            )
            
            plt.show()

        except Exception as ex:
            error_details = traceback.format_exc()
            messagebox.showerror("Error de Ejecución", f"Ocurrió un error:\n{str(ex)}\n\nDetalles:\n{error_details}")

# ==========================================================
# PUNTO DE ENTRADA (ENTRY POINT)
# ==========================================================
if __name__ == "__main__":
    app = TopgalFachadasApp()
    app.mainloop()