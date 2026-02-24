# -*- coding: utf-8 -*-
"""
Topgal / SL - Generador Web (Streamlit) - Versión Definitiva 11.4 (Layout Horizontal)
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import traceback
import os

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA WEB
# ==========================================
st.set_page_config(page_title="Topgal Design Engine", layout="wide")

# ==========================================
# CONSTANTES Y MOTOR MATEMÁTICO
# ==========================================
g = 9.80665
L_MIN_MM, L_MAX_MM, PASO_MM = 500, 4000, 10
PRESIONES_DVP_KGF = np.arange(50.0, 601.0, 50.0)
PLACA_INCLUDE_END_SUPPORTS = True
B_POR_ESPESOR_MM = {16: 998.0, 20: 998.0}
I_POR_ESPESOR_CM4 = {16: 12.71, 20: 19.91}

# Diccionario para mapear el número de apoyos (n) a la imagen del esquema
IMAGENES_TOMAS = {
    1: "esquema_1.png", # 0 tomas (2 apoyos)
    2: "esquema_2.png", # 1 toma (3 apoyos)
    3: "esquema_3.png", # 2 tomas (4 apoyos)
    4: "esquema_4.png", # 3 tomas (5 apoyos)
}

def _beam_element_stiffness(EI, L):
    k = EI / (L ** 3)
    return k * np.array([[12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2], [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]], dtype=float)

def _beam_element_consistent_load(w, L):
    return np.array([w*L/2.0, w*(L**2)/12.0, w*L/2.0, -w*(L**2)/12.0], dtype=float)

def _assemble_beam(nodes_x, EI, w_line):
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

def _apply_constraints(K, F, constrained_dofs):
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

def _shape_functions_v(s, L):
    return np.array([1 - 3*s**2 + 2*s**3, L * (s - 2*s**2 + s**3), 3*s**2 - 2*s**3, L * (-s**2 + s**3)], dtype=float)

def _support_node_indices(n_nodes, include_end_supports):
    idx = [0, n_nodes-1] if include_end_supports else []
    idx += list(range(1, n_nodes-1))
    return sorted(set(idx))

def beam_response_max(nodes_x, E, I, B, p_area, support_node_ids):
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

def invertir_curva_para_excel(L_mm, q_vals):
    mask = np.isfinite(q_vals)
    L_c, q_c = L_mm[mask], q_vals[mask]
    if len(q_c) == 0: return pd.DataFrame({"Presión Viento (kgf/m²)": PRESIONES_DVP_KGF, "Longitud Máx Admisible L (mm)": np.nan})
    sort_idx = np.argsort(q_c)
    L_targets = np.interp(PRESIONES_DVP_KGF, q_c[sort_idx], L_c[sort_idx], left=np.nan, right=np.nan)
    return pd.DataFrame({"Presión Viento (kgf/m²)": PRESIONES_DVP_KGF, "Longitud Máx Admisible L (mm)": np.round(L_targets, 1)})

def placas_q_vs_L(n_tomas, E_Pa, B_m, I_m4, ratio_L_over_delta, sigma_adm_Pa, c_m):
    L_mm = np.arange(L_MIN_MM, L_MAX_MM + PASO_MM, PASO_MM)
    q_def_list, q_ten_list, q_min_list = [], [], []
    for Ltot_mm in L_mm:
        Ltot = Ltot_mm / 1000.0
        delta_adm = (Ltot / (n_tomas + 1)) / ratio_L_over_delta
        v1, M1 = beam_response_max(np.linspace(0.0, Ltot, n_tomas + 2), E_Pa, I_m4, B_m, 1.0, _support_node_indices(n_tomas+2, True))
        
        q_def = ((delta_adm / v1) / g) if v1 > 0 else np.nan
        q_ten = (((sigma_adm_Pa * I_m4 / c_m) / M1) / g) if M1 > 0 else np.nan
        q_min = np.nanmin([q_def, q_ten])
        q_def_list.append(q_def); q_ten_list.append(q_ten); q_min_list.append(q_min)
        
    return pd.DataFrame({"L_total (mm)": L_mm, "q_deflexion (kgf/m2)": np.array(q_def_list), "q_tension (kgf/m2)": np.array(q_ten_list), "q_admisible (kgf/m2)": np.array(q_min_list)})

def conector_q_vs_L(n_tomas, E, Iyy, W_m3, sigma_adm, ratio_montante, fs_aplicado, B_m):
    L_mm = np.arange(L_MIN_MM, L_MAX_MM + PASO_MM, PASO_MM)
    q_def_list, q_ten_list, q_min_list = [], [], []
    for Ltot_mm in L_mm:
        Ltot = Ltot_mm / 1000.0
        delta_adm = min(Ltot / ratio_montante, 50.0 / 1000.0) 
        v1, M1 = beam_response_max(np.linspace(0.0, Ltot, n_tomas + 2), E, Iyy, B_m, 1.0, [0, n_tomas+1])

        q_def = ((delta_adm / v1) / g) / fs_aplicado if v1 > 0 else np.nan
        q_ten = (((sigma_adm * W_m3) / M1) / g) / fs_aplicado if M1 > 0 else np.nan
        q_min = np.nanmin([q_def, q_ten])
        q_def_list.append(q_def); q_ten_list.append(q_ten); q_min_list.append(q_min)

    return pd.DataFrame({"L_total (mm)": L_mm, "q_deflexion (kgf/m2)": np.array(q_def_list), "q_tension (kgf/m2)": np.array(q_ten_list), "q_admisible (kgf/m2)": np.array(q_min_list)})

# ==========================================
# MOTOR DE GRAFICACIÓN ADAPTADO PARA WEB
# ==========================================
def render_plot(L_mm, q_vals, titulo, q_disenos, color_main='blue', q_def=None, q_ten=None):
    """
    Renderiza un gráfico individual con ajuste dinámico de ejes.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Línea principal
    label_main = "q(L) Admisible" if color_main == 'blue' else "q(L) SISTEMA (Envolvente)"
    ax.plot(L_mm, q_vals, linewidth=2.5 if color_main == 'blue' else 3.0, color=color_main, label=label_main)
    
    # Líneas de límite de falla (si existen)
    if q_def is not None and q_ten is not None:
        ax.plot(L_mm, q_def, linestyle=":", color='orange', linewidth=1.5, label="Límite por Flecha")
        ax.plot(L_mm, q_ten, linestyle="--", color='red', linewidth=1.5, label="Límite por Tensión")

    # Ajuste del Eje X e Y
    Q_MAX_GRAFICO = 200.0
    mask_vis = np.isfinite(q_vals) & (q_vals <= Q_MAX_GRAFICO)
    if np.any(mask_vis):
        L_vis = L_mm[mask_vis]
        xmin = max(float(L_vis.min()) - 150, float(L_mm.min()))
        xmax = min(float(L_vis.max()) + 150, float(L_mm.max()))
        if xmax - xmin < 300: xmax = min(xmin + 300, float(L_mm.max()))
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(L_mm.min(), L_mm.max())

    ymax = 600.0
    mask_y = (L_mm >= ax.get_xlim()[0]) & (L_mm <= ax.get_xlim()[1]) & np.isfinite(q_vals)
    if np.any(mask_y):
        ymax = min(float(np.nanmax(q_vals[mask_y])) * 1.15, 600.0)
    ax.set_ylim(0.0, max(ymax, 100.0))

    # Anotaciones de presiones de diseño
    colores = ["red", "orange", "magenta", "purple"]
    y_max_ax = ax.get_ylim()[1]
    for i, q_des in enumerate(q_disenos):
        if q_des > 0:
            ax.axhline(q_des, linestyle="--", color=colores[i], label=f"q viento = {q_des} kgf/m²")
            qmin, qmax = np.nanmin(q_vals), np.nanmax(q_vals)
            if np.isfinite(qmin) and np.isfinite(qmax) and (qmin <= q_des <= qmax):
                idx = int(np.nanargmin(np.abs(q_vals - q_des)))
                Li = float(L_mm[idx])
                ax.axvline(Li, linestyle="--", color=colores[i])
                ax.text(Li + 30, y_max_ax * (0.90 - 0.07 * i), f"L ≈ {Li:.0f} mm", color=colores[i], fontweight="bold")

    ax.set_title(titulo, fontweight="bold", fontsize=10)
    ax.set_xlabel("Altura L [mm]", fontweight="bold")
    ax.set_ylabel("Presión de viento q(L) [kgf/m²]", fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.7)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig

# ==========================================
# FRONTEND - INTERFAZ DE USUARIO (WEB)
# ==========================================
st.title("🏗️ Engine Estructural Topgal - Proyectos Estructurales EIRL")

# PANEL LATERAL DE CONTROLES (SIDEBAR)
st.sidebar.header("⚙️ Parámetros de Diseño")

modo = st.sidebar.radio("Modo de Análisis:", ["Fachada (Vertical)", "Techumbre (Cubierta)"])

st.sidebar.subheader("Panel Topgal")
espesor = st.sidebar.selectbox("Espesor Panel (mm):", [16, 20])
B_panel = st.sidebar.number_input("B tributario (mm):", value=B_POR_ESPESOR_MM[espesor])
sigma_panel = st.sidebar.number_input("Tensión adm. placa (MPa):", value=15.0)
E_panel = st.sidebar.number_input("E placa (GPa):", value=2.4)
I_panel = st.sidebar.number_input("I placa (cm⁴):", value=I_POR_ESPESOR_CM4[espesor])
ratio_panel = st.sidebar.number_input("Deformación Panel (L/δ):", value=50.0)

if modo == "Techumbre (Cubierta)":
    st.sidebar.subheader("Cargas Adicionales")
    peso_pc = st.sidebar.number_input("Peso PC (kgf/m²):", value=1.0 if espesor==16 else 1.25)
    carga_nieve = st.sidebar.number_input("Carga Nieve (kgf/m²):", value=30.0)
    st.sidebar.info("Pendiente fijada al 5%")

st.sidebar.subheader("Sistema de Montante")
mat_montante = st.sidebar.selectbox("Material:", ["Aluminio AA6061-T6", "Policarbonato"])

if mat_montante == "Aluminio AA6061-T6":
    sigma_mont = st.sidebar.number_input("Tensión adm. montante (MPa):", value=142.4)
    E_mont = st.sidebar.number_input("E montante (GPa):", value=70.0)
    I_mont = st.sidebar.number_input("I montante (cm⁴):", value=2.75)
    W_mont = st.sidebar.number_input("Módulo Sec. W (cm³):", value=1.77)
    fs_mat = 1.0
else:
    sigma_mont = st.sidebar.number_input("Tensión adm. montante (MPa):", value=15.0)
    E_mont = st.sidebar.number_input("E montante (GPa):", value=2.4)
    I_mont = st.sidebar.number_input("I montante (cm⁴):", value=25.0)
    W_mont = st.sidebar.number_input("Módulo Sec. W (cm³):", value=12.5)
    fs_mat = 3.0

ratio_mont = st.sidebar.number_input("Deformación Montante (L/δ):", value=50.0)

st.sidebar.subheader("Presiones de Viento (kgf/m²)")
col_q1, col_q2, col_q3 = st.sidebar.columns(3)
q1 = col_q1.number_input("Q1", value=50)
q2 = col_q2.number_input("Q2", value=100)
q3 = col_q3.number_input("Q3", value=150)

# ==========================================
# EJECUCIÓN CENTRAL
# ==========================================
if st.button("🚀 Ejecutar Cálculo y Generar Gráficos", type="primary"):
    
    with st.spinner("Procesando Elementos Finitos..."):
        try:
            q_disenos = [q1, q2, q3]
            
            # Conversiones de Unidades
            E_placa_Pa, B_m, I_placa_m4 = E_panel * 1e9, B_panel / 1000.0, I_panel * 1e-8
            sigma_placa_Pa, c_placa_m = sigma_panel * 1e6, (espesor / 2.0) / 1000.0
            E_mont_Pa, I_mont_m4, W_mont_m3 = E_mont * 1e9, I_mont * 1e-8, W_mont * 1e-6
            sigma_mont_Pa = sigma_mont * 1e6

            # Buffer para exportar Excel sin guardar en disco local
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                
                for n in [1, 2, 3, 4]:
                    tomas_str = f"{n-1} Tomas Int."
                    st.subheader(f"Análisis con {tomas_str}")

                    # 1. CÁLCULO PANEL
                    df_p = placas_q_vs_L(n, E_placa_Pa, B_m, I_placa_m4, ratio_panel, sigma_placa_Pa, c_placa_m)
                    q_pan_arr, L_arr = df_p["q_admisible (kgf/m2)"].values, df_p["L_total (mm)"].values
                    invertir_curva_para_excel(L_arr, q_pan_arr).to_excel(writer, sheet_name=f"Panel_{n}t", index=False)
                    fig_pan = render_plot(L_arr, q_pan_arr, f"PANEL TOPGAL - {tomas_str}", q_disenos, 'blue', df_p["q_deflexion (kgf/m2)"].values, df_p["q_tension (kgf/m2)"].values)
                    
                    # 2. MONTANTE & SISTEMA (Si aplica)
                    fig_mon = None
                    fig_sis = None
                    if n > 1:
                        df_m = conector_q_vs_L(n, E_mont_Pa, I_mont_m4, W_mont_m3, sigma_mont_Pa, ratio_mont, fs_mat, B_m)
                        q_mon_arr = df_m["q_admisible (kgf/m2)"].values
                        invertir_curva_para_excel(L_arr, q_mon_arr).to_excel(writer, sheet_name=f"Mont_{n}t", index=False)
                        fig_mon = render_plot(L_arr, q_mon_arr, f"MONTANTE AISLADO - {tomas_str}", q_disenos, 'blue', df_m["q_deflexion (kgf/m2)"].values, df_m["q_tension (kgf/m2)"].values)

                        q_sis_arr = np.minimum(q_pan_arr, q_mon_arr)
                        invertir_curva_para_excel(L_arr, q_sis_arr).to_excel(writer, sheet_name=f"Sis_{n}t", index=False)
                        fig_sis = render_plot(L_arr, q_sis_arr, f"SISTEMA TOTAL ({mat_montante})", q_disenos, 'black')

                    # --- SECCIÓN DE VISUALIZACIÓN (LAYOUT HORIZONTAL) ---
                    # Creamos columnas con proporciones: 1 parte para esquema, 2 partes para cada gráfico
                    if n > 1:
                        c_esq, c_pan, c_mon, c_sis = st.columns([1, 2, 2, 2])
                    else:
                        # Para el caso de 0 tomas, solo mostramos esquema y panel
                        c_esq, c_pan = st.columns([1, 3])

                    # 1. Columna Esquema (Izquierda)
                    img_filename = IMAGENES_TOMAS.get(n)
                    if img_filename and os.path.exists(img_filename):
                        # use_column_width=True hace que la imagen se adapte al ancho de su columna angosta
                        c_esq.image(img_filename, caption="Esquema Estático", use_column_width=True)
                    else:
                        c_esq.warning("Sin imagen")

                    # 2. Columnas Gráficos (Derecha)
                    c_pan.pyplot(fig_pan)
                    if n > 1:
                        c_mon.pyplot(fig_mon)
                        c_sis.pyplot(fig_sis)
                    # ----------------------------------------------------
                    
                    st.divider() # Línea divisoria entre análisis
                    
            # Descarga de Excel
            excel_data = excel_buffer.getvalue()
            st.success("¡Cálculo Finalizado Exitosamente!")
            st.download_button(
                label="📥 Descargar Reporte Completo en Excel",
                data=excel_data,
                file_name=f"Reporte_Estructural_{modo}.xlsx",
                mime="application/vnd.ms-excel"
            )

        except Exception as e:
            st.error(f"Se produjo un error en el motor de cálculo: {e}")