#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import sys
import os
from matplotlib.colors import LogNorm, Normalize


# --- 1. CONFIGURACIÓN DE ARGUMENTOS ---
parser = argparse.ArgumentParser(description='Análisis Solar Morelia v3.0')
parser.add_argument('archivos', nargs='+', help='Uno o más archivos CSV de entrada')
parser.add_argument('--fmin', type=float, help='Frecuencia mínima (MHz)')
parser.add_argument('--fmax', type=float, help='Frecuencia máxima (MHz)')
parser.add_argument('--start', help='Inicio (AAAA-MM-DD HH:MM)', default=None)
parser.add_argument('--end', help='Fin (AAAA-MM-DD HH:MM)', default=None)
parser.add_argument('--output', '-o', help='Nombre del archivo de salida')

parser.add_argument('--cal_file', help='Archivo CSV externo solo para ruido (Opcional)')
parser.add_argument('--cal_range', nargs=2, help='Rango HH:MM HH:MM para extraer ruido del set actual', default=["03:00", "04:00"])

args = parser.parse_args()

# --- 2. CARGA Y UNIFICACIÓN DE SALTOS (ESTRUCTURA LIMPIA) ---
lista_df = []
print(f"📂 Cargando {len(args.archivos)} archivos...")
for f in args.archivos:
    try:
        temp_df = pd.read_csv(f, header=None, low_memory=False)
        
        # 1. Creamos la columna datetime primero
        temp_df['datetime'] = pd.to_datetime(temp_df[0] + ' ' + temp_df[1]) #
        
        # 2. Convertimos el bloque de datos a numérico de golpe (sin ciclos for col in...)
        # Esto evita el PerformanceWarning y es mucho más rápido
        temp_df.iloc[:, 6:-1] = temp_df.iloc[:, 6:-1].apply(pd.to_numeric, errors='coerce') #
            
        lista_df.append(temp_df)
    except Exception as e:
        print(f"⚠️ Error al leer {f}: {e}")

if not lista_df:
    print("❌ No se pudieron cargar los archivos."); sys.exit(1)

df_raw = pd.concat(lista_df, ignore_index=True)

# --- 2. ALINEACIÓN ULTRA-RÁPIDA (SIN GROUPBY) ---
print("🚀 Alineando saltos de frecuencia con Vectorización...")

# 1. Obtenemos los tiempos únicos y cuántos saltos hay por ciclo (hops)
tiempos_finales = df_raw['datetime'].unique()
tiempos_finales = pd.Series(tiempos_finales).sort_values()
num_hops = df_raw[2].nunique() # Detecta si son 3 o más saltos

# 2. Extraemos solo la matriz de datos numéricos (columnas 6 en adelante)
# Aseguramos que el orden sea cronológico y por frecuencia antes de convertir
df_sorted = df_raw.sort_values(by=['datetime', 2])
data_raw_matrix = df_sorted.iloc[:, 6:-1].values.astype(float)

# 3. El truco de magia: Re-formatear la matriz (Reshape)
# Si tienes 3 saltos, convertimos (N*3, Bins) en (N, 3*Bins)
try:
    total_filas = len(tiempos_finales)
    bins_por_fila = data_raw_matrix.shape[1]
    data_all_numeric = data_raw_matrix.reshape(total_filas, num_hops * bins_por_fila)
except ValueError as e:
    print(f"⚠️ Error en dimensiones: {e}. Reintentando con método seguro...")
    # Si falta algún salto en el archivo, el reshape fallará. 
    # En ese caso, usamos un pivote rápido:
    df_pivot = df_raw.pivot(index='datetime', columns=2, values=list(range(6, df_raw.shape[1]-1)))
    data_all_numeric = df_pivot.fillna(method='ffill').values.astype(float)
    tiempos_finales = df_pivot.index


# --- 3. CALIBRACIÓN DE RUIDO ---
print("🧪 Calibrando ruido de fondo...")
if args.cal_file:
    # (Lógica para archivo externo si lo usas...)
    pass 
else:
    # Extraer ruido del rango de calma (ej. 3:00 AM)
    t_start_c = pd.to_datetime(args.cal_range[0]).time()
    t_end_c = pd.to_datetime(args.cal_range[1]).time()
    mask_cal = (tiempos_finales.dt.time >= t_start_c) & (tiempos_finales.dt.time <= t_end_c)
    noise_matrix = data_all_numeric[mask_cal]

# Perfil de ruido (Mediana por cada columna de frecuencia)
perfil_ruido = np.nanmedian(noise_matrix, axis=0) if noise_matrix.size > 0 else np.nanmedian(data_all_numeric, axis=0)
data_calibrada = data_all_numeric - perfil_ruido


# --- 4. RECORTES Y METADATOS ---
f_min_total = df_raw[2].min() / 1e6
f_step = df_raw.iloc[0, 4] / 1e6
f_max_total = df_raw[3].max() / 1e6

view_min = args.fmin if args.fmin is not None else f_min_total
view_max = args.fmax if args.fmax is not None else f_max_total

col_idx_start = max(0, int((view_min - f_min_total) / f_step))
col_idx_end = min(data_calibrada.shape[1], int((view_max - f_min_total) / f_step))

data_limpia = data_calibrada[:, col_idx_start:col_idx_end]
data_plot = data_limpia.T

potencia_media = np.mean(data_limpia, axis=1) #
potencia_suavizada = pd.Series(potencia_media).rolling(window=15, center=True).mean() #
nivel_base = np.nanmedian(potencia_suavizada) #
potencia_final = potencia_suavizada - nivel_base #

mediana_p = np.nanmedian(potencia_suavizada)

std_p = np.nanstd(potencia_final) #
s1_up = 1 * std_p #
s2_up = 2 * std_p #
s3_up = 3 * std_p #

s1_down = -1 * std_p

# --- 4. GRAFICACIÓN ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [3, 1]})

# --- MARCADORES DE MEDIODÍA ---
dias_unicos = tiempos_finales.dt.date.unique() # Cambiado df por tiempos_finales

for dia in dias_unicos:
    medio_dia = pd.to_datetime(f"{dia} 12:00:00")

    # Verificar si el mediodía está dentro del rango de los datos actuales
    if tiempos_finales.min() <= medio_dia <= tiempos_finales.max():
        # Línea vertical en Espectrograma (ax1)
        ax1.axvline(x=mdates.date2num(medio_dia), color='white', 
                linestyle='--', alpha=0.4, linewidth=1)

        # Línea vertical en Potencia (ax2)
        ax2.axvline(x=medio_dia, color='red', linestyle=':', alpha=0.5)


# X = Frecuencia, Y = Tiempo
x_start = mdates.date2num(tiempos_finales.iloc[0])
x_end = mdates.date2num(tiempos_finales.iloc[-1])
extent = [x_start, x_end, view_min, view_max]

# Espectrograma
v_min, v_max =  0, 15
im = ax1.imshow(data_plot, aspect='auto', extent=extent, cmap='magma', vmin=v_min, vmax=v_max)


# --- REINCORPORAR COLORBAR ---
cbar = fig.colorbar(im, ax=ax1, pad=0.01, aspect=20)
cbar.set_label('Intensidad sobre el fondo (dB)') #

# Configurar ejes de Tiempo (Y en espectrograma, X en potencia)
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)

ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
ax1.set_xlabel("Tiempo (Local)")

ax1.set_title(f"Análisis Radio-Solar: {view_min:.2f}-{view_max:.2f} MHz")

ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
ax1.set_ylabel("Frecuencia [MHz]")

# Curva de Potencia

# --- DIBUJO DE BANDAS SIGMA EN AX2 (CORREGIDO) ---
ax2.fill_between(tiempos_finales, s1_down, s1_up, color='green', alpha=0.1, label='±1σ')
ax2.fill_between(tiempos_finales, s1_up, s2_up, color='yellow', alpha=0.1, label='+2σ')
ax2.fill_between(tiempos_finales, s2_up, s3_up, color='red', alpha=0.1, label='+3σ')

ax2.axhline(y=0, color='blue', linestyle='-', alpha=0.3, label='Nivel Base (0 dB)')

# Línea principal de potencia
ax2.plot(tiempos_finales, potencia_final, color='orange', label='Señal Normalizada')

# Línea base de la Mediana
#ax2.axhline(y=mediana_p, color='blue', linestyle='-', alpha=0.2, label='Mediana')
ax2.axhline(y=0, color='blue', linestyle='-', alpha=0.3, label='Nivel Base (0 dB)') #


ax2.plot(tiempos_finales, potencia_final, color='orange', label='Señal Normalizada')
ax2.fill_between(tiempos_finales, s1_down, s1_up, color='green', alpha=0.1)


ax2.set_xlabel("Tiempo (Local)")
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)
ax2.set_ylabel("Flujo Relativo (dB)") #
ax2.grid(True, alpha=0.3)

for ax in [ax1, ax2]:
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.tick_params(labelsize=10)

plt.subplots_adjust(left=0.1, right=0.85, top=0.92, bottom=0.1, hspace=0.3)

if not args.output:
    # Genera un nombre basado en las fechas reales de los datos procesados
    fecha_inicio = df['datetime'].iloc[0].strftime('%Y%m%d_%H%M')
    fecha_fin = df['datetime'].iloc[-1].strftime('%Y%m%d_%H%M')
    output_file = f"solar_{fecha_inicio}_to_{fecha_fin}.png"
else:
    output_file = args.output if args.output else os.path.splitext(args.archivos[0])[0] + ".png"

plt.rcParams.update({'font.size': 12})

plt.savefig(output_file, dpi=300)
print(f"Éxito: Imagen guardada como {output_file}")
