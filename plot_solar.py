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


# --- REEMPLAZAR EL BLOQUE DE CARGA (Punto 1 en tu script) ---
lista_df = []

print(f"📂 Cargando {len(args.archivos)} archivos...")

for f in args.archivos:
    try:
        temp_df = pd.read_csv(f, header=None)
        lista_df.append(temp_df)
    except Exception as e:
        print(f"⚠️ Error al leer {f}: {e}")

# Concatenar todos los archivos en uno solo
df = pd.concat(lista_df, ignore_index=True)

# Crear columna de tiempo y ordenar
df['datetime'] = pd.to_datetime(df[0] + ' ' + df[1])
df = df.sort_values('datetime')

# --- EXTRACCIÓN DE RUIDO DE FONDO (CORREGIDO) ---
print("🧪 Calibrando ruido de fondo...")

# Seleccionamos solo las columnas que son puramente datos de intensidad
# Ignoramos la col 'datetime' y las primeras 6 columnas de metadatos del CSV
columnas_datos = df.iloc[:, 6:].select_dtypes(include=[np.number])

if args.cal_file:
    # OPCIÓN 2: Usar archivo externo
    df_noise = pd.read_csv(args.cal_file, header=None, low_memory=False)
    noise_matrix = df_noise.iloc[:, 6:].select_dtypes(include=[np.number]).values
else:
    # OPCIÓN 1: Usar rango de calma
    t_start_c = pd.to_datetime(args.cal_range[0]).time()
    t_end_c = pd.to_datetime(args.cal_range[1]).time()
    
    mask_cal = (df['datetime'].dt.time >= t_start_c) & (df['datetime'].dt.time <= t_end_c)
    noise_matrix = df[mask_cal].iloc[:, 6:].select_dtypes(include=[np.number]).values

# Si no hay datos en el rango de calma o el rango falló
if noise_matrix.size == 0:
    print("⚠️ Rango de calibración vacío o no numérico. Usando mediana global de columnas de datos.")
    perfil_ruido = np.nanmedian(columnas_datos.values, axis=0)
else:
    perfil_ruido = np.nanmedian(noise_matrix, axis=0)

# Aplicar la resta solo a las columnas numéricas
data_all_numeric = columnas_datos.values
data_calibrada = data_all_numeric - perfil_ruido

# Actualizar el DataFrame original con los datos limpios
# Usamos el índice de las columnas numéricas para asegurar precisión
df.iloc[:, 6:6+data_calibrada.shape[1]] = data_calibrada


try:
    # Filtrar por tiempo si se solicita
    if args.start:
        df = df[df['datetime'] >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df['datetime'] <= pd.to_datetime(args.end)]

    if df.empty:
        print("Error: El rango seleccionado no contiene datos.")
        sys.exit(1)

    df = df.sort_values('datetime')

    # Resumen en terminal
    duracion = df['datetime'].iloc[-1] - df['datetime'].iloc[0]
    print(f"--- Procesando: {duracion} de datos ---")

except Exception as e:
    print(f"Error al cargar archivo: {e}")
    sys.exit(1)

# --- 3. METADATOS DE FRECUENCIA ---
f_start_file = df.iloc[0, 2]/1e6
f_end_file = df.iloc[0, 3]/1e6
f_step = df.iloc[0, 4]/1e6

# Determinar qué columnas de frecuencia mostrar
view_min = args.fmin if args.fmin else f_start_file
view_max = args.fmax if args.fmax else f_end_file

col_idx_start = max(6, int((view_min - f_start_file) / f_step) + 6)
col_idx_end = min(df.shape[1], int((view_max - f_start_file) / f_step) + 6)

# Extraer matriz de datos y calcular potencia
data_limpia = df.iloc[:, col_idx_start:col_idx_end].values
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
dias_unicos = df['datetime'].dt.date.unique()

for dia in dias_unicos:
    # Crear objeto datetime para el mediodía de ese día
    medio_dia = pd.to_datetime(f"{dia} 12:00:00")

    # Solo dibujar si el mediodía está dentro del rango filtrado
    if df['datetime'].min() <= medio_dia <= df['datetime'].max():
        # Línea HORIZONTAL en Espectrograma (ax1)
        # Usamos date2num porque el eje Y es tiempo
        ax1.axvline(x=mdates.date2num(medio_dia), color='white', 
                linestyle='--', alpha=0.4, linewidth=1)

        ax1.text(view_min + 0.002, mdates.date2num(medio_dia), 'Mediodía Solar', 
                 color='white', fontsize=7, va='bottom', alpha=0.6)

        # Línea VERTICAL en Potencia (ax2)
        ax2.axvline(x=medio_dia, color='red', linestyle=':', alpha=0.5)

# IMPORTANTE: Extent [X_min, X_max, Y_min, Y_max]
# X = Frecuencia, Y = Tiempo
x_start = mdates.date2num(df['datetime'].iloc[0])
x_end = mdates.date2num(df['datetime'].iloc[-1])
extent = [x_start, x_end,  view_min, view_max]

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

# --- DIBUJO DE BANDAS SIGMA EN AX2 ---
# Banda 1 Sigma (Verde tenue - Ruido normal)
ax2.fill_between(df['datetime'], s1_down, s1_up, color='green', alpha=0.1, label='±3σ')
ax2.fill_between(df['datetime'], s1_up, s2_up, color='yellow', alpha=0.1, label='Actividad (+2σ)')
ax2.fill_between(df['datetime'], s2_up, s3_up, color='red', alpha=0.1, label='Evento (+3σ)')

ax2.fill_between(df['datetime'], -s1_up, -s2_up, color='yellow', alpha=0.1, label='Actividad (+2σ)')
ax2.fill_between(df['datetime'], -s2_up, -s3_up, color='red', alpha=0.1, label='Evento (+3σ)')

# Línea base de la Mediana
#ax2.axhline(y=mediana_p, color='blue', linestyle='-', alpha=0.2, label='Mediana')
ax2.axhline(y=0, color='blue', linestyle='-', alpha=0.3, label='Nivel Base (0 dB)') #

ax2.plot(df['datetime'], potencia_final, color='orange', label='Señal Normalizada') #

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
