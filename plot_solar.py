#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import sys
import os

# --- 1. CONFIGURACIÓN DE ARGUMENTOS ---
parser = argparse.ArgumentParser(description='Análisis Solar Morelia v3.0')
parser.add_argument('archivos', nargs='+', help='Uno o más archivos CSV de entrada')
parser.add_argument('--fmin', type=float, help='Frecuencia mínima (MHz)')
parser.add_argument('--fmax', type=float, help='Frecuencia máxima (MHz)')
parser.add_argument('--start', help='Inicio (AAAA-MM-DD HH:MM)', default=None)
parser.add_argument('--end', help='Fin (AAAA-MM-DD HH:MM)', default=None)
parser.add_argument('--output', '-o', help='Nombre del archivo de salida')
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
data = df.iloc[:, col_idx_start:col_idx_end].values
# Resta de ruido simple (mediana de la fila) para resaltar eventos
data_limpia = data - np.median(data, axis=0)

potencia_suavizada = pd.Series(np.mean(data_limpia, axis=1)).rolling(window=20, center=True).mean()

mediana_p = np.nanmedian(potencia_suavizada)
std_p = np.nanstd(potencia_suavizada)

#Banda 1 Sigma (Simétrica para el ruido de fondo)
s3_up = mediana_p + 3*std_p
s3_down = mediana_p - 3*std_p


# --- 4. GRAFICACIÓN ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

# --- MARCADORES DE MEDIODÍA ---
dias_unicos = df['datetime'].dt.date.unique()

for dia in dias_unicos:
    # Crear objeto datetime para el mediodía de ese día
    medio_dia = pd.to_datetime(f"{dia} 12:00:00")
    
    # Solo dibujar si el mediodía está dentro del rango filtrado
    if df['datetime'].min() <= medio_dia <= df['datetime'].max():
        # Línea HORIZONTAL en Espectrograma (ax1)
        # Usamos date2num porque el eje Y es tiempo
        ax1.axhline(y=mdates.date2num(medio_dia), color='white', 
                    linestyle='--', alpha=0.4, linewidth=1)
        ax1.text(view_min + 0.002, mdates.date2num(medio_dia), 'Mediodía Solar', 
                 color='white', fontsize=7, va='bottom', alpha=0.6)

        # Línea VERTICAL en Potencia (ax2)
        ax2.axvline(x=medio_dia, color='red', linestyle=':', alpha=0.5)


# IMPORTANTE: Extent [X_min, X_max, Y_min, Y_max]
# X = Frecuencia, Y = Tiempo
y_start = mdates.date2num(df['datetime'].iloc[0])
y_end = mdates.date2num(df['datetime'].iloc[-1])
extent = [view_min, view_max, y_end, y_start]

# Espectrograma
v_min, v_max = np.percentile(data_limpia, [5, 98])
im = ax1.imshow(data_limpia, aspect='auto', extent=extent, cmap='inferno', vmin=v_min, vmax=v_max)

# Configurar ejes de Tiempo (Y en espectrograma, X en potencia)
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)

ax1.yaxis.set_major_locator(locator)
ax1.yaxis.set_major_formatter(formatter)
ax1.set_ylabel("Tiempo (Local)")
ax1.set_title(f"Análisis Radio-Solar: {view_min:.2f}-{view_max:.2f} MHz")

# Configurar ejes de Frecuencia (X en espectrograma)
ax1.set_xlabel("Frecuencia [MHz]")
ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

# Curva de Potencia

# --- DIBUJO DE BANDAS SIGMA EN AX2 ---
# Banda 1 Sigma (Verde tenue - Ruido normal)
ax2.fill_between(df['datetime'], s3_down, s3_up, color='green', alpha=0.2, label='±3σ')


# Línea base de la Mediana
ax2.axhline(y=mediana_p, color='blue', linestyle='-', alpha=0.2, label='Mediana')

ax2.plot(df['datetime'], potencia_suavizada, color='orange')
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)
ax2.set_ylabel("Potencia (dB)")
ax2.grid(True, alpha=0.3)

# --- 5. GUARDADO ---
plt.tight_layout()

if not args.output:
    # Genera un nombre basado en las fechas reales de los datos procesados
    fecha_inicio = df['datetime'].iloc[0].strftime('%Y%m%d_%H%M')
    fecha_fin = df['datetime'].iloc[-1].strftime('%Y%m%d_%H%M')
    output_file = f"solar_{fecha_inicio}_to_{fecha_fin}.png"
else:
    output_file = args.output if args.output else os.path.splitext(args.archivo)[0] + ".png"

plt.savefig(output_file, dpi=300)
print(f"Éxito: Imagen guardada como {output_file}")
