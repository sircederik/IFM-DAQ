#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import sys

# --- ARGUMENTOS ---
parser = argparse.ArgumentParser(description='Análisis Solar con Espectrograma y Curva de Potencia')
parser.add_argument('archivo', help='Archivo CSV de rtl_power')
parser.add_argument('--fmin', type=float, help='Frecuencia mínima (MHz)')
parser.add_argument('--fmax', type=float, help='Frecuencia máxima (MHz)')
parser.add_argument('--t_cal_start', help='Inicio calibración (HH:MM:SS)', default="03:00:00")
parser.add_argument('--t_cal_end', help='Fin calibración (HH:MM:SS)', default="04:00:00")
args = parser.parse_args()

# 1. Cargar y ordenar datos
try:
    df = pd.read_csv(args.archivo, header=None)
    df['datetime'] = pd.to_datetime(df[0] + ' ' + df[1])
    df = df.sort_values('datetime')
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# 2. Metadatos y Frecuencia
f_start_file, f_end_file, f_step = df.iloc[0, 2]/1e6, df.iloc[0, 3]/1e6, df.iloc[0, 4]/1e6
num_total_cols = df.shape[1]

view_min = args.fmin if args.fmin else f_start_file
view_max = args.fmax if args.fmax else f_end_file

col_idx_start = max(6, int((view_min - f_start_file) / f_step) + 6)
col_idx_end = min(num_total_cols, int((view_max - f_start_file) / f_step) + 6)

# 3. Procesamiento y Calibración
data_full = df.iloc[:, col_idx_start:col_idx_end].values
mask_cal = (df['datetime'].dt.time >= pd.to_datetime(args.t_cal_start).time()) & \
           (df['datetime'].dt.time <= pd.to_datetime(args.t_cal_end).time())

data_cal = df[mask_cal].iloc[:, col_idx_start:col_idx_end].values
perfil_ruido = np.mean(data_cal, axis=0) if data_cal.size > 0 else np.mean(data_full, axis=0)
data_limpia = data_full - perfil_ruido

# Cálculo de Potencia Promedio (La curva de abajo)
potencia_total = np.mean(data_limpia, axis=1)
# Aplicamos media móvil para limpiar el ruido electrónico (jitter)
# Una ventana de 5 a 10 muestras suele ser ideal para 20 MHz
ventana = 20
potencia_suavizada = pd.Series(potencia_total).rolling(window=ventana, center=True).mean()

# Estadísticas para el reescalado dinámico (Zoom automático)
mediana_p = np.nanmedian(potencia_suavizada)
sigma_p = np.nanstd(potencia_suavizada)

# 4. Visualización con 2 Subplots (80% Espectrograma, 20% Curva)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False, 
                               gridspec_kw={'height_ratios': [4, 1]})

t_start, t_end = mdates.date2num(df['datetime'].min()), mdates.date2num(df['datetime'].max())
v_min, v_max = np.percentile(data_limpia, [15, 99.5])

# --- SUBPLOT 1: ESPECTROGRAMA ---
im = ax1.imshow(data_limpia, aspect='auto', 
                extent=[view_min, view_max, t_end, t_start], 
                cmap='magma', vmin=v_min, vmax=v_max)

ax1.set_ylabel('Tiempo (Local)')
ax1.set_xlabel('Frecuencia [MHz]')
ax1.set_title(f'Análisis Radio-Solar: {view_min}-{view_max} MHz')
ax1.yaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d\n%H:%M'))
ax1.yaxis.set_major_locator(mdates.HourLocator(interval=2)) # Salto cada 2 horas
fig.colorbar(im, ax=ax1, label='Relativa (dB)')

# --- SUBPLOT 2: CURVA DE POTENCIA ---
#ax2.plot(df['datetime'], potencia_total, color='orange', linewidth=1)
ax2.plot(df['datetime'], potencia_suavizada, color='orange', linewidth=1.5, label='Potencia neta')
umbral = mediana_p + (3 * sigma_p)
ax2.axhline(y=umbral, color='red', linestyle=':', alpha=0.5, label='Umbral Detección')
ax2.set_ylim(mediana_p - sigma_p, mediana_p + (sigma_p * 10))
ax2.set_ylabel('Potencia (dB)')
ax2.set_xlabel('Tiempo (Local)')
ax2.grid(True, alpha=0.2, linestyle='--')
ax2.legend(loc='upper right', fontsize=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d\n%H:%M'))

# --- LÍNEAS DE REFERENCIA ---
for dia in df['datetime'].dt.date.unique():
    m_time = pd.to_datetime(f"{dia} 12:00:00")
    if df['datetime'].min() <= m_time <= df['datetime'].max():
        # Línea en Espectrograma
        ax1.axhline(y=mdates.date2num(m_time), color='white', linestyle='--', alpha=0.5)
        ax1.text(view_min + 0.01, mdates.date2num(m_time), 'Mediodía Solar', color='white',
            fontsize=8, verticalalignment='bottom', alpha=0.8)


        # Línea en Curva de Potencia
        ax2.axvline(x=m_time, color='red', linestyle='--', alpha=0.5, label='Mediodía')

plt.tight_layout()
plt.savefig(f"solar_full_analysis_{view_min:.1f}.png", dpi=300)
print("Análisis completo generado.")
