#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import csv
import time
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
from collections import deque

# --- CONFIGURACIÓN CENTRALIZADA ---
CONFIG = {
    'PORT': '/dev/ttyUSB0',
    'BAUD': 9600,
    'FILE': "log_temperaturas.csv",
    'MAX_SAMPLES': 20,
    'OFFSETS': [0.0, 0.0, 0.0, 0.0],
    # Nombres personalizados para tus sensores
    'LABELS': ["T. Disipador ", "T. Placa interior", "T. Peltier superior", "T. Ambiente"],
    # Estilos visuales: (Color, Estilo de línea)
    'ESTILOS': [
        ('#FF5733', '-'),  # Naranja rojizo sólido
        ('#82C2DF', '--'), # Verde con guiones
        ('#3357FF', '-.'), # Azul punto-guion
        ('#F333FF', ':')   # Púrpura punteado
    ]
}

class DAQManager:
    def __init__(self, port, baud):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            self.ser.flushInput()
        except serial.SerialException as e:
            print(f"#[!] Error de hardware: {e}")
            sys.exit(1)

    def leer_sensores(self):
        if self.ser.in_waiting > 0:
            try:
                linea = self.ser.readline().decode('utf-8').strip()
                valores = [float(x) for x in linea.split(',')]
                if len(valores) == 4:
                    return [v + CONFIG['OFFSETS'][i] for i, v in enumerate(valores)]
            except (ValueError, UnicodeDecodeError):
                return None
        return None

    def cerrar(self):
        if self.ser: self.ser.close()

def configurar_grafica():
    plt.ion()
    # Usamos un estilo más moderno
    #plt.style.use('seaborn-v0_8-darkgrid') 
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.set_title("Monitoreo Térmico en Tiempo Real - IFM-DAQ", fontsize=14, fontweight='bold')
    ax.set_ylabel("Temperatura (°C)", fontsize=12)
    ax.set_xlabel("Hora Local", fontsize=12)

    # --- Refinar Eje Y (Minor Ticks) ---
    ax.yaxis.set_minor_locator(AutoMinorLocator(5)) # 5 subdivisiones por cada grado
    ax.tick_params(which='both', width=1)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4, color='gray')

    # --- Refinar Eje X (Formato de Tiempo) ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%M/%d\n%H:%M:%S'))
    
    # Deques: Uno para el tiempo (X) y otros para datos (Y)
    time_deque = deque(maxlen=CONFIG['MAX_SAMPLES'])
    data_deques = [deque(maxlen=CONFIG['MAX_SAMPLES']) for _ in range(4)]
    
    # Líneas más gruesas y con estilos distintos
    lineas = []
    for i in range(4):
        ln, = ax.plot([], [], 
                      label=CONFIG['LABELS'][i], 
                      color=CONFIG['ESTILOS'][i][0], 
                      linestyle=CONFIG['ESTILOS'][i][1],
                      linewidth=2.5,  # Línea más notable
                      antialiased=True)
        lineas.append(ln)

    legend = ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=12)

    return fig, ax, time_deque, data_deques, lineas, legend

def actualizar_ui(ax, lineas, time_deque, data_deques, nuevos_datos, ts_obj, legend):
    # Agregar tiempo y datos
    time_deque.append(ts_obj)
    
    for i, d in enumerate(data_deques):
        d.append(nuevos_datos[i])
        lineas[i].set_data(list(time_deque), list(d))
        
        # Actualizar etiquetas de la leyenda con el VALOR ACTUAL
        lineas[i].set_label(f"{CONFIG['LABELS'][i]}: {nuevos_datos[i]:.2f}°C")

    # Refrescar la leyenda para mostrar los nuevos valores
    ax.legend(loc='upper left', frameon=True, shadow=True)

    # Ajuste automático de escalas
    ax.relim()
    ax.autoscale_view()
    
    # Rotar etiquetas del tiempo para mejor lectura
    plt.xticks(rotation=20)
    plt.grid()
    plt.pause(0.01)

def main():
    daq = DAQManager(CONFIG['PORT'], CONFIG['BAUD'])
    fig, ax, time_deque, data_deques, lineas, legend = configurar_grafica()

    try:
        with open(CONFIG['FILE'], 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['Timestamp'] + CONFIG['LABELS'])

            print(f"#[*] Iniciando DAQ. Visualización activa. Ctrl+C para salir.")

            while plt.fignum_exists(fig.number):
                datos = daq.leer_sensores()

                if datos:
                    ahora = datetime.now()
                    ts_str = ahora.strftime('%Y-%m-%d %H:%M:%S')
                    
                    writer.writerow([ts_str] + datos)
                    f.flush()

                    actualizar_ui(ax, lineas, time_deque, data_deques, datos, ahora, legend)
                    print(f"[{ts_str}] {datos}")

    except KeyboardInterrupt:
        print("\n#[!] Captura finalizada.")
    finally:
        daq.cerrar()
        plt.close('all')

if __name__ == "__main__":
    main()
