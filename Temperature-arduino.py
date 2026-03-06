#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import csv
import time
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque

# --- CONFIGURACIÓN CENTRALIZADA ---
CONFIG = {
    'PORT': '/dev/ttyUSB0',
    'BAUD': 9600,
    'FILE': "log_temperaturas.csv",
    'MAX_SAMPLES': 50,
    'OFFSETS': [0.0, 0.0, 0.0, 0.0]  # Ajusta aquí tus valores de calibración
}

class DAQManager:
    """Clase encargada exclusivamente de la comunicación y persistencia."""
    def __init__(self, port, baud):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # Espera reinicio del Nano
            self.ser.flushInput()
        except serial.SerialException as e:
            print(f"#[!] Error de hardware: {e}")
            sys.exit(1)

    def leer_sensores(self):
        """Lee y calibra los datos. Retorna lista de floats o None."""
        if self.ser.in_waiting > 0:
            try:
                linea = self.ser.readline().decode('utf-8').strip()
                valores = [float(x) for x in linea.split(',')]
                if len(valores) == 4:
                    # Aplicar calibración en tiempo real
                    return [v + CONFIG['OFFSETS'][i] for i, v in enumerate(valores)]
            except (ValueError, UnicodeDecodeError):
                return None
        return None

    def cerrar(self):
        if self.ser: self.ser.close()

def configurar_grafica():
    """Inicializa la ventana de Matplotlib."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Sistema de Adquisición de Datos - 4 Canales")
    ax.set_ylabel("Temperatura (°C)")
    ax.grid(True, linestyle='--', alpha=0.6)

    deques = [deque(maxlen=CONFIG['MAX_SAMPLES']) for _ in range(4)]
    lineas = [ax.plot([], [], label=f'Sensor {i+1}')[0] for i in range(4)]
    ax.legend(loc='upper left')

    return fig, ax, deques, lineas

def actualizar_ui(ax, lineas, deques, nuevos_datos):
    """Actualiza los objetos de la gráfica."""
    for i, d in enumerate(deques):
        d.append(nuevos_datos[i])
        lineas[i].set_data(range(len(d)), list(d))

    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01) # Pausa mínima para procesar eventos de la UI

def main():
    daq = DAQManager(CONFIG['PORT'], CONFIG['BAUD'])
    fig, ax, deques, lineas = configurar_grafica()

    try:
        with open(CONFIG['FILE'], 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['Timestamp', 'S1', 'S2', 'S3', 'S4'])

            print(f"#[*] Iniciando DAQ en {CONFIG['PORT']}. Ctrl+C para salir.")

            while plt.fignum_exists(fig.number): # Corre mientras la ventana esté abierta
                datos = daq.leer_sensores()

                if datos:
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([ts] + datos)
                    f.flush()

                    actualizar_ui(ax, lineas, deques, datos)
                    print(f"[{ts}] {datos}")

    except KeyboardInterrupt:
        print("\n#[!] Captura finalizada por el usuario.")
    finally:
        daq.cerrar()
        plt.close('all')

if __name__ == "__main__":
    main()
