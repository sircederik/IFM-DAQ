#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from matplotlib.colors import hsv_to_rgb
import argparse
import sys
import os


class SolarAnalyzer:
    CMAPS_COMUNES = ['charolastra', 'magma', 'inferno', 'viridis', 'plasma', 'jet', 'hot', 'gnuplot2']

    def __init__(self, args):
        self.args = args
        self.df_raw = None
        self.tiempos = None
        self.data_all = None
        self.data_calibrada = None
        self.f_min_total = None
        self.f_max_total = None
        self.f_step = None
        self.potencia_final = None
        self.stats = {}
        self.cmap_final = None

    def cargar_y_limpiar(self):
        """Carga archivos CSV y asegura que los datos sean numéricos."""
        lista_df = []
        print(f"📂 Cargando {len(self.args.archivos)} archivos...")

        for f in self.args.archivos:
            try:
                temp_df = pd.read_csv(f, header=None, low_memory=False)
                # Crear datetime primero para evitar fragmentación
                temp_df['datetime'] = pd.to_datetime(temp_df[0] + ' ' + temp_df[1])
                # Convertir bloque de datos a numérico de golpe (columnas 6 en adelante)
                temp_df.iloc[:, 6:-1] = temp_df.iloc[:, 6:-1].apply(pd.to_numeric, errors='coerce')
                lista_df.append(temp_df)
            except Exception as e:
                print(f"⚠️ Error en {f}: {e}")

        if not lista_df:
            print("❌ No hay datos para procesar."); sys.exit(1)

        self.df_raw = pd.concat(lista_df, ignore_index=True)

    def alinear_espectro(self):
        """Une los saltos de frecuencia  mediante vectorización."""
        print("🚀 Alineando saltos de frecuencia...")
        tiempos_unicos = self.df_raw['datetime'].unique()
        num_hops = self.df_raw[2].nunique()

        # Ordenar para asegurar que el reshape sea coherente
        df_sorted = self.df_raw.sort_values(by=['datetime', 2])
        data_matrix = df_sorted.iloc[:, 6:-1].values.astype(float)

        bins_per_hop = data_matrix.shape[1]
        self.data_all = data_matrix.reshape(len(tiempos_unicos), num_hops * bins_per_hop)
        self.tiempos = pd.Series(tiempos_unicos).sort_values()

        # Extraer metadatos
        self.f_min_total = self.df_raw[2].min() / 1e6
        self.f_max_total = self.df_raw[3].max() / 1e6
        self.f_step = self.df_raw.iloc[0, 4] / 1e6

        print(f"Matriz: {self.data_all.shape} | Hops: {num_hops}| fmin: {self.f_min_total} fmax: {self.f_max_total}")


    def _generar_nombre_default(self):
            """Genera un nombre de archivo basado en fechas, frecuencias y procesos."""
            # Extraer fechas para el nombre (formato YYYYMMDD_HHMM)
            inicio = self.tiempos.min().strftime('%Y%m%d_%H%M')
            fin = self.tiempos.max().strftime('%Y%m%d_%H%M')
            res_khz = int(self.f_step * 1000)
            # Detectar sufijos de procesamiento
            suffix_cal = "_CAL" if self.args.cal is not None else ""
            suffix_norm = "_NORM" if hasattr(self.args, 'norm') and self.args.norm else ""
            
            # Reemplazar caracteres problemáticos en el mapa de color
            cmap_name = self.args.cmap.lower()

            nombre = f"SOLAR_{inicio}-{fin}_{res_khz}k_{cmap_name}{suffix_cal}{suffix_norm}.png"
            return nombre


    def _get_charolastra_cmap(self):
            """
            Genera la paleta 'Charolastra'.
            Basada en la lógica de color HSV de solsticedhiver (https://github.com/solsticedhiver).
            Adaptada para la visualización de ráfagas solares en Morelia.
            """
            def loop(n):
                if n > 1: return 1
                if n < 0: return 1 - abs(n)
                return n

            paleta = []
            for i in range(1024):
                g = i / 1023.0
                # Mapeo HSV: Tono de Azul a Rojo, Brillo creciente
                c = hsv_to_rgb([loop(0.65 - (g - 0.08)), 1, loop(0.2 + g)])
                paleta.append(c)
            return ListedColormap(paleta, name='charolastra')


    def configurar_visualizacion(self):
            """Selecciona el mapa de color según la preferencia del usuario."""
            if self.args.cmap.lower() == 'charolastra':
                self.cmap_final = self._get_charolastra_cmap()
            else:
                # Si no es charolastra, intenta cargar uno de Matplotlib
                try:
                    self.cmap_final = plt.get_cmap(self.args.cmap)
                except ValueError:
                    print(f"⚠️ Colormap '{self.args.cmap}' no encontrado. Usando 'magma'.")
                    self.cmap_final = plt.get_cmap('magma')

    def normalizar_datos(self):
            """
            Normaliza la señal (Z-Score) de forma independiente.
            Divide el residuo por la desviación estándar del ruido.
            """
            if not hasattr(self.args, 'norm') or not self.args.norm:
                return

            print(f"⚖️  Aplicando Normalización Estadística (Z-Score)...")

            # Reutilizamos la lógica de obtención de ruido
            if self.args.cal and len(self.args.cal) == 1 and ":" not in self.args.cal[0]:
                noise_matrix = self._cargar_ruido_archivo(self.args.cal[0])
            else:
                rango = self.args.cal if (self.args.cal and len(self.args.cal) == 2) else ["03:00", "04:00"]
                noise_matrix = self._extraer_ruido_rango(rango)

            if noise_matrix is not None and noise_matrix.size > 0:
                perfil_mediana = np.nanmedian(noise_matrix, axis=0)
                perfil_std = np.nanstd(noise_matrix, axis=0)
                perfil_std[perfil_std <= 0] = 1.0  # Evitar división por cero

                # Aplicamos sobre la data actual (que puede estar calibrada o no)
                self.data_calibrada = (self.data_all - perfil_mediana) / perfil_std
                self.stats['unidad'] = "Sigmas (σ)"
            else:
                print("⚠️ No se pudo normalizar: Referencia de ruido no encontrada.")



    def calibrar_ruido(self):
        """
        Lógica de calibración:
        1. Si '--cal' no está presente en los argumentos -> No calibra.
        2. Si '--cal' está presente pero vacío -> Calibra 03:00 a 04:00.
        3. Si '--cal' tiene 1 argumento -> Se asume que es un archivo CSV.
        4. Si '--cal' tiene 2 argumentos -> Se asume que es un rango de horas.
        """

        # CASO 1: El usuario NO escribió --cal en la terminal
        if self.args.cal is None:
            print("⏭️  Modo: Datos brutos (sin calibración).")
            self.data_calibrada = self.data_all.copy()
            self.stats['modo_cal'] = "Ninguna"
            return

        print("🧪 Iniciando proceso de calibración...")
        noise_matrix = None

        # CASO 2: Escribió --cal pero no puso argumentos (args.cal es una lista vacía [])
        if len(self.args.cal) == 0:
            rango = ["03:00", "04:00"]
            print(f"  -> Usando rango por defecto: {rango}")
            noise_matrix = self._extraer_ruido_rango(rango)
            self.stats['modo_cal'] = f"Default ({rango[0]}-{rango[1]})"

        # CASO 3: Escribió --cal archivo.csv
        elif len(self.args.cal) == 1:
            archivo_n = self.args.cal[0]
            print(f"-> Cargando archivo de ruido externo: {archivo_n}")
            noise_matrix = self._cargar_ruido_archivo(archivo_n)
            self.stats['modo_cal'] = f"Archivo ({archivo_n})"

        # CASO 4: Escribió --cal 12:00 13:00
        elif len(self.args.cal) == 2:
            rango = self.args.cal
            print(f"  -> Usando rango especificado: {rango}")
            noise_matrix = self._extraer_ruido_rango(rango)
            self.stats['modo_cal'] = f"Rango manual ({rango[0]}-{rango[1]})"

        # CÁLCULO FINAL
        if noise_matrix is not None and noise_matrix.size > 0:
            perfil_ruido = np.nanmedian(noise_matrix, axis=0)
            self.data_calibrada = self.data_all - perfil_ruido
        else:
            print("⚠️ No se pudo obtener matriz de ruido. Usando datos brutos.")
            self.data_calibrada = self.data_all.copy()

    def _extraer_ruido_rango(self, rango):
        """Método privado para filtrar por tiempo."""
        t_start = pd.to_datetime(rango[0]).time()
        t_end = pd.to_datetime(rango[1]).time()
        mask = (self.tiempos.dt.time >= t_start) & (self.tiempos.dt.time <= t_end)
        return self.data_all[mask]

    def _cargar_ruido_archivo(self, ruta):
        """Procesa un archivo externo alineando sus saltos de frecuencia."""
        try:
            # 1. Carga básica
            df_n = pd.read_csv(ruta, header=None, low_memory=False)
            df_n['datetime'] = pd.to_datetime(df_n[0] + ' ' + df_n[1])
            df_n.iloc[:, 6:-1] = df_n.iloc[:, 6:-1].apply(pd.to_numeric, errors='coerce')

            # 2. Detectar cuántos saltos (hops) tiene el archivo de ruido
            hops_ruido = df_n[2].nunique()
            hops_datos = self.df_raw[2].nunique()

            if hops_ruido != hops_datos:
                print(f"⚠️ Alerta: El archivo de ruido tiene {hops_ruido} saltos pero los datos tienen {hops_datos}.")

            # 3. ALINEACIÓN (Igual que en los datos principales)
            df_n_sorted = df_n.sort_values(by=['datetime', 2])
            tiempos_n = df_n_sorted['datetime'].unique()
            data_n_raw = df_n_sorted.iloc[:, 6:-1].values.astype(float)

            bins_objetivo = self.data_all.shape[1] // hops_ruido
            data_n_raw = df_n_sorted.iloc[:, 6:6+bins_objetivo].values.astype(float)
            matrix_n = data_n_raw.reshape(len(tiempos_n), hops_ruido * bins_objetivo)

            print(f"Matriz de ruido alineada: {matrix_n.shape}")
            return matrix_n

        except Exception as e:
            print(f"❌ Error al procesar archivo de calibración: {e}")
            return None

    def procesar_potencia(self, data_recortada):
        """Calcula la curva de flujo relativo y estadísticas de ráfagas."""
        potencia_media = np.nanmean(data_recortada, axis=1)
        self.potencia_final = pd.Series(potencia_media).rolling(window=8, center=True).mean()
        mediana=self.potencia_final.median()
        centrada=self.potencia_final-mediana 
        sigma_final=centrada.std()

        if hasattr(self.args, 'norm') and self.args.norm:
            self.potencia_final = centrada / sigma_final
            unidad_label = "Sigmas (σ)"
        else:
            self.potencia_final = centrada
            unidad_label = "dB"
        self.stats.update({'std': sigma_final, 'base_db': mediana, 'unidad': unidad_label})
        return self.potencia_final

    def generar_grafico(self):
        """Crea la visualización final ax1 (espectro) y ax2 (potencia)."""
        # 1. Recorte de frecuencias solicitado
        fmin = self.args.fmin if self.args.fmin else self.f_min_total
        fmax = self.args.fmax if self.args.fmax else self.f_max_total
        idx_s = int((fmin - self.f_min_total) / self.f_step)
        idx_e = int((fmax - self.f_min_total) / self.f_step)
        data_plot = self.data_calibrada[:, idx_s:idx_e]

        # CÁLCULO DINÁMICO DE ESCALA
        # El vmin se ajusta al "piso" de los datos actuales
        v_min_auto = np.nanpercentile(data_plot, 5)   # El 5% más bajo
        # El vmax se ajusta a las ráfagas, dejando un margen
        v_max_auto = np.nanpercentile(data_plot, 99.5) # El tope del 99.5%
        rango = v_max_auto - v_min_auto
        if rango < 5: # Si hay muy poco contraste, forzamos un mínimo de 10dB de rango
            v_max_auto = v_min_auto + 10

        if hasattr(self.args, 'norm') and self.args.norm:
            unidad = "Sigmas (σ)"
            print(f"Escala visual: {v_min_auto:.2f} a {v_max_auto:.2f} Sigmas (σ)")
        else:
            unidad = "dB"
            print(f"Escala visual: {v_min_auto:.2f} a {v_max_auto:.2f} dB")

        potencia = self.procesar_potencia(data_plot)

        # 2. Setup de figura
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})

        # Espectrograma

        extent = [mdates.date2num(self.tiempos.iloc[0]), mdates.date2num(self.tiempos.iloc[-1]), fmin, fmax]
        im = ax1.imshow(data_plot.T, aspect='auto', extent=extent, 
                        cmap=self.cmap_final, 
                        vmin=v_min_auto, 
                        vmax=v_max_auto)

        fig.colorbar(im, ax=ax1, label=f'Intensidad {unidad}')


        # Definimos los niveles según si está normalizado o no
        if hasattr(self.args, 'norm') and self.args.norm:
            # En modo Sigma, las bandas son fijas: 1, 2 y 3
            s1, s2, s3, s4, s5, s6 = 1.0, 2.0, 3.0, 4.0, 5.0 ,6.0
            unidad_txt = "σ"
        else:
            # En modo dB, usamos la desviación estándar calculada
            s1 = self.stats.get('std', 0.1)
            s2, s3 , s4, s5, s6= 2*s1, 3*s1, 4*s1, 5*s1, 6*s1
            unidad_txt = "dB"


        # Potencia y Bandas Sigma

        # Dibujar las bandas de confianza
        ax2.axhspan(-s1, s1, color='gray', alpha=0.15, label=f'1{unidad_txt} (Ruido)')
        ax2.axhspan(s1, s2, color='green', alpha=0.15, label=f'2{unidad_txt} (Cuidado)')
        ax2.axhspan(-s1, -s2, color='green', alpha=0.15, label=f'2{unidad_txt} (Cuidado)')
        ax2.axhspan(s2, s3, color='blue', alpha=0.15, label=f'3{unidad_txt} (Ráfaga!)')
        ax2.axhspan(-s2, -s3, color='blue', alpha=0.15, label=f'3{unidad_txt} (Ráfaga!)')

        # Opcional: Línea en el cero para referencia técnica
        ax2.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
        # Ajustar límites del eje Y dinámicamente
        ymax = max(s3, self.potencia_final.max() * 1.2)
        ymin = min(-s1, self.potencia_final.min() * 1.2)
        ax2.set_ylim(ymin, ymax)

        ax2.plot(self.tiempos, potencia, color='orange', linewidth=1)

        # Formato de tiempo
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.grid(True, alpha=0.2)

        ax1.set_title(f"Análisis Radioastronómico Solar: {fmin}-{fmax} MHz")
        ax2.set_ylabel(f"Flujo Relativo {unidad}")

        if self.args.output:
            output_name = self.args.output
        else:
            output_name = self._generar_nombre_default()

        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado como: {output_name}")
        return output_name


    def imprimir_sumario(self, output_file):
        """Muestra el reporte final en consola."""
        print("\n" + "="*45)
        print("📊 SUMARIO DE PROCESAMIENTO")
        print("="*45)
        print(f"📅 Periodo:   {self.tiempos.min()} -> {self.tiempos.max()}")
        print(f"📡 Espectro:  {self.f_min_total:.2f} a {self.f_max_total:.2f} MHz")
        print(f"📏 Res. Bin:  {self.f_step*1000:.2f} kHz")
        print(f"⚙️  Nivel Base: {self.stats['base_db']:.2f} dB")
        print(f"✅ Resultado: {output_file}")
        print("="*45)

# --- INICIO DEL PROGRAMA ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Analizador Solar Modular - Basado en radioastronomía de baja frecuencia.',
        formatter_class=argparse.RawTextHelpFormatter # Para que respete los saltos de línea en la ayuda
    )


    cmap_help = f"Paleta de colores a utilizar.\nOpciones comunes: {', '.join(SolarAnalyzer.CMAPS_COMUNES)}\n(por defecto: charolastra)"

    parser.add_argument('archivos', nargs='+', help='Archivos CSV')
    parser.add_argument('--fmin', type=float)
    parser.add_argument('--fmax', type=float)
    parser.add_argument('--output', '-o')
    parser.add_argument('--cal', nargs='*', help='Calibración: nada (3-4am), un archivo.csv, o rango "HH:MM HH:MM". Si no se pone --cal, no calibra.')
    parser.add_argument('--cmap', type=str, default='charolastra', help=cmap_help)
    parser.add_argument('--norm', action='store_true', help='Usar normalización estadística (Z-Score)')
    args = parser.parse_args()

    # Flujo de ejecución limpio
    solar = SolarAnalyzer(args)
    solar.cargar_y_limpiar()
    solar.alinear_espectro()
    solar.calibrar_ruido()
    solar.normalizar_datos()
    solar.configurar_visualizacion()
    archivo_final = solar.generar_grafico()
    solar.imprimir_sumario(archivo_final)
