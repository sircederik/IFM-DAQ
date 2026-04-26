#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize
import argparse
import sys
import os

#monitoreamos la memoria
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class SolarAnalyzer:
    CMAPS_COMUNES = ['charolastra', 'magma', 'inferno', 'viridis', 'plasma', 'jet', 'hot', 'gnuplot2','rainbow']
    RES_COMUNES = ['500', '1000', '1500', '2000', '2500', '5000']
    def __init__(self, args):
        self.args = args
        self.df_raw = None
        self.tiempos = None
        self.data_all = None
        self.data_calibrada = None
        self.f_min_total = None
        self.f_max_total = None
        self.f_step = None
        self.potencia = None
        self.stats = {}
        self.cmap_final = None
        self.freqs = None
        self.indices_inicio_archivo = []
        self.limite_vertical=2500

    def cargar_y_limpiar(self):
        """Carga archivos CSV y asegura que los datos sean numéricos."""
        lista_df = []
        self.indices_inicio_archivo = []
        indice_actual=0

        print(f"📂 Cargando {len(self.args.archivos)} archivos...")
        for f in self.args.archivos:
            try:
                temp_df = pd.read_csv(f, header=None, low_memory=False)
                # Crear datetime primero para evitar fragmentación
                temp_df['datetime'] = pd.to_datetime(temp_df[0] + ' ' + temp_df[1])
                # Convertir bloque de datos a numérico de golpe (columnas 6 en adelante)
                temp_df.iloc[:, 6:-1] = temp_df.iloc[:, 6:-1].apply(pd.to_numeric, errors='coerce')
                lista_df.append(temp_df)
                self.indices_inicio_archivo.append(indice_actual)
                indice_actual += temp_df.shape[0]
                self.log_memoria(f'carga de archivos: ', dinamico=True)
                del temp_df
                gc.collect()
            except Exception as e:
                print(f"⚠️ Error en {f}: {e}")

        if not lista_df:
            print("❌ No hay datos para procesar."); sys.exit(1)

        self.df_raw = pd.concat(lista_df, ignore_index=True)
        print(f"✅ Carga completa. {len(lista_df)} archivos unidos.")
        print(f"📍 Costuras detectadas en índices: {self.indices_inicio_archivo}")

    def alinear_espectro(self):
        """Une los saltos de frecuencia  mediante vectorización."""
        print("🚀 Alineando saltos de frecuencia...")

        self.f_min_total = self.df_raw[2].min() / 1e6
        self.f_max_total = self.df_raw[3].max() / 1e6
        self.f_step = self.df_raw.iloc[0, 4] / 1e6
        num_hops = self.df_raw[2].nunique() # Debería ser 3 según tu archivo

        self.df_raw['datetime'] = pd.to_datetime(self.df_raw[0] + ' ' + self.df_raw[1])
        cols_datos = list(range(6, self.df_raw.shape[1] - 1)) # -1 por la nueva col datetime
        self.df_raw = self.df_raw[['datetime', 2] + cols_datos]
        self.df_raw.sort_values(by=['datetime', 2], inplace=True)
        data_matrix = self.df_raw.iloc[:, 2:].values.astype(np.float32)
        tiempos_unicos = self.df_raw['datetime'].unique()
        self.log_memoria(f'Alineacion de espectro...')
        del self.df_raw
        gc.collect()

        bins_per_hop = data_matrix.shape[1]
        self.data_all = data_matrix.reshape(len(tiempos_unicos), num_hops * bins_per_hop)
        num_canales_total = self.data_all.shape[1] 
        
        self.freqs = np.linspace(self.f_min_total, self.f_max_total, num_canales_total)
        print(f"✅ Vector de frecuencias reconstruido: {len(self.freqs)} puntos.")

        self.tiempos = pd.Series(tiempos_unicos).sort_values()
        print(f"✅ Vector de tiempos ordenado.")

    def _generar_nombre_default(self):
            """Genera un nombre de archivo basado en fechas, frecuencias y procesos."""
            # Extraer fechas para el nombre (formato YYYYMMDD_HHMM)
            inicio = self.tiempos.min().strftime('%Y%m%d_%H%M')
            fin = self.tiempos.max().strftime('%Y%m%d_%H%M')
            res_khz = int(self.f_step * 1000)
            # Detectar sufijos de procesamiento
            suffix_cal = "_CAL" if self.args.cal is not None else ""
            suffix_norm = "_NORM" if hasattr(self.args, 'norm') and self.args.norm else ""
            suffix_res = f"_RES{self.args.res}" 
            # Reemplazar caracteres problemáticos en el mapa de color
            cmap_name = self.args.cmap.lower()

            nombre = f"SOLAR_{inicio}-{fin}_{res_khz}k_{cmap_name}{suffix_cal}{suffix_norm}{suffix_res}.png"
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


    def calcular_estadisticos_globales(self):
        print("🌍 Calculando estadísticos globales (referencia fija)...")

        self.perfil_mediana_global = np.nanmedian(self.data_all, axis=0)

        # MAD por canal
        desviaciones = np.abs(self.data_all - self.perfil_mediana_global)
        mad = np.nanmedian(desviaciones, axis=0)

        # Convertir MAD → sigma robusto
        self.perfil_std_global = 1.4826 * mad

        # Seguridad
        self.perfil_std_global[self.perfil_std_global <= 0] = 1.0
        print("✅ Estadísticos globales listos.")


    def aplicar_normalizacion_global(self):
        print("⚖ Aplicando normalización global (Z-score fijo)...")

        #self.data_calibrada = self.data_all.copy()
        #self.data_calibrada -= self.perfil_mediana_global
        #self.data_calibrada /= self.perfil_std_global
        self.data_all -= self.perfil_mediana_global
        self.data_all /= self.perfil_std_global

        self.stats['unidad'] = "Sigmas (σ)"
        self.log_memoria(f'Normalizacion memoria')



    def obtener_limites_raw(self, data):
        """
        Calcula vmin/vmax para datos brutos (dB o cuentas).
        Evita que el ruido base o los picos de interferencia quemen la imagen.
        """
        # El vmin debe estar justo en el 'piso' del ruido (percentil 10)
        # Ignoramos el 10% más bajo para no oscurecer demasiado por 'hoyos' de datos
        v_min_auto = np.nanpercentile(data, 10)

        # El vmax se ajusta al tope del 99.5% de los datos
        v_max_auto = np.nanpercentile(data, 99.5)

        # Añadimos un pequeño margen de maniobra (headroom)
        rango = v_max_auto - v_min_auto
        vmin = v_min_auto
        vmax = v_max_auto + (rango * 0.1) 

        print(f"📊 Escala RAW: {vmin:.2f} a {vmax:.2f} (Unidades originales)")
        return vmin, vmax


    def calibrar_ruido(self):
        """
        Lógica de calibración:
        1. Si '--cal' no está presente en los argumentos -> No calibra.
        2. Si '--cal' está presente pero vacío -> Calibra 03:00 a 04:00.
        3. Si '--cal' tiene 1 argumento -> Se asume que es un archivo CSV.
        4. Si '--cal' tiene 2 argumentos -> Se asume que es un rango de horas.
        """

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
            print(f" -> Usando rango especificado: {rango}")
            noise_matrix = self._extraer_ruido_rango(rango)
            self.stats['modo_cal'] = f"Rango manual ({rango[0]}-{rango[1]})"

        # CÁLCULO FINAL
        if noise_matrix is not None and noise_matrix.size > 0:
            perfil_ruido = np.nanmedian(noise_matrix, axis=0)
            self.data_all = self.data_all - perfil_ruido
        else:
            print("⚠️ No se pudo obtener matriz de ruido. Usando datos brutos.")
            return

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
            print(f"❌ --> Error al procesar archivo de calibración: {e}")
            return None

    def detectar_eventos_transitorios(self, umbral=6.0):
        """
        Localiza píxeles que exceden el umbral de sigmas y devuelve sus coordenadas reales.
        """
        print(f'-> Detentando eventos transitorios...')
        idx_t, idx_f = np.where(self.data_all > umbral)
        
        eventos = []
        if len(idx_t) > 0:
            print(f"🎯 Detectados {len(idx_t)} píxeles sobre {umbral}σ")
            for i in range(len(idx_t)):
                t_idx, f_idx = idx_t[i], idx_f[i]
                evento = {
                    'tiempo': self.tiempos.iloc[t_idx],
                    'frecuencia': self.freqs[f_idx],
                    'intensidad': self.data_all[t_idx, f_idx]
                }
                eventos.append(evento)
                # Solo imprimimos los más significativos si son demasiados
                if i < 30: 
                    print(f"   - [{evento['tiempo'].strftime('%H:%M:%S')}] "
                          f"{evento['frecuencia']:.2f} MHz -> {evento['intensidad']:.2f}σ")
        return eventos


    def procesar_potencia(self):
        """Calcula la potencia de forma consistente (sin efectos de contexto)."""
        print(f'-> Procesando la potencia de la señal integrada...')
        potencia_media = np.nanmean(self.data_all, axis=1)
        # Suavizado
        self.potencia = pd.Series(potencia_media).rolling(
            window=5,
            center=True,
            min_periods=1   
        ).mean()
        print(f'--> Tamaño del vector de potencia: {len(self.potencia) ,len(self.tiempos)}')


    def limpiar_transitorios_de_archivo(self, ancho_segundos=3):
        if self.indices_inicio_archivo is None or len(self.indices_inicio_archivo) == 0:
            return

        print(f"🧹 Limpiando transitorios en {len(self.indices_inicio_archivo)} costuras...")

        # Calculamos una mediana global o por canal de una zona "limpia"
        # Tomamos una muestra de la mitad de la matriz para evitar bordes
        muestra_limpia = self.data_all[10:110, :]
        perfil_relleno = np.nanmedian(muestra_limpia, axis=0)

        for inicio in self.indices_inicio_archivo:
            # El primer archivo (inicio 0) también puede tener ruido de encendido del SDR
            # Yo recomiendo limpiar incluso el inicio 0.

            fin = min(inicio + ancho_segundos, self.data_all.shape[0])

            # REEMPLAZO: Usamos el perfil de la zona limpia
            self.data_all[inicio:fin, :] = perfil_relleno

            # Debug para ver qué estamos haciendo
            print(f" ↳ Costura en índice {inicio}: {ancho_segundos}s reemplazados.")

        print("✨ Datos saneados. Los latigazos han sido neutralizados.")

    def generar_grafico(self):
        print(f'-> Generando gráfico...')
        """Crea la visualización final ax1 (espectro) y ax2 (potencia)."""
        # 1. Calculamos cuántas filas tiene nuestra matriz calibrada
        num_filas_total = self.data_all.shape[0]

        # 2. Calculamos el factor de salto (step)
        # Si num_filas_total < 5000, factor_t será 1 (no cambia nada)
        # Si num_filas_total = 50,000, factor_t será 10 (grafica 1 de cada 10 filas)
        factor_t = max(1, num_filas_total // self.args.res)
        print(f'Número de filas {num_filas_total}, factor de escala {factor_t}')
        # 1. Recorte de frecuencias solicitado
        fmin = self.args.fmin if self.args.fmin else self.f_min_total
        fmax = self.args.fmax if self.args.fmax else self.f_max_total
       
        #idx_s = np.abs(self.freqs - fmin).argmin()
        #idx_e = np.abs(self.freqs - fmax).argmin()

        # 3. Slicing inteligente: Tomamos la vista (no copia) de la matriz
        # [::factor_t, :] salta filas en el tiempo pero mantiene todas las frecuencias
        data_plot = self.data_all[::factor_t]
        tiempos_plot = self.tiempos.iloc[::factor_t]

        print(f"--> Matriz {data_plot.shape}, Tiempos {tiempos_plot.shape}")
        if hasattr(self.args, 'norm') and self.args.norm:
            unidad = "Sigmas (σ)"
            # CÁLCULO DINÁMICO DE ESCALA
            # El vmin se ajusta al "piso" de los datos actuales
            v_min_auto = -1.5 
            # El vmax se ajusta a las ráfagas, dejando un margen
            v_max_auto = 1.50 
            rango = v_max_auto - v_min_auto
        else:
            v_min_auto, v_max_auto = self.obtener_limites_raw(data_plot)
            unidad = "dB"

        if factor_t > 1:
            print(f"--> Optimizando visualización: Factor de decimación {factor_t}x")

        print(f"--> Escala visual: {v_min_auto:.2f} a {v_max_auto:.2f} {unidad}")

        potencia_plot = self.potencia.iloc[::factor_t]
        tiempos_plot = self.tiempos.iloc[::factor_t]

        # 🔥 FORZAR misma longitud
        n = min(len(tiempos_plot), len(potencia_plot))
        tiempos_plot = tiempos_plot.iloc[:n]
        potencia_plot = potencia_plot.iloc[:n]

        print(f"--> Máximo detectado: {np.nanmax(self.data_all):.2f} {unidad}")
        print(f"--> Promedio de los datos: {np.nanmean(self.data_all):.2f} {unidad}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        plt.subplots_adjust(hspace=0.02)

        extent = [mdates.date2num(self.tiempos.iloc[0]), mdates.date2num(self.tiempos.iloc[-1]), fmin, fmax]
        im = ax1.imshow(data_plot.T, 
                        aspect='auto', 
                        extent=extent, 
                        cmap=self.cmap_final, 
                        vmin=v_min_auto, 
                        vmax=v_max_auto,
                        interpolation='nearest',
                        origin='lower')

        if self.args.start: 
            zoom_start = mdates.date2num(pd.to_datetime(self.args.start))
            ax1.set_xlim(left=zoom_start)

        if self.args.end:
            zoom_end = mdates.date2num(pd.to_datetime(self.args.end))
            ax1.set_xlim(right=zoom_end)

        if self.args.fmin and self.args.fmax:
            ax1.set_ylim(float(self.args.fmin), float(self.args.fmax))

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        plt.colorbar(im, cax=cax, label=f'Intensidad {unidad}')


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


        #ax1.set_xlabel(f"Tiempo [LC]")
        ax1.set_title(f"Análisis Radioastronómico Solar: {fmin}-{np.round(fmax)} MHz")
        ax1.set_ylabel(f"Frecuencia [MHz]")
        ax1.tick_params(labelbottom=False) # Quitar etiquetas de ax1 para que no se encimen
        # Dibujar las bandas de confianza
        #ax2.axhspan(-s1, s1, color='gray', alpha=0.15, label=f'1{unidad_txt} (Ruido)')
        #ax2.axhspan(s1, s2, color='green', alpha=0.15, label=f'2{unidad_txt} (Cuidado)')
        #ax2.axhspan(-s1, -s2, color='green', alpha=0.15, label=f'2{unidad_txt} (Cuidado)')
        #ax2.axhspan(s2, s3, color='blue', alpha=0.15, label=f'3{unidad_txt} (Ráfaga!)')
        #ax2.axhspan(-s2, -s3, color='blue', alpha=0.15, label=f'3{unidad_txt} (Ráfaga!)')

        # Ajustar límites del eje Y dinámicamente
        #ymax =max(s3, potencia_plot.max() * 1.5)
        #ymin =min(-s1, potencia_plot.min() * 1.5)
        #ax2.set_ylim(ymin, ymax)
        ax2.margins(x=0)
        ax2.xaxis_date()
        ax2.set_xlabel(f"Tiempo [LT]")

        fig.canvas.draw()
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        ax2.set_position([pos1.x0, pos2.y0, pos1.width, pos2.height])

        locator = mdates.AutoDateLocator()
        ax2.xaxis.set_major_locator(locator)
        ax2.set_ylabel(f"Flujo Relativo {unidad}")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%y/%m/%d'))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.plot(tiempos_plot, potencia_plot, color='red', linewidth=1.3)
        #plt.tight_layout()
        if self.args.output:
            output_name = self.args.output
        else:
            output_name = self._generar_nombre_default()

        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        plt.close('all')
        gc.collect()
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
        print(f"✅ Resultado: {output_file}")
        print("="*45)


    def aplicar_filtro_temporal(self, start_raw, end_raw):
        if not start_raw and not end_raw:
            return

        # 1. Convertir self.tiempos (que ya debe estar decimado si hubo downsampling)
        # Usamos pd.to_datetime para asegurar compatibilidad
        tiempos_dt = pd.to_datetime(self.tiempos)
        fecha_base = tiempos_dt.iloc[0].date()

        def parsear(input_str, default_val):
            if not input_str: return default_val
            try:
                return pd.to_datetime(input_str)
            except:
                h, m = map(int, input_str.split(':'))
                return datetime.combine(fecha_base, time(h, m))

        t_inicio = parsear(start_raw, tiempos_dt.iloc[0])
        t_fin = parsear(end_raw, tiempos_dt.iloc[-1])

        # 2. Crear la máscara basándonos ÚNICAMENTE en el tamaño actual de tiempos_dt
        mask = (tiempos_dt >= t_inicio) & (tiempos_dt <= t_fin)

        # 3. Validar que no quede vacío
        if not any(mask):
            print(f"[!] Ojo: El rango {start_raw}-{end_raw} no existe en los datos procesados.")
            return

        # 4. RECORTAR TODO LO QUE TENGA ESE EJE TEMPORAL (Axis 0)
        self.tiempos = self.tiempos[mask]
        
        if hasattr(self, 'data_all'):
            self.data_all = self.data_all[mask]

        if self.data_calibrada is not None:
            # NumPy aplica la máscara booleana directamente sobre el eje 0 (filas)
            self.data_calibrada = self.data_calibrada[mask]

        print(f"[*] Zoom final aplicado: {len(self.tiempos)} muestras en ventana.")

    def log_memoria(self, etapa, dinamico=False):
        if not HAS_PSUTIL:
            return

        try:
            process = psutil.Process(os.getpid())
            mem_uso_mb = process.memory_info().rss / (1024 * 1024)
            
            # Usamos \r para volver al inicio de la línea y end='' para no saltar
            formato = f"\r📊 [MEMORIA] {etapa}: {mem_uso_mb:.2f} MB"
            
            if dinamico:
                # Rellenamos con espacios al final para limpiar residuos de líneas más largas
                print(f"{formato}".ljust(50), end='', flush=True)
            else:
                # Si no es dinámico, imprimimos normal (un salto de línea)
                print(f"\n{formato}")
        except Exception:
            pass

    # --- INICIO DEL PROGRAMA ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Analizador Solar Modular - Basado en radioastronomía de baja frecuencia.',
        formatter_class=argparse.RawTextHelpFormatter # Para que respete los saltos de línea en la ayuda
    )


    cmap_help = f"Paleta de colores a utilizar.\nOpciones comunes: {', '.join(SolarAnalyzer.CMAPS_COMUNES)}\n(por defecto: charolastra)"
    res_help = f"Resolución de la imagen a generar.\nOpciones comunes: {', '.join(SolarAnalyzer.RES_COMUNES)}\n(por defecto: 2500)"

    parser.add_argument('archivos', nargs='+', help='Archivos CSV')
    parser.add_argument('--fmin', type=float)
    parser.add_argument('--fmax', type=float)
    parser.add_argument('--output', '-o')
    parser.add_argument('--cal', nargs='*', help='Calibración: nada (3-4am), un archivo.csv, o rango "HH:MM HH:MM". Si no se pone --cal, no calibra.')
    parser.add_argument('--cmap', type=str, default='charolastra', help=cmap_help)
    parser.add_argument('--norm', action='store_true', help='Usar normalización estadística (Z-Score)')

    parser.add_argument('--start', type=str, help='Inicio: "YYYY-MM-DD HH:MM" o solo "HH:MM"')
    parser.add_argument('--end', type=str, help='Fin: "YYYY-MM-DD HH:MM" o solo "HH:MM"')

    parser.add_argument('--res', type=int, default=2500, help=res_help)

    args = parser.parse_args()

    # Flujo de ejecución limpio
    solar = SolarAnalyzer(args)
    solar.cargar_y_limpiar()
    solar.alinear_espectro()
    #solar.limpiar_transitorios_de_archivo(ancho_segundos=50)
    #solar.eliminar_rfi_vertical()

    if args.cal:
        solar.calibrar_ruido()

    if args.norm:
        solar.calcular_estadisticos_globales()
        solar.aplicar_normalizacion_global()

    solar.procesar_potencia()

    #solar.configurar_visualizacion()
    archivo_final = solar.generar_grafico()

    solar.detectar_eventos_transitorios(umbral=5)
    solar.imprimir_sumario(archivo_final)

