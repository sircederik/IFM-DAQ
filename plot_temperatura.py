#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import argparse
import sys

class TemperaturePlotter:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.load_data()

    def load_data(self):
        try:
            # Cargamos datos ignorando líneas mal formadas
            self.df = pd.read_csv(self.csv_file, on_bad_lines='skip')
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
            self.df.set_index('Timestamp', inplace=True)
        except Exception as e:
            print(f"#[!] Error cargando el archivo: {e}")
            sys.exit(1)

    def filter_by_date(self, start_date, end_date):
        try:
            if start_date and end_date:
                self.df = self.df.loc[start_date:end_date]
            elif start_date:
                self.df = self.df.loc[start_date:]
            elif end_date:
                self.df = self.df.loc[:end_date]
        except KeyError:
            print("#[!] Advertencia: Rango de fechas no encontrado. Usando datos completos.")

    def create_plot(self, output_file):
        # Usamos r"string" para evitar el SyntaxWarning de LaTeX
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in"
        })

        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
        
        # Estética profesional
        colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
        styles = ['-', '--', '-.', ':']

        for i, col in enumerate(self.df.columns):
            ax.plot(self.df.index, self.df[col], 
                    label=col, 
                    color=colors[i % 4], 
                    linestyle=styles[i % 4], 
                    linewidth=1.8, 
                    alpha=0.9)

        # --- LÓGICA DE ESCALA INTELIGENTE PARA EL EJE Y ---
        y_min, y_max = ax.get_ylim()
        range_y = y_max - y_min

        # Ajuste de ticks para evitar el error de MAXTICKS
        if range_y > 20:
            major_step = 5.0
            minor_step = 0.5  # Décimas si el rango es grande
        elif range_y > 5:
            major_step = 1.0
            minor_step = 0.1
        else:
            major_step = 0.5
            minor_step = 0.01 # Solo centésimas en rangos muy cerrados

        ax.yaxis.set_major_locator(MultipleLocator(major_step))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_step))

        # --- EJE X Y FORMATO ---
        ax.set_ylabel(r"Temperatura ($^\circ$C)", fontweight='bold')
        ax.set_xlabel("Tiempo (HH:MM:SS)", fontweight='bold')
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d\n%H:%M:%S'))
        fig.autofmt_xdate()

        # Grid semitransparente con jerarquía
        ax.grid(True, which='major', linestyle='-', alpha=0.4, color='gray', linewidth=0.8)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='gray', linewidth=0.5)

        ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='black')

        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        print(f"#[*] Gráfico generado con éxito: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CSV de entrada")
    parser.add_argument("output", help="Imagen de salida")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    plotter = TemperaturePlotter(args.input)
    plotter.filter_by_date(args.start, args.end)
    plotter.create_plot(args.output)

if __name__ == "__main__":
    main()
