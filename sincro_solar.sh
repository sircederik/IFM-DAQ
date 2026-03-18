#!/bin/bash

# --- CONFIGURACIÓN ---
FREQ_RANGE="16M:24M:1k"
GAIN="0"
INTEGRATION="10s"
DURATION="8h"
LOCAL_DIR="$HOME/capturas_solar"
REMOTE_USER="space-weather"
REMOTE_IP="148.216.53.34"
REMOTE_DIR="/home/space-weather/capturas/"

mkdir -p "$LOCAL_DIR"

echo "=== Estación Solar Morelia: Modo Captura Continua ==="

while true; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    FILE_NAME="solar_data_$TIMESTAMP.csv"
    FILE_PATH="$LOCAL_DIR/$FILE_NAME"

    echo "[$(date +%T)] Iniciando nueva captura: $FILE_NAME"

    # 1. EJECUCIÓN DE CAPTURA (Este proceso bloquea el script durante 8h)
    rtl_power -f "$FREQ_RANGE" -i "$INTEGRATION" -e "$DURATION" -g "$GAIN" "$FILE_PATH"

    # 2. LANZAR SINCRONIZACIÓN EN SEGUNDO PLANO
    # Al añadir el '&' al final, el script NO espera a que termine el rsync
    # e inmediatamente vuelve al inicio del 'while' para empezar el nuevo rtl_power.
    (
        echo "[$(date +%T)] Sincronización iniciada para $FILE_NAME..."
        rsync -avz --partial --remove-source-files "$LOCAL_DIR/*.csv" "$REMOTE_USER@$REMOTE_IP:$REMOTE_DIR"
        
        if [ $? -eq 0 ]; then
            echo "[$(date +%T)] Sincro OK: $FILE_NAME subido y borrado."
        else
            echo "[$(date +%T)] ERROR en sincro: $FILE_NAME se mantiene en local."
        fi
    ) & 

    # Pequeño margen de seguridad de 1 segundo antes de reabrir el hardware
    sleep 1
done
