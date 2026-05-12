import serial
import csv
import time

nombre_archivo = "captura.CSV"
muestras_esperadas = 5000

ser = serial.Serial('COM3', 921600, timeout=1)
time.sleep(2)
ser.reset_input_buffer()

ser.write(b"CAPTURAR\n")
with open(nombre_archivo, mode='w', newline='') as archivo_csv:
    escritor = csv.writer(archivo_csv)
    
    muestras_recibidas = 0
    while muestras_recibidas < muestras_esperadas:
        linea = ser.readline().decode('utf-8', errors='ignore').strip()
        
        if linea:
            datos = linea.split(',')
            if len(datos) == 3:
                escritor.writerow(datos)
                muestras_recibidas += 1
                print(f"Progreso: {muestras_recibidas}/{muestras_esperadas}")

ser.close()
print("Captura finalizada y guardada.")