const int NUM_MUESTRAS = 1000; 
const int CANTIDAD_PERIODOS = 2;
const float FRECUENCIA_SENAL = 10.0;
const double MICROSEG_IN_SEG = 1000000.0; 

const int pinADC1 = 34;
const int pinADC2 = 35;

unsigned long t_arr[NUM_MUESTRAS];
uint16_t raw1_arr[NUM_MUESTRAS];
uint16_t raw2_arr[NUM_MUESTRAS];

void setup() {
  Serial.begin(921600);
  analogReadResolution(12);
}

void loop() {
  if (Serial.available() > 0) {
    String comando = Serial.readStringUntil('\n');
    comando.trim();
    
    if (comando == "CAPTURAR") {
      double periodo_muestreo_us = (MICROSEG_IN_SEG * CANTIDAD_PERIODOS) / (FRECUENCIA_SENAL * NUM_MUESTRAS);
      
      for (int i = 0; i < NUM_MUESTRAS; i++) {        
        unsigned long tiempo_inicio = micros();
        
        t_arr[i] = tiempo_inicio;
        uint16_t lectura_va_1 = analogRead(pinADC1);
        raw2_arr[i] = analogRead(pinADC2);
        uint16_t lectura_va_2 = analogRead(pinADC1);
        raw1_arr[i] = (lectura_va_1 + lectura_va_2) / 2;
        
        unsigned long tiempo_gastado = micros() - tiempo_inicio;  
        if (periodo_muestreo_us > tiempo_gastado) {
          delayMicroseconds(periodo_muestreo_us - tiempo_gastado);
        }
      }
      
      for (int i = 0; i < NUM_MUESTRAS; i++) {
        double tiempo = t_arr[i] / MICROSEG_IN_SEG; 
        
        float va_leido = raw1_arr[i] * (3.3 / 4095.0);
        float vb_leido = raw2_arr[i] * (3.3 / 4095.0);

        float va = -2.0 * (va_leido - 1.5);
        float vb = -2.0 * (vb_leido - 1.5);
        
        Serial.print(tiempo,6);
        Serial.print(",");
        Serial.print(va,6);
        Serial.print(",");
        Serial.println(vb,6);
      }
    }
  }
}