#include <Arduino_AdvancedAnalog.h>

// Breakout Board pins: (WS, CK, SDI, SDO, MCK)
AdvancedI2S i2s(PB_9, PD_3, PI_2, NC, NC);

void setup() {
    Serial.begin(115200);

    // Initialize I2S with: Mode, Sample Rate, Samples per buffer, Queue depth
    if (!i2s.begin(AN_I2S_MODE_IN, 16000, 32, 64)) {
        Serial.println("Failed to start I2S");
        while (1); 
    }
    
    Serial.println("I2S Initialized");
}

void loop() {
    if (i2s.available()) {
        SampleBuffer buf = i2s.read();

        // Increment by 2 to extract ONLY the Left channel
        for (int i = 0; i < buf.size(); i += 2) {
            
            int16_t sample = (int16_t)buf[i];
            sample = sample * 50; // Apply digital gain
            
            // Send the 16-bit integer as 2 raw binary bytes
            Serial.write((uint8_t*)&sample, 2);
        }
        
        buf.release();
    }
}