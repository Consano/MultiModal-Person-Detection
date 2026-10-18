import serial
import wave

# Configuration
SERIAL_PORT = 'COM11'
BAUD_RATE = 115200 
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
OUTPUT_FILENAME = r"FILEPATH"

print(f"Connecting to {SERIAL_PORT}...")
try:
    # Ideally timeout to RECORD_SECONDS + 2
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=7)
    ser.setDTR(True)
    ser.setRTS(True)
except Exception as e:
    print(f"Error opening port: {e}")
    exit()

# Reset buffer before trying to record
ser.reset_input_buffer()

with wave.open(OUTPUT_FILENAME, 'w') as wav_file:
    wav_file.setnchannels(1)  
    wav_file.setsampwidth(2)  
    wav_file.setframerate(SAMPLE_RATE)
    
    print(f"Recording for {RECORD_SECONDS}s")
    
    # Total bytes to read
    bytes_to_read = SAMPLE_RATE * RECORD_SECONDS * 2
    
    # Read audio
    raw_audio = ser.read(bytes_to_read)
    
    if len(raw_audio) > 0:
        wav_file.writeframesraw(raw_audio)
        print(f"Recording complete. Saved to {OUTPUT_FILENAME}")
    else:
        print("Error: No data received.")

ser.close()