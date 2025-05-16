"""
Simple test script to verify serial communication with Arduino.
This script will send test commands and display responses from Arduino.
"""
import serial
import time

# Try to connect to different possible ports
possible_ports = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/cu.usbmodem*', '/dev/cu.usbserial*']
ser = None

for port in possible_ports:
    try:
        if '*' in port:  # For wildcard ports on macOS
            import glob
            for matched_port in glob.glob(port):
                try:
                    ser = serial.Serial(matched_port, baudrate=9600, timeout=1)
                    print(f"Connected to {matched_port}")
                    break
                except:
                    continue
        else:
            ser = serial.Serial(port, baudrate=9600, timeout=1)
            print(f"Connected to {port}")
        
        if ser:
            break
    except Exception as e:
        print(f"Could not connect to {port}: {e}")

# If no connection was successful
if not ser:
    print("Could not connect to any serial port. Please check connections.")
    exit()

# Wait for Arduino to initialize
time.sleep(2)

# Read any initial data from Arduino
while ser.in_waiting:
    line = ser.readline().decode('utf-8').strip()
    print(f"Arduino says: {line}")

# Test commands to send
test_commands = [
    "Hello Arduino!",
    "food_sandwich",
    "medical_napkin:3",
    "medical_syringe:1",
    "medical_bandage:2"
]

# Send commands and read responses
for cmd in test_commands:
    print(f"\nSending: {cmd}")
    ser.write(f"{cmd}\n".encode())
    time.sleep(0.5)  # Give Arduino time to respond
    
    # Read response
    while ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
        print(f"Arduino says: {line}")
    
    time.sleep(1)  # Wait before sending next command

# Listen for any additional responses
print("\nListening for 10 seconds...")
end_time = time.time() + 10
while time.time() < end_time:
    if ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
        print(f"Arduino says: {line}")
    time.sleep(0.1)

# Clean up
ser.close()
print("\nSerial test completed.")
