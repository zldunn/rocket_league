import serial, time
smartfusion = serial.Serial('COM1', 115200, timeout=.1)
time.sleep(1) #give the connection a second to settle
smartfusion.write("Hello from Python!")
