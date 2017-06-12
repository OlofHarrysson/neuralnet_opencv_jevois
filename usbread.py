import serial

ser = serial.Serial('/dev/tty.usbmodem1413')

print(ser)

print("HEJJ")


while True:
    bytesToRead = ser.inWaiting()
    data = ser.read(bytesToRead)
    print(data)



print("ASDASD")