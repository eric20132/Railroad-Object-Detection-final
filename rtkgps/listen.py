from serial import Serial
stream = Serial('/dev/ttyACM0',9600,timeout = 3000)        
while True:
        line = stream.readline()
        line = str(line,encoding= 'utf-8')
        print(line)
