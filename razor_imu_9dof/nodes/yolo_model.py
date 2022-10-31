import zmq
from pykalman import KalmanFilter
port = '5556'

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect('tcp://localhost:5556')

socket.setsockopt(zmq.SUBSCRIBE, str.encode(''))
x_0 = 0
v_0 = 0
x_val = socket.recv_pyobj()
t_0 = x_val[1][0]
print('t_0:', t_0)
#print(x_val)
#x_val = socket.recv_pyobj()
#print(x_val)

ax_initial = x_val[1][1]
print('ax_initial:', ax_initial)
dt = 0.02
F = [[1,dt,0.5*dt**2],[0,1,dt],[0,0,1]]
print('F:', F)

while True:
   x_val = socket.recv_pyobj()
   print('x_val:', x_val)

        
    
    





#print(x_val[1][0])
    

