import socket
import select


# create object instance socket
sock = socket.socket()
sock.bind(('', 10001))
sock.listen()

# create two connection
conn1, addr = sock.accept()
conn2, addr = sock.accept()

# set unblocking mode
conn1.setblocking(0)
conn2.setblocking(0)

epoll = select.epoll()
epoll.register(conn1.fileno(), select.EPOLLIN | select.EPOLLOUT)
epoll.register(conn2.fileno(), select.EPOLLIN | select.EPOLLOUT)

conn_map = {conn1.fileno(): conn1,
            conn2.fileno(): conn2}

# event loop
while True:
    events = epoll.poll(1)
    for fileno, event in events:
        if event & select.EPOLLIN:
            data = conn_map[fileno].recv(1024)
            print(data.decode('utf8'))
        elif event & select.EPOLLOUT:
            conn_map[fileno].send('ping'.encode('utf8'))
