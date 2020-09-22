# host
import socket


with socket.socket() as sock:
    sock.bind(('', 10001))
    sock.listen()

    while True:
        conn, addr = sock.accept()
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                print(data.decode('utf8'))


# client
import socket


with socket.create_connection(('127.0.0.1', 10001)) as sock:
    sock.sendall('ping'.encode('utf8'))
