import socket


with socket.socket() as sock:
    sock.bind(('', 10001))
    sock.listen()

    # createn several process
    while True:
        # accept between process
        conn, addr = sock.accept()
        # process for conn
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                print(data.encode('utf8'))
