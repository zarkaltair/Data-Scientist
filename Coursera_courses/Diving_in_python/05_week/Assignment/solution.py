import time
import socket


class ClientError(Exception):
    def __init__(self, text):
        self.text = text


class Client:
    def __init__(self, addr_host, port, timeout=None):
        self.addr_host = addr_host
        self.port = port
        self.timeout = timeout
        self.sock = socket.create_connection((self.addr_host, self.port), self.timeout)

    def put(self, metric, value, timestamp=None):
        if timestamp == None:
            timestamp = int(time.time())
        request = f'put {metric} {value} {timestamp}\n'
        try:
            self.sock.sendall(request.encode('utf8'))
        except:
            raise ClientError('ClientError')
        data = self.sock.recv(1024)
        data = data.decode('utf8')
        ans = data.split()
        if ans[0] != 'ok':
            raise ClientError('ClientError')


    def get(self, request):
        request = f'get {request}\n'
        self.sock.sendall(request.encode('utf8'))

        d = {}
        data = self.sock.recv(1024)
        data = data.decode('utf8')
        print(data)
        ans = data.split()[:1]
        arr = data.split()[1:]

        arr_server = [i for ind, i in enumerate(arr) if ind % 3 == 0]
        arr_value = [i for ind, i in enumerate(arr[1:]) if ind % 3 == 0]
        arr_timestamp = [i for ind, i in enumerate(arr[2:]) if ind % 3 == 0]

        if ans[0] == 'ok' and arr == []:
            return d
        elif ans[0] == 'ok' and len(arr) > 1 and 'not_value' not in arr_value and 'not_timestamp' not in arr_timestamp and len(arr_value) == len(arr_timestamp) == len(arr_server):
            for ind, i in enumerate(arr):
                if ind % 3 == 0:
                    try:
                        if i not in d:
                            d[i] = [(int(arr[ind + 2]), float(arr[ind + 1]))]
                        else:
                            d[i] += [(int(arr[ind + 2]), float(arr[ind + 1]))]
                    except:
                        raise ClientError('ClientError')
            for k, v in d.items():
                d[k] = sorted(v)
            return d
        else:
            raise ClientError('ClientError')


# client = Client("127.0.0.1", 8888, timeout=15)

# client.put("palm.cpu", 0.5, timestamp=1150864247)
# client.put("palm.cpu", 2.0, timestamp=1150864248)
# client.put("palm.cpu", 0.5, timestamp=1150864248)
# client.put("eardrum.cpu", 3, timestamp=1150864250)
# client.put("eardrum.cpu", 4, timestamp=1150864251)
# client.put("eardrum.memory", 4200000)

# print(client.get("*"))
# print(client.get("palm.cpu"))
