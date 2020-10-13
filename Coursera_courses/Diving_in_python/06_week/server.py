import sys
import asyncio


def run_server(host, port):
    loop = asyncio.get_event_loop()
    coro = loop.create_server(ClientServerProtocol, host, port)
    server = loop.run_until_complete(coro)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()

d = {}
arr_ts = {}

class ClientServerProtocol(asyncio.Protocol):
    def connection_made(self, transport):
        self.transport = transport

    def process_data(self, data):
        data = data.split()

        if len(data) < 2 or data[0] not in ['get', 'put']:
            return 'error\nwrong command\n\n'

        elif data[0] == 'get' and len(data) != 2:
            return 'error\nwrong command\n\n'

        elif data[0] == 'put' and len(data) != 4:
            return 'error\nwrong command\n\n'

        elif data[0] == 'put':
            try:
                if data[1] not in d:
                    d[data[1]] = [(int(data[3]), float(data[2]))]
                    arr_ts[data[1]] = [int(data[3])]
                else:
                    if int(data[3]) not in arr_ts[data[1]]:
                        d[data[1]] += [(int(data[3]), float(data[2]))]
                        arr_ts[data[1]] += [int(data[3])]
                    else:
                        for ind, i in enumerate(d[data[1]]):
                            if int(data[3]) == i[0]:
                                d[data[1]][ind] = (int(data[3]), float(data[2]))

                for k, v in d.items():
                    d[k] = sorted(v)
                return 'ok\n\n'
            except BaseException as err:
                # print(f'{err.__class__} {err}')
                return 'error\nwrong command\n\n'

        elif data[0] == 'get' and data[1] not in d and data[1] != '*':
            return 'ok\n\n'

        elif data[1] == '*':
            resp = f'ok\n'
            for k, v in d.items():
                for i in v:
                    resp += f'{k} {i[1]} {i[0]}\n'
            return resp + '\n'

        else:
            resp = f'ok\n'
            for i in d[data[1]]:
                resp += f'{data[1]} {i[1]} {i[0]}\n'
            return resp + '\n'

    def data_received(self, data):
        resp = self.process_data(data.decode('utf8'))
        self.transport.write(resp.encode('utf8'))


if __name__ == '__main__':
    host = sys.argv[1]
    port = sys.argv[2]
    run_server(host, port)
