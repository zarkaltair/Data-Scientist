import asyncio


@asyncio.coroutine
def hello_world():
    while True:
        print('Hello world!')
        yield from asyncio.sleep(1.0)


loop = asyncio.get_event_loop()
loop.run_until_complete(hello_world())
loop.close()
