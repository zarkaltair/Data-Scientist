# loop.run_in_executor start in other thread
import asyncio
from urllib.request import urlopen


# a synchronous function
def sync_get_url(url):
    return urlopen(url).read()

async def load_url(url, loop=None):
    future = loop.run_in_executor(None, sync_get_url, url)
    response = await future
    print(len(response))


loop = asyncio.get_event_loop()
loop.run_until_complete(load_url('https://google.com', loop=loop))
