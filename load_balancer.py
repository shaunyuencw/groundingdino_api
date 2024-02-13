from aiohttp import web, ClientSession
import asyncio
import random
import time

PORT = 8000
servers = ["http://localhost:8001", "http://localhost:8002", "http://localhost:8003", "http://localhost:8004"]
server_loads = {server: 0 for server in servers}  # Track active requests for each server

async def forward_request(request):
    # Choose the server with the least load
    server = min(server_loads, key=server_loads.get)
    server_loads[server] += 1  # Increment load for chosen server
    

    # Retrieve the origin from the request headers
    origin = request.headers.get('Origin', 'Unknown Origin')

    gpu_index = servers.index(server)  # Get the index of the server to represent the GPU
    print(f"Request from {origin} forwarding to GPU {gpu_index} (Server: {server})")  # Print the origin and GPU being used
    
    # Debug: Print request path and method
    # print(f"Forwarding request: {request.method} {server + request.path}")

    try:
        data = await request.read()
        headers = {'Content-Type': request.headers.get('Content-Type', '')}
        # print(f"Received response body length: {len(response_body)}")
        # print(f"Snippet of response body: {response_body[:100]}")  # Print the first 100 characters of the response body for a quick peek

        # Debug: Print headers being forwarded
        # print(f"Forwarding headers: {headers}")

        async with ClientSession() as session:
            if request.method == 'POST':
                # Debug: Print a summary of data being forwarded
                # print(f"Forwarding POST request data, length: {len(data)}")

                async with session.post(server + request.path, data=data, headers=headers) as resp:
                    # Debug: Print response status and headers
                    # print(f"Received response status: {resp.status}, headers: {resp.headers}")

                    response = web.Response(status=resp.status, body=await resp.read(), headers={
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type'
                    })
            elif request.method == 'GET':
                async with session.get(server + request.path) as resp:
                    response = web.Response(status=resp.status, body=await resp.read(), headers={
                        'Content-Type': 'text/html',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type'
                    })
        # time.sleep(random.randint(0, 15) * 0.1)
        server_loads[server] -= 1  # Decrement load after request is handled
        return response
    except Exception as e:
        server_loads[server] -= 1  # Ensure load is decremented even if an error occurs
        print(f"Error forwarding request: {e}")
        raise e

async def handle_options(request):
    print("Handling OPTIONS request")
    return web.Response(headers={
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    })

app = web.Application()
app.router.add_route('*', '/{tail:.*}', forward_request)
app.router.add_route('OPTIONS', '/{tail:.*}', handle_options)

web.run_app(app, port=PORT)
