# Asynchronous Request Handling

- Use an async HTTP client (e.g., httpx.AsyncClient)
- For each expert endpoint, send POST request with prompt in parallel
- Set timeout for each request to 5 seconds
- Await all responses using asyncio.gather
- Aggregate results into a single response dictionary
- If a request fails or times out, default that expert's result to 0 