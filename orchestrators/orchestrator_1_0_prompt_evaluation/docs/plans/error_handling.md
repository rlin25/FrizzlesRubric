# Error Handling and Defaults

- If an expert endpoint fails to respond or returns an error, log the error
- If an expert endpoint times out, log the timeout
- If an expert endpoint returns invalid JSON or missing 'result', log the issue
- For any failure, set that expert's result to 0 in the response
- Return partial results for successful experts 