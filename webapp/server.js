const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = 3000;

// Serve static files from the React build directory
app.use(express.static(path.join(__dirname, 'build')));

// Proxy API requests to the orchestrator
app.use('/orchestrate', createProxyMiddleware({
  target: 'http://172.31.48.224:8010',
  changeOrigin: true,
}));

// For any other requests, serve the React app
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
}); 