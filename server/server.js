const express = require('express');
const cors = require('cors');
const { spawnSync } = require('child_process');
const path = require('path');

const app = express();
const PORT = 3000;

app.use(cors());

let lastPrediction = null;
let lastUpdated = 0;
const CACHE_DURATION_MS = 5 * 60 * 1000; // 5 minutes = 300,000 ms

app.get('/soil-data', (req, res) => {
  const now = Date.now();

  if (lastPrediction && now - lastUpdated < CACHE_DURATION_MS) {
    // Serve cached result
    return res.json(lastPrediction);
  }

  try {
    const pythonBin = path.join(__dirname, '..', '.venv', 'bin', 'python');
    const scriptPath = path.join(__dirname, '..', 'model', 'predict.py');

    const result = spawnSync(pythonBin, [scriptPath], { encoding: 'utf8' });

    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error(`Python exited ${result.status}: ${result.stderr}`);

    const payload = JSON.parse(result.stdout);

    lastPrediction = payload;
    lastUpdated = now;

    res.json(payload);
  } catch (err) {
    console.error('Error in /soil-data:', err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server listening on http://0.0.0.0:${PORT}/soil-data`);
});
