// server/server.js
const express    = require('express');
const cors       = require('cors');
const { spawnSync } = require('child_process');
const path       = require('path');

const app  = express();
const PORT = 3000;

app.use(cors());

app.get('/soil-data', (req, res) => {
  try {
    // 1) Build the full path to your venv�s Python interpreter:
    //    project_root/.venv/bin/python
    const pythonBin = path.join(__dirname, '..', '.venv', 'bin', 'python');

    // 2) Build the path to your prediction script:
    //    project_root/model/predict.py
    const scriptPath = path.join(__dirname, '..', 'model', 'predict.py');

    // 3) Spawn the process
    const result = spawnSync(pythonBin, [ scriptPath ], { encoding: 'utf8' });

    if (result.error) {
      throw result.error;
    }
    if (result.status !== 0) {
      throw new Error(`Python exited ${result.status}: ${result.stderr}`);
    }

    // 4) Parse & return the JSON your script prints
    const payload = JSON.parse(result.stdout);
    res.json(payload);

  } catch (err) {
    console.error('Error in /soil-data:', err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`?? Server listening on http://0.0.0.0:${PORT}/soil-data`);
});
