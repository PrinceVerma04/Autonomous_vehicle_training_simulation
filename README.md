# Autonomous Vehicle Training Simulation
<img width="637" height="507" alt="image" src="https://github.com/user-attachments/assets/93c32b91-6817-4fc1-b310-f6e679089820" />


## Project overview

This repository contains a small server application that uses a trained NVIDIA-style convolutional neural network to drive the Udacity Self-Driving Car Simulator in autonomous mode. The server receives images and telemetry from the simulator, runs the model to predict steering angles, and sends steering/throttle commands back to the simulator.

Important: the trained model weights (.h5) are not included in this repository. See "Model weights" below for instructions to obtain or repair the model file.

## Repository 

```
Autonomous_vehicle_training_simulation/
├─ drive.py                # Main server script (runs the model and socket server)
├─ requirements.txt        # Python dependencies
├─ README.md               # This file
└─ model/
   ├─ model.h5             # (EXPECTED) trained model file - removed from repo
   ├─ model_fixed.h5       # (OPTIONAL) fixed/converted model produced from Colab
   └─ model.keras          # alternate Keras model format
```

## Quick summary

- Language: Python 3.8+ (tested with 3.8–3.11)
- Primary libs: TensorFlow / Keras, Flask, python-socketio, eventlet, OpenCV, Pillow, NumPy
- Intended environment: local Windows machine (PowerShell examples provided) or Linux/macOS with equivalent commands

## 1 — Setup (Windows PowerShell)

1. Create a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- If you encounter permission or SSL issues, try running PowerShell as Administrator or use the `--user` flag for pip.

## 2 — Model weights (required)

This project requires a trained model file placed under `model/model.h5` (or one of the alternate names the server searches for). The repository does not include the `.h5` file.

If you already have the trained model (HDF5 .h5), copy it into the `model/` directory and name it `model.h5`.

If your model file was saved in a format that is incompatible with the runtime (common when saving directly from newer TF/Keras), run the small "fix" conversion in Google Colab to produce `model_fixed.h5` (the server's troubleshooting message mentions the same steps). Example snippet to run in Colab:

```python
from tensorflow import keras

# Mount your drive or upload the original model to Colab, then:
model = keras.models.load_model('/path/to/original_model.h5', compile=False)
model.save('/content/model_fixed.h5', include_optimizer=False)
# Download model_fixed.h5 and copy to your local repo: model/model.h5
```

If `model_fixed.h5` works, copy it locally as `model/model.h5`.

## 3 — Run the server

Make sure the model file exists at `model/model.h5` then run:

```powershell
python drive.py
```

Expected behavior:
- The script will search `model/` for several filenames (see top of `drive.py`).
- If no valid model is found the script prints troubleshooting steps and exits.
- If the model loads, the server starts on http://localhost:4567 and waits for the Udacity simulator to connect.

## 4 — Simulator steps (Udacity SDC Simulator)

1. Launch the Udacity Self-Driving Car Simulator.
2. In the simulator UI select `AUTONOMOUS MODE` and choose a track (e.g. Lake Track).
3. The simulator connects to the server and the car should drive using the model's predictions.

## 5 — Common issues & troubleshooting

- "Invalid username or token" when pushing to GitHub: create a PAT with `repo` scope or use `gh auth login` (GitHub CLI).
- `gh` command not found after installation: restart PowerShell or run `& 'C:\Program Files\GitHub CLI\gh.exe'` directly.
- Model load errors (e.g. HDF5 signature, batch_shape issues): run the Colab fix above to produce `model_fixed.h5` and try again.
- If the server prints "NO VALID MODEL FOUND" and then exits, follow the printed troubleshooting steps in `drive.py` to produce `model_fixed.h5` and copy it to `model/model.h5`.

## 6 — Development notes

- Preprocessing function used in `drive.py` (img_preprocess) must match training preprocessing exactly — otherwise steering predictions will be wrong.
- The script loads the model with `compile=False` and then recompiles with Adam and MSE — this prevents some compatibility issues.

I have uploaded a recording of the simulator run to this repository as `simulation.mp4`.

- Preview on GitHub: GitHub will render an embedded video player for `simulation.mp4` when you view the file in the repository.
- Local playback (PowerShell example):

```powershell
# On Windows you can use the default video player
Start-Process .\simulation.mp4
# Or use VLC (if installed)
"C:\Program Files\VideoLAN\VLC\vlc.exe" .\simulation.mp4
```

Caption: "Autonomous driving run — Lake Track — NVIDIA-style CNN model (recorded output)."

# Autonomous_vehicle_training_simulation
