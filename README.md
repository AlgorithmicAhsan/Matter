# PosiSim - Motion Capture to Avatar Pipeline

Real-time human motion capture using MediaPipe, with uncertainty estimation for reliable motion recording. Outputs Blender-ready BVH files.

## Setup Steps

### 1. Install Python & Conda
Download and install Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/

### 2. Create Conda Environment
```bash
conda create -n sim python=3.10 -y
conda activate sim
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

Or install individually:
```bash
pip install mediapipe opencv-python fastapi uvicorn websockets numpy python-multipart
```

### 4. Project Structure
Create folders:
```bash
mkdir backend frontend models output
```

Structure should be:
```
PosiSim/
├── backend/
│   ├── main.py
│   ├── pose.py
│   ├── uncertainty.py
│   ├── bvh_writer.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── avatar.js
│   └── uncertainty_ui.js
├── models/
│   └── avatar.glb
└── output/
```

### 5. Download 3D Avatar Model
- Go to https://www.mixamo.com
- Download a character in **glTF/GLB format** (not FBX)
- Save to `models/avatar.glb`

### 6. Start Backend
```bash
cd backend
uvicorn main:app --reload
```

Backend runs on `http://localhost:8000`

### 7. Open in Browser
```
http://localhost:8000
```

Allow camera permission, click "Start Recording" to begin motion capture.

## How It Works

1. **Pose Extraction** (pose.py)
   - MediaPipe extracts 33 body keypoints from webcam
   - Returns normalized coordinates + visibility scores

2. **Uncertainty Scoring** (uncertainty.py)
   - Aleatoric: Based on visibility scores
   - Epistemic: MC Dropout variance simulation
   - Total score determines if frame is "trusted"

3. **Recording** (bvh_writer.py)
   - Buffers only trusted frames (uncertainty < 0.3)
   - Exports to BVH format on stop

4. **3D Avatar** (Frontend)
   - Three.js renders Mixamo character
   - Joints updated via WebSocket in real-time
   - Green bar = trusted, Red bar = uncertain

5. **Blender Export**
   - BVH file saved to `output/recording.bvh`
   - Import into Blender: File → Import → BVH

## Troubleshooting

**Webcam not accessible:**
```bash
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**Import error:**
```bash
python -c "import mediapipe, cv2, fastapi; print('✓')"
```

**Port 8000 in use:**
```bash
uvicorn main:app --reload --port 8001
```

**Avatar model not found:**
- Ensure `models/avatar.glb` exists
- Verify format is `.glb` (not `.fbx`)

## Dependencies
- mediapipe (pose detection)
- opencv-python (camera)
- fastapi (web server)
- uvicorn (ASGI)
- websockets (real-time data)
- numpy (math)
- Three.js (frontend 3D)

## Detection Models

Beyond the body pose, two additional MediaPipe models run in parallel on the backend (`pose.py`). Each detects its landmarks independently, draws them onto the streamed camera preview, and sends them to the frontend over the WebSocket.

### Face Mesh (`process_face`)
- **Model:** MediaPipe Face Mesh (`refine_landmarks=True`) → **478 landmarks** per face (includes irises).
- **Drawing:** face contours (eyes, brows, lips, face oval, irises) drawn on the live preview.
- **Payload:** sent as `face_landmarks` (list of `{x, y, z}`; no per-point visibility).
- **Mapping:** drives the avatar's **head bone** (nod / turn / tilt). A head basis is built from stable facial points, so the pose self-zeros when facing the camera — no calibration needed. *Facial expressions are not mapped — the Mixamo model has no blendshapes.*

### Hands (`process_hands`)
- **Model:** MediaPipe Hands (`max_num_hands=2`) → **21 landmarks** per hand (full finger articulation).
- **Drawing:** each hand skeleton drawn on the live preview.
- **Payload:** sent as `hand_landmarks` = `{ "left": [...21], "right": [...21] }` (a side is `null` if not visible; no per-point visibility).
- **Mapping:** drives **finger curl** on all 30 finger bones. Per-joint bend angles are computed from consecutive landmark segments — rotation-invariant, so curl reads correctly at any hand orientation.
- **Note:** MediaPipe's Left/Right labels assume a mirrored (selfie) image, so they may be swapped relative to the raw camera frame.
