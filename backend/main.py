import asyncio
import cv2
import json
import os
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pose import PoseExtractor
from uncertainty import UncertaintyScorer
from bvh_writer import BVHWriter
from actions import ActionController, ActionConfig, KeyboardController

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="PosiSim - Markerless Motion Capture")

# Serve frontend files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
app.mount("/models", StaticFiles(directory=str(MODELS_DIR)), name="models")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


# ── Configuration ─────────────────────────────────────────────────────────────
# Set to 0 for default webcam, 1 or 2 for other cameras (like DroidCam)
# Or set to a URL for direct IP camera: "http://192.168.1.XX:4747/video"
CAMERA_SOURCE = 0

# ── Global State ──────────────────────────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.extractor = PoseExtractor()
        self.scorer = UncertaintyScorer()
        self.writer = BVHWriter(fps=30)
        self.cap = None
        self.recording = False
        self.running = False
        self.source = CAMERA_SOURCE
        self.last_landmarks = None
        self.frame_count = 0
        # Action / movement system
        self.action_ctrl = ActionController(ActionConfig())
        self.keyboard_ctrl = KeyboardController(self.action_ctrl)
        self._last_update_time = time.time()

state = SessionState()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Serve the main frontend page."""
    index_path = FRONTEND_DIR / "index.html"
    return FileResponse(str(index_path))


@app.get("/health")
async def health():
    return {"status": "ok", "recording": state.recording, "frames": len(state.writer.frames)}


@app.get("/output/list")
async def list_outputs():
    """List all saved BVH files."""
    files = list(OUTPUT_DIR.glob("*.bvh"))
    return {"files": [f.name for f in sorted(files, reverse=True)]}


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")

    # Open video source
    state.cap = cv2.VideoCapture(state.source)
    if not state.cap.isOpened():
        await websocket.send_text(json.dumps({"error": "Could not open camera"}))
        await websocket.close()
        return

    state.running = True
    state.frame_count = 0

    try:
        while state.running:
            # Check for incoming commands from frontend
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                command = json.loads(data)
                await handle_command(command, websocket)
            except asyncio.TimeoutError:
                pass  # no command this iteration, continue
            except WebSocketDisconnect:
                break

            # Read frame
            ret, frame = state.cap.read()
            if not ret:
                # If video file ended, loop back
                if state.source != 0:
                    state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            # Process frame
            landmarks, annotated = state.extractor.process_frame(frame)
            uncertainty_result = state.scorer.score(landmarks)
            state.last_landmarks = landmarks
            state.frame_count += 1

            # Tick action controller (keeps backend position in sync with keys)
            now = time.time()
            dt = min(now - state._last_update_time, 0.05)
            state._last_update_time = now
            velocity, rotation_delta = state.action_ctrl.update(dt)
            state.action_ctrl.position += velocity * dt
            state.action_ctrl.rotation += rotation_delta

            # Encode annotated frame for preview
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            import base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Add to BVH if recording
            if state.recording:
                state.writer.add_frame(landmarks, trusted=uncertainty_result["trusted"])

            # Build payload to send to frontend
            avatar_state = state.action_ctrl.get_state_dict()
            payload = {
                "frame": state.frame_count,
                "recording": state.recording,
                "image": frame_base64,
                "bvh_frames": len(state.writer.frames),
                "bvh_skipped": state.writer.skipped,
                "uncertainty": {
                    "aleatoric": uncertainty_result["aleatoric"],
                    "epistemic": uncertainty_result["epistemic"],
                    "total": uncertainty_result["total"],
                    "trusted": uncertainty_result["trusted"]
                },
                "landmarks": landmarks if landmarks else None,
                "avatar": avatar_state,   # world position + active actions
            }

            if state.frame_count % 100 == 0:
                print(f"Streaming frame {state.frame_count} (Image size: {len(frame_base64)} bytes)")

            await websocket.send_text(json.dumps(payload))

            # ~30fps cap
            await asyncio.sleep(1 / 30)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        state.running = False
        if state.cap:
            state.cap.release()
        print("Camera released.")


async def handle_command(command: dict, websocket: WebSocket):
    """
    Handle commands sent from the frontend.
    Commands: start_recording, stop_recording, save_bvh, set_source,
              key_down, key_up
    """
    action = command.get("action")

    if action == "start_recording":
        state.writer.reset()
        state.recording = True
        print("Recording started.")
        await websocket.send_text(json.dumps({"event": "recording_started"}))

    elif action == "stop_recording":
        state.recording = False
        print("Recording stopped.")
        await websocket.send_text(json.dumps({
            "event": "recording_stopped",
            "frames": len(state.writer.frames),
            "skipped": state.writer.skipped
        }))

    elif action == "save_bvh":
        state.recording = False
        filepath = state.writer.save()
        filename = Path(filepath).name if filepath else None
        print(f"BVH saved: {filename}")
        await websocket.send_text(json.dumps({
            "event": "bvh_saved",
            "filename": filename,
            "frames": len(state.writer.frames),
            "skipped": state.writer.skipped,
            "coverage_pct": round(
                len(state.writer.frames) /
                max(len(state.writer.frames) + state.writer.skipped, 1) * 100, 1
            )
        }))

    elif action == "set_source":
        source = command.get("source", 0)
        state.source = source
        if state.cap:
            state.cap.release()
        state.cap = cv2.VideoCapture(state.source)
        print(f"Source set to: {state.source}")

    elif action == "key_down":
        # Forward browser key press to Python action controller
        key = command.get("key", "")
        state.keyboard_ctrl.key_down(key)

    elif action == "key_up":
        key = command.get("key", "")
        state.keyboard_ctrl.key_up(key)

    elif action == "get_coverage":
        stats = state.scorer.coverage_stats()
        await websocket.send_text(json.dumps({"event": "coverage_stats", "stats": stats}))


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting PosiSim backend...")
    print("Open http://localhost:8000 in your browser.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)