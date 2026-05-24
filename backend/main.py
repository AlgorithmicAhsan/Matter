import asyncio
import cv2
import json
import os
import time
import base64
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pose import PoseExtractor
from uncertainty import UncertaintyScorer
from bvh_writer import BVHWriter
from actions import ActionController, ActionConfig, KeyboardController
from action_detector import PoseActionDetector          # NEW

app = FastAPI(title="PosiSim - Markerless Motion Capture")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
MODELS_DIR   = Path(__file__).parent.parent / "models"
OUTPUT_DIR   = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
app.mount("/models",   StaticFiles(directory=str(MODELS_DIR)),   name="models")
app.mount("/output",   StaticFiles(directory=str(OUTPUT_DIR)),   name="output")

CAMERA_SOURCE = 0


class SessionState:
    def __init__(self):
        self.extractor    = PoseExtractor()
        self.scorer       = UncertaintyScorer()
        self.writer       = BVHWriter(fps=30)
        self.cap          = None
        self.recording    = False
        self.running      = False
        self.source       = CAMERA_SOURCE
        self.last_landmarks = None
        self.frame_count  = 0

        self.action_ctrl   = ActionController(ActionConfig())
        self.keyboard_ctrl = KeyboardController(self.action_ctrl)
        self.pose_detector = PoseActionDetector()       # NEW
        self.pose_mode     = False                      # NEW: toggled by frontend
        self._last_update_time = time.time()

state = SessionState()


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/health")
async def health():
    return {"status": "ok", "recording": state.recording, "frames": len(state.writer.frames)}

@app.get("/output/list")
async def list_outputs():
    files = list(OUTPUT_DIR.glob("*.bvh"))
    return {"files": [f.name for f in sorted(files, reverse=True)]}


@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    """
    Accept a video upload, run MediaPipe Pose on every frame,
    and return the full landmark sequence as JSON.

    Processing runs in a thread-pool executor so it does NOT block
    the async event loop — the WebSocket camera feed stays alive.
    """
    import tempfile, shutil, concurrent.futures

    # Check extension as fallback — MIME type can be unreliable across browsers
    filename = file.filename or "upload.mp4"
    suffix   = Path(filename).suffix.lower()
    allowed  = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
    if suffix not in allowed and not (file.content_type or "").startswith("video/"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # Write upload to a temp file (UploadFile.file is sync SpooledTemporaryFile)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    def _extract_poses(path: str):
        """CPU-bound: runs in thread pool, safe to block here."""
        ext = PoseExtractor()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None, "Could not open video file."

        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sequence = []
        idx      = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks, _ = ext.process_frame(frame)
            sequence.append({"frame_idx": idx, "landmarks": landmarks})
            idx += 1

        cap.release()
        return {"fps": round(fps, 2), "total_frames": idx, "landmark_sequence": sequence}, None

    try:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result, err = await loop.run_in_executor(pool, _extract_poses, tmp_path)

        if err:
            raise HTTPException(status_code=422, detail=err)

        detected = sum(1 for f in result["landmark_sequence"] if f["landmarks"])
        print(f"[process_video] {filename}: {result['total_frames']} frames @ "
              f"{result['fps']} fps, {detected} with pose detected.")

        return JSONResponse(result)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")

    state.cap = cv2.VideoCapture(state.source)
    if not state.cap.isOpened():
        await websocket.send_text(json.dumps({"error": "Could not open camera"}))
        await websocket.close()
        return

    state.running     = True
    state.frame_count = 0

    try:
        while state.running:
            try:
                data    = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                command = json.loads(data)
                await handle_command(command, websocket)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

            ret, frame = state.cap.read()
            if not ret:
                if state.source != 0:
                    state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            landmarks, annotated = state.extractor.process_frame(frame)
            uncertainty_result   = state.scorer.score(landmarks)
            
            state.last_landmarks = landmarks
            state.frame_count   += 1

            # ── Action tick ──────────────────────────────────────────────────
            now = time.time()
            dt  = min(now - state._last_update_time, 0.05)
            state._last_update_time = now

            pose_transform = None
            detected = []
            if state.pose_mode:
                # Discrete: CROUCH / JUMP only - calibration happens inside if not yet done
                detected = state.pose_detector.detect_actions(landmarks)
                _sync_pose_actions(detected)
                # Continuous: rotation, X position, walking - only if calibrated
                if state.pose_detector.is_calibrated():
                    pose_transform = state.pose_detector.compute_pose_transform(landmarks)

            velocity, rotation_delta = state.action_ctrl.update(dt)
            state.action_ctrl.position += velocity * dt
            state.action_ctrl.rotation += rotation_delta

            # ── Encode frame ─────────────────────────────────────────────────
            _, buffer    = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            if state.recording:
                state.writer.add_frame(landmarks, trusted=uncertainty_result["trusted"])

            payload = {
                "frame":       state.frame_count,
                "recording":   state.recording,
                "image":       frame_base64,
                "bvh_frames":  len(state.writer.frames),
                "bvh_skipped": state.writer.skipped,
                "uncertainty": {
                    "aleatoric": uncertainty_result["aleatoric"],
                    "epistemic": uncertainty_result["epistemic"],
                    "total":     uncertainty_result["total"],
                    "trusted":   uncertainty_result["trusted"],
                },
                "landmarks":      landmarks if landmarks else None,
                "pose_transform":  pose_transform,
                "avatar":         state.action_ctrl.get_state_dict(),
                "detected_actions": [a.value for a in detected],
                # Calibration state for the frontend UI
                "pose_mode":      state.pose_mode,
                "calibrated":     state.pose_detector.is_calibrated(),
                "calib_progress": round(state.pose_detector.calibration_progress(), 3),
            }

            if state.frame_count % 100 == 0:
                print(
                    f"Frame {state.frame_count} | "
                    f"pose_mode={state.pose_mode} | "
                    f"calibrated={state.pose_detector.is_calibrated()}"
                )

            await websocket.send_text(json.dumps(payload))
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


def _sync_pose_actions(detected_actions):
    """
    Apply pose-detected actions into the ActionController.
    Only touches pose-owned actions — keyboard actions are left alone.
    """
    from actions import ActionType
    POSE_OWNED = {
        ActionType.MOVE_FORWARD, ActionType.MOVE_BACKWARD,
        ActionType.STRAFE_LEFT,  ActionType.STRAFE_RIGHT,
        ActionType.TURN_LEFT,    ActionType.TURN_RIGHT,
        ActionType.CROUCH,       ActionType.SPRINT,
        ActionType.JUMP,
    }
    detected_set = set(detected_actions)
    for action in POSE_OWNED:
        if action in detected_set:
            state.action_ctrl.activate_action(action)
        else:
            state.action_ctrl.deactivate_action(action)


async def handle_command(command: dict, websocket: WebSocket):
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
            "event":   "recording_stopped",
            "frames":  len(state.writer.frames),
            "skipped": state.writer.skipped,
        }))

    elif action == "save_bvh":
        state.recording = False
        filepath = state.writer.save()
        filename = Path(filepath).name if filepath else None
        print(f"BVH saved: {filename}")
        await websocket.send_text(json.dumps({
            "event":        "bvh_saved",
            "filename":     filename,
            "frames":       len(state.writer.frames),
            "skipped":      state.writer.skipped,
            "coverage_pct": round(
                len(state.writer.frames) /
                max(len(state.writer.frames) + state.writer.skipped, 1) * 100, 1
            ),
        }))

    elif action == "set_source":
        state.source = command.get("source", 0)
        if state.cap:
            state.cap.release()
        state.cap = cv2.VideoCapture(state.source)
        print(f"Source set to: {state.source}")

    elif action == "key_down":
        state.keyboard_ctrl.key_down(command.get("key", ""))

    elif action == "key_up":
        state.keyboard_ctrl.key_up(command.get("key", ""))

    elif action == "set_pose_mode":
        # Frontend toggles pose-driven control on/off
        state.pose_mode = bool(command.get("enabled", False))
        if state.pose_mode:
            print("Pose mode ON — calibrating..." if not state.pose_detector.is_calibrated() else "Pose mode ON.")
        else:
            print("Pose mode OFF — keyboard only.")
        await websocket.send_text(json.dumps({
            "event":     "pose_mode_changed",
            "pose_mode": state.pose_mode,
        }))

    elif action == "reset_calibration":
        state.pose_detector.reset_calibration()
        await websocket.send_text(json.dumps({"event": "calibration_reset"}))

    elif action == "get_coverage":
        stats = state.scorer.coverage_stats()
        await websocket.send_text(json.dumps({"event": "coverage_stats", "stats": stats}))


if __name__ == "__main__":
    print("Starting PosiSim backend...")
    print("Open http://localhost:8000 in your browser.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)