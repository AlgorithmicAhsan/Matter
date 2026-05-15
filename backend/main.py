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
from starlette.middleware.base import BaseHTTPMiddleware

from pose import PoseExtractor
from uncertainty import UncertaintyScorer
from bvh_writer import BVHWriter
from actions import ActionController, ActionConfig, KeyboardController
from action_detector import PoseActionDetector          # NEW


# ── No-cache middleware: prevents browsers from serving stale JS/CSS ────────
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


app = FastAPI(title="Matter - Markerless Motion Capture")
app.add_middleware(NoCacheMiddleware)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
MODELS_DIR   = Path(__file__).parent.parent / "models"
OUTPUT_DIR   = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
app.mount("/models",   StaticFiles(directory=str(MODELS_DIR)),   name="models")
app.mount("/output",   StaticFiles(directory=str(OUTPUT_DIR)),   name="output")

CAMERA_SOURCE = 0

def _derive_locomotion_state(action_ctrl) -> str:
    actions = set(action_ctrl.active_actions.keys())
    from actions import ActionType
    
    if ActionType.JUMP in actions:
        return "jump"
    if ActionType.CROUCH in actions:
        if ActionType.MOVE_FORWARD in actions or ActionType.MOVE_BACKWARD in actions:
            return "crouch_walk"
        return "crouch"
    if ActionType.SPRINT in actions and ActionType.MOVE_FORWARD in actions:
        return "run"
    if ActionType.MOVE_BACKWARD in actions:
        return "walk_backward"
    if ActionType.TURN_LEFT in actions:
        return "turn_left"
    if ActionType.TURN_RIGHT in actions:
        return "turn_right"
    if ActionType.MOVE_FORWARD in actions:
        return "walk"
    return "idle"

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
        self.pose_mode     = True                      # NEW: toggled by frontend
        self._last_update_time = time.time()

state = SessionState()


@app.get("/")
async def root():
    response = FileResponse(str(FRONTEND_DIR / "index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/health")
async def health():
    return {"status": "ok", "recording": state.recording, "frames": len(state.writer.frames)}

@app.get("/output/list")
async def list_outputs():
    files = list(OUTPUT_DIR.glob("*.bvh"))
    return {"files": [f.name for f in sorted(files, reverse=True)]}


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
            import base64
            _, buffer    = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            if state.recording:
                state.writer.add_frame(landmarks, trusted=uncertainty_result["trusted"])

            payload = {
                "frame":       state.frame_count,
                "recording":   state.recording,
                "image":       frame_base64,
                "active_actions":   [a.value for a in state.action_ctrl.active_actions.keys()],
                "locomotion_state": _derive_locomotion_state(state.action_ctrl),
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
    print("Starting Matter backend...")
    print("Open http://localhost:8000 in your browser.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)