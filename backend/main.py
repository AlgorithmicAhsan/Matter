import asyncio
import cv2
import json
import os
import time
import base64
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pose import PoseExtractor
from uncertainty import UncertaintyScorer
from bvh_writer import BVHWriter

app = FastAPI(title="PosiSim - Markerless Motion Capture")

frontendDir = Path(__file__).parent.parent / "frontend"
modelsDir   = Path(__file__).parent.parent / "models"
outputDir   = Path(__file__).parent.parent / "output"
outputDir.mkdir(exist_ok=True)

app.mount("/frontend", StaticFiles(directory=str(frontendDir)), name="frontend")
app.mount("/models",   StaticFiles(directory=str(modelsDir)),   name="models")
app.mount("/output",   StaticFiles(directory=str(outputDir)),   name="output")

cameraSource = 0


class SessionState:
    def __init__(self):
        self.extractor  = PoseExtractor()
        self.scorer     = UncertaintyScorer()
        self.writer     = BVHWriter(fps=30)
        self.cap        = None
        self.recording  = False
        self.running    = False
        self.source     = cameraSource
        self.frameCount = 0


state = SessionState()


@app.get("/")
async def root():
    return FileResponse(str(frontendDir / "index.html"))

@app.get("/health")
async def health():
    return {"status": "ok", "recording": state.recording, "frames": len(state.writer.frames)}

@app.get("/output/list")
async def listOutputs():
    files = list(outputDir.glob("*.bvh"))
    return {"files": [f.name for f in sorted(files, reverse=True)]}


@app.websocket("/ws")
async def websocketEndpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")

    state.cap = cv2.VideoCapture(state.source)
    if not state.cap.isOpened():
        await websocket.send_text(json.dumps({"error": "Could not open camera"}))
        await websocket.close()
        return

    state.running    = True
    state.frameCount = 0

    try:
        while state.running:
            try:
                data    = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                command = json.loads(data)
                await handleCommand(command, websocket)
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

            resultDict, annotated = state.extractor.process_frame(frame)

            # Uncertainty scoring uses pose2d landmarks (same format as before)
            pose2d = resultDict.get("pose2d")
            uncertaintyResult = state.scorer.score(pose2d)

            state.frameCount += 1

            # BVH recording still uses pose2d landmarks
            if state.recording:
                state.writer.add_frame(pose2d, trusted=uncertaintyResult["trusted"])

            _, buffer    = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frameBase64  = base64.b64encode(buffer).decode("utf-8")

            payload = {
                "frame":      state.frameCount,
                "recording":  state.recording,
                "image":      frameBase64,
                "bvhFrames":  len(state.writer.frames),
                "bvhSkipped": state.writer.skipped,
                "uncertainty": {
                    "aleatoric": uncertaintyResult["aleatoric"],
                    "epistemic": uncertaintyResult["epistemic"],
                    "total":     uncertaintyResult["total"],
                    "trusted":   uncertaintyResult["trusted"],
                },
                # All landmark streams for Kalidokit in browser
                "pose2d":   resultDict["pose2d"],
                "pose3d":   resultDict["pose3d"],
                "face":     resultDict["face"],
                "leftHand": resultDict["leftHand"],
                "rightHand":resultDict["rightHand"],
            }

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


async def handleCommand(command: dict, websocket: WebSocket):
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
            "event":       "bvh_saved",
            "filename":    filename,
            "frames":      len(state.writer.frames),
            "skipped":     state.writer.skipped,
            "coveragePct": round(
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


if __name__ == "__main__":
    print("Starting PosiSim backend...")
    print("Open http://localhost:8000 in your browser.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
