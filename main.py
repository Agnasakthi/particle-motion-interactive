
import os, uuid, base64
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Particle Motion Interactive")

@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")

def encode_f16(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float16)
    return base64.b64encode(arr.tobytes()).decode("ascii")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    allowed = {".mp4", ".webm", ".mov", ".m4v", ".avi"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, "Upload MP4/WebM/MOV/M4V/AVI.")

    job = uuid.uuid4().hex
    video_path = DATA_DIR / f"{job}{ext}"
    video_path.write_bytes(await file.read())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        video_path.unlink(missing_ok=True)
        raise HTTPException(400, "Could not decode this video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if total else 0
    # Keep analysis lightweight enough for a web service.
    sample_fps = min(20.0, max(6.0, fps))
    step = max(1, int(round(fps / sample_fps)))

    target_w, target_h = 96, 54
    prev = None
    flows = []
    luminance = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.2, flags=0
            )
            # Robust normalization prevents a few large vectors dominating.
            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
            scale = np.percentile(mag, 95)
            if scale < 1e-4:
                scale = 1.0
            flow = np.clip(flow / scale, -1.5, 1.5)
            flows.append(flow.astype(np.float16))
            luminance.append(gray.astype(np.uint8))

        prev = gray
        frame_idx += 1

    cap.release()
    video_path.unlink(missing_ok=True)

    if not flows:
        raise HTTPException(400, "The video needs at least two readable frames.")

    flow_arr = np.stack(flows)
    # Compress the visual motion into a small temporal representation.
    # Keep at most 120 field frames.
    if len(flow_arr) > 120:
        ids = np.linspace(0, len(flow_arr)-1, 120).astype(int)
        flow_arr = flow_arr[ids]

    out = DATA_DIR / f"{job}.json"
    payload = {
        "id": job,
        "width": target_w,
        "height": target_h,
        "frames": int(flow_arr.shape[0]),
        "fps": float(sample_fps),
        "flow": encode_f16(flow_arr),
        "format": "float16-interleaved-xy"
    }
    out.write_text(json.dumps(payload))
    return {"id": job, "width": target_w, "height": target_h, "frames": int(flow_arr.shape[0]), "fps": float(sample_fps)}

@app.get("/analysis/{job}")
def get_analysis(job: str):
    path = DATA_DIR / f"{job}.json"
    if not path.exists():
        raise HTTPException(404, "Analysis not found or expired.")
    return FileResponse(path, media_type="application/json")

app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
