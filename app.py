from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from uuid import uuid4
import os

from face_utils import analyze_face

app = FastAPI()

# Simple API key protection
API_KEY_NAME = "X-API-Key"
API_KEY = "imagica-IH096"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key"
        )

# Enable CORS for frontend usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


@app.post("/validate-face")
async def validate_face(
    file: UploadFile | None = File(None),
    _: str = Depends(verify_api_key)
):
    # No file uploaded
    if file is None or file.filename == "":
        raise HTTPException(status_code=400, detail="No image uploaded")

    # File type check
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images allowed"
        )

    # File size check
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Image must be under 10MB"
        )

    file.file.seek(0)

    # Save temporarily
    path = os.path.join(UPLOAD_DIR, f"{uuid4()}.{ext}")

    try:
        with open(path, "wb") as f:
            f.write(contents)

        success, message, yaw_angle, confidence = analyze_face(path)

    finally:
        if os.path.exists(path):
            os.remove(path)

    # Validation failed
    if not success:
        raise HTTPException(
            status_code=422,
            detail={
                "message": message,
                "yaw_angle": round(yaw_angle, 2),
                "confidence": confidence
            }
        )

    # Success
    return {
        "status": "success",
        "message": message,
        "yaw_angle": round(yaw_angle, 2),
        "confidence": confidence
    }
