#!/usr/bin/env python3
"""
FastAPI server that mimics Kutt URL shortener API.
Instead of shortening URLs, it processes videos and returns the processed URL.
"""

import os
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from processor import VideoProcessor, init_log

# Load environment variables
load_dotenv()

app = FastAPI(title="Video Processor (Kutt-compatible)")

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=2)

# Log file path
LOG_PATH = Path(os.getenv("LOG_PATH", "processing_log.html"))

# Initialize processor
processor = VideoProcessor(
    assemblyai_api_key=os.getenv("ASSEMBLYAI_API_KEY"),
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    resemble_api_key=os.getenv("RESEMBLE_API_KEY"),
    resemble_voice_uuid=os.getenv("RESEMBLE_VOICE_UUID"),
    aws_access_key=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    s3_region=os.getenv("S3_REGION"),
    s3_bucket=os.getenv("S3_BUCKET"),
    s3_endpoint=os.getenv("S3_ENDPOINT"),
    log_path=LOG_PATH,
)

# Expected API key
EXPECTED_API_KEY = os.getenv("API_KEY", "your-secret-api-key-change-me")


class LinkRequest(BaseModel):
    target: str
    password: str | None = None  # Dropshare sends this but we ignore it


class LinkResponse(BaseModel):
    link: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to history page"""
    return await history()


@app.get("/history", response_class=HTMLResponse)
async def history():
    """Show processing history"""
    init_log(LOG_PATH)
    return HTMLResponse(content=LOG_PATH.read_text())


@app.post("/api/v2/links")
async def create_link(
    request: LinkRequest,
    x_api_key: str = Header(None, alias="X-API-KEY")
):
    """
    Kutt-compatible endpoint.
    Receives video URL, returns it immediately, processes in background.
    """
    # Verify API key
    if x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    video_url = request.target
    print(f"📨 Received request for: {video_url}")
    
    # Check if it's a video file
    if not any(video_url.lower().endswith(ext) for ext in ['.mp4', '.mov', '.webm', '.mkv']):
        # Not a video - just return original URL (for screenshots etc)
        print(f"   Not a video, returning original URL")
        return {"link": video_url}
    
    # Start background processing
    import threading
    thread = threading.Thread(target=process_video_background, args=(video_url,))
    thread.start()
    
    # Return original URL immediately
    print(f"   Returning original URL, processing in background...")
    return {"link": video_url}


def process_video_background(video_url: str):
    """Process video in background and overwrite original on S3"""
    try:
        print(f"🔄 Background processing started for: {video_url}")
        result = processor.process_and_overwrite(video_url)
        print(f"✅ Background processing complete: {result}")
    except Exception as e:
        print(f"❌ Background processing error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)