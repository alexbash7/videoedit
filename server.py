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
    Receives video URL, processes it, returns processed URL.
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
    
    try:
        # Process video (this is blocking, run in thread pool)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, processor.process, video_url)
        
        return {"link": result["link"]}
    
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        # On error, return original URL so Dropshare doesn't fail
        return {"link": video_url}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)