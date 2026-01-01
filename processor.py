#!/usr/bin/env python3
"""
Video processor - removes fillers, enhances audio, applies voice conversion.
"""

import subprocess
import os
import re
import json
import time
import tempfile
import requests
import boto3
import base64
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, unquote
import threading

# ============== FILLER WORDS ==============
FILLER_WORDS = {"uh", "um", "ah", "er", "eh", "hmm", "hm", "mhm", "erm", "umm", "uhh", "ahh"}

# ============== HTML LOG ==============
LOG_LOCK = threading.Lock()

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Processing Log</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; margin-bottom: 20px; }
        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stat-card .value { font-size: 24px; font-weight: bold; color: #4a90d9; }
        .stat-card .label { color: #666; font-size: 14px; }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        th { 
            background: #4a90d9; 
            color: white; 
            padding: 12px 15px; 
            text-align: left;
            font-weight: 600;
        }
        td { 
            padding: 12px 15px; 
            border-bottom: 1px solid #eee; 
        }
        tr:hover { background: #f8f9fa; }
        tr:last-child td { border-bottom: none; }
        .link-btn {
            background: #4a90d9;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            margin-right: 5px;
            text-decoration: none;
            display: inline-block;
        }
        .link-btn:hover { background: #357abd; }
        .copy-btn {
            background: #6c757d;
            color: white;
            border: none;
            padding: 6px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .copy-btn:hover { background: #545b62; }
        .copy-btn.copied { background: #28a745; }
        .duration { color: #28a745; font-weight: 500; }
        .cost { color: #6c757d; }
        .actions { white-space: nowrap; }
        .empty { text-align: center; padding: 40px; color: #666; }
    </style>
</head>
<body>
    <h1>📹 Video Processing Log</h1>
    <div class="stats">
        <div class="stat-card">
            <div class="value" id="total-videos">0</div>
            <div class="label">Videos Processed</div>
        </div>
        <div class="stat-card">
            <div class="value" id="total-cost">$0.00</div>
            <div class="label">Total Cost</div>
        </div>
        <div class="stat-card">
            <div class="value" id="time-saved">0:00</div>
            <div class="label">Time Saved</div>
        </div>
    </div>
    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Title</th>
                <th>Duration</th>
                <th>Cost</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody id="log-body">
<!-- ROWS -->
        </tbody>
    </table>
    <script>
        function copyUrl(url, btn) {
            navigator.clipboard.writeText(url).then(() => {
                btn.textContent = '✓';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = '📋';
                    btn.classList.remove('copied');
                }, 1500);
            });
        }
    </script>
</body>
</html>"""

ROW_TEMPLATE = """        <tr>
            <td>{date}</td>
            <td>{title} 🎤</td>
            <td class="duration">{orig} → {final} (-{reduction}%)</td>
            <td class="cost">${cost:.2f}</td>
            <td class="actions">
                <a href="{url}" target="_blank" class="link-btn">▶ Open</a>
                <button class="copy-btn" onclick="copyUrl('{url}', this)">📋</button>
            </td>
        </tr>
"""


def init_log(log_path: Path):
    """Initialize log file if it doesn't exist"""
    if not log_path.exists():
        log_path.write_text(HTML_TEMPLATE)


def append_log(log_path: Path, title: str, original_duration: float, final_duration: float, cost: float, s3_url: str):
    """Append entry to HTML log"""
    with LOG_LOCK:
        init_log(log_path)
        
        date = datetime.now().strftime("%Y-%m-%d %H:%M")
        orig = f"{int(original_duration // 60)}:{int(original_duration % 60):02d}"
        final = f"{int(final_duration // 60)}:{int(final_duration % 60):02d}"
        reduction = int((1 - final_duration / original_duration) * 100) if original_duration > 0 else 0
        
        row = ROW_TEMPLATE.format(
            date=date,
            title=title,
            orig=orig,
            final=final,
            reduction=reduction,
            cost=cost,
            url=s3_url
        )
        
        content = log_path.read_text()
        content = content.replace("<!-- ROWS -->", row + "<!-- ROWS -->")
        log_path.write_text(content)


class VideoProcessor:
    def __init__(
        self,
        assemblyai_api_key: str,
        openrouter_api_key: str,
        resemble_api_key: str,
        resemble_voice_uuid: str,
        aws_access_key: str,
        aws_secret_key: str,
        s3_region: str,
        s3_bucket: str,
        s3_endpoint: str,
        log_path: Path = None,
    ):
        self.log_path = log_path or Path("processing_log.html")
        self.assemblyai_api_key = assemblyai_api_key
        self.openrouter_api_key = openrouter_api_key
        self.resemble_api_key = resemble_api_key
        self.resemble_voice_uuid = resemble_voice_uuid
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.s3_region = s3_region
        self.s3_bucket = s3_bucket
        self.s3_endpoint = s3_endpoint

    def log(self, msg: str):
        print(msg)

    def get_video_duration(self, path: str) -> float:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())

    def format_duration(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def get_s3_client(self):
        return boto3.client(
            's3',
            endpoint_url=self.s3_endpoint,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.s3_region
        )

    def download_from_url(self, url: str, local_path: str, max_retries: int = 5, retry_delay: int = 3) -> bool:
        self.log(f"📥 Downloading video...")
        
        for attempt in range(max_retries):
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            elif response.status_code == 404 and attempt < max_retries - 1:
                self.log(f"   ⏳ File not ready yet, waiting {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                self.log(f"   ❌ Download failed with status {response.status_code}")
                return False
        
        return False

    def upload_to_s3(self, local_path: str, s3_key: str, content_type: str = "video/mp4") -> str:
        self.log(f"📤 Uploading to S3...")
        s3 = self.get_s3_client()
        s3.upload_file(
            local_path,
            self.s3_bucket,
            s3_key,
            ExtraArgs={'ACL': 'public-read', 'ContentType': content_type}
        )
        return f"{self.s3_endpoint}/{self.s3_bucket}/{s3_key}"

    def upload_temp_audio_to_s3(self, local_path: str) -> str:
        timestamp = int(time.time())
        s3_key = f"temp/resemble_{timestamp}.mp3"
        return self.upload_to_s3(local_path, s3_key, content_type="audio/mpeg")

    def delete_temp_from_s3(self, s3_url: str):
        try:
            s3 = self.get_s3_client()
            key = s3_url.replace(f"{self.s3_endpoint}/{self.s3_bucket}/", "")
            s3.delete_object(Bucket=self.s3_bucket, Key=key)
        except Exception as e:
            self.log(f"   ⚠️ Could not delete temp file: {e}")

    def extract_audio(self, video_path: str, audio_path: str):
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "libmp3lame",
            "-ar", "44100", "-ac", "1", "-q:a", "2", audio_path
        ]
        subprocess.run(cmd, capture_output=True)

    def transcribe_with_fillers(self, audio_path: str, speech_model: str = "best") -> dict:
        headers = {"authorization": self.assemblyai_api_key}

        with open(audio_path, 'rb') as f:
            response = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers=headers,
                data=f
            )
        audio_url = response.json()["upload_url"]

        config = {
            "audio_url": audio_url,
            "disfluencies": True,
            "speech_model": speech_model
        }

        response = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            headers=headers,
            json=config
        )
        transcript_id = response.json()["id"]

        while True:
            response = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                headers=headers
            )
            data = response.json()
            if data["status"] == "completed":
                return data
            elif data["status"] == "error":
                raise Exception(f"Transcription failed: {data.get('error')}")
            time.sleep(2)

    def find_fillers(self, words: list) -> list:
        fillers = []
        for w in words:
            normalized = re.sub(r'[^\w]', '', w["text"].lower())
            if normalized in FILLER_WORDS:
                fillers.append(w)
        return fillers

    def calculate_clean_segments(self, words: list, fillers: list, total_duration: float, padding: float = 0.03) -> list:
        if not fillers:
            return [(0, total_duration)]

        filler_ranges = []
        for f in fillers:
            start = max(0, f["start"] / 1000 - padding)
            end = f["end"] / 1000 + padding
            filler_ranges.append((start, end))

        filler_ranges.sort()
        merged = [filler_ranges[0]]
        for start, end in filler_ranges[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        clean = []
        prev_end = 0
        for start, end in merged:
            if start > prev_end:
                clean.append((prev_end, start))
            prev_end = end
        if prev_end < total_duration:
            clean.append((prev_end, total_duration))

        return clean

    def cut_video_segments(self, input_path: str, segments: list, output_path: str):
        if not segments:
            return

        filter_parts = []
        concat_inputs = []

        for i, (start, end) in enumerate(segments):
            filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}];")
            filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];")
            concat_inputs.append(f"[v{i}][a{i}]")

        filter_complex = "".join(filter_parts)
        filter_complex += f"{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=1[outv][outa]"

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ]

        subprocess.run(cmd, capture_output=True)

    def remove_fillers_pass(self, video_path: str, output_path: str, pass_name: str, speech_model: str = "best") -> tuple:
        self.log(f"🔍 {pass_name}: transcribing...")

        audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        self.extract_audio(video_path, audio_path)

        duration = self.get_video_duration(video_path)
        cost = (duration / 60) * 0.0062

        transcript = self.transcribe_with_fillers(audio_path, speech_model)
        words = transcript.get("words", [])

        fillers = self.find_fillers(words)
        self.log(f"   Found fillers: {len(fillers)}")

        if fillers:
            segments = self.calculate_clean_segments(words, fillers, duration)
            self.log(f"   Cutting fillers...")
            self.cut_video_segments(video_path, segments, output_path)
        else:
            subprocess.run(["cp", video_path, output_path], capture_output=True)

        os.unlink(audio_path)

        return len(fillers), cost, transcript

    def apply_studio_sound(self, input_path: str, output_path: str):
        self.log(f"🎚️ Applying studio sound...")

        audio_filter = (
            "highpass=f=80,"
            "lowpass=f=12000,"
            "compand=attacks=0.2:decays=0.5:points=-70/-70|-30/-15|-20/-10|0/-6,"
            "volume=0.7"
        )

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", audio_filter,
            "-c:v", "copy",
            output_path
        ]

        subprocess.run(cmd, capture_output=True)

    def apply_resemble_voice(self, input_video: str, output_video: str) -> float:
        self.log(f"🎤 Applying Resemble AI voice conversion...")
        self.log(f"   Voice: {self.resemble_voice_uuid}")

        audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        self.extract_audio(input_video, audio_path)

        audio_duration = self.get_video_duration(audio_path)
        self.log(f"   Audio duration: {self.format_duration(audio_duration)}")

        if audio_duration > 300:
            self.log(f"   ⚠️ Audio > 5 min, Resemble will trim to 5 min!")

        self.log(f"   Uploading audio to S3...")
        audio_url = self.upload_temp_audio_to_s3(audio_path)
        self.log(f"   URL: {audio_url}")

        self.log(f"   Sending to Resemble AI...")
        response = requests.post(
            "https://f.cluster.resemble.ai/synthesize",
            headers={
                "Authorization": f"Bearer {self.resemble_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "voice_uuid": self.resemble_voice_uuid,
                "data": f'<resemble:convert src="{audio_url}"></resemble:convert>',
                "sample_rate": 44100,
                "output_format": "mp3"
            },
            timeout=300
        )

        data = response.json()

        if not data.get("success"):
            raise Exception(f"Resemble AI error: {data}")

        self.log(f"   ✅ Got response from Resemble AI")

        new_audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        audio_bytes = base64.b64decode(data["audio_content"])
        with open(new_audio_path, "wb") as f:
            f.write(audio_bytes)

        self.log(f"   Replacing audio in video...")
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-i", new_audio_path,
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_video
        ]
        subprocess.run(cmd, capture_output=True)

        os.unlink(audio_path)
        os.unlink(new_audio_path)
        self.delete_temp_from_s3(audio_url)

        cost = (audio_duration / 60) * 0.03
        self.log(f"   💰 Resemble cost: ${cost:.3f}")

        return cost

    def generate_title(self, transcript_text: str) -> str:
        self.log(f"📝 Generating title...")

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }

        prompt = f"""This is a transcript of a tutorial video. Create a short title (up to 50 characters) in English.
Format: just the title, no quotes or explanations.

Transcript (first 500 chars):
{transcript_text[:500]}"""

        payload = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        return "Untitled Tutorial"

    def process(self, url: str) -> dict:
        """
        Main processing function.
        Returns dict with 'link' field for Kutt-compatible response.
        """
        self.log(f"\n{'='*60}")
        self.log(f"🎬 PROCESSING VIDEO")
        self.log(f"{'='*60}\n")

        total_cost = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            original_name = unquote(urlparse(url).path.split("/")[-1])
            input_video = tmpdir / original_name

            if not self.download_from_url(url, str(input_video)):
                raise Exception("Failed to download video")

            original_duration = self.get_video_duration(str(input_video))
            self.log(f"   Duration: {self.format_duration(original_duration)}")

            # Pass 1 - remove fillers
            pass1_output = tmpdir / "pass1.mp4"
            fillers1, cost1, transcript1 = self.remove_fillers_pass(
                str(input_video), str(pass1_output), "Pass 1", "best"
            )
            total_cost += cost1

            # Pass 2 - catch remaining fillers
            pass2_output = tmpdir / "pass2.mp4"
            fillers2, cost2, _ = self.remove_fillers_pass(
                str(pass1_output), str(pass2_output), "Pass 2", "best"
            )
            total_cost += cost2

            # Apply studio sound
            studio_output = tmpdir / "studio.mp4"
            self.apply_studio_sound(str(pass2_output), str(studio_output))

            # Apply Resemble AI voice conversion
            resemble_output = tmpdir / "resemble.mp4"
            resemble_cost = self.apply_resemble_voice(str(studio_output), str(resemble_output))
            total_cost += resemble_cost

            final_duration = self.get_video_duration(str(resemble_output))

            # Generate title
            title = self.generate_title(transcript1.get("text", ""))
            total_cost += 0.001

            # Create safe filename
            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:50]
            date_prefix = datetime.now().strftime("%Y-%m-%d")
            final_name = f"{date_prefix}_{safe_title}.mp4"

            # Upload to S3
            s3_key = f"processed/{final_name}"
            s3_url = self.upload_to_s3(str(resemble_output), s3_key)

            reduction = int((1 - final_duration / original_duration) * 100)

            # Append to HTML log
            append_log(self.log_path, title, original_duration, final_duration, total_cost, s3_url)

            self.log(f"\n{'='*60}")
            self.log(f"✅ {title}")
            self.log(f"📎 {s3_url}")
            self.log(f"💰 ${total_cost:.2f} | {self.format_duration(original_duration)} → {self.format_duration(final_duration)} (-{reduction}%)")
            self.log(f"{'='*60}\n")

            return {
                "link": s3_url,
                "title": title,
                "original_duration": original_duration,
                "final_duration": final_duration,
                "reduction_percent": reduction,
                "cost": total_cost
            }