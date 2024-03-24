import os
import uuid
import base64
import shutil
import asyncio
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from typing import List
from aiohttp import ClientSession
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from io import BytesIO
import aiofiles
import httpx
import time
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw
import cv2
from collections import deque

from PIL import Image, ImageFilter

import torch
import groundingdino.datasets.transforms as T

app = FastAPI()
app.router.redirect_slashes = False

# Set up CORS middleware options
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.51:3000",
        "*"
        ], # Whitelist
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>G-DINO Master</title>
        </head>
        <body>
            <h1>GroundingDINO Demo Master</h1>
        </body>
    </html>
    """

TEMP_STORAGE = "temp_storage"
# SLAVE_ENDPOINTS = ["http://localhost:8001"]
SLAVE_ENDPOINTS = ["http://localhost:8001", "http://localhost:8002", "http://localhost:8003", "http://localhost:8004"]
ENDPOINT_QUEUE = deque(SLAVE_ENDPOINTS)

MAX_REQUESTS_PER_ENDPOINT = 5 
MAX_SIZE = 1024

async def get_next_endpoint():
    while True:
        if ENDPOINT_QUEUE:
            endpoint = ENDPOINT_QUEUE.popleft()
            return endpoint
        else:
            # Wait for an endpoint to become available
            await asyncio.sleep(0.1)

async def release_endpoint(endpoint):
    ENDPOINT_QUEUE.append(endpoint)

def extract_frames(video_path, output_folder, fps):
    """
    Extracts frames from a video file at a specified fps.
    """
    # Ensure the output folder exists, clear it if it does
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Load the video file
    clip = VideoFileClip(video_path)
    print(f"Original duration: {clip.duration}")

    max_width = 1024
    max_height = 1024

    # Calculate the aspect ratio of the video
    aspect_ratio = clip.w / clip.h
    
    # Determine the new dimensions based on the max_width and max_height
    if clip.w > max_width or clip.h > max_height:
        if aspect_ratio > 1:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
    else:
        new_width = clip.w
        new_height = clip.h
    
    # Resize the video while maintaining the aspect ratio
    resized_clip = clip.resize((new_width, new_height))
    print(f"Duration: {resized_clip.duration}")

    total_frames = int(resized_clip.duration * fps)

    print(f"Clip resized: {resized_clip.w} * {resized_clip.h}")
    start_time = time.perf_counter()
    for i in range(total_frames):
        # Extract the frame at the desired timestamp
        frame = resized_clip.get_frame(i / fps)

        # Save each frame as an image
        frame_image = Image.fromarray(frame)
        frame_image.save(os.path.join(output_folder, f"{i+1:08d}.jpg"))

    print(f"Extracted {total_frames} frames to {output_folder}, {time.perf_counter() - start_time:.3f}s elapsed")

    return output_folder

def draw_bounding_boxes(image, bounding_boxes, labels=None):
    """
    Draw bounding boxes on an image. The bounding boxes are expected to be in the format
    (x_min, y_min, x_max, y_max), normalized to [0, 1].
    """
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    for i, box in enumerate(bounding_boxes):
        x_center, y_center, width, height = box
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        left = x_min * img_width
        top = y_min * img_height
        right = x_max * img_width
        bottom = y_max * img_height
        draw.rectangle([left, top, right, bottom], outline="blue", width=2)

    return image

async def resize_image(image_file):

    if isinstance(image_file, str):
        # If image_file is a path, open the file and read its content
        with open(image_file, "rb") as f:
            image_data = f.read()
    else:
        # If image_file is a file object, read its content
        image_data = await image_file.read()

    try:
        # Resize the image while preserving the aspect ratio
        image = Image.open(BytesIO(image_data))
        width, height = image.size

        if width > height:
            new_width = MAX_SIZE
            new_height = int(height * (MAX_SIZE / width))
        else:
            new_height = MAX_SIZE
            new_width = int(width * (MAX_SIZE / height))

        resized_image = image.resize((new_width, new_height))

        resized_image_bytes = BytesIO()
        resized_image.save(resized_image_bytes, format='PNG')
        resized_image_bytes = base64.b64encode(resized_image_bytes.getvalue()).decode('utf-8')

        return resized_image_bytes

    except Exception as e:
        print(f"Error resizing image: {str(e)}")
        return None


import os
import subprocess

def frames_to_video(frames_path, fps, job_id, output_folder):
    """Compiles images from a specified folder into a video file at a given fps."""
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    output_file = f"{output_folder}/{job_id}.mp4"

    ffmpeg_command = [
        "/usr/local/bin/ffmpeg",
        "-framerate", str(fps),
        "-i", f"{frames_path}/%08d.jpg",
        "-c:v", "libx264",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_file,
        "-loglevel", "error",
        "-hide_banner",
        "-nostats"
    ]

    try:
        result = subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        print(f"Video saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg command failed with error code: {e.returncode}")
        print(f"Error output: {e.stderr}")


async def send_to_slave(image_file, text_prompt, box_threshold, text_threshold):
    endpoint = await get_next_endpoint()

    async with httpx.AsyncClient() as client:
        resized_image_bytes = await resize_image(image_file)

        if resized_image_bytes is None:
            # Return an error response as JSON
            error_response = {"error": "Failed to resize image"}
            return error_response

        payload = {
            "image": resized_image_bytes,
            "text_prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
        }

        headers = {"X-Forwarded-Prefix": "/"}
        response = await client.post(f"{endpoint}/detect", json=payload, headers=headers)

        await release_endpoint(endpoint)
        return response.json()

@app.post("/image_inference")
async def detect_objects(image: UploadFile = File(...), text_prompt: str = Form(...), 
                         box_threshold: float = Form(0.3), text_threshold: float = Form(0.3)):

    start_time = time.perf_counter()
    # image_data = await image.read()
    # _, image_tensor = load_image(image_data)

    text_prompt = text_prompt.lower() # Convert to lower case

    result = await send_to_slave(image, text_prompt, box_threshold, text_threshold)
    request_time = f"{time.perf_counter() - start_time:.3f}"
    print(f"Request took {request_time}s")

    if "error" in result:
        # Image resizing failed, return the error response
        return JSONResponse(content=result, status_code=400)

    # TODO Check IOU? If IOU is too large, keep the highest confidence (Solves issue where the model thinks a same object is two different things)

    return result

@app.post("/video_inference")
async def video_inference(video: UploadFile = File(...), text_prompt: str = Form(...), 
                         box_threshold: float = Form(0.3), text_threshold: float = Form(0.3),
                         fps: int = Form(6)):
    print(f"Receive video request")
    # Generate a uuid as the job id, create a folder in 'temp_storage'
    job_id = str(uuid.uuid4())
    job_folder = os.path.join(TEMP_STORAGE, job_id)
    os.makedirs(job_folder, exist_ok=True)

    # Save the uploaded video to the job folder
    video_path = os.path.join(job_folder, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Create a frames folder in its job folder
    frames_folder = os.path.join(job_folder, "frames")
    os.makedirs(frames_folder, exist_ok=True)

    # Split the video into frames and save them in the frames folder
    extract_frames(video_path, frames_folder, fps)

    # Call process_video
    preview_url, download_url = await process_video(job_id, frames_folder, text_prompt, box_threshold, text_threshold, fps)

    return {"preview_url": preview_url, "download_url": download_url, "error": "Placeholder"}

    # return JSONResponse(content=error_response, status_code=400)

async def process_video(job_id, frames_folder, text_prompt, box_threshold, text_threshold, fps):
    # Create detections.txt in job folder
    detections_file = os.path.join(TEMP_STORAGE, job_id, "detections.txt")

    # Load frames
    frame_files = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Infer each frame with load balancing
    processed_frames_folder = os.path.join(TEMP_STORAGE, job_id, "processed_frames")
    os.makedirs(processed_frames_folder, exist_ok=True)

    async with aiofiles.open(detections_file, "w") as f:
        tasks = []
        for i, frame_file in enumerate(frame_files):
            task = asyncio.ensure_future(process_frame(i, frame_file, text_prompt, box_threshold, text_threshold, processed_frames_folder, f))
            tasks.append(task)

        await asyncio.gather(*tasks)

    # Piece frames with bounding boxes into video
    output_video_path = os.path.join(TEMP_STORAGE, job_id, f"{job_id}.mp4")
    frames_to_video(processed_frames_folder, fps, job_id, os.path.dirname(output_video_path))

    # Return preview url and download url
    preview_url = f"/temp_storage/{job_id}/{job_id}.mp4"
    download_url = f"/temp_storage/{job_id}"
    return preview_url, download_url

async def process_frame(i, frame_file, text_prompt, box_threshold, text_threshold, processed_frames_folder, f):
    try:
        # Infer the frame using the send_to_slave function
        result = await send_to_slave(frame_file, text_prompt, box_threshold, text_threshold)

        # Draw bounding boxes on the frame
        frame_image = Image.open(frame_file)
        processed_frame = draw_bounding_boxes(frame_image, result["bounding_boxes"], result["labels"])

        # Save the processed frame
        processed_frame_file = os.path.join(processed_frames_folder, f"{i+1:08d}.jpg")
        processed_frame.save(processed_frame_file)

        # Write detections to detections.txt
        for j, box in enumerate(result["bounding_boxes"]):
            label = result["labels"][j] if result["labels"] else ""
            x_min, y_min, x_max, y_max = box
            confidence_score = result["confidence"][j]
            await f.write(f"{i+1},{label},{x_min},{y_min},{x_max},{y_max},{confidence_score}\n")

    except Exception as e:
        print(f"Error processing frame {i+1}: {str(e)}")
        # Assume no detections for this frame

@app.get("/preview/{job_id}")
async def serve_video(job_id: str):
    video_path = os.path.join(f"{TEMP_STORAGE}/{job_id}", f"{job_id}.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")