import os
import uuid
import base64
import random
import time
import shutil
from collections import deque
from uuid import uuid4

from io import BytesIO
import asyncio
import aiofiles
import httpx

from apscheduler.schedulers.background import BackgroundScheduler

from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw

import groundingdino.datasets.transforms as T

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

app = FastAPI()
app.router.redirect_slashes = False

# Global task registry to track ongoing tasks
task_registry = {}

TEMP_STORAGE = "temp_storage"
JOB_EXPIRATION_TIME = 600 # 10 minutes

SLAVE_ENDPOINTS = ["http://localhost:8001"]
# SLAVE_ENDPOINTS = ["http://localhost:8001", "http://localhost:8002", "http://localhost:8003", "http://localhost:8004"]
ENDPOINT_QUEUE = deque(SLAVE_ENDPOINTS)

MAX_REQUESTS_PER_ENDPOINT = 5 
MAX_SIZE = 1024

# Function to clear expired jobs
def clear_expired_jobs():
    current_time = time.time()
    job_folders = os.listdir(TEMP_STORAGE)
    for job_folder in job_folders:
        job_path = os.path.join(TEMP_STORAGE, job_folder)
        status_file_path = os.path.join(job_path, "status.txt")
        
        if os.path.exists(status_file_path):
            modification_time = os.path.getmtime(status_file_path)
            if (current_time - modification_time) > JOB_EXPIRATION_TIME:
                try:
                    shutil.rmtree(job_path)
                    print(f"Deleted expired job: {job_folder}")
                except Exception as e:
                    print(f"Error deleting job {job_folder}: {e}")

def generate_short_uuid():
    # Generate a UUID
    uuid_val = uuid4()

    # Convert the UUID to bytes
    uuid_bytes = uuid_val.bytes

    # Encode the bytes to Base64 and remove any padding
    short_uuid = base64.urlsafe_b64encode(uuid_bytes).rstrip(b'=').decode('utf-8')

    return short_uuid

def generate_color():
    """
    Generate a random color that avoids overly light or overly dark shades.
    This is achieved by limiting the RGB values to a certain range.
    """
    # Define the minimum and maximum range for RGB values to avoid light colors
    min_val = 50  # To avoid colors that are too dark
    max_val = 200  # To avoid colors that are too light

    return (
        random.randint(min_val, max_val),
        random.randint(min_val, max_val),
        random.randint(min_val, max_val),
    )

# Start the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(clear_expired_jobs, "interval", seconds=JOB_EXPIRATION_TIME)  # Run every hour
scheduler.start()

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

def draw_bounding_boxes(image, bounding_boxes, labels, colors_list):
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

        # Get the corresponding color for the current label
        label = labels[i]
 
        # If the label is not in the colors list, assign a random color to it
        if label not in colors_list:
            colors_list[label] = generate_color()
        
        color = colors_list[label]

        draw.rectangle([left, top, right, bottom], outline=color, width=3)

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
        "/usr/bin/ffmpeg",
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
    job_id = str(generate_short_uuid())
    job_folder = os.path.join(TEMP_STORAGE, job_id)
    os.makedirs(job_folder, exist_ok=True)

    job_status_file = os.path.join(job_folder, "status.txt")
    with open(job_status_file, "w") as f:
        f.write("processing")

    # Save the uploaded video to the job folder
    video_path = os.path.join(job_folder, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Create a frames folder in its job folder
    frames_folder = os.path.join(job_folder, "frames")
    os.makedirs(frames_folder, exist_ok=True)

    # Split the video into frames and save them in the frames folder
    extract_frames(video_path, frames_folder, fps)

    task = asyncio.create_task(process_video(job_id, frames_folder, text_prompt, box_threshold, text_threshold, fps))
    task_registry[job_id] = task

    return {"job_id": job_id}

async def process_video(job_id, frames_folder, text_prompt, box_threshold, text_threshold, fps):
    try:
        detections_file = os.path.join(TEMP_STORAGE, job_id, "detections.txt")
        frame_files = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        processed_frames_folder = os.path.join(TEMP_STORAGE, job_id, "processed_frames")
        os.makedirs(processed_frames_folder, exist_ok=True)

        max_failed_frames = int(0.1 * len(frame_files))  # Set the threshold for allowed failed frames
        failed_frames = 0  # Keep track of the number of failed frames

        colors_list = {}

        async with aiofiles.open(detections_file, "w") as f:
            tasks = []
            for i, frame_file in enumerate(frame_files):
                task = asyncio.ensure_future(
                    process_frame(i, frame_file, text_prompt, box_threshold, text_threshold, processed_frames_folder, f, colors_list)
                )
                tasks.append(task)

            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)  # Allows exceptions to be returned
            
            # Check for failed frames
            failed_frames = sum(1 for task in completed_tasks if isinstance(task, Exception))
            
            if failed_frames > max_failed_frames:
                # If failed frames exceed the threshold, set status to "failed"
                with open(os.path.join(TEMP_STORAGE, job_id, "status.txt"), "w") as status_file:
                    status_file.write("failed")
                return  # Exit early

        # If all tasks complete successfully, set status to "success"
        if failed_frames <= max_failed_frames:
            output_video_path = os.path.join(TEMP_STORAGE, job_id, f"{job_id}.mp4")
            frames_to_video(processed_frames_folder, fps, job_id, os.path.dirname(output_video_path))
            with open(os.path.join(TEMP_STORAGE, job_id, "status.txt"), "w") as status_file:
                status_file.write("success")

    except asyncio.CancelledError:
        print(f"Task for job {job_id} was cancelled")
        raise  # Re-raise to handle proper cancellation


async def process_frame(i, frame_file, text_prompt, box_threshold, text_threshold, processed_frames_folder, f, colors_list):
    try:
        # Infer the frame using the send_to_slave function
        result = await send_to_slave(frame_file, text_prompt, box_threshold, text_threshold)

        # Draw bounding boxes on the frame
        frame_image = Image.open(frame_file)
        processed_frame = draw_bounding_boxes(frame_image, result["bounding_boxes"], result["labels"], colors_list)

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

def get_job_status(job_id):
    job_folder = os.path.join(TEMP_STORAGE, job_id)
    job_status_file = os.path.join(job_folder, "status.txt")
    if not os.path.exists(job_folder) or not os.path.exists(job_status_file):
        return "failed"

    with open(job_status_file, "r") as f:
        status = f.read()

    return status

@app.delete("/delete_job/{job_id}")
async def delete_job(job_id: str):
    job_folder = f"temp_storage/{job_id}"

    # Stop the ongoing task if it's running
    if job_id in task_registry:
        task_registry[job_id].cancel()  # Cancel the task
        del task_registry[job_id]  # Remove from the registry
    
    # Delete the job folder
    try:
        shutil.rmtree(job_folder)  # Delete the folder and its contents
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting job: {e}")

@app.get("/check_job/{job_id}")
async def check_job(job_id: str):
    status = get_job_status(job_id)
    return {"status": status}

@app.get("/preview/{job_id}")
async def serve_video(job_id: str):
    video_path = os.path.join(f"{TEMP_STORAGE}/{job_id}", f"{job_id}.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")

# TODO New endpoint to packages detections with frames and vidoe file for download.