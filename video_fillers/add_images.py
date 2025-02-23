#input_transcription_of_audio = X_transcription
#input_audio = X_audio_with_music

#1) Get the generated topics from identified_topics.json
#1) For each topic, browse the internet for relevant images and choose one
#4) Add the images to the X_audio and put them together to generate a mp4
#5) Export X_audio_with_music_and_images

#output = X_audio_with_music_and_images

from typing import List, Dict
from dataclasses import dataclass
import os
import json
import requests
from dotenv import load_dotenv
import subprocess
import tempfile
from PIL import Image
import io
import numpy as np
import fal_client  # New import for AI image generation using Fal AI
import time
from functools import wraps
import logging

# Image parameters for vertical social media format (9:16 aspect ratio)
IMAGE_WIDTH = 1080  # Width for vertical video
IMAGE_HEIGHT = 1920  # Height for vertical video (9:16 aspect ratio)
TRANSITION_DURATION = 1.0  # Duration of fade transition between images in seconds
MIN_IMAGE_DURATION = 3.0  # Minimum duration to show each image

# Optimization constants
TARGET_FPS = 120  # Increased FPS for smoother transitions
ENCODING_PRESET = 'ultrafast'  # Fastest encoding preset
ENCODING_THREADS = 2  # Number of threads for encoding
AUDIO_FPS = 44100  # Standard audio sampling rate

# Update constants
FFMPEG_PRESET = 'ultrafast'  # FFmpeg encoding preset
VIDEO_BITRATE = '4M'  # Target video bitrate
AUDIO_BITRATE = '192k'  # Target audio bitrate

# Ken Burns effect parameters
ZOOM_START_SCALE = 1.0
ZOOM_END_SCALE = 1.05  # Reduced from 1.3 for subtler effect
PAN_SPEED = 0.000000005   # Same pan speed for refined smoothness at higher FPS
ZOOM_SPEED = 0.0015       # Same zoom speed for refined smoothness at higher FPS
INTERPOLATION_FACTOR = 2.0  # Increased from 1.5 to generate more intermediate frames for smoother transition

# Prompt templates for easy modification
IMAGE_GENERATION_PROMPT_TEMPLATE = (
    "Generate a high quality, artistic image representing the concept of '{topic_name}' "
    "depicted with elements that allude to the words: '{topic_words}' "
    "in a vertical format suitable for social media. Use vivid colors, detailed textures, "
    "and modern design elements. Do not include any text in the image."
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def timer_decorator(func):
    """Decorator to measure and log the execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"{func.__name__} took {duration:.2f} seconds to execute")
        return result
    return wrapper

@dataclass
class TopicImage:
    """Represents an image associated with a topic"""
    topic_name: str
    topic_index: int  # Added to track duplicate topics
    image_url: str
    start_time: float
    end_time: float
    local_path: str = ""

def find_env_file(start_path: str) -> str:
    """
    Find the .env file by traversing up the directory tree.
    
    Args:
        start_path (str): Starting directory path
        
    Returns:
        str: Path to the found .env file
        
    Raises:
        FileNotFoundError: If .env file cannot be found
    """
    current_path = os.path.abspath(start_path)
    while current_path != '/':
        env_path = os.path.join(current_path, '.env')
        print(f"[DEBUG] Checking for .env at: {env_path}")
        if os.path.isfile(env_path):
            return env_path
        current_path = os.path.dirname(current_path)
    raise FileNotFoundError("Could not find .env file in any parent directory")

@timer_decorator
def load_topics() -> List[tuple[str, List[str]]]:
    """
    Load topics from identified_topics.json file and preserve order.
    
    Returns:
        List[tuple[str, List[str]]]: List of (topic_name, topic_words) pairs ordered as in file
        
    Raises:
        FileNotFoundError: If topics file doesn't exist
    """
    topics_file = "identified_topics.json"
    print(f"[DEBUG] load_topics: Loading topics from {topics_file}")
    
    assert os.path.exists(topics_file), f"Topics file not found: {topics_file}"
    
    with open(topics_file, 'r') as f:
        # Load as OrderedDict to preserve file order
        topics_data = json.load(f, object_pairs_hook=lambda pairs: pairs)
    
    assert isinstance(topics_data, list), "Topics must be loaded as ordered pairs"
    print(f"[DEBUG] load_topics: Loaded {len(topics_data)} topics in order")
    
    # Return the ordered list of tuples directly
    return topics_data

@timer_decorator
def download_image(url: str, topic_name: str, topic_index: int) -> str:
    """
    Download image from URL, process it to fit target dimensions, and save locally.
    
    Args:
        url (str): URL of the image to download
        topic_name (str): Name of the topic for file naming
        topic_index (int): Index to handle duplicate topics
        
    Returns:
        str: Path to the saved image
        
    The function will:
    1. Download the image
    2. Calculate the best crop to maintain aspect ratio
    3. Center the crop on the most interesting part of the image
    4. Scale to target dimensions
    """
    print(f"[DEBUG] download_image: Downloading image for topic '{topic_name}' (index: {topic_index})")
    
    images_dir = "topic_images"
    os.makedirs(images_dir, exist_ok=True)
    
    safe_name = ''.join(c if c.isalnum() else '_' for c in topic_name)
    image_path = f"{images_dir}/{safe_name}_{topic_index}.jpg"
    
    # Download image
    response = requests.get(url)
    response.raise_for_status()
    
    # Open image
    img = Image.open(io.BytesIO(response.content))
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Calculate target aspect ratio
    target_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
    
    # Get current image dimensions
    img_width, img_height = img.size
    img_ratio = img_width / img_height
    
    if img_ratio > target_ratio:
        # Image is wider than target ratio - crop width
        new_width = int(img_height * target_ratio)
        left = (img_width - new_width) // 2
        img = img.crop((left, 0, left + new_width, img_height))
    else:
        # Image is taller than target ratio - crop height
        new_height = int(img_width / target_ratio)
        top = (img_height - new_height) // 2
        img = img.crop((0, top, img_width, top + new_height))
    
    # Now scale to target size
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
    
    # Save with high quality
    img.save(image_path, "JPEG", quality=95, optimize=True)
    
    print(f"[DEBUG] download_image: Saved processed image to {image_path}")
    return image_path

@timer_decorator
def create_video_from_images(
    topic_images: List[TopicImage],
    audio_path: str,
    output_path: str
) -> None:
    """
    Create optimized video using FFmpeg with smooth Ken Burns effect.
    """
    print("[DEBUG] create_video_from_images: Creating video using FFmpeg")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # First, create individual video clips with Ken Burns effect
        clip_paths = []
        for i, topic_img in enumerate(topic_images):
            duration = topic_img.end_time - topic_img.start_time

            # Alternate pan directions deterministically based on clip index
            if i % 4 == 0:
                # Pan vertically upward: no horizontal movement
                pan_x = None
                pan_y = "up"
            elif i % 4 == 1:
                # Pan horizontally rightward: no vertical movement
                pan_x = "right"
                pan_y = None
            elif i % 4 == 2:
                # Pan vertically downward: no horizontal movement
                pan_x = None
                pan_y = "down"
            else:  # i % 4 == 3
                # Pan horizontally leftward: no vertical movement
                pan_x = "left"
                pan_y = None

            # Calculate pan expressions based on direction
            if pan_x is None:
                x_expr = "'(iw - iw/zoom)/2'"
            else:
                x_expr = f"'iw/2 - (iw/zoom)/2{'+' if pan_x == 'right' else '-'}x*{PAN_SPEED}'"
            
            if pan_y is None:
                y_expr = "'(ih - ih/zoom)/2'"
            else:
                y_expr = f"'ih/2 - (ih/zoom)/2{'+' if pan_y == 'down' else '-'}x*{PAN_SPEED}'"
            
            clip_path = os.path.join(temp_dir, f'clip_{i}.mp4')
            clip_paths.append(clip_path)
            
            # Create individual clip with Ken Burns effect with more intermediate frames:
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', topic_img.local_path,
                '-t', str(duration),
                '-vf', (
                    f'scale={IMAGE_WIDTH}:{IMAGE_HEIGHT}:force_original_aspect_ratio=decrease,'
                    f'pad={IMAGE_WIDTH}:{IMAGE_HEIGHT}:(ow-iw)/2:(oh-ih)/2,'
                    f'zoompan=z=\'{ZOOM_START_SCALE}+{ZOOM_SPEED}*on\':'
                    f'x={x_expr}:y={y_expr}:'
                    f's={IMAGE_WIDTH}x{IMAGE_HEIGHT}:d={int(duration*TARGET_FPS*INTERPOLATION_FACTOR)},'
                    f'fps={TARGET_FPS}'
                ),
                '-c:v', 'libx264',
                '-preset', FFMPEG_PRESET,
                '-b:v', VIDEO_BITRATE,
                '-r', str(TARGET_FPS),
                clip_path
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        
        # Create concat file for processed clips
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w') as f:
            for clip_path in clip_paths:
                f.write(f"file '{clip_path}'\n")
        
        # Concatenate all clips and add audio
        final_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-i', audio_path,
            '-c:v', 'libx264',             # Re-encode to apply consistent 120 FPS
            '-preset', FFMPEG_PRESET,
            '-r', str(TARGET_FPS),
            '-b:v', VIDEO_BITRATE,
            '-c:a', 'aac',
            '-b:a', AUDIO_BITRATE,
            '-shortest',
            output_path
        ]
        
        try:
            result = subprocess.run(final_cmd, check=True, capture_output=True, text=True)
            print(f"[DEBUG] FFmpeg stdout: {result.stdout}")
            print(f"[DEBUG] create_video_from_images: Video saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg error: {e.stderr}")
            raise

def get_word_timestamps(jsonl_path: str) -> List[Dict[str, str]]:
    """Load word timestamps from JSONL file."""
    timestamps = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            timestamps.append(json.loads(line))
    print(f"[DEBUG] Loaded {len(timestamps)} word timestamps")
    print(f"[DEBUG] First few timestamps: {timestamps[:5]}")
    return timestamps

def find_topic_timing(
    topic_words: List[str], 
    word_timestamps: List[Dict[str, str]], 
    current_idx: int
) -> tuple[float, float, int]:
    """Find start and end times for a topic based on its words."""
    # Initialize variables
    start_time = None
    end_time = None
    idx = current_idx
    words_found = 0
    
    # Normalize words for comparison
    normalized_topic_words = [w.lower() for w in topic_words]
    print(f"\n[DEBUG] Looking for topic words: {normalized_topic_words}")
    print(f"[DEBUG] Starting search from index {current_idx}")
    
    while idx < len(word_timestamps) and words_found < len(topic_words):
        timestamp_entry = word_timestamps[idx]
        current_word = timestamp_entry["word"].lower()
        
        print(f"[DEBUG] Comparing word {words_found}: '{current_word}' with '{normalized_topic_words[words_found]}'")
        
        if current_word == normalized_topic_words[words_found]:
            # Found matching word
            if start_time is None:
                start_time = float(timestamp_entry["timestamp"])
                print(f"[DEBUG] Found first word '{current_word}' at time {start_time}s")
            end_time = float(timestamp_entry["timestamp"])
            print(f"[DEBUG] Found word '{current_word}' at time {end_time}s")
            words_found += 1
        
        idx += 1
    
    print(f"[DEBUG] Words found: {words_found}/{len(topic_words)}")
    print(f"[DEBUG] Final timing - start: {start_time}s, end: {end_time}s")
    
    assert words_found == len(topic_words), f"Could not find all words for topic. Found {words_found} of {len(topic_words)}"
    assert start_time is not None and end_time is not None, "Could not determine topic timing"
    
    # Add a small buffer to end_time to avoid abrupt transitions
    end_time = end_time + 0.01
    print(f"[DEBUG] Added 0.5s buffer. New end time: {end_time}s")
    print(f"[DEBUG] Next search will start from index {idx}\n")
    
    return start_time, end_time, idx

@timer_decorator
def generate_ai_image(topic_name: str, topic_words: List[str]) -> str:
    """
    Generate an image for a given topic using AI text-to-image model via Fal AI.
    
    Args:
        topic_name (str): The topic to generate the image for.
        topic_words (List[str]): The actual words being said in the topic.
    
    Returns:
        str: URL of the generated image.
        
    Raises:
        AssertionError: If generation does not return a valid image URL.
    """
    # Build the prompt by including the topic words
    prompt = IMAGE_GENERATION_PROMPT_TEMPLATE.format(
        topic_name=topic_name,
        topic_words=", ".join(topic_words)
    )
    
    def on_queue_update(update: fal_client.InProgress) -> None:
        """Callback to log progress of image generation."""
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(log["message"])
    
    result = fal_client.subscribe(
        "fal-ai/flux-pro/v1.1-ultra-finetuned",
        arguments={
            "prompt": prompt,
            "finetune_id": "",
            "finetune_strength": 1.0
        },
        with_logs=True,
        on_queue_update=on_queue_update
    )
    
    # Ensure result is a dict with a valid 'images' key representing generated images
    assert isinstance(result, dict), "Expected result to be a dictionary from fal_client.subscribe"
    images = result.get("images")
    assert isinstance(images, list) and len(images) > 0, "No images generated"
    image_data = images[0]
    image_url = image_data.get("url")
    assert isinstance(image_url, str) and image_url, "Generated image URL is not valid"
    return image_url

@timer_decorator
def image_exists_for_topic(topic_name: str, topic_index: int) -> tuple[bool, str]:
    """
    Check if an image already exists for the given topic.
    
    Args:
        topic_name (str): Name of the topic
        topic_index (int): Index of the topic
        
    Returns:
        tuple[bool, str]: (exists, path_if_exists)
    """
    images_dir = "topic_images"
    safe_name = ''.join(c if c.isalnum() else '_' for c in topic_name)
    image_path = f"{images_dir}/{safe_name}_{topic_index}.jpg"
    
    exists = os.path.exists(image_path)
    logging.info(f"Checking for existing image at {image_path}: {'Found' if exists else 'Not found'}")
    
    return exists, image_path

@timer_decorator
def process_video_with_images(
    audio_path: str,
    output_path: str
) -> None:
    """Main function to process audio into video with images."""
    total_start_time = time.time()
    logging.info("Starting video creation process")
    
    # Load environment variables
    try:
        env_start_time = time.time()
        env_path = find_env_file(os.path.dirname(__file__))
        load_dotenv(env_path)
        logging.info(f"Environment loading took {time.time() - env_start_time:.2f} seconds")
    except FileNotFoundError as e:
        logging.error(f"{e}")
        raise
    
    # Load topics and word timestamps
    topics_list = load_topics()
    logging.info(f"Loaded {len(topics_list)} topics")
    
    word_timestamps = get_word_timestamps("audio_to_timestamp.jsonl")
    
    # Process each topic
    topic_images = []
    current_idx = 0
    
    for topic_index, (topic_name, topic_words) in enumerate(topics_list):
        topic_start_time = time.time()
        logging.info(f"Processing topic {topic_index + 1}/{len(topics_list)}: {topic_name}")
        
        # Find timing for this topic
        timing_start = time.time()
        start_time, end_time, current_idx = find_topic_timing(
            topic_words, 
            word_timestamps, 
            current_idx
        )
        logging.info(f"Topic timing calculation took {time.time() - timing_start:.2f} seconds")
        
        # Check if image already exists
        image_exists, local_path = image_exists_for_topic(topic_name, topic_index)
        
        if not image_exists:
            # Generate and download image using AI only if it doesn't exist
            image_gen_start = time.time()
            image_url = generate_ai_image(topic_name, topic_words)
            logging.info(f"AI image generation took {time.time() - image_gen_start:.2f} seconds")
            
            download_start = time.time()
            local_path = download_image(image_url, topic_name, topic_index)
            logging.info(f"Image download and processing took {time.time() - download_start:.2f} seconds")
        else:
            logging.info(f"Using existing image for topic '{topic_name}' at {local_path}")
            image_url = ""  # Empty string since we're using existing image
        
        topic_image = TopicImage(
            topic_name=topic_name,
            topic_index=topic_index,
            image_url=image_url,
            start_time=start_time,
            end_time=end_time,
            local_path=local_path
        )
        topic_images.append(topic_image)
        logging.info(f"Total time for topic {topic_name}: {time.time() - topic_start_time:.2f} seconds")
    
    # Create video
    video_start_time = time.time()
    create_video_from_images(topic_images, audio_path, output_path)
    logging.info(f"Video creation took {time.time() - video_start_time:.2f} seconds")
    
    total_time = time.time() - total_start_time
    logging.info(f"Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    process_video_with_images(
        "audio.mp3",
        "final_video.mp4"
    )