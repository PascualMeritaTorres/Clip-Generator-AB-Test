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

# Image parameters for vertical social media format (9:16 aspect ratio)
IMAGE_WIDTH = 1080  # Width for vertical video
IMAGE_HEIGHT = 1920  # Height for vertical video (9:16 aspect ratio)
TRANSITION_DURATION = 1.0  # Duration of fade transition between images in seconds
MIN_IMAGE_DURATION = 3.0  # Minimum duration to show each image

# Optimization constants
TARGET_FPS = 30  # Standard for social media
ENCODING_PRESET = 'ultrafast'  # Fastest encoding preset
ENCODING_THREADS = 4  # Number of threads for encoding
AUDIO_FPS = 44100  # Standard audio sampling rate

# Update constants
FFMPEG_PRESET = 'ultrafast'  # FFmpeg encoding preset
VIDEO_BITRATE = '4M'  # Target video bitrate
AUDIO_BITRATE = '192k'  # Target audio bitrate

# Ken Burns effect parameters
ZOOM_START_SCALE = 1.0
ZOOM_END_SCALE = 1.3
ZOOM_DURATION = 15.0  # Duration of the zoom effect in seconds

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

def search_pixabay_image(topic_name: str, api_key: str) -> str:
    """
    Search for an image on Pixabay using the topic name.
    
    Args:
        topic_name (str): Topic to search for
        api_key (str): Pixabay API key
        
    Returns:
        str: URL of the selected image
        
    Raises:
        RuntimeError: If no suitable images found
    """
    print(f"[DEBUG] search_pixabay_image: Searching for images with topic: {topic_name}")
    
    # Convert topic name to URL-friendly search query
    search_query = topic_name.replace(' ', '+')
    print(search_query)
    
    base_url = "https://pixabay.com/api/"
    params = {
        "key": api_key,
        "q": search_query,
        "order": "popular"  # Get most popular images first
    }
    
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    
    data = response.json()
    hits = data.get("hits", [])
    
    if not hits:
        raise RuntimeError(f"No images found for topic: {topic_name}")
    
    # Select the first image (highest relevance)
    image_url = hits[0]["largeImageURL"]
    print(f"[DEBUG] search_pixabay_image: Selected image URL: {image_url}")
    return image_url

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

def create_video_from_images(
    topic_images: List[TopicImage],
    audio_path: str,
    output_path: str
) -> None:
    """
    Create optimized video using FFmpeg with Ken Burns effect.
    
    Args:
        topic_images: List of TopicImage objects containing image info
        audio_path: Path to the audio file
        output_path: Path where the output video will be saved
    """
    print("[DEBUG] create_video_from_images: Creating video using FFmpeg")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create FFmpeg concat file with absolute paths
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w') as f:
            for topic_img in topic_images:
                abs_path = os.path.abspath(topic_img.local_path)
                duration = topic_img.end_time - topic_img.start_time
                f.write(f"file '{abs_path}'\n")
                f.write(f"duration {duration}\n")
        
        abs_audio_path = os.path.abspath(audio_path)
        
        # Calculate zoompan parameters
        frames_per_transition = int(ZOOM_DURATION * TARGET_FPS)
        zoom_increment = (ZOOM_END_SCALE - ZOOM_START_SCALE) / frames_per_transition
        
        # FFmpeg command with zoompan effect using defined scale constants
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-i', abs_audio_path,
            '-c:v', 'libx264',
            '-preset', FFMPEG_PRESET,
            '-b:v', VIDEO_BITRATE,
            '-c:a', 'aac',
            '-b:a', AUDIO_BITRATE,
            '-r', str(TARGET_FPS),
            '-vf', (
                f'scale={IMAGE_WIDTH}:{IMAGE_HEIGHT}:force_original_aspect_ratio=decrease,'
                f'pad={IMAGE_WIDTH}:{IMAGE_HEIGHT}:(ow-iw)/2:(oh-ih)/2,'
                f'zoompan=z=\'if(eq(on,0),{ZOOM_START_SCALE},min({ZOOM_END_SCALE},zoom+{zoom_increment}))\':'
                f'd={frames_per_transition}:'
                f'x=\'iw/2-(iw/zoom/2)\':y=\'ih/2-(ih/zoom/2)\':'
                f's={IMAGE_WIDTH}x{IMAGE_HEIGHT}'
            ),
            '-shortest',
            output_path
        ]
        
        print(f"[DEBUG] Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
        
        try:
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            print(f"[DEBUG] FFmpeg stdout: {result.stdout}")
            print(f"[DEBUG] create_video_from_images: Video saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg error: {e.stderr}")
            with open(concat_file, 'r') as f:
                print(f"[DEBUG] Concat file contents:\n{f.read()}")
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

def process_video_with_images(
    audio_path: str,
    output_path: str
) -> None:
    """Main function to process audio into video with images."""
    print("\n[DEBUG] process_video_with_images: Starting video creation process")
    
    # Load environment variables
    try:
        env_path = find_env_file(os.path.dirname(__file__))
        load_dotenv(env_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        raise
    
    pixabay_key = os.getenv('PIXABAY_API_KEY')
    assert pixabay_key is not None, "PIXABAY_API_KEY not found in environment variables"
    
    # Load topics and word timestamps
    topics_list = load_topics()
    print(f"\n[DEBUG] Loaded topics: {[t[0] for t in topics_list]}")
    
    word_timestamps = get_word_timestamps("audio_to_timestamp.jsonl")
    
    # Process each topic
    topic_images = []
    current_idx = 0
    
    for topic_index, (topic_name, topic_words) in enumerate(topics_list):
        print(f"\n[DEBUG] Processing topic: {topic_name} (index: {topic_index})")
        print(f"[DEBUG] Topic words: {topic_words}")
        
        # Find timing for this topic
        start_time, end_time, current_idx = find_topic_timing(
            topic_words, 
            word_timestamps, 
            current_idx
        )
        
        print(f"[DEBUG] Final topic timing - start: {start_time}s, end: {end_time}s")
        print(f"[DEBUG] Duration: {end_time - start_time:.2f}s")
        
        # Search and download image
        print(f"[DEBUG] Searching for image with topic: {topic_name}")
        image_url = search_pixabay_image(topic_name, pixabay_key)
        print(f"[DEBUG] Found image URL: {image_url}")
        
        local_path = download_image(image_url, topic_name, topic_index)
        print(f"[DEBUG] Saved image to: {local_path}")
        
        topic_image = TopicImage(
            topic_name=topic_name,
            topic_index=topic_index,
            image_url=image_url,
            start_time=start_time,
            end_time=end_time,
            local_path=local_path
        )
        topic_images.append(topic_image)
        print(f"[DEBUG] Added topic image to list. Total images: {len(topic_images)}")
    
    print("\n[DEBUG] All topics processed. Creating final video...")
    # Create video
    create_video_from_images(topic_images, audio_path, output_path)
    print("[DEBUG] process_video_with_images: Video creation complete")

if __name__ == "__main__":
    process_video_with_images(
        "audio.mp3",
        "final_video.mp4"
    )