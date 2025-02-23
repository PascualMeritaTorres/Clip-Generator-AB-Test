#input_transcription_of_audio = X_transcription
#input_audio = X_audio_with_music

#1) Get the generated topics from identified_topics.json
#1) For each topic, browse the internet for relevant images and choose one
#4) Add the images to the X_audio and put them together to generate a mp4
#5) Export X_audio_with_music_and_images

#output = X_audio_with_music_and_images

from typing import List, Dict, Tuple
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

# NEW HYPERPARAMETER: Set to False to disable saving images persistently. When False, images are saved into temporary files.
SAVE_IMAGES = True

# Update the prompt template to be more story-focused
STORY_CONTEXT_PROMPT = """You are a visual storytelling expert. I will provide you with a sequence of topics and their associated words from a video/audio script. 
Your task is to create a cohesive visual narrative by generating image prompts that flow naturally from one topic to the next.

For each topic, generate an artistic image prompt that will be passed to an AI image generation model and should:
1. Capture the essence of the topic and its associated words
2. Maintain visual consistency with the previous and next topics
3. Follow a coherent color scheme and visual style
4. Is suitable for vertical video format (9:16)
5. Does not include any text elements

Topics and their words:
{topics_data}

Respond with a JSON array of objects. Each object must have exactly these two fields:
- "topic_name": The exact topic name as provided
- "image_prompt": Your crafted prompt for image generation

Keep each prompt focused and specific, around 2-3 sentences.

Example of the exact JSON format required:
[
  {{
    "topic_name": "Early Computers",
    "image_prompt": "A vintage room bathed in warm amber light, featuring a massive early computer with glowing vacuum tubes and brass details. The composition draws the eye upward, with floating mathematical equations and circuit patterns creating a vertical flow in a retro-futuristic style."
  }},
  {{
    "topic_name": "Digital Revolution",
    "image_prompt": "Streams of luminous binary code cascading down like a digital waterfall against a deep blue backdrop, transforming into modern devices. Maintains the warm lighting accents from the previous scene while introducing cool cyber-blue elements."
  }}
]

Important: Your response must be exactly in this JSON format, with no additional text before or after."""

# Logging configuration
logging.basicConfig(
    level=logging.WARNING,
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
    
    safe_name = ''.join(c if c.isalnum() else '_' for c in topic_name)
    if SAVE_IMAGES:
        images_dir = "topic_images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = f"{images_dir}/{safe_name}_{topic_index}.jpg"
    else:
        # When saving is disabled, use a temporary file.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image_path = tmp.name
    
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

def normalize_word(word: str) -> str:
    """
    Normalize a word by removing punctuation and standardizing ellipsis.
    
    Args:
        word (str): Word to normalize
        
    Returns:
        str: Normalized word
    """
    # Remove any whitespace
    word = word.strip()
    
    # Convert ellipsis variations to standard form
    word = word.replace('...', '')
    word = word.replace('..', '')
    
    # Remove punctuation except apostrophes
    word = ''.join(c for c in word if c.isalnum() or c == "'")
    
    return word.lower()

def find_topic_timing(topic_words: List[str], timestamp_data: List[Dict[str, str]], current_idx: int) -> Tuple[float, float, int]:
    """
    Find start and end timestamps for a topic by matching words.
    
    Args:
        topic_words (List[str]): List of words in the topic
        timestamp_data (List[Dict[str, str]]): List of word timestamps
        current_idx (int): Current position in timestamp data
        
    Returns:
        Tuple[float, float, int]: Start time, end time, and new current index
    """
    print(f"[DEBUG] Looking for topic words: {topic_words}")
    print(f"[DEBUG] Starting search from index {current_idx}")
    
    words_found = 0
    start_time = None
    end_time = None
    i = current_idx
    first_word_idx = None
    
    # Stop scanning once all topic words are found.
    while i < len(timestamp_data) and words_found < len(topic_words):
        for topic_idx, topic_word in enumerate(topic_words):
            if words_found > topic_idx:
                continue
                
            timestamp_word = timestamp_data[i]['word']
            
            # Normalize both words for comparison
            norm_topic_word = normalize_word(topic_word)
            norm_timestamp_word = normalize_word(timestamp_word)
            
            print(f"[DEBUG] Comparing word {topic_idx}: '{timestamp_word}' / '{norm_timestamp_word}' with '{topic_word}'")
            
            # Handle contractions
            if "'" in topic_word and i < len(timestamp_data) - 2:
                # Try combining current word with next ones for contractions
                combined = timestamp_word
                look_ahead = 1
                while look_ahead <= 2 and i + look_ahead < len(timestamp_data):
                    combined += timestamp_data[i + look_ahead]['word']
                    if normalize_word(combined) == norm_topic_word:
                        if start_time is None:
                            first_word_idx = i
                            # For first word, get previous timestamp if not first topic
                            if current_idx > 0 and i > 0:
                                start_time = float(timestamp_data[i - 1]['timestamp'])
                            else:
                                start_time = float(timestamp_data[i]['timestamp'])
                        end_time = float(timestamp_data[i + look_ahead]['timestamp'])
                        words_found += 1
                        i += look_ahead
                        break
                    look_ahead += 1
            # Direct word match
            elif norm_timestamp_word == norm_topic_word:
                if start_time is None:
                    first_word_idx = i
                    # For first word, get previous timestamp if not first topic
                    if current_idx > 0 and i > 0:
                        start_time = float(timestamp_data[i - 1]['timestamp'])
                    else:
                        start_time = float(timestamp_data[i]['timestamp'])
                end_time = float(timestamp_data[i]['timestamp'])
                words_found += 1
                print(f"[DEBUG] Found word '{topic_word}' at time {end_time}s")
                break
        # If we have matched all words, break early.
        if words_found == len(topic_words):
            break
        i += 1
    
    print(f"[DEBUG] Words found: {words_found}/{len(topic_words)}")
    
    if words_found >= 1:
        if end_time is None:
            end_time = start_time
            
        # If this is the last word of the last topic, use the final timestamp
        if i >= len(timestamp_data) - 1 or current_idx >= len(timestamp_data) - len(topic_words):
            end_time = float(timestamp_data[-1]['timestamp'])
            print(f"[DEBUG] Using final timestamp for last topic: {end_time}s")
            
        # Log the final timestamps being used
        print(f"[DEBUG] Using timestamps for topic: start={start_time:.3f}s, end={end_time:.3f}s")
        if first_word_idx is not None and first_word_idx > 0:
            print(f"[DEBUG] First word '{timestamp_data[first_word_idx]['word']}' at {float(timestamp_data[first_word_idx]['timestamp']):.3f}s")
            print(f"[DEBUG] Previous word '{timestamp_data[first_word_idx-1]['word']}' at {float(timestamp_data[first_word_idx-1]['timestamp']):.3f}s")
            
        # Advance pointer one more token if we're not at the end.
        new_idx = i + 1 if i < len(timestamp_data) else i
        return start_time, end_time, new_idx
        
    raise AssertionError(f"Could not find enough words for topic. Found {words_found} of {len(topic_words)}")

@timer_decorator
def generate_coherent_prompts(topics_list: List[tuple[str, List[str]]]) -> List[Dict[str, str]]:
    """
    Generate coherent image prompts for all topics using Claude.
    
    Args:
        topics_list: List of (topic_name, topic_words) pairs
        
    Returns:
        List[Dict[str, str]]: List of dicts with topic names and their image prompts
    """
    # Format topics data for Claude in a clearer way
    topics_formatted = []
    for topic_name, topic_words in topics_list:
        topic_str = f"Topic: {topic_name}\nAssociated words: {', '.join(topic_words)}"
        topics_formatted.append(topic_str)
    
    topics_data = "\n\n".join(topics_formatted)
    
    logging.info(f"Formatted topics for prompt:\n{topics_data}")
    
    # Call Claude via fal-ai to generate coherent prompts
    try:
        result = fal_client.subscribe(
            "fal-ai/any-llm",
            arguments={
                "model": "anthropic/claude-3.5-sonnet",
                "prompt": STORY_CONTEXT_PROMPT.format(topics_data=topics_data),
                "system_prompt": "You are a visual storytelling expert who creates cohesive image generation prompts. Always respond with valid JSON."
            }
        )
        
        # Parse the JSON response from Claude
        try:
            prompts_data = json.loads(result["output"])
            logging.info(f"Received response from Claude: {result['output']}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from Claude's response: {result['output']}")
            raise
        
        # Validate the response format
        assert isinstance(prompts_data, list), f"Expected list of prompts from Claude, got {type(prompts_data)}"
        assert len(prompts_data) == len(topics_list), f"Mismatch in number of prompts. Expected {len(topics_list)}, got {len(prompts_data)}"
        
        for i, prompt in enumerate(prompts_data):
            assert isinstance(prompt, dict), f"Expected dict for prompt {i}, got {type(prompt)}"
            assert "topic_name" in prompt, f"Missing topic_name in prompt {i}"
            assert "image_prompt" in prompt, f"Missing image_prompt in prompt {i}"
            
        return prompts_data
        
    except Exception as e:
        logging.error(f"Error generating coherent prompts: {str(e)}")
        logging.error(f"Full error details: {e}")
        raise

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
    
    # Generate coherent prompts for all topics
    prompts_start_time = time.time()
    coherent_prompts = generate_coherent_prompts(topics_list)
    logging.info(f"Generated coherent prompts in {time.time() - prompts_start_time:.2f} seconds")
    
    word_timestamps = get_word_timestamps("audio_to_timestamp.jsonl")
    
    # Process each topic
    topic_images = []
    current_idx = 0
    
    for topic_index, ((topic_name, topic_words), prompt_data) in enumerate(zip(topics_list, coherent_prompts)):
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
            # Use the coherent prompt instead of generating a new one
            image_url = generate_ai_image_with_prompt(prompt_data["image_prompt"])
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

@timer_decorator
def generate_ai_image_with_prompt(prompt: str) -> str:
    """
    Generate an image using the provided prompt via Fal AI.
    
    Args:
        prompt (str): The detailed prompt for image generation
        
    Returns:
        str: URL of the generated image
    """
    result = fal_client.subscribe(
        "fal-ai/flux-pro/v1.1-ultra",
        arguments={
            "prompt": prompt,
            "finetune_id": "",
            "finetune_strength": 1.0
        },
        with_logs=True
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
    if not SAVE_IMAGES:
        logging.info("Intermediate image saving is disabled. Skipping existence check.")
        return False, ""
    images_dir = "topic_images"
    safe_name = ''.join(c if c.isalnum() else '_' for c in topic_name)
    image_path = f"{images_dir}/{safe_name}_{topic_index}.jpg"
    
    exists = os.path.exists(image_path)
    logging.info(f"Checking for existing image at {image_path}: {'Found' if exists else 'Not found'}")
    
    return exists, image_path

if __name__ == "__main__":
    process_video_with_images(
        "audio.mp3",
        "final_video.mp4"
    )