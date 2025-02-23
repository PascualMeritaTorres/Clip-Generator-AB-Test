"""
This script overlays text from a JSONL file onto a video using MoviePy.
Each line in the JSONL file should contain a word and a timestamp.
"""

import json
from moviepy import *
from typing import List, Dict, Any

# Hyperparameters
FONT_FILE = "Source_Sans_3/static/SourceSans3-Bold.ttf"  # Path to a TTF font file
FONT_SIZE = 82
FONT_COLOR = "white"
DEFAULT_CAPTION_DURATION = 2.0

INPUT_VIDEO = "final_video.mp4"
TRANSCRIPTION_PATH = "audio_to_timestamp.jsonl"
OUTPUT_VIDEO = "final_video_with_text.mp4"

def load_transcription(transcription_path: str) -> List[Dict[str, Any]]:
    """
    Loads the transcription from a JSONL file and adjusts timestamps to display text in pairs of words.
    
    Args:
        transcription_path (str): Path to the transcription file.
        
    Returns:
        List[Dict[str, Any]]: List of caption dictionaries.
    """
    captions = []
    with open(transcription_path, "r") as f:
        lines = f.readlines()

    current_group_timestamp = None
    current_group_words = []
    previous_timestamp = None  # To store the previous word's timestamp

    for line in lines:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        word = str(entry["word"])
        timestamp = float(entry["timestamp"])

        # Use the previous word's timestamp for the current word
        if previous_timestamp is not None:
            timestamp = previous_timestamp

        if current_group_timestamp is None:
            current_group_timestamp = timestamp
            current_group_words = [word]
        elif len(current_group_words) < 2:
            current_group_words.append(word)
        else:
            caption_text = " ".join(current_group_words)
            captions.append({
                "start_time": current_group_timestamp,
                "end_time": timestamp,
                "text": caption_text,
            })
            current_group_timestamp = timestamp
            current_group_words = [word]

        # Update the previous timestamp
        previous_timestamp = float(entry["timestamp"])

    if current_group_words:
        caption_text = " ".join(current_group_words)
        captions.append({
            "start_time": current_group_timestamp,
            "end_time": current_group_timestamp + DEFAULT_CAPTION_DURATION,
            "text": caption_text,
        })

    print(f"Loaded {len(captions)} captions from {transcription_path}")
    return captions

def create_text_clips(captions: List[Dict[str, Any]], video_height: int) -> List[TextClip]:
    """
    Creates a list of TextClip objects for each caption with a black border around the white text.
    
    Args:
        captions (List[Dict[str, Any]]): List of caption entries.
        video_height (int): Height of the video to position text.
        
    Returns:
        List[TextClip]: List of text clips.
    """
    text_clips = []
    for caption in captions:
        text_clip = TextClip(
            text=caption["text"],
            font_size=FONT_SIZE,
            color=FONT_COLOR,
            font=FONT_FILE,
            stroke_color="black",  # Add black stroke
            stroke_width=2         # Width of the stroke
        ).with_position(("center", video_height * 0.75)).with_start(caption["start_time"]).with_duration(caption["end_time"] - caption["start_time"])
        text_clips.append(text_clip)
    return text_clips

def overlay_text_on_video(input_video: str, transcription_path: str, output_video: str) -> None:
    """
    Overlays text on the video using MoviePy.
    
    Args:
        input_video (str): Path to the input video.
        transcription_path (str): Path to the transcription JSONL file.
        output_video (str): Path where the output video will be saved.
    """
    print(f"Processing video: {input_video}")
    print(f"Using transcription: {transcription_path}")
    print(f"Output will be saved to: {output_video}")
    video = VideoFileClip(input_video)
    captions = load_transcription(transcription_path)
    text_clips = create_text_clips(captions, video.h)
    
    final_video = CompositeVideoClip([video] + text_clips)
    final_video.write_videofile(output_video, codec="libx264", audio_codec="aac")
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    overlay_text_on_video(INPUT_VIDEO, TRANSCRIPTION_PATH, OUTPUT_VIDEO) 