"""
This module overlays transcription text onto a video file.
It loads a transcription file (JSON format expected) that contains a list
of caption entries with "start_time", "end_time", and "text" keys.
For each caption, FFmpeg's drawtext filter is used to overlay the text automatically
during the specified time interval.
"""
from typing import List, Tuple, BinaryIO, Dict, Any

import os
import json
import subprocess
import logging

# Hyperparameters for text overlay (update as needed)
FONT_FILE = "Source_sans_3/SourceSans3-Bold.ttf"  # Path to a TTF font file (must exist)
FONT_SIZE = 78                      # Size of the text
FONT_COLOR = "white"                # Color of the text
TEXT_X = "(w-text_w)/2"            # Horizontal centering
TEXT_Y = "h*0.75-text_h/2"         # Position text in the bottom 3/4 of the video frame

# Add border parameters for text
BORDER_WIDTH = 6                    # Increased border width to make text appear more bold
BORDER_COLOR = "black"              # Color of the text border

# NEW: Hyperparameter for JSONL caption duration when no explicit end time is available.
DEFAULT_CAPTION_DURATION = 2.0   # Duration (in seconds) for the final caption group in JSONL processing

# NEW: File path constants for video processing
INPUT_VIDEO = "final_video.mp4"         
TRANSCRIPTION_PATH = "audio_to_timestamp.jsonl"  # Now using a JSONL file instead of a JSON file.
OUTPUT_VIDEO = "final_video_with_text.mp4"         

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def escape_text(text: str) -> str:
    """
    Escapes special characters in text for use in FFmpeg drawtext filter.
    
    Args:
        text (str): The text to be escaped.
        
    Returns:
        str: The escaped text.
    """
    assert isinstance(text, str), "Expected text to be a string"
    # First escape backslashes, then colons and single quotes.
    text = text.replace('\\', '\\\\')
    text = text.replace(':', r'\:')
    text = text.replace("'", r"\'")
    return text

def load_transcription(transcription_path: str) -> List[Dict[str, Any]]:
    """
    Loads the transcription file that includes an array of caption entries.
    Supports both JSON and JSONL formats.
    
    For JSON format:
        Expects a list of dictionaries with "start_time", "end_time", and "text" keys.
    
    For JSONL format:
        Expects each line to be a JSON object with "word" and "timestamp" keys.
        Consecutive lines with the same timestamp are grouped together to form a caption.
        The caption's "start_time" is set to the group's timestamp; "end_time" is set to the next
        group's timestamp (or start_time + DEFAULT_CAPTION_DURATION for the final group), and
        "text" is the concatenation of all words in the group.
    
    Args:
        transcription_path (str): Path to the transcription file.
        
    Returns:
        List[Dict[str, Any]]: List of caption dictionaries.
        
    Raises:
        AssertionError: if the file or expected fields are not found.
    """
    if transcription_path.endswith(".jsonl"):
        captions = []
        with open(transcription_path, "r") as f:
            lines = f.readlines()

        current_group_timestamp = None
        current_group_words = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Error parsing JSON line: {line}") from exc

            # Validate JSONL entry format
            assert "word" in entry and "timestamp" in entry, (
                "Each JSONL entry must have 'word' and 'timestamp' keys"
            )
            word = str(entry["word"])
            timestamp = float(entry["timestamp"])

            if current_group_timestamp is None:
                current_group_timestamp = timestamp
                current_group_words = [word]
            elif abs(timestamp - current_group_timestamp) < 1e-6:
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

        # Add the last group if present with a default duration.
        if current_group_timestamp is not None:
            caption_text = " ".join(current_group_words)
            captions.append({
                "start_time": current_group_timestamp,
                "end_time": current_group_timestamp + DEFAULT_CAPTION_DURATION,
                "text": caption_text,
            })

        # Validate the caption entries
        for caption in captions:
            assert "start_time" in caption and "end_time" in caption and "text" in caption, (
                "Each caption must have 'start_time', 'end_time', and 'text' keys"
            )
            assert isinstance(caption["start_time"], (float, int))
            assert isinstance(caption["end_time"], (float, int))
            assert isinstance(caption["text"], str)
                
        return captions
    else:
        # Original JSON format processing remains unchanged
        assert os.path.exists(transcription_path), f"Transcription file not found: {transcription_path}"
        with open(transcription_path, "r") as f:
            captions = json.load(f)
        
        assert isinstance(captions, list), "Transcription file should contain a list of captions"
        for entry in captions:
            assert "start_time" in entry and "end_time" in entry and "text" in entry, (
                "Each caption entry must have 'start_time', 'end_time', and 'text' keys"
            )
        return captions

def build_drawtext_filters(captions: List[Dict[str, Any]]) -> str:
    """
    Constructs the FFmpeg filter string that includes a drawtext filter
    for each caption in the transcription.
    
    Args:
        captions (List[Dict[str, Any]]): List of caption entries.
        
    Returns:
        str: The combined FFmpeg video filter string.
    """
    filter_parts = []
    
    for caption in captions:
        start_time = float(caption["start_time"])
        end_time = float(caption["end_time"])
        text = str(caption["text"])
        escaped_text = escape_text(text)
        
        # Updated drawtext filter with border instead of box and shadow
        drawtext_filter = (
            f"drawtext=fontfile='{FONT_FILE}':"
            f"text='{escaped_text}':"
            f"fontcolor={FONT_COLOR}:fontsize={FONT_SIZE}:"
            f"bordercolor={BORDER_COLOR}:borderw={BORDER_WIDTH}:"
            f"x={TEXT_X}:y={TEXT_Y}:"
            f"enable='between(t,{start_time},{end_time})'"
        )
        filter_parts.append(drawtext_filter)
    
    # Chain all drawtext filters using a comma
    filter_chain = ",".join(filter_parts)
    
    # Assert that the filter chain is a non-empty string
    assert isinstance(filter_chain, str) and len(filter_chain) > 0, "Filter chain was not constructed correctly"
    
    return filter_chain

def process_video_with_text_overlay(input_video: str, transcription_path: str, output_video: str) -> None:
    """
    Overlays the transcription text on the input video and writes the output video.
    
    Args:
        input_video (str): Path to the input video (e.g. mp4 file).
        transcription_path (str): Path to the transcription JSON file.
        output_video (str): Path where the output video will be saved.
    """
    assert os.path.exists(input_video), f"Input video file not found: {input_video}"
    
    captions = load_transcription(transcription_path)
    logging.info(f"Loaded {len(captions)} transcription captions")
    
    filter_chain = build_drawtext_filters(captions)
    logging.info("Constructed FFmpeg drawtext filter chain")
    
    # Build the FFmpeg command. We use '-c:a copy' to copy the audio stream.
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", filter_chain,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "copy",  # Copy audio stream without re-encoding
        output_video
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        logging.info(f"FFmpeg output: {result.stdout}")
        logging.info(f"Video with text overlay saved to {output_video}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        raise

def main() -> None:
    """Main function to process video overlay."""
    process_video_with_text_overlay(INPUT_VIDEO, TRANSCRIPTION_PATH, OUTPUT_VIDEO)

if __name__ == "__main__":
    main() 