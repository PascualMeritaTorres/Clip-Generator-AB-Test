"""
Final pipeline orchestrating the preprocessing, generation of topics, sound effects, images, and text overlay.

Flow:
1. Preprocess MP3 timestamps to create word timestamps and transcription files
2. Load transcription text from audio_transcription.jsonl.
3. Generate topics using generate_topics.py.
4. Generate audio with sound effects using add_sound_effects.py.
5. Generate video with images from the sound-effect processed audio using add_images.py.
6. Overlay transcription text onto the video using add_text_overlay.py.
"""

# Ensure the parent directory is in the Python module search path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import json
import logging
import time
from typing import Any, Dict

# Import functions from the respective modules
from video_fillers.preprocess_input import process_timestamps
from video_fillers.generate_topics import generate_topics
from video_fillers.add_sound_effects import process_audio_with_effects, save_audio
from video_fillers.add_images import process_video_with_images
from video_fillers.add_text_overlay import process_video_with_text_overlay

# Hyperparameters / file constants
AUDIO_INPUT: str = "audio.mp3"
MP3_TIMESTAMPS_FILE: str = "timestamps.json"
TIMESTAMP_FILE: str = "audio_to_timestamp.jsonl"
TRANSCRIPTION_FILE: str = "audio_transcription.jsonl"
TOPICS_OUTPUT_FILE: str = "identified_topics.json"
AUDIO_WITH_EFFECTS: str = "audio_with_sound_effects.mp3"
VIDEO_WITH_IMAGES: str = "video_with_images.mp4"
FINAL_VIDEO: str = "final_video_with_text.mp4"

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def main() -> None:
    """Orchestrates the full video creation pipeline."""
    # Record overall start time
    total_start = time.time()

    # Validate input files exist
    assert os.path.exists(AUDIO_INPUT), f"Audio file not found: {AUDIO_INPUT}"
    assert os.path.exists(MP3_TIMESTAMPS_FILE), f"MP3 timestamps file not found: {MP3_TIMESTAMPS_FILE}"

    # Step 0: Preprocess MP3 timestamps to create word timestamps and transcription
    preprocess_start_time = time.time()
    logging.info("Preprocessing MP3 timestamps...")
    timestamp_path, transcription_path = await asyncio.to_thread(
        process_timestamps, 
        MP3_TIMESTAMPS_FILE, 
        "."  # Use current directory
    )
    preprocess_time = time.time() - preprocess_start_time
    logging.info(f"Preprocessing completed. Time taken: {preprocess_time:.2f} seconds")

    # Validate generated files exist
    assert os.path.exists(timestamp_path), f"Generated timestamp file not found: {timestamp_path}"
    assert os.path.exists(transcription_path), f"Generated transcription file not found: {transcription_path}"

    # Load transcription text from the transcription JSONL file
    logging.info("Loading transcription text...")
    with open(transcription_path, "r") as f:
        data: Dict[str, Any] = json.load(f)
    assert "text" in data, "Transcription file must contain a 'text' key"
    transcription_text: str = data["text"]
    logging.info("Transcription text loaded successfully.")

    # Step 1: Generate topics from the transcription text.
    topics_start_time = time.time()
    logging.info("Generating topics using generate_topics...")
    await asyncio.to_thread(generate_topics, transcription_text, TOPICS_OUTPUT_FILE)
    topics_time = time.time() - topics_start_time
    logging.info(f"Topics generated and saved to {TOPICS_OUTPUT_FILE}. Time taken: {topics_time:.2f} seconds.")

    """
    # Step 2: Generate audio with sound effects.
    sound_start_time = time.time()
    logging.info("Processing sound effects on audio...")
    audio_effects = await asyncio.to_thread(process_audio_with_effects, AUDIO_INPUT, transcription_text, timestamp_path)
    # Save processed audio with sound effects.
    await asyncio.to_thread(save_audio, audio_effects, AUDIO_WITH_EFFECTS)
    sound_time = time.time() - sound_start_time
    logging.info(f"Audio with sound effects saved to {AUDIO_WITH_EFFECTS}. Time taken: {sound_time:.2f} seconds.")
    """

    # Step 3: Create video using images.
    # Use the sound effects audio as input for the video generation.
    images_start_time = time.time()
    logging.info("Generating video with images...")
    await asyncio.to_thread(process_video_with_images, AUDIO_INPUT, VIDEO_WITH_IMAGES)
    images_time = time.time() - images_start_time
    logging.info(f"Video with images saved to {VIDEO_WITH_IMAGES}. Time taken: {images_time:.2f} seconds.")

    # Step 4: Overlay transcription text (captions) onto the video.
    overlay_start_time = time.time()
    logging.info("Adding text overlay to the video...")
    await asyncio.to_thread(process_video_with_text_overlay, VIDEO_WITH_IMAGES, timestamp_path, FINAL_VIDEO)
    overlay_time = time.time() - overlay_start_time
    logging.info(f"Final video with text overlay saved to {FINAL_VIDEO}. Time taken: {overlay_time:.2f} seconds.")

    total_time = time.time() - total_start
    logging.info(f"Total processing time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(main()) 