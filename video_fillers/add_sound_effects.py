#input_transcription_of_audio = X_transcription
#input_audio = X_audio_with_music

#1) Divide the X_audio_with_music into chunks (arbitrarily)
#2) For each chunk, identify both where to add the sound effect and what sound effect to add (generate a prompt for elevenlabs)
#3) Generate and add the sound effect to the chunk
#4) Export X_audio_with_music_and_sound_effects

#output = X_audio_with_music_and_sound_effects

from typing import List, Tuple, BinaryIO, Dict
from dataclasses import dataclass
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os
from pydub import AudioSegment
import json
from openai import OpenAI
import math
import time  # <-- ADDED: Import time to measure processing duration
import concurrent.futures
import logging

# NEW IMPORT: Load topics from generate_topics.py instead of generating them here
from video_fillers.generate_topics import load_topics

# Hyperparameter: Toggle saving of intermediate files (chunks, sound effects, topics, etc.)
SAVE_INTERMEDIATE_FILES = True

# Volume control parameters (in dB)
MAIN_AUDIO_VOLUME_DB = 0  # 0 means no change, negative values reduce volume, positive values increase it
SOUND_EFFECT_VOLUME_DB = -20  # Sound effects will be 10dB quieter than main audio by default
MIN_VOLUME_DB = -30
MAX_VOLUME_DB = -10

# Reverb parameters
REVERB_INTENSITY = 1.0   # Overall reverb intensity (0-1), lower = more subtle
REVERB_DELAY_MS = 30      # Reduced from 50 to 30 for subtler effect
REVERB_DECAY = 0.3        # Reduced from 0.5 to 0.3 for faster decay
REVERB_REPEATS = 2        # Reduced from 3 to 2 repeats

@dataclass
class AudioChunk:
    """Represents a chunk of audio with its associated metadata"""
    start_time: float  # in seconds
    end_time: float   # in seconds
    audio_data: AudioSegment
    transcription: str

@dataclass
class Topic:
    """Represents a topic segment with its words and timing"""
    name: str
    words: List[str]
    start_time: float
    end_time: float

def load_audio(audio_path: str) -> AudioSegment:
    """
    Load an MP3 audio file.
    
    Args:
        audio_path (str): Path to the MP3 file
        
    Returns:
        AudioSegment: Loaded audio data
        
    Raises:
        AssertionError: If file doesn't exist or isn't an MP3
    """
    print(f"[DEBUG] load_audio: Loading audio file from {audio_path}")
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"
    assert audio_path.endswith('.mp3'), "Audio file must be MP3 format"
    return AudioSegment.from_mp3(audio_path)

def generate_effect_prompt(chunk: AudioChunk) -> Tuple[str, float]:
    """
    Analyze chunk transcription to determine sound effect prompt and timing.
    
    Given an audio chunk with transcription, this function invokes an LLM (via fal_client)
    to determine if a sound effect is needed. The LLM returns trigger words and a detailed
    sound effect prompt focusing on filler sounds.
    The duration is computed based on the number of trigger words.
    
    Args:
        chunk (AudioChunk): Audio chunk with transcription
        
    Returns:
        Tuple[str, float]: A tuple containing the sound effect prompt and the computed duration in seconds.
    """
    print(f"[DEBUG] generate_effect_prompt: Processing chunk from {chunk.start_time} s to {chunk.end_time} s")
    import fal_client
    description = f"""
Analyze the following transcript and determine if a sound effect is needed. If so, create a prompt which then will be passed to a text-to-sound-effect model.
The sound effects should focus on filler background sounds for ambientation such as:
  - Risers for tension,
  - Deep sounds for very philosophical or deep moments,
  - Cliffhanger sounds when something important is about to be revealed or when the clip ends on a cliffhanger.

Return the result as a JSON object with the following structure:
{{
  "trigger_words": ["word or phrase", ...],
  "sound_effect_prompt": "Brief description of desired sound effect. Start with 'Generate a background sound effect:' followed by the effect description. Do not include any voice or speech elements."
}}

If no sound effect is needed, return an empty list for "trigger_words" and an empty string for "sound_effect_prompt".

Transcription: {chunk.transcription}
    """
    try:
        result = fal_client.subscribe(
            "fal-ai/any-llm",
            arguments={
                "model": "anthropic/claude-3.5-sonnet",
                "prompt": description + "\n\nOutput valid JSON only."
            },
            with_logs=False
        )
        data = json.loads(result["output"])
    except Exception as e:
        print(f"[DEBUG] generate_effect_prompt: Exception encountered: {e}. Defaulting to no sound effect.")
        data = {"trigger_words": [], "sound_effect_prompt": ""}
    
    # Validate the response
    assert isinstance(data, dict), "LLM response must be a dictionary"
    assert "trigger_words" in data, "LLM response missing 'trigger_words'"
    assert "sound_effect_prompt" in data, "LLM response missing 'sound_effect_prompt'"
    
    trigger_words = data["trigger_words"]
    sound_effect_prompt = data["sound_effect_prompt"]

    if not trigger_words or sound_effect_prompt.strip() == "":
        return ("", 0.0)
    
    # Add explicit instruction to avoid voice/speech
    sound_effect_prompt += ". No voice or speech elements, only ambient sounds."
    
    print(f"[DEBUG] generate_effect_prompt: LLM returned trigger_words: {trigger_words}, sound_effect_prompt: {sound_effect_prompt}")
    
    # Compute duration based on the number of trigger words
    duration = 5 * len(trigger_words)
    return (sound_effect_prompt, duration)

def generate_sound_effect(
    client: ElevenLabs,
    prompt: str,
    duration: float,
    chunk_start_time: float  # Added parameter to help with filename
) -> bytes:
    """
    Generate sound effect using ElevenLabs API and save to folder.
    
    Args:
        client (ElevenLabs): ElevenLabs client
        prompt (str): Description of the sound effect
        duration (float): Desired duration in seconds
        chunk_start_time (float): Start time of the chunk (for filename)
        
    Returns:
        bytes: Generated sound effect audio data
    """
    assert 0.1 <= duration <= 30.0, "Duration must be between 0.1 and 30 seconds"
    print(f"[DEBUG] generate_sound_effect: Generating effect with prompt: '{prompt}' for duration: {duration} seconds")
    
    # Create sound_effects directory if it doesn't exist
    effects_dir = "sound_effects"
    os.makedirs(effects_dir, exist_ok=True)
    
    result = client.text_to_sound_effects.convert(
        text=prompt,
        duration_seconds=duration,
        prompt_influence=0.8
    )
    
    # Convert bytes to AudioSegment for saving
    effect_bytes = b''.join(result)
    from io import BytesIO
    effect_audio = AudioSegment.from_file(BytesIO(effect_bytes), format="mp3")
    
    # Create a safe filename from the prompt
    safe_prompt = ''.join(c if c.isalnum() else '_' for c in prompt[:30])  # First 30 chars
    effect_filename = f"{effects_dir}/effect_{safe_prompt}_{chunk_start_time:.2f}s.mp3"
    
    # Conditionally save the effect file if enabled
    if SAVE_INTERMEDIATE_FILES:
        effect_audio.export(effect_filename, format='mp3')
        print(f"[DEBUG] generate_sound_effect: Saved sound effect to {effect_filename}")
    else:
        print(f"[DEBUG] generate_sound_effect: Skipping saving effect file.")
    
    print("[DEBUG] generate_sound_effect: Sound effect generated successfully.")
    return effect_bytes

def add_reverb(audio: AudioSegment) -> AudioSegment:
    """
    Add subtle reverb effect to an audio segment.
    
    Args:
        audio (AudioSegment): Input audio segment
        
    Returns:
        AudioSegment: Audio with reverb effect
    """
    print("[DEBUG] add_reverb: Adding subtle reverb effect to audio")
    
    # Validate parameters
    assert 0 < REVERB_DELAY_MS <= 200, "Reverb delay must be between 1-200ms"
    assert 0 < REVERB_DECAY < 1, "Reverb decay must be between 0-1"
    assert 0 < REVERB_REPEATS <= 5, "Reverb repeats must be between 1-5"
    assert 0 <= REVERB_INTENSITY <= 1, "Reverb intensity must be between 0-1"
    
    # Create reverb effect by overlaying delayed copies with decreasing volume
    output = audio
    current_delay = REVERB_DELAY_MS
    current_volume = REVERB_DECAY * REVERB_INTENSITY  # Scale initial volume by intensity
    
    for _ in range(REVERB_REPEATS):
        # Create delayed copy with reduced volume
        delayed = audio._spawn(audio.raw_data)  # Create copy
        
        # Calculate volume reduction in dB, scaled by intensity
        volume_reduction = 20 * math.log10(1/current_volume)
        delayed = delayed - volume_reduction  # Adjust volume
        
        # Overlay delayed copy
        output = output.overlay(delayed, position=current_delay, gain_during_overlay=-3)
        
        # Update delay and volume for next iteration
        current_delay += REVERB_DELAY_MS
        current_volume *= REVERB_DECAY
    
    print(f"[DEBUG] add_reverb: Added reverb with intensity {REVERB_INTENSITY}")
    return output

def mix_effect_with_chunk(
    chunk: AudioChunk,
    effect: bytes,
    timing: float
) -> AudioChunk:
    """
    Mix generated sound effect with original audio chunk.
    
    Args:
        chunk (AudioChunk): Original audio chunk
        effect (bytes): Sound effect audio data
        timing (float): When to insert the effect (in seconds from the start of the chunk)
        
    Returns:
        AudioChunk: The updated audio chunk with the sound effect mixed in.
    """
    from io import BytesIO
    print(f"[DEBUG] mix_effect_with_chunk: Mixing sound effect into audio chunk starting at {chunk.start_time} s")
    
    # Validate volume parameters
    assert MIN_VOLUME_DB <= MAIN_AUDIO_VOLUME_DB <= 0, \
        f"Main audio volume {MAIN_AUDIO_VOLUME_DB}dB must be between {MIN_VOLUME_DB}dB and {MAX_VOLUME_DB}dB"
    assert MIN_VOLUME_DB <= SOUND_EFFECT_VOLUME_DB <= MAX_VOLUME_DB, \
        f"Sound effect volume {SOUND_EFFECT_VOLUME_DB}dB must be between {MIN_VOLUME_DB}dB and {MAX_VOLUME_DB}dB"
    
    # Convert effect bytes to AudioSegment
    effect_audio = AudioSegment.from_file(BytesIO(effect), format="mp3")
    
    # Add reverb to the sound effect
    effect_audio = add_reverb(effect_audio)
    
    # Apply volume adjustments with safety check
    adjusted_chunk_audio = chunk.audio_data + MAIN_AUDIO_VOLUME_DB
    
    # Calculate current volume level of effect_audio (approximate)
    effect_volume = effect_audio.dBFS
    target_volume = effect_volume + SOUND_EFFECT_VOLUME_DB
    
    # If target volume would exceed MAX_VOLUME_DB, adjust the volume difference
    if target_volume > MAX_VOLUME_DB:
        volume_adjustment = MAX_VOLUME_DB - effect_volume
        print(f"[DEBUG] mix_effect_with_chunk: Reducing sound effect volume from {SOUND_EFFECT_VOLUME_DB}dB to {volume_adjustment}dB to stay under maximum")
        adjusted_effect_audio = effect_audio + volume_adjustment
    else:
        adjusted_effect_audio = effect_audio + SOUND_EFFECT_VOLUME_DB
    
    # Calculate chunk duration in seconds
    chunk_duration = len(adjusted_chunk_audio) / 1000.0  # Convert ms to seconds
    
    # Adjust timing if it exceeds chunk duration
    if timing > chunk_duration:
        print(f"[DEBUG] mix_effect_with_chunk: Adjusting timing from {timing}s to fit within chunk duration {chunk_duration}s")
        # Place effect at the start of the chunk
        timing = 0.0
        
        # Trim effect duration if needed
        effect_duration = len(adjusted_effect_audio) / 1000.0
        if effect_duration > chunk_duration:
            print(f"[DEBUG] mix_effect_with_chunk: Trimming effect from {effect_duration}s to {chunk_duration}s")
            adjusted_effect_audio = adjusted_effect_audio[:int(chunk_duration * 1000)]
    
    # Save the mixed version of the effect
    effects_dir = "sound_effects"
    mix_filename = f"{effects_dir}/mixed_effect_{chunk.start_time:.2f}s_at_{timing:.2f}s.mp3"
    if SAVE_INTERMEDIATE_FILES:
        adjusted_effect_audio.export(mix_filename, format='mp3')
        print(f"[DEBUG] mix_effect_with_chunk: Saved mixed effect to {mix_filename}")
    else:
        print(f"[DEBUG] mix_effect_with_chunk: Skipping saving mixed effect file.")
    
    # Convert timing from seconds to milliseconds for pydub
    offset_ms = int(timing * 1000)
    assert offset_ms >= 0, f"Invalid timing: {timing} seconds"
    
    # Overlay the effect on top of the chunk audio
    mixed_audio = adjusted_chunk_audio.overlay(adjusted_effect_audio, position=offset_ms)
    
    # Verify the mixed audio duration matches original
    assert len(mixed_audio) == len(chunk.audio_data), "Mixed audio duration differs from original chunk"
    
    new_chunk = AudioChunk(
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        audio_data=mixed_audio,
        transcription=chunk.transcription
    )
    return new_chunk

# New: Process an individual audio chunk concurrently.
def process_chunk(chunk: AudioChunk, client: ElevenLabs) -> AudioChunk:
    """
    Process an individual audio chunk concurrently.
    Generates a sound effect based on LLM output and mixes it with the chunk if applicable.
    
    Args:
        chunk (AudioChunk): AudioChunk to process.
        client (ElevenLabs): Initialized ElevenLabs client.
        
    Returns:
        AudioChunk: Processed audio chunk with sound effect mixed in if needed.
    """
    prompt, timing = generate_effect_prompt(chunk)
    if prompt:
        print(f"[DEBUG] process_chunk: Sound effect prompt generated for chunk starting at {chunk.start_time} s.")
        effect = generate_sound_effect(client, prompt, timing, chunk.start_time)
        return mix_effect_with_chunk(chunk, effect, timing)
    else:
        print(f"[DEBUG] process_chunk: No sound effect needed for chunk starting at {chunk.start_time} s.")
        return chunk

def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase, removing punctuation,
    and standardizing whitespace.
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    import re
    # Convert to lowercase
    text = text.lower()
    # Remove all punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace (convert multiple spaces to single space)
    text = ' '.join(text.split())
    return text

def get_sequential_topic_timestamps(topic_words, timestamp_data):
    """
    Get start and end timestamps for each topic based on sequential word matching.
    Handles contractions and apostrophes flexibly.
    
    Args:
        topic_words (dict): Dictionary mapping topic names to lists of words
        timestamp_data (list): List of dictionaries containing word and timestamp data
    
    Returns:
        dict: Dictionary mapping topic names to (start_time, end_time) tuples
    """
    def clean_word(word):
        # Remove punctuation but preserve apostrophes for contractions
        cleaned = word.lower()
        # Handle special cases for contractions
        cleaned = cleaned.replace("'s", "s")  # Convert "there's" to "theres"
        cleaned = cleaned.replace("'ll", "ll")  # Convert "it'll" to "itll"
        # Remove any remaining punctuation
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c.isspace())
        return cleaned
    
    def find_word_match(target_word, timestamp_words):
        """Find matching word considering contractions and combined words"""
        target_clean = clean_word(target_word)
        
        # Try direct match first
        for i, item in enumerate(timestamp_words):
            if clean_word(item['word']) == target_clean:
                return float(item['timestamp'])
            
        # Try matching contractions by combining consecutive words
        if "'" in target_word:  # If target is a contraction
            for i in range(len(timestamp_words) - 1):
                combined = timestamp_words[i]['word'] + timestamp_words[i + 1]['word']
                if clean_word(combined) == target_clean:
                    return float(timestamp_words[i]['timestamp'])
        
        return None

    topic_timestamps = {}
    
    for topic_name, words in topic_words.items():
        logging.debug(f"Processing topic: {topic_name} with {len(words)} words")
        
        start_time = None
        end_time = None
        
        # Find the first word
        first_word = words[0]
        last_word = words[-1]
        
        start_time = find_word_match(first_word, timestamp_data)
        end_time = find_word_match(last_word, timestamp_data)
        
        if start_time is None:
            # Try finding the next available word if first word fails
            for word in words[1:]:
                start_time = find_word_match(word, timestamp_data)
                if start_time is not None:
                    break
                    
        if end_time is None:
            # Try finding the previous word if last word fails
            for word in reversed(words[:-1]):
                end_time = find_word_match(word, timestamp_data)
                if end_time is not None:
                    break
        
        assert start_time is not None, f"Could not find timestamp for any words at start of topic '{topic_name}'"
        assert end_time is not None, f"Could not find timestamp for any words at end of topic '{topic_name}'"
        
        topic_timestamps[topic_name] = (start_time, end_time)
        logging.debug(f"Topic '{topic_name}' timing: {start_time:.2f}s to {end_time:.2f}s")
    
    return topic_timestamps

def chunk_audio_by_topics(
    audio: AudioSegment,
    transcription: str,
    timestamp_file: str
) -> List['AudioChunk']:
    """
    Divide audio into chunks based on topic analysis with sequential timing.
    
    Args:
        audio (AudioSegment): Input audio
        transcription (str): Full transcription text
        timestamp_file (str): Path to timestamp JSON file
        
    Returns:
        List[AudioChunk]: List of audio chunks by topic
    """
    print("[DEBUG] chunk_audio_by_topics: Starting to chunk audio by topics.")
    chunks_dir = "audio_chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Load timestamp data
    with open(timestamp_file, 'r') as f:
        timestamp_data = [json.loads(line) for line in f if line.strip()]
    
    # Instead of generating topics (with an LLM) here,
    # load the pre-generated topics from the JSON file.
    topic_words = load_topics()
    
    # Get sequential timestamps for all topics
    topics = get_sequential_topic_timestamps(topic_words, timestamp_data)
    
    chunks = []
    for topic_name, (start_time, end_time) in topics.items():
        print(f"[DEBUG] chunk_audio_by_topics: Creating chunk for topic '{topic_name}' from {start_time}s to {end_time}s")
        
        # Convert to milliseconds for pydub
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        # Ensure we don't exceed audio length
        end_ms = min(end_ms, len(audio))
        
        # Extract audio chunk
        chunk_audio = audio[start_ms:end_ms]
        
        # Verify chunk duration
        chunk_duration = len(chunk_audio) / 1000.0  # Convert to seconds
        print(f"[DEBUG] chunk_audio_by_topics: Chunk duration: {chunk_duration:.2f} seconds")
        
        # Create chunk object
        chunk = AudioChunk(
            start_time=start_time,
            end_time=end_time,
            audio_data=chunk_audio,
            transcription=' '.join(topic_words[topic_name])
        )
        
        # Conditionally save chunk to file if enabled
        safe_topic_name = ''.join(c if c.isalnum() else '_' for c in topic_name)
        chunk_filename = f"{chunks_dir}/chunk_{safe_topic_name}_{start_time:.2f}_{end_time:.2f}.mp3"
        if SAVE_INTERMEDIATE_FILES:
            chunk_audio.export(chunk_filename, format='mp3')
            print(f"[DEBUG] chunk_audio_by_topics: Exported chunk to {chunk_filename}")
        else:
            print(f"[DEBUG] chunk_audio_by_topics: Skipping saving chunk file for topic {topic_name}.")
        
        chunks.append(chunk)
    
    assert len(chunks) > 0, "No chunks were created"
    print("[DEBUG] chunk_audio_by_topics: Completed chunk creation.")
    return chunks

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

def process_audio_with_effects(
    input_audio_path: str,
    transcription: str,
    timestamp_file: str
) -> AudioSegment:
    """
    Main function to process audio with sound effects.
    
    Args:
        input_audio_path (str): Path to input MP3 file
        transcription (str): Text transcription of the audio
        timestamp_file (str): Path to timestamp JSON file
        
    Returns:
        AudioSegment: Processed audio with added sound effects
    """
    print("[DEBUG] process_audio_with_effects: Starting processing of audio with sound effects.")
    
    # Load environment variables and initialize client
    try:
        env_path = find_env_file(os.path.dirname(__file__))
        print(f"[DEBUG] Found .env file at: {env_path}")
        load_dotenv(env_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        raise
    
    api_key = os.getenv('ELEVENLABS_API_KEY')
    assert api_key is not None, "ELEVENLABS_API_KEY not found in environment variables"
    
    client = ElevenLabs(api_key="sk_0b281c6f098d228856648d1cf8fe04f14001bbfe7279b4d0")
    
    # Load and process audio
    audio = load_audio(input_audio_path)
    chunks = chunk_audio_by_topics(audio, transcription, timestamp_file)
    print(f"[DEBUG] process_audio_with_effects: {len(chunks)} chunks created.")
    
    # Process chunks concurrently instead of sequentially
    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_chunks = list(executor.map(lambda chunk: process_chunk(chunk, client), chunks))
    
    # Combine processed chunks
    final_audio = AudioSegment.empty()
    for chunk in processed_chunks:
        final_audio += chunk.audio_data
    
    print("[DEBUG] process_audio_with_effects: Completed processing all chunks.")
    return final_audio

def save_audio(audio: AudioSegment, output_path: str) -> None:
    """
    Save processed audio to file.
    
    Args:
        audio (AudioSegment): Processed audio
        output_path (str): Output file path
    """
    print(f"[DEBUG] save_audio: Saving audio to file: {output_path}")
    assert output_path.endswith('.mp3'), "Output file must be MP3 format"
    audio.export(output_path, format='mp3')

def read_transcription(jsonl_path: str) -> str:
    """
    Read transcription from a JSONL file.
    
    Args:
        jsonl_path (str): Path to the JSONL file
        
    Returns:
        str: Transcription text
        
    Raises:
        AssertionError: If file doesn't exist or isn't a JSONL file
    """
    print(f"[DEBUG] read_transcription: Reading transcription from file: {jsonl_path}")
    assert os.path.exists(jsonl_path), f"Transcription file not found: {jsonl_path}"
    assert jsonl_path.endswith('.jsonl'), "Transcription file must be JSONL format"
    
    import json
    with open(jsonl_path, 'r') as f:
        data = json.load(f)
    assert isinstance(data, dict) and 'text' in data, "Invalid JSONL format"
    print("[DEBUG] read_transcription: Finished reading transcription.")
    return data['text']

if __name__ == "__main__":
    start_time = time.time()  # Start timing the whole process
    
    transcription = read_transcription("audio_transcription.jsonl")
    print("[DEBUG] Main: Transcription loaded successfully.")
    
    final_audio = process_audio_with_effects(
        "audio.mp3",
        transcription,
        "audio_to_timestamp.jsonl"
    )
    print("[DEBUG] Main: Audio processing complete, saving file 'audio_with_sound_effects.mp3'")
    
    save_audio(final_audio, "audio_with_sound_effects.mp3")
    print("[DEBUG] Main: Audio with sound effects saved successfully.")
    
    total_duration = time.time() - start_time
    print(f"[DEBUG] Total processing time: {total_duration:.2f} seconds")