import json
from typing import Dict, List, Tuple
import os
from pathlib import Path

def load_mp3_timestamps(filepath: str) -> Dict:
    """
    Load MP3 timestamps from JSON file.
    
    Args:
        filepath (str): Path to the JSON file containing MP3 timestamps
        
    Returns:
        Dict: Dictionary containing the MP3 timestamps data for the first transcription
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    
    transcription_data = data
    assert isinstance(transcription_data, list) and len(transcription_data) > 0, "Invalid transcription data format"
    
    return transcription_data[0]  # Return the first transcription segment

def create_word_timestamps(
    characters: List[str], start_times: List[float], end_times: List[float]
) -> List[Dict[str, str]]:
    """
    Create word-level timestamps from character-level data with millisecond precision.
    Treats punctuation marks as separate entries.
    
    For the first token, the timestamp is based on the start time of its first character;
    for every subsequent token the timestamp is taken from the end time of its last character.
    
    Args:
        characters (List[str]): List of characters.
        start_times (List[float]): List of start times for each character.
        end_times (List[float]): List of end times for each character.
        
    Returns:
        List[Dict[str, str]]: List of word timestamps in the format {"word": str, "timestamp": str}
                              where timestamp has millisecond precision (e.g. "1.234").
    """
    assert len(characters) == len(start_times) == len(end_times), "Characters, start times and end times must have the same length"
    
    # Define punctuation marks to treat separately.
    PUNCTUATION = {',', '.', ':', ';', '!', '?', '-', '"', "'", '(', ')', '[', ']', '{', '}'}
    
    result = []
    current_word = []
    token_start_index = None
    
    for i, char in enumerate(characters):
        if char == " ":
            if current_word:
                token_end_index = i - 1
                # For the first token, use the start time; else use the end time of the last character.
                ts = start_times[token_start_index] if len(result) == 0 else end_times[token_end_index]
                result.append({
                    "word": "".join(current_word),
                    "timestamp": f"{ts:.3f}"
                })
                current_word = []
                token_start_index = None
        elif char in PUNCTUATION:
            if current_word:
                token_end_index = i - 1
                ts = start_times[token_start_index] if len(result) == 0 else end_times[token_end_index]
                result.append({
                    "word": "".join(current_word),
                    "timestamp": f"{ts:.3f}"
                })
                current_word = []
                token_start_index = None
            # For punctuation token itself, decide timestamp based on token order.
            ts = start_times[i] if len(result) == 0 else end_times[i]
            result.append({
                "word": char,
                "timestamp": f"{ts:.3f}"
            })
        else:
            if not current_word:
                token_start_index = i
            current_word.append(char)
    
    # Add last token if it exists.
    if current_word:
        token_end_index = len(characters) - 1
        ts = start_times[token_start_index] if len(result) == 0 else end_times[token_end_index]
        result.append({
            "word": "".join(current_word),
            "timestamp": f"{ts:.3f}"
        })
    
    return result

def create_transcription(characters: List[str]) -> Dict[str, str]:
    """
    Create full transcription from characters.
    
    Args:
        characters (List[str]): List of characters
        
    Returns:
        Dict[str, str]: Dictionary containing the full text transcription
    """
    return {"text": "".join(characters)}

def save_jsonl(data: List[Dict], filepath: str) -> None:
    """
    Save data to JSONL format.
    
    Args:
        data (List[Dict]): Data to save
        filepath (str): Output filepath
    """
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def save_transcription(data: Dict, filepath: str) -> None:
    """
    Save transcription data to JSONL format.
    
    Args:
        data (Dict): Transcription data
        filepath (str): Output filepath
    """
    with open(filepath, 'w') as f:
        f.write(json.dumps(data) + '\n')

def process_timestamps(input_filepath: str, output_dir: str = None) -> Tuple[str, str]:
    """
    Process MP3 timestamps and create word timestamps and transcription files.
    
    Args:
        input_filepath (str): Path to input MP3 timestamps JSON file.
        output_dir (str, optional): Directory to save output files. Defaults to current directory.
        
    Returns:
        Tuple[str, str]: Paths to the created timestamp and transcription files.
    """
    # Use current directory if output_dir not specified.
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
    
    input_filepath = Path(input_filepath)
    assert input_filepath.exists(), f"Input file {input_filepath} does not exist"
    
    # Load input data.
    data = load_mp3_timestamps(str(input_filepath))
    
    # Extract characters, start times and end times.
    characters = data["characters"]
    start_times = data["character_start_times_seconds"]
    end_times = data["character_end_times_seconds"]
    
    # Create word timestamps using end times for every element except the first.
    word_timestamps = create_word_timestamps(characters, start_times, end_times)
    
    # Create transcription.
    transcription = create_transcription(characters)
    
    # Define output paths.
    timestamp_path = output_dir / "audio_to_timestamp.jsonl"
    transcription_path = output_dir / "audio_transcription.jsonl"
    
    # Save outputs.
    save_jsonl(word_timestamps, str(timestamp_path))
    save_transcription(transcription, str(transcription_path))
    
    return str(timestamp_path), str(transcription_path)

if __name__ == "__main__":
    input_file = "timestamps.json"
    
    timestamp_file, transcription_file = process_timestamps(input_file)
    print(f"Created timestamp file: {timestamp_file}")
    print(f"Created transcription file: {transcription_file}") 