from typing import Optional
from .speech_to_text import SpeechToText
from .input_helpers import extract_audio_from_video
import os

def process_input(file_path: str, file_type: str) -> Optional[str]:
    """
    Process different types of input files and return transcription if applicable.
    
    Args:
        file_path (str): Path to the input file
        file_type (str): MIME type of the file (e.g., 'video/mp4', 'audio/mpeg', 'audio/mp3', 'text/plain')
        
    Returns:
        Optional[str]: Transcription text if successful, None if processing fails
    """
    try:
        # Handle markdown/text files
        if file_type == "text/plain":
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        # Handle video files
        elif file_type == "video/mp4":
            # Extract audio from video
            audio_path = file_path.replace('.mp4', '.mp3')
            extract_audio_from_video(file_path, audio_path)
            
            # Process the extracted audio
            transcriber = SpeechToText()
            transcription = transcriber.transcribe_audio(audio_path)
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            return transcription

        # Handle audio files (both mp3 and mpeg)
        elif file_type in ["audio/mpeg", "audio/mp3"]:
            transcriber = SpeechToText()
            return transcriber.transcribe_audio(file_path)

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    except Exception as e:
        print(f"Error processing file: {e}")
        return None
