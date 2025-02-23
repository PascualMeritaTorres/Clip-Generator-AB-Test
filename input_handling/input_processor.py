import asyncio
import fal_client
import moviepy as mp
import os
from typing import Optional

# Constants
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
MAX_AUDIO_SIZE = 25 * 1024 * 1024   # 25MB
SUPPORTED_AUDIO_TYPES = ["audio/mpeg", "audio/mp3"]
SUPPORTED_VIDEO_TYPES = ["video/mp4"]
SUPPORTED_TEXT_TYPES = ["text/plain"]

# Set FAL API key directly
os.environ["FAL_KEY"] = "7f5d1518-7cca-495e-9a2d-55984ccb5966:090b6d85ebc5b716ef5bc0464a14a1a8"

class InputProcessor:
    """
    A class to handle processing of different input types (video, audio, text) 
    and convert them to text format when applicable.
    
    This class combines audio extraction, speech-to-text conversion, and general
    input processing capabilities.
    
    Requirements:
        - FAL_KEY environment variable must be set for speech-to-text functionality
        - moviepy library for video processing
    """
    
    @staticmethod
    def extract_audio_from_video(
        video_file: str, 
        output_audio_file: str,
        max_video_size: int = MAX_VIDEO_SIZE,
        max_audio_size: int = MAX_AUDIO_SIZE
    ) -> None:
        """
        Extracts audio from a video file and saves it to an output audio file.

        Args:
            video_file (str): Path to the input video file
            output_audio_file (str): Path where the extracted audio will be saved
            max_video_size (int): Maximum allowed video file size in bytes
            max_audio_size (int): Maximum allowed audio file size in bytes

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If file size exceeds limits
            RuntimeError: If audio extraction fails
        """
        assert isinstance(video_file, str), "video_file must be a string"
        assert isinstance(output_audio_file, str), "output_audio_file must be a string"
        
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file '{video_file}' does not exist")

        if os.path.getsize(video_file) > max_video_size:
            raise ValueError(f"Video file exceeds {max_video_size / (1024 * 1024)}MB limit")

        try:
            with mp.VideoFileClip(video_file) as video:
                video.audio.write_audiofile(output_audio_file)

            if os.path.getsize(output_audio_file) > max_audio_size:
                os.remove(output_audio_file)
                raise ValueError(f"Extracted audio file exceeds {max_audio_size / (1024 * 1024)}MB limit")
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio from video: {e}")

    async def _transcribe(self, audio_file: str) -> dict:
        """
        Internal method to handle audio transcription using Fal client.

        Args:
            audio_file (str): Path to the audio file

        Returns:
            dict: Transcription result from the Fal API
        """
        assert isinstance(audio_file, str), "audio_file must be a string"
        
        try:
            # Create new event loop for this context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            audio_url = fal_client.upload_file(audio_file)
            
            result = await fal_client.subscribe_async(
                "fal-ai/whisper",
                arguments={
                    "audio_url": audio_url,
                    "task": "transcribe",
                    "language": "en",
                    "response_format": "text",
                }
            )
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio: {e}")
        finally:
            loop.close()

    async def transcribe_audio_async(self, audio_file: str) -> str:
        """
        Async method to transcribe audio file to text.
        
        Args:
            audio_file (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        assert isinstance(audio_file, str), "audio_file must be a string"
        assert os.path.exists(audio_file), f"Audio file {audio_file} does not exist"
        
        result = await self._transcribe(audio_file)
        return result.get("text", "").strip()

    def transcribe_audio(self, audio_file: str) -> str:
        """
        Synchronous wrapper for transcribe_audio_async.
        
        Args:
            audio_file (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Create new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async function
            result = loop.run_until_complete(self._transcribe(audio_file))
            return result.get("text", "").strip()
        finally:
            loop.close()

    async def process_input_async(self, file_path: str, file_type: str) -> Optional[str]:
        """
        Async version of process_input.
        
        Args:
            file_path (str): Path to the input file
            file_type (str): MIME type of the file
            
        Returns:
            Optional[str]: Transcription text if successful, None if processing fails
        """
        assert isinstance(file_path, str), "file_path must be a string"
        assert isinstance(file_type, str), "file_type must be a string"
        assert os.path.exists(file_path), f"File {file_path} does not exist"
        
        try:
            # Handle text files
            if file_type in SUPPORTED_TEXT_TYPES:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            # Handle video files
            elif file_type in SUPPORTED_VIDEO_TYPES:
                audio_path = file_path.replace('.mp4', '.mp3')
                self.extract_audio_from_video(file_path, audio_path)
                
                try:
                    transcription = await self.transcribe_audio_async(audio_path)
                finally:
                    # Clean up temporary audio file
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                        
                return transcription

            # Handle audio files
            elif file_type in SUPPORTED_AUDIO_TYPES:
                return await self.transcribe_audio_async(file_path)

            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:
            print(f"Error processing file: {e}")
            return None

    def process_input(self, file_path: str, file_type: str) -> Optional[str]:
        """
        Process different types of input files and return transcription if applicable.
        
        Args:
            file_path (str): Path to the input file
            file_type (str): MIME type of the file
            
        Returns:
            Optional[str]: Transcription text if successful, None if processing fails
        """
        assert isinstance(file_path, str), "file_path must be a string"
        assert isinstance(file_type, str), "file_type must be a string"
        assert os.path.exists(file_path), f"File {file_path} does not exist"
        
        try:
            # Handle text files
            if file_type in SUPPORTED_TEXT_TYPES:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            # Handle video files
            elif file_type in SUPPORTED_VIDEO_TYPES:
                audio_path = file_path.replace('.mp4', '.mp3')
                self.extract_audio_from_video(file_path, audio_path)
                
                try:
                    return self.transcribe_audio(audio_path)
                finally:
                    # Clean up temporary audio file
                    if os.path.exists(audio_path):
                        os.remove(audio_path)

            # Handle audio files
            elif file_type in SUPPORTED_AUDIO_TYPES:
                return self.transcribe_audio(file_path)

            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:
            print(f"Error processing file: {e}")
            return None 