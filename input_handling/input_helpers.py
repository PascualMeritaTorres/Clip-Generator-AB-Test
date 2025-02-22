
import moviepy.editor as mp
import os

MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25MB

def extract_audio_from_video(video_file, output_audio_file, max_video_size=MAX_VIDEO_SIZE, max_audio_size=MAX_AUDIO_SIZE):
    """
    Extracts audio from a video file and saves it to an output audio file.

    Parameters:
    video_file (str): Path to the input video file.
    output_audio_file (str): Path to the output audio file where the extracted audio will be saved.
    max_video_size (int, optional): Maximum allowed size of the video file in bytes. Defaults to MAX_VIDEO_SIZE.
    max_audio_size (int, optional): Maximum allowed size of the extracted audio file in bytes. Defaults to MAX_AUDIO_SIZE.

    Raises:
    FileNotFoundError: If the video file does not exist.
    ValueError: If the video file exceeds the maximum allowed size or if the extracted audio file exceeds the maximum allowed size.
    RuntimeError: If there is an error during the audio extraction process.

    Notes:
    - The supported video formats depend on the `moviepy` library, which typically includes formats like MP4, AVI, MOV, etc.
    - The extracted audio is saved in the format specified by the `output_audio_file` extension, typically MP3 or WAV.
    """
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