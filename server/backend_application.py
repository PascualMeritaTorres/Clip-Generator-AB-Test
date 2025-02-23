import logging
import uvicorn
from pydantic import BaseModel
import json
from input_handling.input_helpers import extract_audio_from_video
from input_handling.speech_to_text import SpeechToText
from speech_generation.generate_audio import AudioGenerator
from content_generation import main as cg
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File
from fastapi import FastAPI, HTTPException
from youtube_receiver import YoutubeReceiver


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="YouTube Video Details API",
    description="A simple API to get YouTube video details.",
    version="0.1.0",
)

@app.post("/transcribe_and_generate/")
async def transcribe_and_generate(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"temp/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Extract audio if the file is an mp4
        if file.filename.endswith(".mp4"):
            audio_file = extract_audio_from_video(file_location)
        else:
            audio_file = file_location

        # Transcribe the audio
        transcriber = SpeechToText()
        transcription = transcriber.transcribe(audio_file)

        # Generate content
        content = cg(transcription)

        # Generate audio from content
        audio_generator = AudioGenerator()
        output_dir = "output"
        lst = audio_generator.generate_audio_from_transcriptions(content, output_dir)

        return JSONResponse(content={"message": "Content generated successfully", "output_file": lst})

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/get_video_with_id/{video_id}")
def get_video_with_id(video_id: str):
    try:
        youtube_receiver = YoutubeReceiver(output_dir="output")
        video_details = youtube_receiver.get_video_details(video_id=video_id)
        if video_details is None:
            raise HTTPException(status_code=404, detail="Video not found.")
        return JSONResponse(content=video_details)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
