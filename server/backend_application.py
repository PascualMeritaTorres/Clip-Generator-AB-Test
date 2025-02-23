import logging
import uvicorn
from pydantic import BaseModel
from fastapi.responses import JSONResponse
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
