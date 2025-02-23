import asyncio
import fal_client
import time

class SpeechToText:
    """Speech to Text class using the Fal client.
    It is required to have a FAL_KEY set in the environment variables.
    """
    def transcribe_audio(self, audio_file) -> str:
        result = asyncio.run(self._subscribe(audio_file))
        return result.get("text", "").strip()

    async def _subscribe(self, audio_file):
        audio_url = fal_client.upload_file(audio_file)

        handler = await fal_client.submit_async(
            "fal-ai/whisper",
            arguments={
                "audio_url": audio_url,
                "task": "transcribe",
                "language": "en",
                "response_format": "text",
            },
        )

        result = await handler.get()

        return result

# if __name__ == "__main__":
#     stt = SpeechToText()
#     start_time = time.time()
#     transcription = stt.transcribe_audio("Will An Asteroid Hit Earth in 2032.mp3")
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     with open("Will An Asteroid Hit Earth in 2032.txt", "w") as f:
#         f.write(transcription)
#     print(f"Transcription completed in {elapsed_time:.2f} seconds")