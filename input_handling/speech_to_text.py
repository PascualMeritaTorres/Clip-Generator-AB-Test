import asyncio
import fal_client

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
