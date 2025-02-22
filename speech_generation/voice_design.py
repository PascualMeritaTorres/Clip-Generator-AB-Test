from elevenlabs import ElevenLabs
import os
import random

class VoiceDesigner:
    """
    A class used to design and create voice previews and voices using the ElevenLabs API.

    Attributes
    ----------
    client : ElevenLabs
        An instance of the ElevenLabs client initialized with the API key.

    Methods
    -------
    _create_voice_previews(voice_description: str, text: str) -> str
        Creates voice previews based on the provided voice description and text, and returns the generated voice ID.

    create_voice(voice_name: str, voice_description: str, text: str)
        Creates a voice using the provided voice name, voice description, and text by generating a voice preview first.
    """

    def __init__(self):
        self.client = ElevenLabs(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        )

    def _create_voice_previews(self, voice_description: str, text: str) -> str:
        """
        Creates voice previews based on the provided voice description and text, and returns the generated voice ID.
        """
        response = self.client.text_to_voice.create_previews(
            voice_description=voice_description,
            text=text
        )
        previews = response.previews
        return random.choice(previews).generated_voice_id  # Access the generated_voice_id attribute from a random preview

    def create_voice(self, voice_name: str, voice_description: str, text: str):
        """
        Creates a voice using the provided voice name, voice description, and text.
        """
        return self.client.text_to_voice.create_voice_from_preview(
            voice_name= voice_name,
            voice_description=voice_description,
            generated_voice_id=self._create_voice_previews(
                voice_description=voice_description,
                text=text
            )
        ).voice_id


