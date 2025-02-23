from elevenlabs import ElevenLabs
import os
import random

class VoiceDesigner:
    """
    A class for creating and managing voice designs using ElevenLabs API.
    """
    
    def __init__(self):
        """Initialize the VoiceDesigner with ElevenLabs client."""
        self.client = ElevenLabs(
            api_key=os.getenv("ELEVENLABS_API_KEY")
        )

    async def create_voice(self, voice_name: str, voice_description: str, text: str) -> str:
        """
        Create a new voice using ElevenLabs API.
        
        Args:
            voice_name (str): Name for the new voice
            voice_description (str): Description of the voice characteristics
            text (str): Sample text used to generate the voice
            
        Returns:
            str: The ID of the created voice
        """
        # Create a new voice using the ElevenLabs API
        voice = await self.client.voices.add(
            name=voice_name,
            description=voice_description,
            text=text
        )
        
        return voice.voice_id

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


