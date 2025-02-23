from elevenlabs import ElevenLabs
import os
from voice_design import VoiceDesigner
import io
from pydub import AudioSegment
import time
import base64
import json

class AudioGenerator:
    """
    AudioGenerator is a class responsible for generating audio files from text transcriptions using the ElevenLabs API.

    Attributes:
        client (ElevenLabs): An instance of the ElevenLabs client initialized with the API key.
        voices (dict): A dictionary to store the mapping of speaker names to their corresponding voice IDs.
        voice_designer (VoiceDesigner): An instance of the VoiceDesigner class used to create new voices.

    Methods:
        __init__():
            Initializes the AudioGenerator instance with the ElevenLabs client and VoiceDesigner.

        _generate_voices(data):
            Generates or retrieves voices for each speaker in the provided data.

        _generate_audio(voice_id, text):
            Converts the given text to speech using the specified voice ID and returns the audio result.

        generate_audio_from_transcriptions(data, output_dir, pause_duration_ms=500):
            Generates audio files from the provided transcriptions and saves them to the specified output directory.
    """

    def __init__(self):
        self.nice_voice_ids = ["UgBBYS2sOqTuMpoF3BR0", "NOpBlnGInO9m6vDvFkFC", "56AoDkrOh6qfVPDXZ7Pt"]
        self.client = ElevenLabs(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        )
        self.voices = {}
        self.voice_designer = VoiceDesigner()

    async def generate_audio_from_transcriptions(self, data, output_dir, pause_duration_ms=500, preset_voices=False):
        await self._generate_voices(data, preset_voices=preset_voices)  # Ensure voices are populated before generating audio
        silence = AudioSegment.silent(duration=pause_duration_ms)  # Create a silence segment

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = []

        for idx, entry in enumerate(data):
            combined_audio = AudioSegment.silent(duration=0)  # Start with empty audio
            transcription = entry["transcription"]

            for i, segment in enumerate(transcription):
                speaker = segment["speaker"]
                text = segment["text"]

                # Generate audio for the speaker-text pair
                response = await self._generate_audio(voice_id=self.voices[speaker], text=text)

                audio_bytes = base64.b64decode(response["audio_base64"])  # Decode the base64 content
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

                # Append to the combined audio, adding silence except for the first segment
                if i > 0:
                    combined_audio += silence  # Add pause between speakers

                combined_audio += audio_segment

            # Export final concatenated audio
            final_audio_path = os.path.join(output_dir, f"transcription_{idx}.mp3")
            combined_audio.export(final_audio_path, format="mp3")
            segment_alignments = []
            for segment in transcription:
                response = await self._generate_audio(voice_id=self.voices[segment["speaker"]], text=segment["text"])
                segment_alignments.append(response["alignment"])
            results.append({final_audio_path: segment_alignments})

            # Dump the results to a JSON file periodically
            results_path = os.path.join(output_dir, "mp3_timestamps.json")
            with open(results_path, 'w') as results_file:
                json.dump(results, results_file, indent=4)

        return results

    async def _generate_voices(self, data, preset_voices=False):
        start_time = time.time()
        if preset_voices:
            print("Using preset voices")
            speakers = {segment["speaker"] for item in data for segment in item["transcription"]}
            for i, speaker in enumerate(speakers):
                if speaker not in self.voices:
                    self.voices[speaker] = self.nice_voice_ids[i % len(self.nice_voice_ids)]
                    print(f"Assigned preset voice ID {self.voices[speaker]} to speaker {speaker}")
        else:
            for item in data:
                for speaker_info in item["speaker_voice_descriptions"]:
                    speaker = speaker_info["speaker"]
                    description = speaker_info["description"]
                    all_voices = await self.client.voices.get_all().voices
                    matched_voice = next((voice for voice in all_voices if voice.name == speaker), None)

                    if matched_voice:
                        self.voices[speaker] = matched_voice.voice_id
                        print(f"Matched existing voice for speaker: {speaker}")
                    else:
                        if speaker not in self.voices:
                            speaker_text = "".join(t["text"] for t in item["transcription"] if t["speaker"] == speaker)
                            print(f"Creating new voice for speaker: {speaker}. Speaker text: {speaker_text}")
                            self.voices[speaker] = await self.voice_designer.create_voice(voice_name=speaker,
                                                                                          voice_description=description, text=speaker_text)
                            print(f"Created new voice for speaker: {speaker}")
        end_time = time.time()
        print(f"_generate_voices took {end_time - start_time:.2f} seconds")

    async def _generate_audio(self, voice_id, text):
        start_time = time.time()
        result = self.client.text_to_speech.convert_with_timestamps(
            voice_id=voice_id,
            output_format="mp3_44100_128",
            text=text,
            model_id="eleven_turbo_v2_5"
        )
        end_time = time.time()
        print(f"_generate_audio took {end_time - start_time:.2f} seconds")
        return result

import asyncio

if __name__ == "__main__":
    ag = AudioGenerator()
    with open('neil_small.json', 'r') as file:
        data = json.load(file)
    asyncio.run(ag.generate_audio_from_transcriptions(data, "audio_generation_output", preset_voices=True))