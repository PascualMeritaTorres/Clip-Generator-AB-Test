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

    """

    def __init__(self):
        self.client = ElevenLabs(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        )
        self.voices = {}
        self.voice_designer = VoiceDesigner()

    def _generate_voices(self, data):
        start_time = time.time()
        for item in data:
            for speaker_info in item["speaker_voice_descriptions"]:
                speaker = speaker_info["speaker"]
                description = speaker_info["description"]
                all_voices = self.client.voices.get_all().voices
                matched_voice = next((voice for voice in all_voices if voice.name == speaker), None)

                if matched_voice:
                    self.voices[speaker] = matched_voice.voice_id
                    print(f"Matched existing voice for speaker: {speaker}")
                else:
                    if speaker not in self.voices:
                        speaker_text = "".join(t["text"] for t in item["transcription"] if t["speaker"] == speaker)
                        print(f"Creating new voice for speaker: {speaker}. Speaker text: {speaker_text}")
                        self.voices[speaker] = self.voice_designer.create_voice(voice_name=speaker,
                                                                                voice_description=description, text=speaker_text)
                        print(f"Created new voice for speaker: {speaker}")
        end_time = time.time()
        print(f"_generate_voices took {end_time - start_time:.2f} seconds")

    def _generate_audio(self, voice_id, text):
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

    def generate_audio_from_transcriptions(self, data, output_dir, pause_duration_ms=500):
        self._generate_voices(data)
        output_files = []
        silence = AudioSegment.silent(duration=pause_duration_ms)  # Create a silence segment

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, entry in enumerate(data):
            combined_audio = AudioSegment.silent(duration=0)  # Start with empty audio
            transcription = entry["transcription"]

            for i, segment in enumerate(transcription):
                speaker = segment["speaker"]
                text = segment["text"]

                # Generate audio for the speaker-text pair
                response = self._generate_audio(voice_id=self.voices[speaker], text=text)

                audio_bytes = base64.b64decode(response["audio_base64"])  # Decode the base64 content
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

                # Append to the combined audio, adding silence except for the first segment
                if i > 0:
                    combined_audio += silence  # Add pause between speakers

                combined_audio += audio_segment

            # Export final concatenated audio
            final_audio_path = os.path.join(output_dir, f"transcription_{idx}.mp3")
            combined_audio.export(final_audio_path, format="mp3")
            output_files.append(final_audio_path)

        return output_files


if __name__ == "__main__":
    ag = AudioGenerator()
    with open('transcription.json', 'r') as file:
        data = json.load(file)
    ag.generate_audio_from_transcriptions(data, "audio_generation_output")