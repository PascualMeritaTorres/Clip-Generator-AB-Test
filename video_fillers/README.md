Task C.1 = Identify places to put music + what type of music + what specific track
Task C.2 = Identify places to put sound effects + generate prompt for elevenlabs for sound effect
Task C.3 = Identify main topics + Search internet for images

How to run

- Run final_pipeline.py
- Note that you need to have two files: the audio.mp3 and the timestamps.json for the timestamps

## File Structure

- `final_pipeline.py` - Main execution script
- `preprocess_input.py` - Handles initial data processing from Iñaki code to my code format
- `add_sound_effects.py` - Manages sound effect generation and placement
- `add_images.py` - Handles AI image generation and integration
- `timestamps.json` - Contains speech segment timing information
