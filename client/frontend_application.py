import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Rest of the imports
import streamlit as st
from services.db_services import DbServices

# Comment out the import since we're ignoring authentication
# from config import get_authenticated_service
from services.backend_services import BackendServices
import tempfile
import asyncio
from input_handling.input_processor import InputProcessor
from youtube_interaction.youtube_receiver import YoutubeReceiver
from youtube_interaction.youtube_uploader import YoutubeUploader
from youtube_interaction.config import get_authenticated_service
from content_generation.content_generation import main as generate_content_variations
from speech_generation.generate_audio import AudioGenerator
from content_rating.content_rating import generate_rating
from video_fillers.video_fillers_pipeline import main as process_video_pipeline

# Add at the top with other constants
DEMO_MODE = True  # Set to False for actual backend processing
DEMO_ASSETS_DIR = Path(__file__).parent / "demo_assets"
OUTPUTS_DIR = Path(__file__).parent / "outputs"


def main():
    if "youtube_authenticated_client" not in st.session_state:
        st.session_state.youtube_authenticated_client = None
    # Sidebar navigation
    st.sidebar.title("Virl Labs")
    st.sidebar.markdown("## Menu")
    page = st.sidebar.radio("Go to", ["Content Lab", "Analytics"])

    if page == "Content Lab":
        if st.session_state.youtube_authenticated_client is None:
            st.session_state.youtube_authenticated_client = get_authenticated_service()
        st.title("Welcome to Virl Labs")
        st.markdown(
            """
        ### <span style="color: red;">Transform Your Podcasts into Viral Short-Form Content</span>
        
        Upload your podcast audio and let our AI:
        - 🎯 Extract the most engaging clips
        - 🔄 Create multiple viral variations
        - 🎵 Add perfect sound effects
        - 🖼️ Match with compelling visuals
        - 📊 A/B test performance
        - 🧠 Self-improve through performance analytics to optimize future content
        """, unsafe_allow_html=True
        )

        # File uploader section
        uploaded_file = st.file_uploader("Upload your file", type=["mp3", "mp4", "wav"])

        if uploaded_file is not None:
            # Create columns for file details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("📁 **Filename:**", uploaded_file.name)
            with col2:
                st.write("📋 **Type:**", uploaded_file.type)
            with col3:
                st.write("📊 **Size:**", f"{uploaded_file.size/1024:.1f} KB")

            # Configuration section
            st.markdown("### Content Generation Settings")
            col1, col2 = st.columns(2)
            with col1:
                num_clips = st.slider("Number of clips to generate", 1, 10, 3)
                variations_per_clip = st.slider("Variations per clip", 1, 5, 2)
            with col2:
                platform = st.selectbox(
                    "Target Platform",
                    ["YouTube Shorts", "TikTok", "Instagram Reels"],
                    index=0,
                )  # Set YouTube Shorts as default
                content_style = st.selectbox(
                    "Content Style",
                    ["Educational", "Entertaining", "News", "Commentary"],
                    index=0,
                )  # Set Educational as default

            # Create temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name

            try:
                # Process audio files
                if uploaded_file.type in [
                    "audio/mpeg",
                    "audio/mp3",
                    "audio/wav",
                    "video/mp4",
                ]:
                    # Create temporary file
                    st.audio(uploaded_file)
                    if st.button("🚀 Generate Viral Variations"):
                        with st.spinner("Processing your content..."):
                            if DEMO_MODE:
                                # Demo Mode: load pre-saved transcription and variations
                                transcription_file = (
                                    DEMO_ASSETS_DIR
                                    / "audio_transcription_20250223_091224.json"
                                )
                                variations_file = (
                                    DEMO_ASSETS_DIR
                                    / "audio_variations_20250223_091239.json"
                                )
                                with open(
                                    transcription_file, "r", encoding="utf-8"
                                ) as f:
                                    transcription_data = json.load(f)
                                with open(variations_file, "r", encoding="utf-8") as f:
                                    variations_data = json.load(f)
                                transcription = transcription_data["content"]["transcription"]
                                st.markdown("### Transcription")
                                st.text(transcription)

                                variations = variations_data["variations"]
                            else:
                                # Non-demo mode: process the uploaded audio file
                                temp_path = None
                                with tempfile.NamedTemporaryFile(
                                    delete=False,
                                    suffix=os.path.splitext(uploaded_file.name)[1],
                                ) as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    temp_path = tmp_file.name
                                processor = InputProcessor()
                                transcription = processor.process_input(
                                    temp_path, uploaded_file.type
                                )
                                if transcription:
                                    # Save transcription
                                    transcription_path = save_transcription(
                                        transcription, uploaded_file.name
                                    )
                                    st.success(
                                        f"Transcription saved to {transcription_path}"
                                    )
                                    st.markdown("### Transcription")
                                    st.text(transcription)

                                    # Generate content variations
                                    with st.spinner("Generating viral variations..."):
                                        variations = generate_content_variations(
                                            transcription
                                        )

                                        # Save variations
                                        variations_path = save_variations(
                                            variations, uploaded_file.name
                                        )
                                        st.success(
                                            f"Variations saved to {variations_path}"
                                        )
                                else:
                                    st.error(
                                        "Failed to process content. Please try again."
                                    )

                        # Display generated variations (handle demo and non-demo modes)
                        st.markdown("### Generated Content Variations")
                        if DEMO_MODE:
                            for i, variation in enumerate(variations, 1):
                                with st.expander(f"Variation {i}", expanded=True):
                                    st.markdown("#### Script")
                                    for segment in variation["content"][
                                        "transcription"
                                    ]:
                                        st.markdown(
                                            f"**{segment['speaker']}:** {segment['text']}"
                                        )
                                    st.markdown("#### Voice Descriptions")
                                    for speaker, desc in variation["content"][
                                        "speaker_voice_descriptions"
                                    ].items():
                                        st.markdown(f"**{speaker}:** {desc}")
                                    st.markdown("#### Content Details")
                                    st.markdown(
                                        f"**Title:** {variation['content']['params']['title']}"
                                    )
                                    st.markdown(
                                        f"**Description:** {variation['content']['params']['description']}"
                                    )
                                    st.markdown(
                                        f"**Modifications:** {variation['content']['params']['modifications']}"
                                    )
                                    st.markdown(
                                        f"**Summary:** {variation['content']['params']['short_modifications']}"
                                    )
                                    st.markdown(
                                        f"**ID:** {variation['content']['params']['id']}"
                                    )
                        else:
                            for i, variation in enumerate(variations, 1):
                                with st.expander(f"Variation {i}", expanded=True):
                                    st.markdown("#### Script")
                                    for segment in variation["transcription"]:
                                        st.markdown(
                                            f"**{segment['speaker']}:** {segment['text']}"
                                        )
                                    st.markdown("#### Voice Descriptions")
                                    for speaker, desc in variation[
                                        "speaker_voice_descriptions"
                                    ].items():
                                        st.markdown(f"**{speaker}:** {desc}")
                                    st.markdown("#### Content Details")
                                    st.markdown(
                                        f"**Title:** {variation['params']['title']}"
                                    )
                                    st.markdown(
                                        f"**Description:** {variation['params']['description']}"
                                    )
                                    st.markdown(
                                        f"**Modifications:** {variation['params']['modifications']}"
                                    )
                                    st.markdown(
                                        f"**Summary:** {variation['params']['short_modifications']}"
                                    )
                                    st.markdown(f"**ID:** {variation['params']['id']}")

                        # ----- Audio Generation Section -----
                        with st.spinner("Processing audio for variations..."):
                            # Assert that variations is a list
                            assert isinstance(
                                variations, list
                            ), "variations should be a list"

                            try:
                                # Process audio for variations
                                audio_results = asyncio.run(
                                    process_variations_to_audio(
                                        variations, str(OUTPUTS_DIR)
                                    )
                                )

                                if audio_results:
                                    st.success("Audio processing complete!")

                                    st.markdown("### Generated Content")
                                    for i, result in enumerate(audio_results, 1):
                                        with st.expander(
                                            f"Variation {i}", expanded=True
                                        ):
                                            # Display variation content
                                            variation = variations[i - 1]
                                            if DEMO_MODE:
                                                content = variation.get("content", {})
                                                transcription = content.get(
                                                    "transcription", []
                                                )
                                                speaker_descriptions = content.get(
                                                    "speaker_voice_descriptions", {}
                                                )
                                                params = content.get("params", {})
                                            else:
                                                transcription = variation.get(
                                                    "transcription", []
                                                )
                                                speaker_descriptions = variation.get(
                                                    "speaker_voice_descriptions", {}
                                                )
                                                params = variation.get("params", {})

                                            # Display script
                                            st.markdown("#### Script")
                                            for segment in transcription:
                                                st.markdown(
                                                    f"**{segment['speaker']}:** {segment['text']}"
                                                )

                                            # Display voice descriptions
                                            st.markdown("#### Voice Descriptions")
                                            for (
                                                speaker,
                                                desc,
                                            ) in speaker_descriptions.items():
                                                st.markdown(f"**{speaker}:** {desc}")

                                            # Display content details
                                            st.markdown("#### Content Details")
                                            st.markdown(
                                                f"**Title:** {params.get('title', 'N/A')}"
                                            )
                                            st.markdown(
                                                f"**Description:** {params.get('description', 'N/A')}"
                                            )
                                            st.markdown(
                                                f"**Modifications:** {params.get('modifications', 'N/A')}"
                                            )
                                            st.markdown(
                                                f"**Summary:** {params.get('short_modifications', 'N/A')}"
                                            )

                                            # Display audio if available
                                            if result.get("audio_path"):
                                                try:
                                                    with open(
                                                        result["audio_path"], "rb"
                                                    ) as af:
                                                        audio_bytes = af.read()
                                                    st.audio(
                                                        audio_bytes, format="audio/mp3"
                                                    )
                                                    st.caption(
                                                        f"Audio file: {Path(result['audio_path']).name}"
                                                    )
                                                except Exception as e:
                                                    st.error(
                                                        f"Error loading audio: {str(e)}"
                                                    )

                                            # Display video if available
                                            if (
                                                result.get("video_path")
                                                and Path(result["video_path"]).exists()
                                            ):
                                                try:
                                                    with open(
                                                        result["video_path"], "rb"
                                                    ) as vf:
                                                        video_bytes = vf.read()
                                                    st.video(video_bytes)
                                                    st.caption(
                                                        f"Video file: {Path(result['video_path']).name}"
                                                    )
                                                except Exception as e:
                                                    st.error(
                                                        f"Error loading video: {str(e)}"
                                                    )

                            except Exception as e:
                                st.error(f"Error processing variations: {str(e)}")
            finally:
                # Clean up temporary file only in non-demo mode
                if not DEMO_MODE and "temp_path" in locals() and temp_path:
                    os.unlink(temp_path)

    elif page == "Analytics":
        st.title("Performance Analytics 📈")

        # Get data from the database
        db_services = DbServices()
        db_data = db_services.get_db_data()
        video_ids = [video["video_id"] for video in db_data.get("videos", [])]
        video_ratings = db_services.get_video_ratings(
            st.session_state.youtube_authenticated_client, video_ids
        )
        st.write(video_ratings)
        total_views = sum([int(video["viewCount"]) for video in video_ratings.values()])
        comment_counts = sum(
            [int(video["commentCount"]) for video in video_ratings.values()]
        )
        rating_results = generate_rating(video_ratings)
        st.write(rating_results)

        # Analytics Overview
        st.markdown("### Content Performance Overview")

        if not db_data.get("videos", []):
            st.info(
                "No content has been published yet. Generate some variations to get started!"
            )
        else:
            # Add metrics overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Clips", len(db_data.get("videos", [])))
            with col2:
                st.metric("Total Views", total_views)
            with col3:
                st.metric("Comment Count", comment_counts)
            with col4:
                st.metric("A/B Tests Running", "5")

            # Show detailed performance
            for video in db_data.get("videos", []):
                with st.expander(f"📊 {video['video_name']}", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    col1.markdown(f"**ID:** {video['video_id']}")
                    col2.markdown(
                        f"**Performance Score:** {video.get('performance_score', 'N/A')}"
                    )
                    if col3.button("View Analytics", key=video["video_id"]):
                        youtube_receiver = YoutubeReceiver(
                            output_dir="output",
                            youtube_authenticated_client=st.session_state.youtube_authenticated_client,
                        )
                        video_details = youtube_receiver.get_video_details(
                            video_id=video["video_id"]
                        )
                        if video_details is None:
                            st.write("Video not found.")
                        else:
                            st.write("### Detailed Analytics")
                            st.json(video_details)


async def process_variations_to_audio(variations: list, output_dir: str) -> list:
    """
    Convert content variations to audio files and process them through the video pipeline.
    
    If in demo mode and pre-saved demo files are available, load them.
    Otherwise (or if demo assets are missing), process the variation in production mode.
    
    Args:
        variations (list): List of content variation dictionaries.
        output_dir (str): Directory where audio files are saved.
        
    Returns:
        list: A list of dictionaries with keys 'audio_path', 'alignments', and 'video_path'
              indicating the generated results for each variation.
    """
    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create an empty placeholder at the bottom of the page
    status_container = st.empty()
    status_container.info("Your content generation requests are being processing. Please wait...")
    
    for i, variation in enumerate(variations, 1):
        if DEMO_MODE:
            demo_audio = DEMO_ASSETS_DIR / f"demo_variation_{i}.mp3"
            demo_alignments = DEMO_ASSETS_DIR / f"demo_variation_{i}_alignments.json"
            demo_video = DEMO_ASSETS_DIR / f"demo_variation_{i}_final.mp4"
            
            if demo_audio.exists() and demo_alignments.exists():
                try:
                    with open(demo_alignments, 'r', encoding='utf-8') as f:
                        alignments = json.load(f)
                    
                    result = {
                        'audio_path': str(demo_audio),
                        'alignments': alignments,
                        'video_path': str(demo_video) if demo_video.exists() else None
                    }
                    
                    results.append(result)
                    # Skip to next variation if demo assets are successfully loaded
                    continue
                except (json.JSONDecodeError, OSError) as e:
                    status_container.error(f"Error loading demo files for variation {i}: {str(e)}. Falling back to production processing.")
            else:
                status_container.info(f"Processing variation {i}")
                
        # Production (non-demo) processing or fallback when demo assets are not available
        variation_dir = output_path / f"variation_{i}"
        variation_dir.mkdir(exist_ok=True)
        
        audio_gen = AudioGenerator()
        # Assert input data consistency
        assert isinstance(variation.get('speaker_voice_descriptions', {}), dict), \
            "Variation's speaker_voice_descriptions must be a dictionary"
        speaker_descriptions = [
            {"speaker": speaker, "description": desc}
            for speaker, desc in variation.get('speaker_voice_descriptions', {}).items()
        ]
        
        audio_data = [{
            "transcription": variation.get('transcription', []),
            "speaker_voice_descriptions": speaker_descriptions
        }]
        
        try:
            # Update status message for each variation
            status_container.info(f"Processing variation {i} of {len(variations)}...")
            
            # Generate audio and get alignments
            audio_result = await audio_gen.generate_audio_from_transcriptions(
                data=audio_data,
                output_dir=str(variation_dir),
                pause_duration_ms=500,
                preset_voices=False,
            )

            if not audio_result:
                status_container.error(f"Failed to generate audio for variation {i}")
                continue
                
            # Ensure the result structure is valid (one key/value pair)
            audio_path = list(audio_result[0].keys())[0]
            alignments = list(audio_result[0].values())[0]
            
            # Save alignments to file
            alignments_file = variation_dir / f"variation_{i}_alignments.json"
            with open(alignments_file, 'w', encoding='utf-8') as f:
                json.dump(alignments, f, indent=4)
            
            # Generate a timestamps file for the video pipeline
            timestamps_file = variation_dir / "timestamps.json"
            timestamps_data = {
                "words": [
                    {
                        "word": segment["text"],
                        "start": segment["start"],
                        "end": segment["end"],
                    }
                    for segment in alignments
                ]
            }
            
            with open(timestamps_file, 'w', encoding='utf-8') as f:
                json.dump(timestamps_data, f, indent=4)

            # Process through video pipeline
            with st.spinner(f"Generating video for variation {i}..."):
                try:
                    final_video_path = await process_video_pipeline(
                        audio_path, str(timestamps_file)
                    )
                    
                    # Save generated files as demo assets for future use
                    import shutil

                    DEMO_ASSETS_DIR.mkdir(exist_ok=True)
                    demo_audio_copy = DEMO_ASSETS_DIR / f"demo_variation_{i}.mp3"
                    demo_alignments_copy = DEMO_ASSETS_DIR / f"demo_variation_{i}_alignments.json"
                    demo_video_copy = DEMO_ASSETS_DIR / f"demo_variation_{i}_final.mp4"
                    
                    shutil.copy2(audio_path, demo_audio_copy)
                    shutil.copy2(alignments_file, demo_alignments_copy)
                    shutil.copy2(final_video_path, demo_video_copy)
                    
                    results.append({
                        'audio_path': str(audio_path),
                        'alignments': alignments,
                        'video_path': str(final_video_path)
                    })
                    
                    status_container.success(f"Generated video for variation {i}")
                    
                except Exception as e:
                    status_container.error(f"Error generating video for variation {i}: {str(e)}")
                    continue

        except Exception as e:
            status_container.error(f"Error processing variation {i}: {str(e)}")
            continue
            
    # Clear the status message when done
    status_container.empty()
            
    return results


def save_transcription(transcription: str, filename: str) -> Path:
    """
    Save transcription to a JSON file in the demo_assets directory.

    Args:
        transcription (str): The transcription text
        filename (str): Original filename of the uploaded file

    Returns:
        Path: Path to the saved transcription file
    """
    if not DEMO_ASSETS_DIR.exists():
        DEMO_ASSETS_DIR.mkdir(parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(filename).stem
    save_path = DEMO_ASSETS_DIR / f"{base_name}_transcription_{timestamp}.json"

    # Enhanced data structure with all necessary frontend information
    data = {
        "metadata": {
            "original_filename": filename,
            "timestamp": timestamp,
            "file_type": Path(filename).suffix,
            "processing_status": "completed",
        },
        "content": {
            "transcription": transcription,
            "word_count": len(transcription.split()),
            "duration_seconds": None,  # This would need to be passed from the audio processing
        },
        "display_config": {"show_timestamps": True, "show_speakers": True},
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return save_path


def save_variations(variations: list, filename: str) -> Path:
    """
    Save content variations to a JSON file in the demo_assets directory.

    Args:
        variations (list): List of variation dictionaries
        filename (str): Original filename of the uploaded file

    Returns:
        Path: Path to the saved variations file
    """
    if not DEMO_ASSETS_DIR.exists():
        DEMO_ASSETS_DIR.mkdir(parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(filename).stem
    save_path = DEMO_ASSETS_DIR / f"{base_name}_variations_{timestamp}.json"

    # Enhanced data structure with all necessary frontend information
    processed_variations = []
    for i, variation in enumerate(variations, 1):
        processed_variation = {
            "variation_id": f"var_{timestamp}_{i}",
            "metadata": {
                "creation_timestamp": timestamp,
                "version": "1.0",
                "status": "generated",
            },
            "content": {
                "transcription": variation["transcription"],
                "speaker_voice_descriptions": variation["speaker_voice_descriptions"],
                "params": variation["params"],
            },
            "audio": {
                "path": (
                    str(DEMO_ASSETS_DIR / f"demo_variation_{i}.mp3")
                    if DEMO_MODE
                    else None
                ),
                "duration_seconds": None,
                "format": "mp3",
            },
            "visual": {
                "background_image": None,
                "text_overlay_config": {
                    "font": "Arial",
                    "size": 24,
                    "color": "#FFFFFF",
                },
            },
            "performance_metrics": {"views": 0, "likes": 0, "comments": 0, "shares": 0},
            "frontend_display": {
                "expanded": True,
                "show_script": True,
                "show_voice_descriptions": True,
                "show_metadata": True,
            },
        }
        processed_variations.append(processed_variation)

    data = {
        "metadata": {
            "original_filename": filename,
            "timestamp": timestamp,
            "total_variations": len(variations),
        },
        "generation_config": {
            "platform": "YouTube Shorts",  # This should be dynamic based on user selection
            "content_style": "Educational",  # This should be dynamic based on user selection
            "target_duration": 60,  # in seconds
        },
        "variations": processed_variations,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return save_path


def load_saved_data(base_name: str) -> tuple:
    """
    Load the saved transcription and variations data.

    Args:
        base_name (str): Base name of the file to load

    Returns:
        tuple: (transcription_data, variations_data)
    """
    # Find the most recent files for this base_name
    transcription_files = list(
        DEMO_ASSETS_DIR.glob(f"{base_name}_transcription_*.json")
    )
    variations_files = list(DEMO_ASSETS_DIR.glob(f"{base_name}_variations_*.json"))

    transcription_data = None
    variations_data = None

    if transcription_files:
        latest_transcription = max(transcription_files, key=lambda p: p.stat().st_mtime)
        with open(latest_transcription, "r", encoding="utf-8") as f:
            transcription_data = json.load(f)

    if variations_files:
        latest_variations = max(variations_files, key=lambda p: p.stat().st_mtime)
        with open(latest_variations, "r", encoding="utf-8") as f:
            variations_data = json.load(f)

    return transcription_data, variations_data


if __name__ == "__main__":
    main()
