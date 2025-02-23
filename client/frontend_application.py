import os
import sys
from pathlib import Path

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

# Add at the top with other constants
DEMO_MODE = False  # Set to False for actual backend processing
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
        ### Transform Your Podcasts into Viral Short-Form Content
        
        Upload your podcast audio and let our AI:
        - 🎯 Extract the most engaging clips
        - 🔄 Create multiple viral variations
        - 🎵 Add perfect sound effects
        - 🖼️ Match with compelling visuals
        - 📊 A/B test performance
        - 🧠 Self-improve through performance analytics to optimize future content
        """
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
                                st.success("Content processed successfully!")

                                # Display the transcription
                                st.markdown("### Transcription")
                                st.text(transcription)

                                # Generate content variations
                                with st.spinner("Generating viral variations..."):
                                    variations = generate_content_variations(
                                        transcription
                                    )

                                    # Display generated variations
                                    st.markdown("### Generated Content Variations")
                                    for i, variation in enumerate(variations, 1):
                                        with st.expander(
                                            f"Variation {i}", expanded=True
                                        ):
                                            # Display transcription
                                            st.markdown("#### Script")
                                            for segment in variation["transcription"]:
                                                st.markdown(
                                                    f"**{segment['speaker']}:** {segment['text']}"
                                                )

                                            # Display voice descriptions
                                            st.markdown("#### Voice Descriptions")
                                            for speaker, desc in variation[
                                                "speaker_voice_descriptions"
                                            ].items():
                                                st.markdown(f"**{speaker}:** {desc}")

                                            # Display metadata
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
                                            st.markdown(
                                                f"**ID:** {variation['params']['id']}"
                                            )
                            else:
                                st.error("Failed to process content. Please try again.")

            finally:
                # Clean up temporary file
                os.unlink(temp_path)

    elif page == "Analytics":
        st.title("Performance Analytics 📈")

        # Get data from the database
        db_services = DbServices()
        db_data = db_services.get_db_data()

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
                st.metric("Best Performing", "2.1M views")
            with col3:
                st.metric("Avg. Engagement", "12.3%")
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


async def process_variations_to_audio(variations, output_dir):
    """
    Convert content variations to audio files.
    
    Args:
        variations (List[Dict]): List of content variations
        output_dir (str): Directory to save audio files
    
    Returns:
        List[Dict]: List of audio results with paths and timing alignments
    """
    if DEMO_MODE:
        # Check for demo audio files in demo_assets directory
        demo_results = []
        for i, variation in enumerate(variations):
            demo_file = DEMO_ASSETS_DIR / f"demo_variation_{i+1}.mp3"
            if demo_file.exists():
                # Return demo file path and mock timing alignments
                demo_results.append({
                    str(demo_file): {
                        "timing_alignments": [
                            {"start": 0, "end": 5, "text": "Demo audio segment"}
                        ]
                    }
                })
            else:
                st.warning(f"Demo file {demo_file} not found. Please ensure demo assets are in place.")
        return demo_results
    
    # Real processing mode
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        
    # Reformat the data for the audio generator
    audio_data = []
    for variation in variations:
        speaker_descriptions = [
            {"speaker": speaker, "description": desc}
            for speaker, desc in variation['speaker_voice_descriptions'].items()
        ]
        
        audio_data.append({
            "transcription": variation['transcription'],
            "speaker_voice_descriptions": speaker_descriptions
        })
    
    # Initialize audio generator and generate audio
    audio_gen = AudioGenerator()
    results = await audio_gen.generate_audio_from_transcriptions(
        data=audio_data,
        output_dir=str(output_path),
        pause_duration_ms=500,
        preset_voices=False
    )
    
    # Save results to outputs directory with meaningful names
    if not OUTPUTS_DIR.exists():
        OUTPUTS_DIR.mkdir(parents=True)
        
    for i, result in enumerate(results):
        audio_path = list(result.keys())[0]
        new_path = OUTPUTS_DIR / f"variation_{i+1}_{Path(audio_path).name}"
        Path(audio_path).rename(new_path)
        results[i] = {str(new_path): result[audio_path]}
    
    return results


if __name__ == "__main__":
    main()
