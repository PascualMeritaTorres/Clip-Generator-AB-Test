import streamlit as st
from services.db_services import DbServices

# Comment out the import since we're ignoring authentication
# from config import get_authenticated_service
from services.backend_services import BackendServices
import os
import tempfile
import sys
from pathlib import Path

# Add the project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from input_handling.input_processor import InputProcessor
from youtube_interaction.youtube_receiver import YoutubeReceiver
from youtube_interaction.youtube_uploader import YoutubeUploader
from youtube_interaction.config import get_authenticated_service


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
                    "Target Platform", ["TikTok", "YouTube Shorts", "Instagram Reels"]
                )
                content_style = st.selectbox(
                    "Content Style",
                    ["Entertaining", "Educational", "News", "Commentary"],
                )

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
                            result = processor.process_input(
                                temp_path, uploaded_file.type
                            )
                            if result:
                                st.success("Content processed successfully!")

                                # Display generated clips
                                st.markdown("### Generated Content Variations")
                                for i, clip in enumerate(result.get("clips", [])):
                                    with st.expander(f"Clip {i+1}", expanded=True):
                                        st.markdown(
                                            f"**Transcript:** {clip.get('transcript', 'N/A')}"
                                        )
                                        st.markdown(
                                            f"**Duration:** {clip.get('duration', 'N/A')}s"
                                        )
                                        if clip.get("audio_url"):
                                            st.audio(clip["audio_url"])

                                        # Show variations
                                        st.markdown("#### Variations")
                                        for j, variation in enumerate(
                                            clip.get("variations", [])
                                        ):
                                            st.markdown(f"**Variation {j+1}**")
                                            st.json(variation)
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


if __name__ == "__main__":
    main()
