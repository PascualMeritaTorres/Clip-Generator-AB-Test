import streamlit as st
from services.db_services import DbServices
from config import get_authenticated_service
from services.backend_services import BackendServices


def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("## Menu")  # Add a header for the menu
    page = st.sidebar.radio("Go to", ["Home", "Upload", "Statistics"])

    if page == "Home":
        st.write("## Welcome to AutoTrend!")
    elif page == "Upload":
        # File uploader
        st.write("## Upload a file")

        ################################## FILE UPLOADER ##################################
        uploaded_file = st.file_uploader("Choose a file", type=["mp4", "mp3", "txt"])

        if uploaded_file is not None:
            # Display file details
            st.write("Filename:", uploaded_file.name)
            st.write("File type:", uploaded_file.type)
            st.write("File size:", uploaded_file.size, "bytes")

            # Process the file based on its type
            if uploaded_file.type == "video/mp4":
                st.video(uploaded_file)
                if st.button("Process Video"):
                    st.write("Processing video...")
            elif uploaded_file.type == "audio/mpeg":
                st.audio(uploaded_file)
                if st.button("Process Audio"):
                    st.write("Processing audio...")
            elif uploaded_file.type == "text/plain":
                st.text(uploaded_file.read().decode("utf-8"))
                if st.button("Process Text"):
                    st.write("Processing text...")
        #####################################################################################

    elif page == "Statistics":
        # Get data from the database
        db_services = DbServices()
        db_data = db_services.get_db_data()

        st.write("## Video Statistics")

        ############################# TABLE OF VIDEO STATISTICS #############################
        col1, col2, col3 = st.columns([3, 2, 1])
        col1.markdown(
            "<p style='margin-bottom: 30px; margin-top: 50px'><strong>Video Name</strong></p>",
            unsafe_allow_html=True,
        )
        col2.markdown(
            "<p style='margin-bottom: 30px; margin-top: 50px'><strong>Video ID</strong></p>",
            unsafe_allow_html=True,
        )
        col3.markdown(
            "<p style='margin-bottom: 30px; margin-top: 50px'><strong></strong></p>",
            unsafe_allow_html=True,
        )
        for video in db_data.get("videos", []):
            col1, col2, col3 = st.columns([3, 2, 1])
            col1.markdown(
                f"<p style='margin-bottom: 0;'>{video['video_name']}</p>",
                unsafe_allow_html=True,
            )
            col2.markdown(
                f"<p style='margin-bottom: 0;'>{video['video_id']}</p>",
                unsafe_allow_html=True,
            )
            if col3.button("Go to Video", key=video["video_id"]):
                backend_services = BackendServices()
                video_details = backend_services.get_video_with_id_service(
                    video_id=video["video_id"]
                )
                st.write(f"Redirecting to video {video['video_id']}...")
            st.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
        #####################################################################################


if __name__ == "__main__":
    main()
