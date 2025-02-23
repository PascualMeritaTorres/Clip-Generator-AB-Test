import os
import json
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from config import CLIENT_SECRETS_FILE, SCOPES, API_SERVICE_NAME, API_VERSION


os.chdir(os.path.dirname(os.path.abspath(__file__)))


class YoutubeReceiver:

    def __init__(self, output_dir: str, youtube_authenticated_client):
        # Set the data folder
        self.output_dir = output_dir
        # Authorize the request and store authorization credentials.
        self.youtube_authenticated_client = youtube_authenticated_client
        # self.youtube_authenticated_client = self.get_authenticated_service()

    # Authorize the request and store authorization credentials.
    def get_authenticated_service(self):
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

    def get_video_details(self, video_id: str):
        request = self.youtube_authenticated_client.videos().list(
            part="snippet,statistics,contentDetails", id=video_id
        )
        response = request.execute()
        return response["items"][0]

    def read_video_ids(self, file_path: str):
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        # Extract the list of video IDs
        video_ids = [video["video_id"] for video in data["uploaded_videos"]]
        return video_ids

    def run_get_video_pipeline(self):
        video_details_result = {}
        for filename in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, filename)
            if os.path.isfile(file_path):
                try:
                    video_ids = self.read_video_ids(file_path=file_path)
                    for video_id in video_ids:
                        video_details = self.get_video_details(video_id=video_id)
                        video_details_result[video_id] = video_details
                    return video_details_result
                except Exception as e:
                    raise e


if __name__ == "__main__":
    youtuber_receiver = YoutubeReceiver(output_dir=os.path.join(os.getcwd(), "output"))
    video_details_result = youtuber_receiver.run_get_video_pipeline()
