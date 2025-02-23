import os
import json
from typing import List
from youtube_interaction.youtube_receiver import YoutubeReceiver

database_dir = os.chdir(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)


class DbServices:
    def __init__(self):
        self.database_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir,
            "database/database.json",
        )

    def get_db_data(self):
        db_videos = {}
        with open(self.database_path, "r") as json_file:
            data = json.load(json_file)
            db_videos["videos"] = data["videos"]
        return db_videos

    def get_video_ratings(self, youtube_authenticated_client, video_ids: List[str]):
        video_statistcs = {}
        youtube_receiver = YoutubeReceiver(
            output_dir=os.path.join(os.getcwd(), "output"),
            youtube_authenticated_client=youtube_authenticated_client,
        )
        for video_id in video_ids:
            video_details = youtube_receiver.get_video_details(video_id=video_id)
            video_statistcs[video_id] = video_details
        return video_statistcs


if __name__ == "__main__":
    db_services = DbServices()
    db_data = db_services.get_db_data()
    print(db_data)
