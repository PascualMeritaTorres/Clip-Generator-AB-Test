import os
import time
import json
import random
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from config import (
    CLIENT_SECRETS_FILE,
    SCOPES,
    API_SERVICE_NAME,
    API_VERSION,
    MAX_RETRIES,
    RETRIABLE_EXCEPTIONS,
    RETRIABLE_STATUS_CODES,
)


os.chdir(os.path.dirname(os.path.abspath(__file__)))


class YoutubeUploader:

    def __init__(self, data_dir: str, youtube_authenticated_client):
        # Set the data folder
        self.data_dir = data_dir
        # Authorize the request and store authorization credentials.
        self.youtube_authenticated_client = youtube_authenticated_client
        # self.youtube_client = self.get_authenticated_service()

    # Authorize the request and store authorization credentials.
    def get_authenticated_service(self):
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

    def resumable_upload(self, request):
        response = None
        error = None
        retry = 0
        while response is None:
            try:
                status, response = request.next_chunk()
                if response is not None:
                    if "id" in response:
                        print(
                            'Video id "%s" was successfully uploaded.' % response["id"]
                        )
                    else:
                        exit(
                            "The upload failed with an unexpected response: %s"
                            % response
                        )
            except HttpError as e:
                if e.resp.status in RETRIABLE_STATUS_CODES:
                    error = "A retriable HTTP error %d occurred:\n%s" % (
                        e.resp.status,
                        e.content,
                    )
                else:
                    raise
            except RETRIABLE_EXCEPTIONS as e:
                error = "A retriable error occurred: %s" % e

            if error is not None:
                print(error)
                retry += 1
                if retry > MAX_RETRIES:
                    exit("No longer attempting to retry.")

                max_sleep = 2**retry
                sleep_seconds = random.random() * max_sleep
                print("Sleeping %f seconds and then retrying..." % sleep_seconds)
                time.sleep(sleep_seconds)

        return {"video_id": response["id"], "video_name": response["snippet"]["title"]}

    def initialize_upload(
        self,
        file_path: str,
        title: str,
        description: str,
        keywords: str,
        category: str,
        privacy_status: str,
    ):
        print(f"Initializing upload for {file_path} to YouTube...")
        tags = None
        if keywords:
            tags = keywords.split(",")

        body = dict(
            snippet=dict(
                title=title,
                description=description,
                tags=tags,
                categoryId=category,
            ),
            status=dict(privacyStatus=privacy_status),
        )

        # Call the API's videos.insert method to create and upload the video.
        insert_request = self.youtube_client.videos().insert(
            part=",".join(body.keys()),
            body=body,
            # The chunksize parameter specifies the size of each chunk of data, in
            # bytes, that will be uploaded at a time. Set a higher value for
            # reliable connections as fewer chunks lead to faster uploads. Set a lower
            # value for better recovery on less reliable connections.
            #
            # Setting 'chunksize' equal to -1 in the code below means that the entire
            # file will be uploaded in a single HTTP request. (If the upload fails,
            # it will still be retried where it left off.) This is usually a best
            # practice, but if you're using Python older than 2.6 or if you're
            # running on App Engine, you should set the chunksize to something like
            # 1024 * 1024 (1 megabyte).
            media_body=MediaFileUpload(file_path, chunksize=-1, resumable=True),
        )

        uploaded_video_data = self.resumable_upload(insert_request)

        return uploaded_video_data

    def run_upload_videos_pipeline(self):

        uploded_videos_response_list = []
        # Loop through all files in the data folder and upload them to YouTube
        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            if os.path.isfile(file_path):
                try:
                    uploaded_video_data = self.initialize_upload(
                        file_path=file_path,
                        title=f"{filename}#Shorts",
                        description=f"Upload of {filename}#Shorts",
                        keywords="",
                        category="22",
                        privacy_status="private",
                    )
                    uploded_videos_response_list.append(uploaded_video_data)
                except HttpError as e:
                    print("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))

        response = {"uploaded_videos": uploded_videos_response_list}

        # Ensure the output directory exists
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)

        # Write response into json file
        output_file_path = os.path.join(output_dir, "upload_responses.json")
        with open(output_file_path, "w") as json_file:
            json.dump(response, json_file, indent=4)
        print(f"Upload responses have been written to { {output_file_path}}")


if __name__ == "__main__":
    youtube_uploader = YoutubeUploader(data_dir=os.path.join(os.getcwd(), "data"))
    youtube_client = youtube_uploader.run_upload_videos_pipeline()
