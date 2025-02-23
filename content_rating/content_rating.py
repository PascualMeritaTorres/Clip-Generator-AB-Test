"""
Module to generate and load topics from transcription using Fal AI.
The module uses fal_client to generate topics and saves the topics into a JSON file.
Other modules (e.g. add_images.py and add_sound_effects.py) can load the topics from this file.
"""

from typing import List, Dict
import json
import os
import time
import logging
import fal_client  # Ensure correct installation via pipenv install fal-client
from collections import OrderedDict


# Logging configuration for generate_topics
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Global filter to suppress any log containing "HTTP Request:" in its message.
class SuppressHttpRequestFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "HTTP Request:" not in record.getMessage()


# Add the filter to the root logger so it applies to all log messages.
logging.getLogger().addFilter(SuppressHttpRequestFilter())

# Additionally, ensure that fal_client's logger uses WARNING level.
logging.getLogger("fal_client").setLevel(logging.WARNING)


def generate_rating(videos_stats: dict):
    sorted_videos_stats = OrderedDict(
        sorted(
            videos_stats.items(),
            key=lambda item: int(item[1]["viewCount"]),
            reverse=True,
        )
    )
    return sorted_videos_stats
    # videos_stats_str = str(videos_stats)
    # base_prompt = f"""
    # You will receive a list of video details in the form of a dictionary. These are videos that come from Youtube. Each video is listed with a unique id. Based,
    # on the video statistics and content, you need to rate each video on a scale of 1 to 5. The rating should be based on the video's quality, relevance, and
    # engagement. The rating should be an integer between 1 and 5. Eseentially, you need to put numbers depending on the viewCount, so, the video with the highest viewCount
    # will have the score of 5, the rest will have a lower score

    # Rules:
    # 1. The rating should be an integer between 1 and 5.
    # 2. If you are unsure about a video, you can skip it by not providing a rating.
    # 3. The ratings should be stored in a dictionary where the key is the video id and the value is the rating.

    # Return ONLY a JSON where:
    # - Each key is a video id.
    # - Each value is the rating for that video.

    # Example:
    #     Input: ""6fZ8hEQgXqE":
    #     "viewCount":"1"
    #     "likeCount":"0"
    #     "dislikeCount":"0"
    #     "favoriteCount":"0"
    #     "commentCount":"0"
    #     ,
    #     "Icgtl2HIZjs":
    #     "viewCount":"3"
    #     "likeCount":"0"
    #     "dislikeCount":"0"
    #     "favoriteCount":"0"
    #     "commentCount":"0"
    #     "

    #     In this case, the video with id Icgtl2HIZjs has 3 views whilst 6fZ8hEQgXqE has 1, so we you MUST attribute a score of 5 to Icgtl2HIZjs, and you can give a score of 1 to 6fZ8hEQgXqE
    #     Output: "6fZ8hEQgXqE": 1, "Icgtl2HIZjs": 5

    # Here is the dictionary of video details:
    # Videos: {videos_stats_str}
    # """

    # def on_queue_update(update: fal_client.InProgress) -> None:
    #     """Callback to log progress of image generation.

    #     Disabled logging to avoid printing HTTP request logs.
    #     """
    #     pass

    # max_attempts = 3
    # for attempt in range(max_attempts):
    #     logging.info(f"generate_ratings: Attempt {attempt + 1}")
    #     try:
    #         result = fal_client.subscribe(
    #             "fal-ai/any-llm",
    #             arguments={
    #                 "model": "anthropic/claude-3.5-sonnet",
    #                 "prompt": (
    #                     base_prompt + "\n\nRemember: Output only valid JSON."
    #                     if attempt > 0
    #                     else base_prompt
    #                 ),
    #             },
    #             with_logs=True,
    #             on_queue_update=on_queue_update,
    #         )
    #         result_json = json.loads(result["output"])
    #         assert isinstance(result_json, dict), "LLM response must be a dictionary"
    #         return result_json
    #     except Exception as e:
    #         logging.error(
    #             f"generate_ratings: Attempt {attempt + 1} failed with error: {e}"
    #         )
    #         if attempt == max_attempts - 1:
    #             raise RuntimeError(
    #                 f"Failed to generate ratings after {max_attempts} attempts"
    #             ) from e
    # raise RuntimeError("generate_ratings: Unexpected error")


if __name__ == "__main__":
    # Example usage
    videos_stats = {
        "6fZ8hEQgXqE": {
            "viewCount": "1",
            "likeCount": "0",
            "dislikeCount": "0",
            "favoriteCount": "0",
            "commentCount": "0",
        },
        "Icgtl2HIZjs": {
            "viewCount": "3",
            "likeCount": "0",
            "dislikeCount": "0",
            "favoriteCount": "0",
            "commentCount": "0",
        },
    }
    ratings = generate_rating(videos_stats)
    logging.info(f"Ratings: {ratings}")
    print(ratings)
