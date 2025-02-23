import requests


BACKEND_URL = "http://localhost:8000"


class BackendServices:
    def __init__(self):
        pass

    def get_video_with_id_service(self, video_id: str):
        response = requests.get(f"{BACKEND_URL}/get_video_with_id/{video_id}")
        return response.json()


if __name__ == "__main__":
    backend_services = BackendServices()
    db_data = backend_services.get_db_data()
    print(db_data)
