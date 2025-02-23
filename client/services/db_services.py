import os
import json

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


class DbServices:
    def __init__(self):
        self.database_dir = "database"

    def get_db_data(self):
        db_videos = {}
        for filename in os.listdir(self.database_dir):
            file_path = os.path.join(self.database_dir, filename)
            if os.path.isfile(file_path):
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                    db_videos["videos"] = data["videos"]
        return db_videos


if __name__ == "__main__":
    db_services = DbServices()
    db_data = db_services.get_db_data()
    print(db_data)
