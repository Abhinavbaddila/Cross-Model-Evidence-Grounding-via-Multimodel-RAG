import json
import os

class ResultsSaver:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump([], f)

    def save(self, result):
        with open(self.path, "r") as f:
            data = json.load(f)
        data.append(result)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)
