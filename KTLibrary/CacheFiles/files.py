import json

def read_cache_file(self, file_path: str):
    print("Read:", file_path)
    json_file = None
    try:
        with open(file_path, "r") as file:
            json_data = json.load(file)
            print("JSON data loaded successfully")
    except FileNotFoundError:
        print("File not found")
    except json.JSONDecodeError:
        print("Invalid JSON format.")
    return json_data